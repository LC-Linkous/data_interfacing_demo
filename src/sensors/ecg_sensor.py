"""
ecg_sensor.py
=============
Simulates an ECG sensor streaming data over a serial port.

Signal properties (matches slide deck):
  - Amplitude:  0.5 - 4 mV
  - Bandwidth:  0.05 - 150 Hz
  - Sample rate: 500 Hz (common clinical ECG rate)
  - Heart rate:  60 - 100 BPM (configurable)

Noise model (see add_realistic_noise):
  The clean PQRST waveform is corrupted with the same artifact types you
  see on a real bedside monitor, so the downstream filters actually have
  something to remove:
    - white sensor/thermal noise      (broadband)
    - 60 Hz mains + 120/180 Hz harmonics  (a single notch only kills 60 Hz)
    - baseline wander                 (multi-tone; part of it sits INSIDE the
                                       passband, so a high-pass can't fully
                                       remove it)
    - motion-artifact bursts          (broadband + a low-frequency swing that
                                       overlaps the QRS band -> a band-pass
                                       CANNOT remove these)
  The overall amount is chosen by NOISE_LEVEL ("off"|"low"|"medium"|"high").

Packet format (9 bytes per sample):
  [0xAA] [0xBB] [timestamp_ms: 4 bytes, big-endian uint32]
  [sample_uV: 2 bytes, big-endian int16] [0xFF]

  Preamble : 0xAA 0xBB  (2 bytes) - sync / framing
  Timestamp: uint32      (4 bytes) - milliseconds since stream start
  Sample   : int16       (2 bytes) - ECG voltage in microvolts (uV)
             e.g. 1.25 mV -> 1250 uV -> int16(1250)
  Footer   : 0xFF        (1 byte)  - end of packet

Usage (standalone):
    python ecg_sensor.py --port COM3 --baud 115200 --hr 72 --level medium
"""

try:
    import serial
except ImportError:
    serial = None   # not needed when using serial_override (mock transport)
import struct
import time
import math
import random
import argparse


class ECGSensor:
    """
    Generates a synthetic ECG waveform and streams packets over a serial port.

    The PQRST morphology is synthesized using a sum of Gaussian curves,
    a technique commonly used in ECG simulation literature.
    """

    SAMPLE_RATE_HZ  = 500       # samples per second
    BAUD_RATE       = 115200
    PREAMBLE        = b'\xAA\xBB'
    FOOTER          = b'\xFF'
    PACKET_FORMAT   = '>I h'    # big-endian: uint32 timestamp, int16 sample
    PACKET_SIZE     = 9         # 2 (preamble) + 4 (ts) + 2 (sample) + 1 (footer)

    # Gaussian parameters for PQRST: (amplitude_mV, center_fraction, width)
    # center_fraction is position within one cardiac cycle [0.0, 1.0]
    _PQRST_GAUSSIANS = [
        ( 0.20, 0.10, 0.025),   # P wave
        (-0.05, 0.18, 0.010),   # Q wave (small negative deflection)
        ( 2.50, 0.20, 0.008),   # R wave (tall positive spike)
        (-0.35, 0.23, 0.010),   # S wave (negative after R)
        ( 0.35, 0.35, 0.030),   # T wave
    ]

    # --------------------------------------------------------------------------
    # Noise presets. Each value is an amplitude in mV.
    #   white_mv  : std-dev of broadband Gaussian noise
    #   mains_mv  : amplitude of the 60 Hz fundamental (harmonics scale off this)
    #   wander_mv : amplitude of the dominant baseline-wander tone
    #   motion_mv : peak amplitude of a motion-artifact burst
    # R-wave is ~2.5 mV, so "high" motion (~0.9 mV) is a big, obvious excursion.
    # --------------------------------------------------------------------------
    #
    # Tuning note: at the paired 0.5-70 Hz band-pass + 60 Hz notch, "low" leaves
    # a signal the reference detector still reads correctly (~72 BPM), while
    # "medium"/"high" intentionally overwhelm naive percentile peak-detection --
    # a built-in lesson that filtering alone is not enough.
    NOISE_PRESETS = {
        "off":    dict(white_mv=0.000, mains_mv=0.00, wander_mv=0.00, motion_mv=0.00),
        "low":    dict(white_mv=0.012, mains_mv=0.05, wander_mv=0.04, motion_mv=0.10),
        "medium": dict(white_mv=0.030, mains_mv=0.10, wander_mv=0.08, motion_mv=0.30),
        "high":   dict(white_mv=0.060, mains_mv=0.18, wander_mv=0.18, motion_mv=0.70),
    }

    def __init__(self, port: str = None, heart_rate_bpm: int = 72,
                 noise_level: str = "medium", baud_rate: int = BAUD_RATE,
                 serial_override=None):
        """
        Args:
            port             : Serial port string (e.g. 'COM3'). Ignored when
                               serial_override is provided.
            heart_rate_bpm   : Simulated heart rate in beats per minute
            noise_level      : "off" | "low" | "medium" | "high"
            baud_rate        : Serial baud rate
            serial_override  : A MockSerial or pyserial Serial object. When set,
                               no real port is opened.
        """
        self.port             = port or "MOCK"
        self.heart_rate_bpm   = heart_rate_bpm
        self.baud_rate        = baud_rate
        self._serial_override = serial_override

        if noise_level not in self.NOISE_PRESETS:
            raise ValueError(f"noise_level must be one of "
                             f"{list(self.NOISE_PRESETS)}, got {noise_level!r}")
        self.noise_level = noise_level
        self._noise      = dict(self.NOISE_PRESETS[noise_level])

        # Mains gets a fixed-but-arbitrary phase so successive runs differ a little
        self._mains_phase = random.uniform(0, 2 * math.pi)

        # Motion-artifact burst state (see _motion_artifact)
        self._motion_remaining = 0      # samples left in the current burst
        self._motion_len       = 1      # length of the current burst (samples)
        self._motion_amp       = 0.0    # amplitude of the current burst (mV)
        self._motion_freq      = 5.0    # in-band swing frequency of the burst (Hz)
        # Trigger a new burst on average once every ~3.5 s
        self._motion_prob      = 1.0 / (3.5 * self.SAMPLE_RATE_HZ)

        self._serial          = None
        self._running         = False
        self._start_time_ms   = None

    # ------------------------------------------------------------------
    # Waveform synthesis
    # ------------------------------------------------------------------

    def _cardiac_cycle_duration_s(self) -> float:
        return 60.0 / self.heart_rate_bpm

    def _ecg_sample(self, t_s: float) -> float:
        """
        Return a single ECG voltage (mV) at time t_s seconds.
        Synthesized as a sum of Gaussians over a repeating cardiac cycle,
        plus realistic sensor noise (see add_realistic_noise).
        """
        cycle_duration = self._cardiac_cycle_duration_s()
        # Phase within the current cycle [0.0, 1.0)
        phase = (t_s % cycle_duration) / cycle_duration

        voltage = 0.0
        for amp, center, width in self._PQRST_GAUSSIANS:
            exponent = -((phase - center) ** 2) / (2 * width ** 2)
            voltage += amp * math.exp(exponent)

        voltage += self.add_realistic_noise(t_s)
        return voltage

    # ------------------------------------------------------------------
    # Noise / artifact model
    # ------------------------------------------------------------------

    def add_realistic_noise(self, t_s: float) -> float:
        """
        Return the total additive noise (mV) at time t_s, per the active
        NOISE_PRESETS entry. Sum of four independent artifact sources:

          1. white     - broadband thermal/sensor noise. The band-pass removes
                          most of its power, but it's always present.
          2. mains      - 60 Hz + 120 Hz + 180 Hz. The 60 Hz notch kills only
                          the fundamental; the harmonics teach that one notch
                          is not enough (and why clinical front-ends chain
                          several, or use a comb filter).
          3. wander     - baseline drift. Two tones: 0.2 Hz respiration (below a
                          0.5 Hz high-pass corner) and a 0.7 Hz drift that sits
                          INSIDE the passband, so the high-pass can't fully
                          remove it.
          4. motion     - intermittent bursts whose energy overlaps the QRS
                          band. A band-pass CANNOT remove these -- this is the
                          key lesson that filtering has limits.
        """
        p = self._noise
        n = 0.0

        # 1. Broadband white noise
        if p["white_mv"] > 0.0:
            n += random.gauss(0.0, p["white_mv"])

        # 2. Power-line interference: 60 Hz fundamental + harmonics
        if p["mains_mv"] > 0.0:
            a = p["mains_mv"]
            ph = self._mains_phase
            n += a        * math.sin(2 * math.pi * 60.0  * t_s + ph)
            n += 0.40 * a * math.sin(2 * math.pi * 120.0 * t_s + ph)
            n += 0.20 * a * math.sin(2 * math.pi * 180.0 * t_s + ph)

        # 3. Baseline wander (respiration + in-band drift)
        if p["wander_mv"] > 0.0:
            w = p["wander_mv"]
            n += w        * math.sin(2 * math.pi * 0.20 * t_s)          # below HP corner
            n += 0.5 * w  * math.sin(2 * math.pi * 0.70 * t_s + 1.0)    # inside passband

        # 4. Motion-artifact bursts
        if p["motion_mv"] > 0.0:
            n += self._motion_artifact(t_s, p["motion_mv"])

        return n

    def _motion_artifact(self, t_s: float, peak_mv: float) -> float:
        """
        Stateful motion-artifact generator. Occasionally starts a burst that
        lasts ~0.15-0.6 s. Each burst is a low-frequency swing (1.5-5 Hz, in the
        signal band) plus broadband noise, shaped by a smooth Hann envelope so it
        ramps on and off rather than clicking. Because the swing overlaps the
        signal band, the 0.5-70 Hz band-pass cannot remove it.
        """
        # Not currently in a burst -> maybe start one
        if self._motion_remaining <= 0:
            if random.random() >= self._motion_prob:
                return 0.0
            self._motion_len       = random.randint(int(0.15 * self.SAMPLE_RATE_HZ),
                                                     int(0.60 * self.SAMPLE_RATE_HZ))
            self._motion_remaining = self._motion_len
            self._motion_amp       = peak_mv * random.uniform(0.5, 1.0)
            self._motion_freq      = random.uniform(1.5, 5.0)   # slow in-band heave

        # Position within the burst -> raised-cosine (Hann) envelope
        pos = 1.0 - (self._motion_remaining / self._motion_len)
        env = 0.5 * (1.0 - math.cos(2.0 * math.pi * pos))
        self._motion_remaining -= 1

        swing     = math.sin(2.0 * math.pi * self._motion_freq * t_s)
        broadband = random.gauss(0.0, 0.15)
        return self._motion_amp * env * (swing + broadband)

    # ------------------------------------------------------------------
    # Packet framing
    # ------------------------------------------------------------------

    def _build_packet(self, timestamp_ms: int, sample_mv: float) -> bytes:
        """
        Pack one ECG sample into the wire format:
          0xAA 0xBB | uint32 timestamp_ms | int16 sample_uV | 0xFF
        """
        sample_uv = int(sample_mv * 1000)                   # mV -> uV, int16 range +-32767
        sample_uv = max(-32767, min(32767, sample_uv))      # clamp

        payload = struct.pack(self.PACKET_FORMAT, timestamp_ms, sample_uv)
        return self.PREAMBLE + payload + self.FOOTER

    # ------------------------------------------------------------------
    # Serial streaming
    # ------------------------------------------------------------------

    def connect(self):
        """Open the serial port (or attach the mock serial override)."""
        if self._serial_override is not None:
            self._serial = self._serial_override
        else:
            if serial is None:
                raise RuntimeError(
                    "pyserial is not installed. Either pip install pyserial "
                    "or pass serial_override=<MockSerial> to use the mock transport."
                )
            self._serial = serial.Serial(
                port     = self.port,
                baudrate = self.baud_rate,
                bytesize = serial.EIGHTBITS,
                parity   = serial.PARITY_NONE,
                stopbits = serial.STOPBITS_ONE,
                timeout  = 1,
            )
        self._start_time_ms = int(time.time() * 1000)
        print(f"[ECG]  Connected on {self.port} @ {self.baud_rate} baud")
        print(f"[ECG]  Heart rate: {self.heart_rate_bpm} BPM | "
              f"Sample rate: {self.SAMPLE_RATE_HZ} Hz | "
              f"Noise level: {self.noise_level}")

    def disconnect(self):
        """Close the serial port and stop streaming."""
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
        print("[ECG]  Disconnected.")

    def stream(self):
        """
        Blocking loop - generates and transmits ECG samples at SAMPLE_RATE_HZ.
        Call connect() before this. Stop with KeyboardInterrupt.
        """
        if not self._serial or not self._serial.is_open:
            raise RuntimeError("Call connect() before stream().")

        self._running = True
        interval_s    = 1.0 / self.SAMPLE_RATE_HZ
        t_s           = 0.0

        print("[ECG]  Streaming... (Ctrl+C to stop)")
        try:
            while self._running:
                loop_start = time.perf_counter()

                timestamp_ms = int(time.time() * 1000) - self._start_time_ms
                sample_mv    = self._ecg_sample(t_s)
                packet       = self._build_packet(timestamp_ms, sample_mv)

                self._serial.write(packet)

                t_s += interval_s

                # Maintain sample rate by sleeping the remaining interval
                elapsed = time.perf_counter() - loop_start
                sleep_s = interval_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except KeyboardInterrupt:
            print("\n[ECG]  Stream interrupted by user.")
        finally:
            self.disconnect()


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ECG Sensor Simulator - streams synthetic ECG over serial"
    )
    parser.add_argument("--port",  default="COM3",    help="Serial port (default: COM3)")
    parser.add_argument("--baud",  default=115200, type=int, help="Baud rate (default: 115200)")
    parser.add_argument("--hr",    default=72,     type=int, help="Heart rate BPM (default: 72)")
    parser.add_argument("--level", default="medium",
                        choices=list(ECGSensor.NOISE_PRESETS),
                        help="Noise level (default: medium)")
    args = parser.parse_args()

    sensor = ECGSensor(
        port           = args.port,
        heart_rate_bpm = args.hr,
        noise_level    = args.level,
        baud_rate      = args.baud,
    )
    sensor.connect()
    sensor.stream()


if __name__ == "__main__":
    main()