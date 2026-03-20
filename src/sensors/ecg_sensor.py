"""
ecg_sensor.py
=============
Simulates an ECG sensor streaming data over a serial port.

Signal properties (matches slide deck):
  - Amplitude: 0.5 - 4 mV
  - Bandwidth:  0.05 - 150 Hz
  - Sample rate: 500 Hz (common clinical ECG rate)
  - Heart rate:  60 - 100 BPM (configurable)

Packet format (8 bytes per sample):
  [0xAA] [0xBB] [timestamp_ms: 4 bytes, big-endian uint32]
  [sample_mv: 2 bytes, big-endian int16, scaled x1000] [0xFF]

  Preamble : 0xAA 0xBB  (2 bytes) - sync / framing
  Timestamp: uint32      (4 bytes) - milliseconds since stream start
  Sample   : int16       (2 bytes) - ECG voltage in microvolts (uV)
             e.g. 1.25 mV -> 1250 uV -> int16(1250)
  Footer   : 0xFF        (1 byte)  - end of packet

Usage (standalone):
    python ecg_sensor.py --port COM3 --baud 115200 --hr 72
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
    PACKET_SIZE     = 8         # 2 (preamble) + 4 (ts) + 2 (sample) + 1 (footer) = 9
                                # Note: struct.pack gives 6 bytes, total = 9

    # Gaussian parameters for PQRST: (amplitude_mV, center_fraction, width)
    # center_fraction is position within one cardiac cycle [0.0, 1.0]
    _PQRST_GAUSSIANS = [
        ( 0.20, 0.10, 0.025),   # P wave
        (-0.05, 0.18, 0.010),   # Q wave (small negative deflection)
        ( 2.50, 0.20, 0.008),   # R wave (tall positive spike)
        (-0.35, 0.23, 0.010),   # S wave (negative after R)
        ( 0.35, 0.35, 0.030),   # T wave
    ]

    def __init__(self, port: str = None, heart_rate_bpm: int = 72,
                 noise_mv: float = 0.04, baud_rate: int = BAUD_RATE,
                 serial_override=None):
        """
        Args:
            port             : Serial port string (e.g. 'COM3'). Ignored when
                               serial_override is provided.
            heart_rate_bpm   : Simulated heart rate in beats per minute
            noise_mv         : Standard deviation of additive Gaussian noise (mV)
            baud_rate        : Serial baud rate
            serial_override  : A MockSerial or pyserial Serial object. When set,
                               no real port is opened.
        """
        self.port             = port or "MOCK"
        self.heart_rate_bpm   = heart_rate_bpm
        self.noise_mv         = noise_mv
        self.baud_rate        = baud_rate
        self._serial_override = serial_override

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
        Synthesized as a sum of Gaussians over a repeating cardiac cycle.
        """
        cycle_duration = self._cardiac_cycle_duration_s()
        # Phase within the current cycle [0.0, 1.0)
        phase = (t_s % cycle_duration) / cycle_duration

        voltage = 0.0
        for amp, center, width in self._PQRST_GAUSSIANS:
            exponent = -((phase - center) ** 2) / (2 * width ** 2)
            voltage += amp * math.exp(exponent)

        # Additive white Gaussian noise
        voltage += random.gauss(0, self.noise_mv)

        # Slow baseline wander (breathing artifact ~0.2 Hz)
        voltage += 0.05 * math.sin(2 * math.pi * 0.2 * t_s)

        return voltage

    # ------------------------------------------------------------------
    # Packet framing
    # ------------------------------------------------------------------

    def _build_packet(self, timestamp_ms: int, sample_mv: float) -> bytes:
        """
        Pack one ECG sample into the wire format:
          0xAA 0xBB | uint32 timestamp_ms | int16 sample_uV | 0xFF
        """
        sample_uv = int(sample_mv * 1000)                   # mV -> uV, int16 range ±32767
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
              f"Noise: {self.noise_mv} mV")

    def disconnect(self):
        """Close the serial port and stop streaming."""
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
        print("[ECG]  Disconnected.")

    def stream(self):
        """
        Blocking loop — generates and transmits ECG samples at SAMPLE_RATE_HZ.
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
        description="ECG Sensor Simulator — streams synthetic ECG over serial"
    )
    parser.add_argument("--port",  default="COM3",    help="Serial port (default: COM3)")
    parser.add_argument("--baud",  default=115200, type=int, help="Baud rate (default: 115200)")
    parser.add_argument("--hr",    default=72,    type=int, help="Heart rate BPM (default: 72)")
    parser.add_argument("--noise", default=0.04,  type=float, help="Noise std dev mV (default: 0.04)")
    args = parser.parse_args()

    sensor = ECGSensor(
        port           = args.port,
        heart_rate_bpm = args.hr,
        noise_mv       = args.noise,
        baud_rate      = args.baud,
    )
    sensor.connect()
    sensor.stream()


if __name__ == "__main__":
    main()
