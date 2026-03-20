"""
spo2_sensor.py
==============
Simulates a pulse oximeter (SpO2) sensor streaming PPG data over serial.

Signal properties (matches slide deck):
  - Amplitude: 1 - 10 mV AC component
  - Bandwidth:  0.5 - 10 Hz
  - Sample rate: 100 Hz (typical for PPG)
  - SpO2 range: 94 - 100 % (configurable)

The PPG waveform has two components:
  - DC component: slow-moving baseline (~1.0 V in real hardware, normalized here)
  - AC component: pulsatile signal riding on top of DC (~1-10 mV amplitude)

Packet format (11 bytes per sample):
  [0xCC] [0xDD]                          2 bytes  - preamble / sync
  [timestamp_ms: uint32, big-endian]     4 bytes  - ms since stream start
  [ppg_red: int16, big-endian]           2 bytes  - red channel (660 nm), scaled uV
  [ppg_ir: int16, big-endian]            2 bytes  - IR channel (940 nm), scaled uV
  [spo2_pct: uint8]                      1 byte   - SpO2 percentage (0-100)
  [0xFE]                                 1 byte   - footer

  Total: 12 bytes per packet

SpO2 is estimated from the ratio of red/IR AC:DC components (R value).
Students will recompute SpO2 from the raw PPG channels in their TODO.

Usage (standalone):
    python spo2_sensor.py --port COM5 --baud 115200 --spo2 98
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


class SpO2Sensor:
    """
    Generates synthetic PPG (photoplethysmography) waveforms for red and
    infrared channels and streams packets over a serial port.

    Red channel   (660 nm) : more absorbed by deoxygenated Hb
    IR channel    (940 nm) : more absorbed by oxygenated Hb
    The ratio of (AC/DC)_red to (AC/DC)_ir encodes SpO2.
    """

    SAMPLE_RATE_HZ = 100
    BAUD_RATE      = 115200
    PREAMBLE       = b'\xCC\xDD'
    FOOTER         = b'\xFE'
    PACKET_FORMAT  = '>I hh B'  # uint32 ts, int16 red, int16 ir, uint8 spo2

    # Empirical calibration curve coefficients (simplified Beer-Lambert):
    # R = (AC/DC)_red / (AC/DC)_ir
    # SpO2 ≈ A - B * R  (linearized approximation, A=110, B=25 is common)
    _CAL_A = 110.0
    _CAL_B = 25.0

    def __init__(self, port: str = None, spo2_pct: float = 98.0,
                 heart_rate_bpm: int = 72, noise_scale: float = 0.005,
                 baud_rate: int = BAUD_RATE, serial_override=None):
        """
        Args:
            port             : Serial port string. Ignored when serial_override is set.
            spo2_pct         : Target SpO2 percentage (94.0 - 100.0)
            heart_rate_bpm   : Underlying pulse rate for PPG morphology
            noise_scale      : Fractional noise on PPG amplitude
            baud_rate        : Serial baud rate
            serial_override  : MockSerial or pyserial Serial object.
        """
        self.port             = port or "MOCK"
        self.spo2_pct         = max(80.0, min(100.0, spo2_pct))
        self.heart_rate_bpm   = heart_rate_bpm
        self.noise_scale      = noise_scale
        self.baud_rate        = baud_rate
        self._serial_override = serial_override

        self._serial          = None
        self._running         = False
        self._start_time_ms   = None

        # Derive the red/IR ratio that corresponds to target SpO2
        # R = (A - SpO2) / B
        self._target_R = (self._CAL_A - self.spo2_pct) / self._CAL_B

    # ------------------------------------------------------------------
    # Waveform synthesis
    # ------------------------------------------------------------------

    def _pulse_phase(self, t_s: float) -> float:
        """
        Return the normalized pulse phase [0, 1) at time t_s.
        """
        cycle_s = 60.0 / self.heart_rate_bpm
        return (t_s % cycle_s) / cycle_s

    def _ppg_ac_component(self, phase: float) -> float:
        """
        Return the AC (pulsatile) component of a PPG waveform.
        Shape: rapid systolic upstroke, dicrotic notch, slower diastolic decay.
        Amplitude normalized to 1.0 peak.
        """
        if phase < 0.15:
            # Systolic upstroke
            ac = math.sin(math.pi * phase / 0.15) ** 2
        elif phase < 0.25:
            # Dicrotic notch region — slight dip
            notch_phase = (phase - 0.15) / 0.10
            ac = 0.85 - 0.10 * math.sin(math.pi * notch_phase)
        elif phase < 0.45:
            # Diastolic peak (smaller secondary hump)
            diastole_phase = (phase - 0.25) / 0.20
            ac = 0.75 + 0.10 * math.sin(math.pi * diastole_phase)
        else:
            # Exponential decay back to baseline
            ac = 0.75 * math.exp(-5.0 * (phase - 0.45))

        return max(0.0, ac)

    def _channel_sample(self, t_s: float, ac_amplitude: float,
                        dc_level: float = 1000.0,
                        motion_phase_offset: float = 0.0) -> float:
        """
        Return one channel sample in normalized units (will be scaled to int16).
          total = DC + AC_component * amplitude + noise + motion

        motion_phase_offset gives each channel an independent motion phase so
        the artifact does not coherently bias the R-ratio estimate.
        """
        phase = self._pulse_phase(t_s)
        ac    = self._ppg_ac_component(phase) * ac_amplitude
        noise = random.gauss(0, ac_amplitude * self.noise_scale)
        # Slow motion artifact (0.3 Hz) — small amplitude, independent phase per channel
        motion = ac_amplitude * 0.008 * math.sin(2 * math.pi * 0.3 * t_s + motion_phase_offset)
        # Subtract AC: during systole more blood absorbs more light → lower intensity.
        # This gives the familiar upward-peak PPG shape when plotted (dc - ac peaks upward
        # because ac is maximum when the vessel is most full).
        # Note: we negate so the pulsatile peak appears as a positive deflection on screen.
        return dc_level - ac + noise + motion

    def _compute_channels(self, t_s: float):
        """
        Compute red and IR channel samples.

        R = (AC/DC)_red / (AC/DC)_ir = self._target_R

        Both channels share the same DC level so the ratio recovers cleanly:
          (AC/DC)_ir = k  =>  AC_ir = k * DC
          (AC/DC)_red = target_R * k  =>  AC_red = target_R * k * DC

        Modulation index k = 0.05 (5%), typical for reflective PPG sensors.
        """
        dc      = 1000.0
        ac_ir   = dc * 0.05                   # 5% modulation
        ac_red  = ac_ir * self._target_R      # enforces target R-ratio

        ir_sample  = self._channel_sample(t_s, ac_ir,  dc_level=dc, motion_phase_offset=0.0)
        red_sample = self._channel_sample(t_s, ac_red, dc_level=dc, motion_phase_offset=1.1)

        # Scale to int16 (x10 keeps one decimal place of precision)
        ir_int  = max(-32767, min(32767, int(ir_sample  * 10)))
        red_int = max(-32767, min(32767, int(red_sample * 10)))

        return red_int, ir_int

    # ------------------------------------------------------------------
    # Packet framing
    # ------------------------------------------------------------------

    def _build_packet(self, timestamp_ms: int, red: int, ir: int) -> bytes:
        """
        Pack one SpO2 sample:
          0xCC 0xDD | uint32 ts | int16 red | int16 ir | uint8 spo2 | 0xFE
        """
        # Add small physiological variation (+/- 1%) to reported SpO2
        reported_spo2 = int(self.spo2_pct + random.gauss(0, 0.3))
        reported_spo2 = max(0, min(100, reported_spo2))

        payload = struct.pack(self.PACKET_FORMAT,
                              timestamp_ms, red, ir, reported_spo2)
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
        print(f"[SpO2] Connected on {self.port} @ {self.baud_rate} baud")
        print(f"[SpO2] SpO2 target: {self.spo2_pct:.1f}% | "
              f"Heart rate: {self.heart_rate_bpm} BPM | "
              f"Sample rate: {self.SAMPLE_RATE_HZ} Hz")

    def disconnect(self):
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
        print("[SpO2] Disconnected.")

    def stream(self):
        """
        Blocking loop — generates and transmits SpO2 packets at SAMPLE_RATE_HZ.
        Call connect() before this. Stop with KeyboardInterrupt.
        """
        if not self._serial or not self._serial.is_open:
            raise RuntimeError("Call connect() before stream().")

        self._running = True
        interval_s    = 1.0 / self.SAMPLE_RATE_HZ
        t_s           = 0.0

        print("[SpO2] Streaming... (Ctrl+C to stop)")
        try:
            while self._running:
                loop_start = time.perf_counter()

                timestamp_ms    = int(time.time() * 1000) - self._start_time_ms
                red, ir         = self._compute_channels(t_s)
                packet          = self._build_packet(timestamp_ms, red, ir)

                self._serial.write(packet)

                t_s += interval_s

                elapsed = time.perf_counter() - loop_start
                sleep_s = interval_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except KeyboardInterrupt:
            print("\n[SpO2] Stream interrupted by user.")
        finally:
            self.disconnect()


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SpO2 Sensor Simulator — streams synthetic PPG over serial"
    )
    parser.add_argument("--port",  default="COM5",   help="Serial port (default: COM5)")
    parser.add_argument("--baud",  default=115200, type=int)
    parser.add_argument("--spo2",  default=98.0,  type=float, help="SpO2 %% (default: 98.0)")
    parser.add_argument("--hr",    default=72,    type=int,   help="Heart rate BPM (default: 72)")
    parser.add_argument("--noise", default=0.005, type=float, help="Noise scale (default: 0.005)")
    args = parser.parse_args()

    sensor = SpO2Sensor(
        port           = args.port,
        spo2_pct       = args.spo2,
        heart_rate_bpm = args.hr,
        noise_scale    = args.noise,
        baud_rate      = args.baud,
    )
    sensor.connect()
    sensor.stream()


if __name__ == "__main__":
    main()
