"""
bp_sensor.py
============
Simulates an invasive arterial blood pressure (IBP) sensor streaming over serial.

Signal properties (matches slide deck):
  - Amplitude:   10 - 300 mV (from pressure transducer Wheatstone bridge)
  - Bandwidth:   DC - 40 Hz
  - Sample rate: 200 Hz (standard for arterial waveform fidelity)
  - Pressure:    mmHg (typical adult: systolic 120, diastolic 80)

The arterial waveform includes:
  - Rapid systolic upstroke
  - Incisura (dicrotic notch) — aortic valve closure, key clinical landmark
  - Diastolic decay (Windkessel exponential)
  - Slow respiratory variation (~0.25 Hz, mimics intrathoracic pressure)

Packet format (11 bytes per sample):
  [0xBB] [0xCC]                         2 bytes  - preamble
  [timestamp_ms: uint32, big-endian]    4 bytes  - ms since stream start
  [pressure_mmhg: int16, big-endian]    2 bytes  - pressure in 0.1 mmHg units
                                                   e.g. 120.5 mmHg -> 1205
  [systolic: uint8]                     1 byte   - last detected systolic (mmHg)
  [diastolic: uint8]                    1 byte   - last detected diastolic (mmHg)
  [0xEE]                                1 byte   - footer
  Total: 11 bytes

Students will detect systolic/diastolic peaks themselves in their TODO.
The sensor reports them as a convenience for cross-checking.

Usage (standalone):
    python bp_sensor.py --port COM7 --baud 115200 --sys 120 --dia 80
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


class InvasiveBPSensor:
    """
    Generates a synthetic arterial blood pressure waveform and streams
    packets over a serial port.

    Waveform is synthesized using a physiologically plausible model:
      - Systolic upstroke: half-sine rise
      - Peak & early decay: smooth rolloff
      - Dicrotic notch: brief dip at ~1/3 of cycle
      - Diastolic decay: exponential decay (Windkessel model)
    """

    SAMPLE_RATE_HZ = 200
    BAUD_RATE      = 115200
    PREAMBLE       = b'\xBB\xCC'
    FOOTER         = b'\xEE'
    PACKET_FORMAT  = '>I h BB'  # uint32 ts, int16 pressure*10, uint8 sys, uint8 dia

    def __init__(self, port: str = None,
                 systolic_mmhg: float  = 120.0,
                 diastolic_mmhg: float = 80.0,
                 heart_rate_bpm: int   = 72,
                 noise_mmhg: float     = 1.5,
                 baud_rate: int        = BAUD_RATE,
                 serial_override       = None):
        """
        Args:
            port             : Serial port string. Ignored when serial_override is set.
            systolic_mmhg    : Peak systolic pressure in mmHg
            diastolic_mmhg   : Diastolic (baseline) pressure in mmHg
            heart_rate_bpm   : Heart rate controlling cycle duration
            noise_mmhg       : Std dev of additive pressure noise (mmHg)
            baud_rate        : Serial baud rate
            serial_override  : MockSerial or pyserial Serial object.
        """
        self.port             = port or "MOCK"
        self.systolic_mmhg    = systolic_mmhg
        self.diastolic_mmhg   = diastolic_mmhg
        self.heart_rate_bpm   = heart_rate_bpm
        self.noise_mmhg       = noise_mmhg
        self.baud_rate        = baud_rate
        self._serial_override = serial_override

        self._serial          = None
        self._running         = False
        self._start_time_ms   = None

        # Pulse pressure: range of the AC waveform
        self._pulse_pressure = systolic_mmhg - diastolic_mmhg

    # ------------------------------------------------------------------
    # Waveform synthesis
    # ------------------------------------------------------------------

    def _cardiac_cycle_s(self) -> float:
        return 60.0 / self.heart_rate_bpm

    def _arterial_waveform(self, phase: float) -> float:
        """
        Return normalized arterial pressure waveform (0.0 = diastolic, 1.0 = systolic)
        at cardiac cycle phase [0.0, 1.0).

        Phases:
          0.00 - 0.15  Isovolumic contraction + rapid ejection (upstroke)
          0.15 - 0.30  Reduced ejection (rolloff from peak)
          0.30 - 0.38  Dicrotic notch (aortic valve closes)
          0.38 - 1.00  Diastolic runoff (Windkessel exponential decay)
        """
        if phase < 0.15:
            # Rapid systolic upstroke — half sine
            norm = math.sin(math.pi / 2.0 * (phase / 0.15))
            return norm

        elif phase < 0.30:
            # Reduced ejection — decay from peak
            decay_phase = (phase - 0.15) / 0.15
            return 1.0 - 0.20 * decay_phase

        elif phase < 0.34:
            # Dicrotic notch — brief dip
            notch_phase = (phase - 0.30) / 0.04
            notch_depth = 0.12 * math.sin(math.pi * notch_phase)
            return 0.80 - notch_depth

        elif phase < 0.42:
            # Dicrotic wave — small secondary hump after notch
            wave_phase = (phase - 0.34) / 0.08
            return 0.68 + 0.06 * math.sin(math.pi * wave_phase)

        else:
            # Diastolic Windkessel decay
            # Fit exponential from ~0.74 at phase=0.42 back to 0.0 at phase=1.0
            tau   = 0.25   # time constant in cycle fractions
            decay = 0.74 * math.exp(-(phase - 0.42) / tau)
            return max(0.0, decay)

    def _pressure_sample(self, t_s: float) -> float:
        """
        Return absolute arterial pressure in mmHg at time t_s.
        Includes respiratory variation and noise.
        """
        cycle_s = self._cardiac_cycle_s()
        phase   = (t_s % cycle_s) / cycle_s

        # Base waveform
        norm_wave = self._arterial_waveform(phase)
        pressure  = self.diastolic_mmhg + norm_wave * self._pulse_pressure

        # Respiratory variation (~0.25 Hz, ±3 mmHg — normal physiological range)
        pressure += 3.0 * math.sin(2 * math.pi * 0.25 * t_s)

        # Beat-to-beat variation (±2 mmHg systolic variability)
        beat_index = int(t_s / cycle_s)
        random.seed(beat_index)             # same noise per beat
        pressure  += random.gauss(0, 1.2)
        random.seed()                       # restore random state

        # High-frequency noise
        pressure += random.gauss(0, self.noise_mmhg)

        return pressure

    # ------------------------------------------------------------------
    # Packet framing
    # ------------------------------------------------------------------

    def _build_packet(self, timestamp_ms: int, pressure_mmhg: float) -> bytes:
        """
        Pack one IBP sample:
          0xBB 0xCC | uint32 ts | int16 pressure*10 | uint8 sys | uint8 dia | 0xEE
        """
        p_scaled  = int(pressure_mmhg * 10)
        p_scaled  = max(-32767, min(32767, p_scaled))

        sys_byte  = max(0, min(255, int(self.systolic_mmhg)))
        dia_byte  = max(0, min(255, int(self.diastolic_mmhg)))

        payload   = struct.pack(self.PACKET_FORMAT,
                                timestamp_ms, p_scaled, sys_byte, dia_byte)
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
        print(f"[IBP]  Connected on {self.port} @ {self.baud_rate} baud")
        print(f"[IBP]  BP target: {self.systolic_mmhg:.0f}/{self.diastolic_mmhg:.0f} mmHg | "
              f"HR: {self.heart_rate_bpm} BPM | "
              f"Sample rate: {self.SAMPLE_RATE_HZ} Hz")

    def disconnect(self):
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
        print("[IBP]  Disconnected.")

    def stream(self):
        """
        Blocking loop — generates and transmits IBP packets at SAMPLE_RATE_HZ.
        Call connect() before this. Stop with KeyboardInterrupt.
        """
        if not self._serial or not self._serial.is_open:
            raise RuntimeError("Call connect() before stream().")

        self._running = True
        interval_s    = 1.0 / self.SAMPLE_RATE_HZ
        t_s           = 0.0

        print("[IBP]  Streaming... (Ctrl+C to stop)")
        try:
            while self._running:
                loop_start = time.perf_counter()

                timestamp_ms  = int(time.time() * 1000) - self._start_time_ms
                pressure_mmhg = self._pressure_sample(t_s)
                packet        = self._build_packet(timestamp_ms, pressure_mmhg)

                self._serial.write(packet)

                t_s += interval_s

                elapsed = time.perf_counter() - loop_start
                sleep_s = interval_s - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except KeyboardInterrupt:
            print("\n[IBP]  Stream interrupted by user.")
        finally:
            self.disconnect()


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Invasive BP Sensor Simulator — streams synthetic arterial waveform over serial"
    )
    parser.add_argument("--port",  default="COM7",   help="Serial port (default: COM7)")
    parser.add_argument("--baud",  default=115200, type=int)
    parser.add_argument("--sys",   default=120.0, type=float, help="Systolic mmHg (default: 120)")
    parser.add_argument("--dia",   default=80.0,  type=float, help="Diastolic mmHg (default: 80)")
    parser.add_argument("--hr",    default=72,    type=int,   help="Heart rate BPM (default: 72)")
    parser.add_argument("--noise", default=1.5,   type=float, help="Noise std dev mmHg (default: 1.5)")
    args = parser.parse_args()

    sensor = InvasiveBPSensor(
        port            = args.port,
        systolic_mmhg   = args.sys,
        diastolic_mmhg  = args.dia,
        heart_rate_bpm  = args.hr,
        noise_mmhg      = args.noise,
        baud_rate       = args.baud,
    )
    sensor.connect()
    sensor.stream()


if __name__ == "__main__":
    main()
