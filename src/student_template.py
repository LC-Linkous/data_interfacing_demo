"""
student_template.py  (in-process version)
==========================================
BME Testbench Exercise — Medical Device Data Processing

NO SETUP REQUIRED. Just run this file:
    python student_template.py

The sensor runs in a background thread and sends data directly into
this script through an in-memory queue — no COM ports, no drivers,
no second terminal needed.

Change SENSOR at the top to switch between ECG, SpO2, and IBP.

Your job: complete the three TODO functions below.
Everything else is already written — read it, understand it.

Dependencies:
    pip install numpy scipy matplotlib
    (pyserial is NOT required for this version)
"""

import struct
import threading
import time
import collections

import numpy as np
from scipy import signal as sp_signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mock_serial import MockSerialPair
from sensors.ecg_sensor  import ECGSensor
from sensors.spo2_sensor import SpO2Sensor
from sensors.bp_sensor   import InvasiveBPSensor

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — change SENSOR to switch between ecg / spo2 / bp
# ══════════════════════════════════════════════════════════════════════════════

SENSOR        = "ecg"    # "ecg" | "spo2" | "bp"
HEART_RATE    = 72       # BPM  (applies to all sensors)
NOISE_LEVEL   = "low"    # off|low|medium|high (all sensors). low: filters recover
                         # the target; medium/high intentionally stress naive detection.
SPO2_TARGET   = 98.0     # %%   (SpO2 sensor only)
SYS_MMHG      = 120.0    # mmHg (BP sensor only)
DIA_MMHG      = 80.0     # mmHg (BP sensor only)

PLOT_WINDOW_S = 6        # seconds of waveform to show on screen at once

# ══════════════════════════════════════════════════════════════════════════════
# SENSOR PROTOCOL DEFINITIONS
# These describe the binary packet format each sensor puts on the wire.
# The PacketParser below uses these to frame and unpack each packet.
# ══════════════════════════════════════════════════════════════════════════════

PROTOCOLS = {
    "ecg": {
        "preamble"    : b'\xAA\xBB',
        "footer"      : b'\xFF',
        "packet_size" : 9,
        "format"      : '>I h',       # uint32 timestamp_ms, int16 sample_uV
        "sample_rate" : 500,
        "unit"        : "uV",
        "label"       : "ECG",
    },
    "spo2": {
        "preamble"    : b'\xCC\xDD',
        "footer"      : b'\xFE',
        "packet_size" : 12,
        "format"      : '>I hh B',    # uint32 ts, int16 red, int16 ir, uint8 spo2
        "sample_rate" : 100,
        "unit"        : "AU",
        "label"       : "SpO2 (IR channel)",
    },
    "bp": {
        "preamble"    : b'\xBB\xCC',
        "footer"      : b'\xEE',
        "packet_size" : 11,
        "format"      : '>I h BB',    # uint32 ts, int16 pressure*10, uint8 sys, uint8 dia
        "sample_rate" : 200,
        "unit"        : "mmHg",
        "label"       : "Arterial BP",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PACKET PARSER
# Handles framing: scans for preamble, reads fixed-length packet, checks footer.
# Works with any object that has a .read(n) method — serial port, mock, socket.
# ══════════════════════════════════════════════════════════════════════════════

class PacketParser:
    def __init__(self, readable, protocol: dict):
        self.readable     = readable
        self.preamble     = protocol["preamble"]
        self.footer       = protocol["footer"]
        self.packet_size  = protocol["packet_size"]
        self.fmt          = protocol["format"]
        self.payload_size = struct.calcsize(self.fmt)

    def read_packet(self):
        """Block until one valid packet arrives. Returns unpacked tuple or None."""
        # Scan byte-by-byte until preamble is found
        buf = b''
        while True:
            byte = self.readable.read(1)
            if not byte:
                continue
            buf += byte
            if len(buf) > len(self.preamble):
                buf = buf[-len(self.preamble):]
            if buf == self.preamble:
                break

        # Read the rest of the packet
        remaining = self.payload_size + len(self.footer)
        raw = b''
        while len(raw) < remaining:
            chunk = self.readable.read(remaining - len(raw))
            if chunk:
                raw += chunk

        payload_bytes = raw[:self.payload_size]
        footer_bytes  = raw[self.payload_size:]

        if footer_bytes != self.footer:
            return None

        try:
            return struct.unpack(self.fmt, payload_bytes)
        except struct.error:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# DIGITAL FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def build_bandpass_filter(sample_rate_hz, lowcut_hz, highcut_hz, order=4):
    """Butterworth bandpass filter coefficients."""
    nyq  = 0.5 * sample_rate_hz
    b, a = sp_signal.butter(order, [lowcut_hz/nyq, highcut_hz/nyq], btype='band')
    return b, a

def build_notch_filter(sample_rate_hz, notch_hz=60.0, quality_factor=30.0):
    """IIR notch filter to remove power-line interference."""
    b, a = sp_signal.iirnotch(notch_hz, quality_factor, fs=sample_rate_hz)
    return b, a


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — THIS IS YOUR TODO
# ══════════════════════════════════════════════════════════════════════════════

def compute_heart_rate(timestamps_ms: np.ndarray,
                       samples: np.ndarray,
                       sample_rate_hz: int):
    """
    Estimate heart rate (BPM) from an ECG or PPG waveform.

    Args:
        timestamps_ms  : 1-D array of timestamps in milliseconds
        samples        : 1-D array of signal samples (same length)
        sample_rate_hz : Sample rate in Hz

    Returns:
        Heart rate in BPM as a float, or None if it cannot be determined.

    ── TODO ──────────────────────────────────────────────────────────────────
    Step 1: Find peaks (R-waves in ECG, pulse peaks in PPG).
            Use scipy.signal.find_peaks():
              - height   : minimum peak height — try the 60th percentile
              - distance : minimum samples between peaks
                           at 72 BPM and 500 Hz, one beat = 500*(60/72) ≈ 417 samples
                           set distance to ~80% of that as a minimum

    Step 2: Extract the timestamps of those peaks.

    Step 3: Compute the mean RR interval (time between consecutive peaks).

    Step 4: Convert to BPM:  bpm = 60000.0 / mean_rr_ms

    Starter skeleton:
        peak_indices, _ = sp_signal.find_peaks(
            samples,
            height   = np.percentile(samples, 60),
            distance = int(sample_rate_hz * 60 / 150),   # max 150 BPM spacing
        )
        if len(peak_indices) < 2:
            return None
        peak_times_ms = timestamps_ms[peak_indices]
        rr_intervals  = np.diff(peak_times_ms)
        mean_rr_ms    = np.mean(rr_intervals)
        return 60000.0 / mean_rr_ms

    Verify: run the script and check the displayed BPM vs HEART_RATE = {hr}.
    ──────────────────────────────────────────────────────────────────────────
    """
    # TODO: implement this function
    return None


def compute_spo2(red_samples: np.ndarray, ir_samples: np.ndarray,
                 sample_rate_hz: int):
    """
    Estimate SpO2 (%) from red and IR PPG channels.

    Args:
        red_samples    : Red channel (660 nm) amplitude array
        ir_samples     : IR channel (940 nm) amplitude array
        sample_rate_hz : Sample rate in Hz

    Returns:
        SpO2 as a float percentage, or None if undetermined.

    ── TODO ──────────────────────────────────────────────────────────────────
    Background: SpO2 is encoded in the ratio R of the two channels:
        R = (AC/DC)_red / (AC/DC)_ir
    Then:
        SpO2 ≈ 110 - 25 * R    (empirical calibration, simplified Beer-Lambert)

    Steps:
      1. DC component of each channel ≈ mean over the window
         AC component of each channel ≈ standard deviation
      2. R = (AC_red / DC_red) / (AC_ir / DC_ir)
      3. SpO2 = 110.0 - 25.0 * R
      4. Clamp to [80, 100] (physiologically plausible range)

    Starter skeleton:
        dc_red = np.mean(red_samples)
        dc_ir  = np.mean(ir_samples)
        ac_red = np.std(red_samples)
        ac_ir  = np.std(ir_samples)
        if dc_red == 0 or dc_ir == 0 or ac_ir == 0:
            return None
        R    = (ac_red / dc_red) / (ac_ir / dc_ir)
        spo2 = 110.0 - 25.0 * R
        return max(80.0, min(100.0, spo2))

    Verify: displayed SpO2 should be close to SPO2_TARGET = {spo2}.
    ──────────────────────────────────────────────────────────────────────────
    """
    # TODO: implement this function
    return None


def compute_blood_pressure(timestamps_ms: np.ndarray,
                           pressure_samples: np.ndarray):
    """
    Detect systolic and diastolic blood pressure from an arterial waveform.

    Args:
        timestamps_ms    : Array of timestamps in ms
        pressure_samples : Array of pressure values in mmHg

    Returns:
        (systolic_mmhg, diastolic_mmhg) as a tuple of floats, or None.

    ── TODO ──────────────────────────────────────────────────────────────────
    Two options — pick whichever makes more sense to you:

    Option A — percentile method (simpler, works well):
        systolic  = np.percentile(pressure_samples, 95)
        diastolic = np.percentile(pressure_samples, 10)
        return (systolic, diastolic)

    Option B — peak/trough detection (more like real clinical algorithms):
        peaks,  _ = sp_signal.find_peaks(pressure_samples,  distance=...)
        troughs,_ = sp_signal.find_peaks(-pressure_samples, distance=...)
        if len(peaks) < 1 or len(troughs) < 1:
            return None
        return (np.mean(pressure_samples[peaks]),
                np.mean(pressure_samples[troughs]))

    Verify: output should match SYS_MMHG={sys} / DIA_MMHG={dia}.
    ──────────────────────────────────────────────────────────────────────────
    """
    # TODO: implement this function
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR LAUNCHER  (background thread)
# ══════════════════════════════════════════════════════════════════════════════

def start_sensor(sensor_name: str):
    """Start the chosen sensor in a daemon thread. Returns the read end."""
    write_end, read_end = MockSerialPair.create(sensor_name.upper())

    if sensor_name == "ecg":
        sensor = ECGSensor(heart_rate_bpm=HEART_RATE, noise_level=NOISE_LEVEL,
                           serial_override=write_end)
    elif sensor_name == "spo2":
        sensor = SpO2Sensor(spo2_pct=SPO2_TARGET, heart_rate_bpm=HEART_RATE,
                            noise_level=NOISE_LEVEL, serial_override=write_end)
    elif sensor_name == "bp":
        sensor = InvasiveBPSensor(systolic_mmhg=SYS_MMHG, diastolic_mmhg=DIA_MMHG,
                                  heart_rate_bpm=HEART_RATE, noise_level=NOISE_LEVEL,
                                  serial_override=write_end)
    else:
        raise ValueError(f"Unknown sensor: {sensor_name!r}")

    def _run():
        sensor.connect()
        sensor.stream()

    threading.Thread(target=_run, daemon=True, name=f"{sensor_name}-sensor").start()
    return read_end


# ══════════════════════════════════════════════════════════════════════════════
# DATA RECEIVER  (background thread)
# ══════════════════════════════════════════════════════════════════════════════

class DataReceiver:
    def __init__(self, read_end, protocol: dict):
        self.protocol    = protocol
        self.sample_rate = protocol["sample_rate"]
        maxlen           = self.sample_rate * PLOT_WINDOW_S * 2

        self.timestamps  = collections.deque(maxlen=maxlen)
        self.samples     = collections.deque(maxlen=maxlen)
        self.red_samples = collections.deque(maxlen=maxlen)
        self.ir_samples  = collections.deque(maxlen=maxlen)

        self._parser     = PacketParser(read_end, protocol)
        self._running    = False
        self._thread     = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            try:
                fields = self._parser.read_packet()
                if fields is None:
                    continue
                if SENSOR == "ecg":
                    ts_ms, sample_uv = fields
                    self.timestamps.append(ts_ms)
                    self.samples.append(sample_uv)
                elif SENSOR == "spo2":
                    ts_ms, red, ir, _ = fields
                    self.timestamps.append(ts_ms)
                    self.samples.append(ir)
                    self.red_samples.append(red)
                    self.ir_samples.append(ir)
                elif SENSOR == "bp":
                    ts_ms, pressure_x10, _, _ = fields
                    self.timestamps.append(ts_ms)
                    self.samples.append(pressure_x10 / 10.0)
            except Exception as e:
                if self._running:
                    print(f"[Receiver] Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE PLOT
# ══════════════════════════════════════════════════════════════════════════════

def run_live_plot(receiver: DataReceiver, proto: dict):
    sample_rate    = proto["sample_rate"]
    window_samples = sample_rate * PLOT_WINDOW_S

    if SENSOR == "ecg":
        # Passband widened to 70 Hz (was 40). At a 40 Hz cutoff the band-pass
        # already suppresses 60 Hz, so the notch is redundant; at 70 Hz the mains
        # sits IN-band and the 60 Hz notch does real work. Set NOISE_LEVEL and
        # toggle the notch below to see it. (Sensor bandwidth is 0.05-150 Hz.)
        b_bp, a_bp = build_bandpass_filter(sample_rate, 0.5, 70.0)
        b_n,  a_n  = build_notch_filter(sample_rate, 60.0)
    elif SENSOR == "spo2":
        # SpO2: skip the bandpass here — we remove DC via a rolling mean in the
        # plot instead, which avoids the large startup transient caused by
        # lfilter initializing against a ~10,000 AU DC offset.
        b_bp, a_bp = None, None
        b_n,  a_n  = None, None
    elif SENSOR == "bp":
        # BP: notch-only filtering so the signal stays on its absolute mmHg scale.
        # A bandpass would strip the DC mean (~100 mmHg), making systolic/diastolic
        # unreadable. We only need to remove 60 Hz power-line noise.
        b_bp, a_bp = None, None
        b_n,  a_n  = build_notch_filter(sample_rate, 60.0)

    fig, (ax_raw, ax_filt) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor("#0D1B2A")
    for ax in (ax_raw, ax_filt):
        ax.set_facecolor("#1A2D40")
        ax.tick_params(colors="#C8D6E0")
        ax.yaxis.label.set_color("#C8D6E0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2E4057")

    line_raw,  = ax_raw.plot([], [], color="#0F8B8D", lw=1.0)
    line_filt, = ax_filt.plot([], [], color="#E8A838", lw=1.0)
    ax_raw.set_ylabel(f"Raw ({proto['unit']})", color="#C8D6E0")
    filt_label = "AC component (AU)" if SENSOR == "spo2" else f"Filtered ({proto['unit']})"
    ax_filt.set_ylabel(filt_label, color="#C8D6E0")
    ax_filt.set_xlabel("Time (s)",                    color="#C8D6E0")

    feature_text = ax_raw.text(
        0.01, 0.92, "Waiting for data...",
        transform=ax_raw.transAxes, color="#E8A838", fontsize=11, va="top"
    )
    plt.tight_layout(pad=1.5)

    def animate(_frame):
        if len(receiver.samples) < 10:
            return line_raw, line_filt, feature_text

        samples_arr = np.array(list(receiver.samples)[-window_samples:])
        ts_arr      = np.array(list(receiver.timestamps)[-window_samples:])
        if len(samples_arr) < 10:
            return line_raw, line_filt, feature_text

        t_s = (ts_arr - ts_arr[0]) / 1000.0

        filtered = samples_arr.copy()
        if len(samples_arr) > 15:
            if b_bp is not None:
                # Initialize filter state from first sample to suppress startup transient
                zi_bp = sp_signal.lfilter_zi(b_bp, a_bp) * samples_arr[0]
                filtered, _ = sp_signal.lfilter(b_bp, a_bp, samples_arr, zi=zi_bp)
            if b_n is not None:
                zi_n = sp_signal.lfilter_zi(b_n, a_n) * filtered[0]
                filtered, _ = sp_signal.lfilter(b_n, a_n, filtered, zi=zi_n)
            if SENSOR == "spo2":
                # Remove DC via scipy uniform_filter1d — a centered rolling mean
                # that handles boundaries with 'reflect' padding, which avoids the
                # edge spikes that appear when convolve or cumsum pads with zeros.
                from scipy.ndimage import uniform_filter1d
                window = min(150, len(filtered) // 2)   # ~1.5 cardiac cycles
                rolling_mean = uniform_filter1d(filtered.astype(float),
                                                size=window, mode='reflect')
                filtered = filtered - rolling_mean

        line_raw.set_data(t_s, samples_arr)
        line_filt.set_data(t_s, filtered)

        # Raw panel always autoscales to true signal range
        ax_raw.relim()
        ax_raw.autoscale_view()

        # Filtered panel: for SpO2 the bandpass removes the large DC baseline,
        # leaving only the small AC component (~200-500 AU). Fix the y-axis
        # symmetrically around zero so the waveform shape is clear.
        ax_filt.relim()
        ax_filt.autoscale_view()
        if SENSOR == "spo2":
            ac_range = max(abs(filtered.max()), abs(filtered.min())) * 1.4 + 10
            ax_filt.set_ylim(-ac_range, ac_range)

        # Call your TODO functions —————————————————————————————————————————
        feature_str = ""
        if SENSOR == "ecg":
            hr = compute_heart_rate(ts_arr, filtered, sample_rate)
            feature_str = (f"Heart Rate: {hr:.1f} BPM" if hr is not None
                           else "Heart Rate: implement compute_heart_rate() ↑")

        elif SENSOR == "spo2":
            red_arr = np.array(list(receiver.red_samples)[-window_samples:])
            ir_arr  = np.array(list(receiver.ir_samples)[-window_samples:])
            if len(red_arr) == len(ir_arr) and len(red_arr) > 10:
                spo2 = compute_spo2(red_arr, ir_arr, sample_rate)
                hr   = compute_heart_rate(ts_arr, filtered, sample_rate)
                spo2_str = f"{spo2:.1f}%" if spo2 is not None else "implement compute_spo2() ↑"
                hr_str   = f"{hr:.1f} BPM" if hr is not None else "implement compute_heart_rate() ↑"
                feature_str = f"SpO2: {spo2_str}   |   HR: {hr_str}"

        elif SENSOR == "bp":
            bp = compute_blood_pressure(ts_arr, samples_arr)
            feature_str = (f"BP: {bp[0]:.0f}/{bp[1]:.0f} mmHg" if bp is not None
                           else "BP: implement compute_blood_pressure() ↑")

        feature_text.set_text(feature_str)
        target = _target_label()
        fig.suptitle(
            f"{proto['label']} — Live Stream  |  Target: {target}  |  {len(samples_arr)} samples",
            color="#FFFFFF", fontsize=12
        )
        return line_raw, line_filt, feature_text

    ani = animation.FuncAnimation(
        fig, animate, interval=100, blit=False, cache_frame_data=False
    )
    plt.show()
    return ani


def _target_label():
    if SENSOR == "ecg":  return f"HR {HEART_RATE} BPM"
    if SENSOR == "spo2": return f"SpO2 {SPO2_TARGET}%  HR {HEART_RATE} BPM"
    if SENSOR == "bp":   return f"{SYS_MMHG:.0f}/{DIA_MMHG:.0f} mmHg  HR {HEART_RATE} BPM"
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Patch TODO docstrings with actual config values at runtime
    compute_heart_rate.__doc__ = compute_heart_rate.__doc__.format(hr=HEART_RATE)
    compute_spo2.__doc__       = compute_spo2.__doc__.format(spo2=SPO2_TARGET)
    compute_blood_pressure.__doc__ = compute_blood_pressure.__doc__.format(
        sys=SYS_MMHG, dia=DIA_MMHG)

    proto = PROTOCOLS[SENSOR]
    print("=" * 55)
    print(f"  Medical Device Testbench  (no hardware needed)")
    print(f"  Sensor : {SENSOR.upper()}  —  {proto['label']}")
    print(f"  Target : {_target_label()}")
    print("=" * 55)
    print("  Close the plot window to exit.\n")

    read_end = start_sensor(SENSOR)
    time.sleep(0.3)

    receiver = DataReceiver(read_end, proto)
    receiver.start()
    time.sleep(1.5)

    try:
        run_live_plot(receiver, proto)
    finally:
        receiver.stop()
        print("Done.")


if __name__ == "__main__":
    main()