"""
student_template.py  (ANSWER KEY)
==========================================
BME Testbench Exercise — Medical Device Data Processing

NO SETUP REQUIRED. Just run this file:
    python student_template.py

Dependencies:
    pip install numpy scipy matplotlib
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
SPO2_TARGET   = 98.0     # %%   (SpO2 sensor only)
SYS_MMHG      = 120.0    # mmHg (BP sensor only)
DIA_MMHG      = 80.0     # mmHg (BP sensor only)

PLOT_WINDOW_S = 6        # seconds of waveform to show on screen at once

# ══════════════════════════════════════════════════════════════════════════════
# SENSOR PROTOCOL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

PROTOCOLS = {
    "ecg": {
        "preamble"    : b'\xAA\xBB',
        "footer"      : b'\xFF',
        "packet_size" : 9,
        "format"      : '>I h',
        "sample_rate" : 500,
        "unit"        : "uV",
        "label"       : "ECG",
    },
    "spo2": {
        "preamble"    : b'\xCC\xDD',
        "footer"      : b'\xFE',
        "packet_size" : 12,
        "format"      : '>I hh B',
        "sample_rate" : 100,
        "unit"        : "AU",
        "label"       : "SpO2 (IR channel)",
    },
    "bp": {
        "preamble"    : b'\xBB\xCC',
        "footer"      : b'\xEE',
        "packet_size" : 11,
        "format"      : '>I h BB',
        "sample_rate" : 200,
        "unit"        : "mmHg",
        "label"       : "Arterial BP",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PACKET PARSER
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
# FEATURE EXTRACTION — ANSWER KEY
# ══════════════════════════════════════════════════════════════════════════════

def compute_heart_rate(timestamps_ms: np.ndarray,
                       samples: np.ndarray,
                       sample_rate_hz: int):
    """
    Estimate heart rate (BPM) from an ECG or PPG waveform.

    Approach: detect R-wave peaks (ECG) or pulse peaks (PPG) using
    scipy.signal.find_peaks, then compute the mean RR interval.

    Key parameter choices:
      - height threshold at 60th percentile filters out noise and baseline
        without being so high that real beats are missed
      - distance set to max-180-BPM spacing ensures we don't double-detect
        within a single beat, while still allowing tachycardia
    """
    if len(samples) < 2 or len(timestamps_ms) < 2:
        return None

    # Step 1: detect peaks
    # distance = minimum samples between beats (at max plausible HR of 180 BPM)
    min_distance = int(sample_rate_hz * 60 / 150)  # min spacing for max 150 BPM

    peak_indices, _ = sp_signal.find_peaks(
        samples,
        height   = np.percentile(samples, 60),   # ignore low-amplitude noise
        distance = min_distance,
    )

    # Need at least 2 peaks to compute an interval
    if len(peak_indices) < 2:
        return None

    # Step 2: get peak timestamps
    peak_times_ms = timestamps_ms[peak_indices]

    # Step 3: RR intervals = time between consecutive peaks
    rr_intervals_ms = np.diff(peak_times_ms)

    # Guard against degenerate intervals (e.g. very short windows)
    if len(rr_intervals_ms) == 0 or np.mean(rr_intervals_ms) <= 0:
        return None

    # Step 4: convert mean RR interval to BPM
    mean_rr_ms = np.mean(rr_intervals_ms)
    heart_rate_bpm = 60000.0 / mean_rr_ms

    # Clamp to physiologically plausible range
    if not (20 < heart_rate_bpm < 300):
        return None

    return heart_rate_bpm


def compute_spo2(red_samples: np.ndarray, ir_samples: np.ndarray,
                 sample_rate_hz: int):
    """
    Estimate SpO2 (%) from red and IR PPG channels using the R-ratio method.

    The Beer-Lambert law relates light absorption to oxygenated vs.
    deoxygenated hemoglobin. In practice, an empirical calibration curve
    (A - B*R) is fit to co-oximeter reference measurements.

    AC/DC decomposition:
      DC ≈ mean of the signal (slowly varying baseline)
      AC ≈ std dev of the signal (pulsatile component)

    This works because the cardiac pulse creates periodic variation on top
    of a slower-moving baseline — std dev captures the pulsatile energy
    without needing to explicitly bandpass filter here.
    """
    if len(red_samples) < 10 or len(ir_samples) < 10:
        return None

    # Step 1: DC component — mean of each channel over the window
    dc_red = np.mean(red_samples)
    dc_ir  = np.mean(ir_samples)

    if dc_red == 0 or dc_ir == 0:
        return None

    # Step 2: AC component — bandpass filter then peak-to-peak per cycle.
    # std() over-estimates AC when noise or motion artifact is present because
    # it conflates pulsatile energy with wideband noise. Peak-to-peak on the
    # filtered signal isolates only the cardiac-frequency AC component.
    nyq = 0.5 * sample_rate_hz
    b, a = sp_signal.butter(4, [0.5 / nyq, 8.0 / nyq], btype='band')
    red_filt = sp_signal.filtfilt(b, a, red_samples.astype(float))
    ir_filt  = sp_signal.filtfilt(b, a, ir_samples.astype(float))

    min_dist = int(sample_rate_hz * 60 / 180)   # max 180 BPM

    def _peak_to_peak(sig):
        peaks,   _ = sp_signal.find_peaks( sig, distance=min_dist)
        troughs, _ = sp_signal.find_peaks(-sig, distance=min_dist)
        n = min(len(peaks), len(troughs))
        if n < 2:
            return np.std(sig)   # fallback if not enough cycles
        return np.mean(sig[peaks[:n]] - sig[troughs[:n]])

    ac_red = _peak_to_peak(red_filt)
    ac_ir  = _peak_to_peak(ir_filt)

    if ac_ir == 0:
        return None

    # Step 3: R ratio
    R = (ac_red / dc_red) / (ac_ir / dc_ir)

    # Step 4: empirical calibration (simplified linear, A=110, B=25)
    spo2 = 110.0 - 25.0 * R

    # Step 5: clamp to physiological range
    return max(80.0, min(100.0, spo2))


def compute_blood_pressure(timestamps_ms: np.ndarray,
                           pressure_samples: np.ndarray):
    """
    Detect systolic and diastolic blood pressure from the arterial waveform.

    Two approaches shown:
      - Option A (percentile): robust, simple, works well on clean signals
      - Option B (peak detection): more analogous to clinical algorithms,
        handles cases where respiratory variation skews the percentiles

    We use Option B here as it's more instructive, with Option A as a fallback.
    """
    if len(pressure_samples) < 10:
        return None

    # Minimum beat spacing at max 180 BPM, 200 Hz sample rate
    # 200 Hz * 60s / 180 BPM ≈ 67 samples minimum
    sample_rate_hz = 200   # IBP sensor rate
    min_distance   = int(sample_rate_hz * 60 / 180)

    # --- Option B: peak / trough detection ---

    # Systolic peaks: tall positive peaks
    sys_peaks, _ = sp_signal.find_peaks(
        pressure_samples,
        height   = np.percentile(pressure_samples, 60),
        distance = min_distance,
    )

    # Diastolic troughs: invert signal and find peaks.
    # Use the same distance constraint AND a prominence threshold to avoid
    # detecting the dicrotic notch (a shallow mid-cycle dip) as a trough.
    pulse_pressure_est = float(np.percentile(pressure_samples, 90) -
                               np.percentile(pressure_samples, 10))
    dia_troughs, _ = sp_signal.find_peaks(
        -pressure_samples,
        distance   = min_distance,
        prominence = pulse_pressure_est * 0.3,   # must be a real trough, not a notch
    )

    # Fall back to percentile method if peak detection finds nothing
    if len(sys_peaks) < 1 or len(dia_troughs) < 1:
        systolic  = float(np.percentile(pressure_samples, 95))
        diastolic = float(np.percentile(pressure_samples, 5))
        return (systolic, diastolic)

    systolic  = float(np.mean(pressure_samples[sys_peaks]))
    diastolic = float(np.mean(pressure_samples[dia_troughs]))

    # Sanity check: systolic must be higher than diastolic
    if systolic <= diastolic:
        return None

    return (systolic, diastolic)


# ══════════════════════════════════════════════════════════════════════════════
# SENSOR LAUNCHER  (background thread)
# ══════════════════════════════════════════════════════════════════════════════

def start_sensor(sensor_name: str):
    """Start the chosen sensor in a daemon thread. Returns the read end."""
    write_end, read_end = MockSerialPair.create(sensor_name.upper())

    if sensor_name == "ecg":
        sensor = ECGSensor(heart_rate_bpm=HEART_RATE, serial_override=write_end)
    elif sensor_name == "spo2":
        sensor = SpO2Sensor(spo2_pct=SPO2_TARGET, heart_rate_bpm=HEART_RATE,
                            serial_override=write_end)
    elif sensor_name == "bp":
        sensor = InvasiveBPSensor(systolic_mmhg=SYS_MMHG, diastolic_mmhg=DIA_MMHG,
                                  heart_rate_bpm=HEART_RATE, serial_override=write_end)
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
        b_bp, a_bp = build_bandpass_filter(sample_rate, 0.5, 40.0)
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

        ax_raw.relim()
        ax_raw.autoscale_view()
        ax_filt.relim()
        ax_filt.autoscale_view()
        if SENSOR == "spo2":
            ac_range = max(abs(filtered.max()), abs(filtered.min())) * 1.4 + 10
            ax_filt.set_ylim(-ac_range, ac_range)

        feature_str = ""
        if SENSOR == "ecg":
            hr = compute_heart_rate(ts_arr, filtered, sample_rate)
            feature_str = (f"Heart Rate: {hr:.1f} BPM" if hr is not None
                           else "Heart Rate: not enough data yet...")

        elif SENSOR == "spo2":
            red_arr = np.array(list(receiver.red_samples)[-window_samples:])
            ir_arr  = np.array(list(receiver.ir_samples)[-window_samples:])
            if len(red_arr) == len(ir_arr) and len(red_arr) > 10:
                spo2 = compute_spo2(red_arr, ir_arr, sample_rate)
                hr   = compute_heart_rate(ts_arr, filtered, sample_rate)
                spo2_str = f"{spo2:.1f}%" if spo2 is not None else "..."
                hr_str   = f"{hr:.1f} BPM" if hr is not None else "..."
                feature_str = f"SpO2: {spo2_str}   |   HR: {hr_str}"

        elif SENSOR == "bp":
            bp = compute_blood_pressure(ts_arr, samples_arr)
            feature_str = (f"BP: {bp[0]:.0f}/{bp[1]:.0f} mmHg" if bp is not None
                           else "BP: not enough data yet...")

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
    proto = PROTOCOLS[SENSOR]
    print("=" * 55)
    print(f"  Medical Device Testbench  (ANSWER KEY)")
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