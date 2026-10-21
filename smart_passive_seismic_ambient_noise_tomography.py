import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.cross_correlation import correlate

# =========================================================
# Smart Passive Seismic + Ambient Noise Tomography
# Versi siap dipakai untuk data seismic .mseed
# =========================================================

DATA_DIR = "data"


def preprocess_trace(tr, freqmin=0.1, freqmax=2.0):
    tr = tr.copy()
    tr.detrend("linear")
    tr.taper(max_percentage=0.05)
    tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax)
    return tr


def load_station_traces(data_dir=DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, "*.mseed")))
    files = [
        f for f in files
        if not os.path.basename(f).lower().startswith("gempa_")
        and not os.path.basename(f).lower().endswith("_event.mseed")
    ]
    if not files:
        raise FileNotFoundError(f"Tidak ada file .mseed ambient noise di folder: {data_dir}")

    traces = {}
    for file in files:
        st = read(file)
        station_name = os.path.basename(file).split(".")[0]
        comps = st.select(component="Z")
        if len(comps) == 0:
            comps = st.select(component="E")
        if len(comps) == 0:
            comps = st.select(component="N")
        if len(comps) == 0:
            raise ValueError(f"Tidak ada komponen valid di file {file}")

        tr = preprocess_trace(comps[0])
        traces[station_name] = tr
        print(f"Loaded: {station_name} -> {tr.id}, SR={tr.stats.sampling_rate} Hz")

    return traces


def get_common_window(traces):
    start = max(tr.stats.starttime for tr in traces.values())
    end = min(tr.stats.endtime for tr in traces.values())
    for name, tr in traces.items():
        traces[name] = tr.trim(starttime=start, endtime=end)
    return traces


def compute_pairwise_correlation(traces):
    station_names = list(traces.keys())
    results = []

    for i in range(len(station_names)):
        for j in range(i + 1, len(station_names)):
            s1, s2 = station_names[i], station_names[j]
            tr1 = traces[s1]
            tr2 = traces[s2]

            if tr1.stats.sampling_rate != tr2.stats.sampling_rate:
                sr = max(tr1.stats.sampling_rate, tr2.stats.sampling_rate)
                tr1 = tr1.resample(sr)
                tr2 = tr2.resample(sr)

            data1 = tr1.data.astype(float)
            data2 = tr2.data.astype(float)

            cc = correlate(data1, data2, shift=len(data1) - 1)
            lags = np.arange(-len(data1) + 1, len(data1))
            idx = np.argmax(np.abs(cc))
            lag = lags[idx]
            peak = cc[idx]

            dt = lag / tr1.stats.sampling_rate
            t0 = 1.0
            dv_v = -dt / t0

            results.append({
                "pair": (s1, s2),
                "lag": lag,
                "dt": dt,
                "dv_v": dv_v,
                "cc_peak": peak,
            })

    return results


def estimate_vs_anomaly(pair_results):
    if not pair_results:
        raise ValueError("Tidak ada pasangan stasiun yang bisa dihitung")

    dv_values = [abs(r["dv_v"]) for r in pair_results]
    avg_anomaly = np.mean(dv_values)
    baseline_vs = 450.0
    vs_final = baseline_vs - avg_anomaly * baseline_vs

    return {
        "avg_anomaly": avg_anomaly,
        "baseline_vs": baseline_vs,
        "vs_final": vs_final,
    }


def plot_results(traces, pair_results, estimate):
    fig, axs = plt.subplots(2, 1, figsize=(12, 9))

    # Plot waveform samples
    for i, (name, tr) in enumerate(traces.items()):
        t = np.arange(len(tr.data)) / tr.stats.sampling_rate
        axs[0].plot(t, tr.data + i * 0.5, label=name)

    axs[0].set_title("Waveform Pasca-Preprocessing")
    axs[0].set_xlabel("Waktu (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Plot travel time shift
    dt_values = [r["dt"] for r in pair_results]
    x = np.arange(len(dt_values))
    axs[1].bar(x, dt_values, color="tab:orange")
    axs[1].set_title("Travel Time Shift antar Stasiun")
    axs[1].set_xlabel("Pair index")
    axs[1].set_ylabel("dt (s)")
    axs[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot Vs comparison
    plt.figure(figsize=(6, 4))
    plt.bar(["Vs baseline", "Vs final"], [estimate["baseline_vs"], estimate["vs_final"]], color=["tab:green", "tab:red"])
    plt.ylabel("Vs (m/s)")
    plt.title("Perbandingan Vs Baseline vs Vs Final")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


# =========================================================
# RUN PROGRAM
# =========================================================
if __name__ == "__main__":
    traces = load_station_traces(DATA_DIR)
    traces = get_common_window(traces)
    pair_results = compute_pairwise_correlation(traces)
    estimate = estimate_vs_anomaly(pair_results)

    print("\nHasil estimasi:")
    for item in pair_results:
        print(f"Pair {item['pair']} -> dt={item['dt']:.6f}s, dv/v={item['dv_v']:.6f}")

    print("\nModel Vs:")
    print(f"Average anomaly = {estimate['avg_anomaly']:.6f}")
    print(f"Vs baseline = {estimate['baseline_vs']:.2f} m/s")
    print(f"Vs final = {estimate['vs_final']:.2f} m/s")

    penurunan_persen = ((estimate['baseline_vs'] - estimate['vs_final']) / estimate['baseline_vs']) * 100
    print(f"Penurunan Vs = {penurunan_persen:.2f}%")

    if penurunan_persen >= 20:
        kategori = "CRITICAL"
        interpretasi = "Zona deformasi sangat kuat terdeteksi. Perlu tindakan segera."
    elif penurunan_persen >= 10:
        kategori = "WARNING"
        interpretasi = "Ada indikasi deformasi yang perlu ditinjau lebih lanjut."
    else:
        kategori = "NORMAL"
        interpretasi = "Belum ada indikasi deformasi yang signifikan."

    print(f"Status: {kategori}")
    print(f"Interpretasi: {interpretasi}")

    plot_results(traces, pair_results, estimate)
