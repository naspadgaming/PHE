import os
import numpy as np
from obspy import UTCDateTime, Trace, Stream

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sample_rate = 20.0
npts = 3600 * 2  # 2 jam data
start = UTCDateTime("2024-01-01T00:00:00")

rng = np.random.default_rng(42)

def make_station(name, phase_shift=0, amplitude=1.0, trend=0.0):
    t = np.arange(npts) / sample_rate
    # ambient noise-like signal: colored random + weak sinusoid
    noise = rng.normal(0, 1.0, npts)
    signal = 0.6 * np.sin(2 * np.pi * 0.25 * t + phase_shift) + 0.2 * np.sin(2 * np.pi * 0.7 * t)
    data = amplitude * (noise + signal + trend * np.linspace(0, 1, npts))

    tr = Trace(data=data)
    tr.stats.network = "IU"
    tr.stats.station = name
    tr.stats.location = "00"
    tr.stats.channel = "LHZ"
    tr.stats.starttime = start
    tr.stats.sampling_rate = sample_rate
    tr.stats.npts = len(data)
    return tr

# generate two stations with similar ambient noise but different phase and amplitude
station_a = make_station("ANMO", phase_shift=0.0, amplitude=1.0)
station_b = make_station("ANMB", phase_shift=0.35, amplitude=0.92)

st_a = Stream(traces=[station_a])
st_b = Stream(traces=[station_b])

st_a.write(os.path.join(OUTPUT_DIR, "noise_station_a.mseed"), format="MSEED")
st_b.write(os.path.join(OUTPUT_DIR, "noise_station_b.mseed"), format="MSEED")

print("Created ambient-noise style MSEED files:")
print(" -", os.path.join(OUTPUT_DIR, "noise_station_a.mseed"))
print(" -", os.path.join(OUTPUT_DIR, "noise_station_b.mseed"))
