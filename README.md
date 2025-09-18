# CSP_for_CongestedEM
Below is a quick guide for running the end-to-end pipeline that simulates a congested 2.4 GHz scene, stores IQ into MySQL, then load and run cyclostationary detection to produce an SCF heatmap.

# WHAT IS INCLUDED: 
cochannel_uav_sim.py – builds a scene with three fully overlapped, co-channel signals (OFDM Wi-Fi signal with Cyclic Prefix, GFSK burst train, and a DSSS-BPSK UAV link), plus noise. Saves quick-look PSD/spectrogram PNGs and returns (x, signals, fs, fc, T). \
\
Write_to_MySQL.py – takes that simulated capture and writes it to a MySQL time-series schema: \
\
rf_capture(capture_id, fs_sps, duration_s, samples, center_frequency) \
rf_sample(capture_id, signal_name, sample_idx, i_val, q_val) \
\
Streams the composite and (optionally) each component (ofdm, gfsk, uav, noise) in batches; prints a capture_id for later analysis. \
\
cochannel_uav_detect.py – loads IQ back from MySQL for a given capture_id and signal_name, runs an FFT-accumulation-method SCF, whitens, auto-scales, and saves a heatmap PNG labeled by alpha (cyclic frequency) vs. f (offset from fc).

# QUICK START
Generate an RF collect and write it to mySQL database using CLI (bash): \
\
python Write_to_MySQL.py

Analyze the collection with the following CLI (bash) commands: \
\
python cochannel_uav_detect.py \\
\
  --capture_id 1 \\
  --signal_name composite \\
  --nfft 16384 --overlap 0.5 \\
  --alpha_start 0 --alpha_stop 1.2e6 --alpha_step 1.0e4 \\
  --prefix scf_db
