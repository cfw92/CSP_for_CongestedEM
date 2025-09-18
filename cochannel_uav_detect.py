#
# Read IQ from MySQL (rf_db) with schema:
#   rf_capture(capture_id, fs_sps, duration_s, samples, center_frequency)
#   rf_sample(capture_id, signal_name, sample_idx, i_val, q_val)
# Compute SCF (FAM-style) and CAF slices for cyclostationary detection.
#
import matplotlib.pyplot as plt
import argparse, numpy as np
import mysql.connector
from scipy.ndimage import uniform_filter1d
#
# ----------------------- DB loader -----------------------
# Connects to rf_db via mySQL and the mysql_cfg parameters hardcoded in main() for redundancy. 
# Extracts generic features from rf_capture table. 
# Extracts I/Q from rf_samples table.
# Returns Composite signal, x, sample freq, fs, and center freq, fc.
#
def load_iq_from_mysql(
    config: dict,
    capture_id: int,
    signal_name: str,
    start_idx: int | None = None,
    end_idx: int | None = None,
    fetch_chunk: int = 200_000,
):

    conn = mysql.connector.connect(**config)
    cur  = conn.cursor()

    # --- capture header ---
    cur.execute(
        "SELECT fs_sps, duration_s, samples, center_frequency "
        "FROM rf_capture WHERE capture_id=%s",
        (capture_id,)
    )
    row = cur.fetchone()
    if row is None:
        cur.close(); conn.close()
        raise ValueError(f"capture_id={capture_id} not found.")

    fs, duration_s, total_samples, fc = float(row[0]), float(row[1]), int(row[2]), float(row[3])

    # --- windowing ---
    s0 = 0 if start_idx is None else max(0, int(start_idx)) #-------------------------------- [Identify start idx value as default, 0, or an explicit val]
    s1 = total_samples - 1 if end_idx is None else min(total_samples - 1, int(end_idx)) #---- [Identify end idx value as default, end_idx-1, or explicit val]
    if s1 < s0: #---------------------------------------------------------------------------- [Check if end_idx > start_idx]
        cur.close(); conn.close()
        raise ValueError(f"end_idx ({s1}) must be >= start_idx ({s0})")

    Nwin = s1 - s0 + 1 #--------------------------------------------------------------------- [Create window length]
    if Nwin <= 0: 
        cur.close(); conn.close()
        raise ValueError("Selected window length is zero")

    # Zero pad output buffer
    x = np.zeros(Nwin, dtype=np.complex64)

    # --- stream samples in order ---
    cur.arraysize = min(fetch_chunk, 1_000_000)
    cur.execute(
        "SELECT sample_idx, i_val, q_val "
        "FROM rf_sample "
        "WHERE capture_id=%s AND signal_name=%s AND sample_idx BETWEEN %s AND %s "
        "ORDER BY sample_idx ASC",
        (capture_id, signal_name, s0, s1)
    )

    filled = 0
    while True:
        rows = cur.fetchmany()  
        if not rows:
            break
        for (k, I, Q) in rows: #------------------------------------------------------------ [iterate over sample idx, Inphase, and Quadtrature val]
            pos = k - s0 #------------------------------------------------------------------ [scales pos vector in the event that start idx (s0) is non-zero]
            if 0 <= pos < Nwin:
                i = float(I); q = float(Q) #------------------------------------------------ [ensure float values are loaded]
                if np.isfinite(i) and np.isfinite(q):
                    x[pos] = np.complex64(np.float32(i) + 1j*np.float32(q)) #--------------- [combine I/Q into one complex valued x[pos] input]
                    filled += 1

    cur.close(); conn.close()

    # --- completeness check/DEBUGGING ---
    if filled != Nwin:
        missing = Nwin - filled
        raise RuntimeError(
            f"Expected {Nwin} samples in [{s0},{s1}], but loaded {filled} "
            f"(missing {missing}). Check table continuity and indices."
        )

    return x, fs, fc

# ----------------------- DSP Core -----------------------
#
# Apply short time fourier transform with hanning window, x(t) -> X(f).
# Estimate spectral correlation function via FFT accumulation method. 
# Apply alpha/2 shift up/down to each frequency bin, and compute the conjugate product, X(f+a/2) * conj X(f-a/2).
# Apply second fast fourier transform, X(f +/- a/2) -> X(a) complex-valued
# Apply whitening to SCF magnitude, |X(a)|^2, and convert to dB. Plot a vs f.
#
def stft_frames(x, nfft, overlap):
    hop = max(1, int(nfft*(1-overlap))) #------------------------------------- [Compute Num. of new samples between frames]
    n_frames = 1 + (len(x)-nfft)//hop if len(x) >= nfft else 0 #-------------- [Compute Num. of complete frames]
    w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(nfft)/(nfft-1)) # ----------------- [Hanning window definition]
    frames = np.lib.stride_tricks.as_strided(x, shape=(n_frames, nfft),
        strides=(x.strides[0]*hop, x.strides[0])).copy() #-------------------- [Create a 2D view of overlapped frames]
    frames *= w.astype(frames.dtype) #---------------------------------------- [Scale by the window]
    X = np.fft.fft(frames, n=nfft, axis=1) #---------------------------------- [FFT across rows]
    return X

def whiten(S, smooth_bins=129, floor_q=0.30):
    H2 = np.abs(S)**2 #------------------------------------------------------- [Compute power-like magnitude, |SCF[α, f]|^2]
    P  = np.nanmedian(H2, axis=0).astype(np.float64)   #---------------------- [find the median value across each cyclic freq, alpha bin]
    if smooth_bins > 1:
        P = uniform_filter1d(P, size=smooth_bins, mode="nearest") #----------- [Apply a 1-dim smoothing filter from scipy library]
    floor = max(np.nanquantile(P, floor_q), 1e-20)
    G = 1.0 / np.sqrt(np.maximum(P, floor))        #-------------------------- [Compute whitening gain for each f]
    Hw = np.abs(S) * G[None, :]; Hw[~np.isfinite(S)] = np.nan #--------------- [apply frequency whitening gain to magnitudes] 
    return Hw

def fam_scf(X, fs, nfft, alpha_list_Hz):
    """
    FAM-style SCF.
    Returns complex SCF rows.
    """
    df = fs / nfft
    k  = np.arange(nfft, dtype=np.int32)
    rows = []

    for alpha in np.atleast_1d(alpha_list_Hz): #------------------------------ [Iterate over cyclic frequencies, alpha]
        a_bins = int(np.round(alpha / df)) #---------------------------------- [Snaps the steps between alphas to rounded integers]

        # symmetric split of α bins
        h_up =  (a_bins + 1) // 2   #------------------------------------------ [Prepare shift up, ceil(a_bins/2)]
        h_dn =  a_bins // 2         #------------------------------------------ [Prepare shift down, floor(a_bins/2)]
        ip = k + h_up               #------------------------------------------ [Apply positive shift f + α/2]
        im = k - h_dn               #------------------------------------------ [Apply negative shift f - α/2]

        # keep only columns where both bins are inside [0, nfft)
        valid = (ip >= 0) & (ip < nfft) & (im >= 0) & (im < nfft) #------------ [Create valid mask, prevents displaying of V-notch]
        row = np.full(nfft, np.nan, np.complex64) #---------------------------- [fill invalid columns with NaN, plot will ignore these]
        if valid.any(): 
            prod = X[:, ip[valid]] * np.conj(X[:, im[valid]]) #---------------- [Compute the conjugate product, P = X * conj(X)] 
            row[valid] = prod.mean(axis=0, dtype=np.complex128).astype(np.complex64)
        rows.append(np.fft.fftshift(row)) # ----------------------------------- [Apply Second FFT to transform into cyclic frequency space]

    scf = np.stack(rows, axis=0)  #-------------------------------------------- [n_alpha, nfft] complex-valued
    freq_Hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/fs)).astype(np.float32) #[Apply appropriate shifts to align frequency too]
    return scf, freq_Hz

# ----------------------- Plot -----------------------

def plot_scf(prefix, scf, alpha_Hz, freq_Hz):
    H = whiten(scf, smooth_bins=129, floor_q=0.30) #-------------------------- [Apply whitening to SCF, makes subtle ridges more pronounced.]
    dB = 10*np.log10(np.clip(H, 1e-20, None)) #------------------------------- [Convert to dB]
    i0 = int(np.argmin(np.abs(alpha_Hz)))#------------------------------------ [Remove stationary power bleed over from a=0 bin, helps with weak features.]
    dB_scale = np.delete(dB, i0, axis=0) if dB.shape[0] > 1 else dB #--------- [Deletes the dB associated with a=0 bin]
    vmax = np.nanpercentile(dB_scale, 98.0) #--------------------------------- [Define max value from 98percentile in dB_scale] 
    vmin = vmax - 10.0 #------------------------------------------------------ [Focus only on the strongest features within 30dB]
    dB_ma = np.ma.array(dB, mask=~np.isfinite(dB)) #-------------------------- [If any invalid dB values (NaN/inf), ignores and applies transparent color map]
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='none', alpha=0.0)
    extent = [float(freq_Hz[0])/1e6, float(freq_Hz[-1])/1e6,
              float(alpha_Hz[0])/1e3, float(alpha_Hz[-1])/1e3] #-------------- [Scales the x/y-axis]
    plt.figure(figsize=(9,5))
    plt.imshow(dB, origin='lower', aspect='auto',
               extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.xlabel("Frequency offset from fc (MHz)")
    plt.ylabel("Cyclic frequency α (kHz)")
    plt.title("SCF magnitude (dB), whitened")  
    plt.tight_layout()
    plt.savefig(prefix+"_scf_heatmap.png", dpi=150)
    plt.close()

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="SCF/CAF from MySQL rf_db (UAV cyclostationary detection).")
    # CLI input controls:
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", type=str)
    ap.add_argument("--database", default="rf_db")
    ap.add_argument("--capture_id", type=int, required=True)
    ap.add_argument("--signal_name", default="composite", help="composite | ofdm | gfsk | uav | noise")
    # Windowing / limits
    ap.add_argument("--start_idx", type=int, default=None, help="first sample index (inclusive)")
    ap.add_argument("--end_idx", type=int, default=None, help="last sample index (inclusive)")
    # DSP parameters
    ap.add_argument("--nfft", type=int, default=16384)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--alpha_start", type=float, default=0.0)
    ap.add_argument("--alpha_stop", type=float, default=1.2e6)
    ap.add_argument("--alpha_step", type=float, default=10e3)
    ap.add_argument("--alpha_target", type=str, default="1e6,977.517")
    ap.add_argument("--prefix", default="scf_db")
    args = ap.parse_args()
    out_prefix = f"{args.prefix}_cap{args.capture_id}_{args.signal_name}"
    alpha_Hz = np.arange(args.alpha_start, 
                         args.alpha_stop + 0.5*abs(args.alpha_step), 
                         args.alpha_step, float) #--------------------------------------- [Cyclic freq. range (alpha grid)]
    
    # MySQL config / Pull from DB / define alpha range
    mysql_cfg = {
        "host": "localhost", 
        "user": "root", 
        "password": "P@$$w0rd_123", 
        "database": "rf_db"
    }
    x, fs, fc = load_iq_from_mysql(
        mysql_cfg,
        capture_id=args.capture_id, 
        signal_name=args.signal_name,
        start_idx=args.start_idx, 
        end_idx=args.end_idx, 
        fetch_chunk=200_000
    )
    print(f"IQ loaded from database.\n")
    
    # Compute Spectral Correlation Function via FFT Accumulation Method
    X = stft_frames(x, args.nfft, args.overlap)
    scf, freq_Hz = fam_scf(X, fs, args.nfft, alpha_Hz)
    print(f"Complex SCF computed.\n")

    # Plot SCF map
    plot_scf(out_prefix, scf, alpha_Hz, freq_Hz)
    print(f"SCF plotted.\n")
  
    print(f"Done. Saved SCF plot with prefix: {out_prefix}. fs={fs/1e6:.3f} Msps, fc={fc/1e9:.3f} GHz, N={len(x)}")

if __name__ == "__main__":
    main()