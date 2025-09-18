
# cochannel_uav_sim_24ghz.py
# Three fully co-channel signals overlapped in frequency, labeled around the 2.4 GHz Wi‑Fi band.
# IQ is complex baseband; interpret as downconverted around an RF center fc in 2.4 GHz.
#
# Signals:
#   - OFDM-like (with CP)        -> strong cyclostationary lines at 1/Tsym
#   - GFSK burst train           -> periodicity at symbol rate and burst rate
#   - UAV control link (hidden): DSSS-BPSK (PN-1023) -> lines at chip_rate and chip_rate/1023

import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window

def rand_bits(n, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.integers(0, 2, size=n, dtype=np.int8)

# ---------------------------- OFDM-WiFi Signal (with CP) ----------------------------
# Introducing a cyclic prefix creates a periodic signal that is can be recognizable in a SCF at alpha = sym_rate
# The random QPSK modulation per symbol simulates a busy wi-fi signal
# Randomly selecting active / inactive subcarriers simulates allocation/re-allocation of sub-bands
#
def ofdm_with_cp(fs, N, n_subc, used_frac, sym_rate, cp_frac, rng=None):

    # Define basic parameters
    rng = np.random.default_rng(1) if rng is None else rng
    T_sym = 1.0/sym_rate #---------------------- [symbol duration, seconds]
    sps = round(fs*T_sym) #--------------------- [samples / symbol] 
    n_syms = N//sps + 2  #---------------------- [# of symbols]
    df = fs/n_subc #---------------------------- [evenly spaced, orthogonal frequency subbands]
    x = np.zeros(N,dtype=np.complex64) #-------- [prepare output vector scaled to N samples]
    t = np.arange(N)/fs #----------------------- [initialize a time vector, seconds] 

    # Create subcarrier range and assign a random fraction of the subcarriers as actively used
    subc = np.arange(-n_subc//2, n_subc//2) #--- [define subcarrier range] 
    active = rng.random(len(subc)) < used_frac # [creates an array of randomly selected FP values < 0.85]
    active_idx = subc[active] #----------------- [identify which subcarriers will be used] 

    # Symbol loop assigns QPSK modulated symbols to each of the active subcarriers
    for k in range(n_syms):
        
        start = k*sps #------------------------- [identify sample # to work on]
        if start >= N: break #------------------ [breaks loop when max symbol count is reached]
        end = min(start+sps, N) #--------------- [identify sample # to end on]
        tk = t[start:end] #--------------------- [create local time vector, seconds]
        
        bits = rand_bits(2*np.sum(active), rng) #------------------------------------- [initialize random array of bits]
        qpsk = (1-2*bits[0::2] + 1j*(1-2*bits[1::2]))/np.sqrt(2) #-------------------- [2 bits per sym; even bit idx = real values, odd bit idx = imag values]
        
        sym = np.zeros(end-start,dtype=np.complex64) #-------------------------------- [initialize sym array]
        for qk, sc in zip(qpsk, active_idx): #---------------------------------------- [iterate through each 2-bit symbol & apply it to a subcarrier]
            f = sc*df #--------------------------------------------------------------- [identify the frequency component] 
            sym += qk * np.exp(1j*2*np.pi*f*tk).astype(np.complex64) #---------------- [apply the phase shift] 
            
        if len(qpsk) > 0: #----------------------------------------------------------- [normalize to keep power per symbol constant]
            sym /= np.sqrt(len(qpsk))

        # Create the cyclic prefix by copying the the end of the symbol and creating a replica header
        cp_len = int(round(cp_frac*(end-start))) #------------------------------------ [identify length of the CP]
        if cp_len > 0:
            sym_with_cp = np.concatenate([sym[-cp_len:], sym]) #---------------------- [write CP to start of symbol]
        else:
            sym_with_cp = sym #------------------------------------------------------- [write CP to start of symbol]
        write_len = min(len(sym_with_cp), N-start) 
        x[start:start+write_len] += sym_with_cp[:write_len].astype(np.complex64) #---- [write CP+symbol to output array]
    return x

# ---------------------------- GFSK bursts -----------------------------------
# The Burst structure gives cyclostationary energy at the burst repetition rate (period) and the symbol rate (~sym_rate).
# The gaussian smoothing of the envelop replicates a bluetooth-like signal and limits occupied bandwidth.
#
def gfsk_bursts(fs, N, sym_rate, h, bt, n_bursts, duty, rng=None):

    # Define basic parameters
    rng = np.random.default_rng(2) if rng is None else rng
    x = np.zeros(N, dtype=np.complex64) #--------------------------------------------- [prepare the output array, scaled to N samples]
    sps = max(1, round(fs/sym_rate)) #------------------------------------------------ [samples per symbol]
    
    # Build the Gaussian Shaped Pulse
    span = 6 #------------------------------------------------------------------------ [pulse will span +/- 3symbol durations, common for gfsk]
    tg = np.linspace(-span/2, span/2, span*sps, endpoint=False) #--------------------- [create local time index]
    alpha = np.sqrt(np.log(2))/(2*bt*np.pi) #----------------------------------------- [maps the pulse effective bandwidth, bt, to the guassian std-dev]
    g = np.exp(-(tg**2)/(2*alpha**2)); g /= np.sum(g) #------------------------------- [normalizes the shaped pulses] 
    
    # Schedule the bursts
    period = N//n_bursts #------------------------------------------------------------ [Samples per burst]
    on = int(duty*period) #----------------------------------------------------------- [Num. of Tx Samples]
    
    # Build each burst package
    for b in range(n_bursts):

        # identify start and end timestamp
        start = b*period
        end = min(start+on, N)
        if start >= N: break
        n_syms = max(1, (end-start)//sps) #------------------------------------------- [Num. of symbols per burst]
        bits = rand_bits(n_syms, rng) #----------------------------------------------- [generate random bits to fill symbols]
        nrz = 2*bits.astype(np.float32)-1 #------------------------------------------- [map bits to non-return zero (0-bit:-1, 1-bit:+1)
        m = np.convolve(np.repeat(nrz, sps), g, mode='full')[:end-start] #------------ [hold bit for 'sps' samples, apply gaussian shaping filter 'g']
        kf = 0.5*h*sym_rate #--------------------------------------------------------- [scales the frequency deviation]
        phase = 2*np.pi*np.cumsum(kf*m)/fs #------------------------------------------ [convert frequency into phase; phase = integral(freq deviation)dt] 
        burst = np.exp(1j*phase).astype(np.complex64) #------------------------------- [form the burst envelop, exp^(j*phase)]
        x[start:start+len(burst)] += burst #------------------------------------------ [add the burst into the output array]

    # Normalize the output and return
    rms = np.sqrt(np.mean(np.abs(x)**2) + 1e-12)
    if rms > 0: x /= rms
    return x

# ---------------------------- DSSS-BPSK (UAV) --------------------------------

# Create Linear Feedback Shift Register (LFSR).
# Taps at positions 10 and 3, allow for maximum sequence length (2^10 - 1 = 1023) before repeating.
# Initial state, all 1s.
# Return value: 0-bit: -1 or 1-bit: +1
# DSSS multiplication uses +/-1 chips to invert or keep the data symbol.
def mseq(length=1023):
    reg = np.ones(10, dtype=np.int8)
    out = []
    for _ in range(length):
        out.append(reg[-1])
        new = reg[9] ^ reg[2]
        reg[1:] = reg[:-1]
        reg[0] = new
    return (2*np.array(out, dtype=np.int8)-1) #.astype(np.int8)

def dsss_bpsk_hidden(fs, N, data_rate, chip_rate, rng=None):

    # Build basic parameters
    rng = np.random.default_rng(3) if rng is None else rng
    pn = mseq(1023) #----------------------------------------------------------------- [generate deterministic Pseudo-Noise sequence]
    chips_per_bit = chip_rate//data_rate #-------------------------------------------- [identify chips per bit, 1e6 / 20e3 = 50 chips per bit]
    n_bits = int(np.ceil(N/fs*data_rate)) #------------------------------------------- [identify how many bits will fit in N samples]
    total_chips = chips_per_bit * n_bits #-------------------------------------------- [identify Num. of chips]

    # Build the spread spectrum chip stream
    data = 2*rand_bits(n_bits, rng)-1 #----------------------------------------------- [create the +/-1 data stream]
    data_rep = np.repeat(data, chips_per_bit) #--------------------------------------- [repeat each bit for 50 chips]
    pn_rep  = np.tile(pn, int(np.ceil(total_chips/1023)))[:total_chips] #------------- [loop pseudo-noise LFSR as necessary]
    spread = data_rep * pn_rep #------------------------------------------------------ [Result is a +/-1 chip sequence at the chip rate]

    # Apply rectangular pulse shape per chip & normalize
    sps_chip = round(fs/chip_rate) #-------------------------------------------------- [identify samples per chip]
    x = np.repeat(spread, sps_chip)[:N].astype(np.float32) + 0j #--------------------- [create rectangular shaped pulses across each chip]
    x = x.astype(np.complex64)
    rms = np.sqrt(np.mean(np.abs(x)**2) + 1e-12)
    if rms > 0: x /= rms
    return x

# ---------------------------- Scene builder ----------------------------------
def build_scene(fs, T, amps, noise_sigma, fc, rng=None):
    
    # Identify num. of samples, N
    N = int(round(fs*T)) 
    
    # Build each signal by calling their function definitions and scale with signal amplitude
    ofdm = amps[0] * ofdm_with_cp(fs, N, n_subc=64, used_frac=0.85, sym_rate=400_000, cp_frac=1/4, rng=rng)
    gfsk = amps[1] * gfsk_bursts(fs, N, sym_rate=150_000, h=0.75, bt=0.75, n_bursts=10, duty=0.5, rng=rng)
    uav  = amps[2] * dsss_bpsk_hidden(fs, N, data_rate=20_000, chip_rate=1_000_000, rng=rng)
    
    # Generate a normal distribution of white noise to simulate environmental effects
    noise = (np.random.default_rng(99).normal(0, noise_sigma, N) + 1j*np.random.default_rng(100).normal(0, noise_sigma, N)).astype(np.complex64)
    
    # All signals mixed together as the receiver would see it
    x = (ofdm + gfsk + uav + noise).astype(np.complex64)

    # All signal snippets isolated for ground truth
    signals = {"ofdm": ofdm, "gfsk": gfsk, "uav": uav, "noise": noise}
    return x, signals

def quicklook(x, fs, title):
    # ----------------- Power Spectral Density -----------------
    x -= np.mean(x) #--------------------------------- [Remove DC component]
    N = len(x) # ------------------------------------- [Num. of samples
    w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(N)/(N-1)) # [Apply a hanning window]
    X = np.fft.fftshift(np.fft.fft(x*w)) #------------ [FFT & shift]
    U = (np.sum(w**2)/N) #---------------------------- [Compute window power]
    Pxx = (np.abs(X)**2)/(fs*N*U + 1e-15) #----------- [Compute PSD and Normalize]
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs)) #-- [shift frequency bins]
    # Plot
    plt.figure(figsize=(9,4)); plt.plot(f/1e6, 10*np.log10(Pxx+1e-18))
    plt.xlabel("Frequency offset from fc (MHz)"); plt.ylabel("PSD (dB/Hz)"); 
    plt.title(title); 
    plt.grid(True); plt.tight_layout();
    plt.savefig(title + "_PSD_quicklook.png", dpi=150)
    plt.close()
    print(f"{title + "_PSD_quicklook.png"} saved.\n")
    
    # ----------------- Spectrogram -----------------
    # Settings (tweak as needed)    
    nfft_spec = 4096 #------------------------------ [Establish nfft length]
    overlap   = 0.5 #------------------------------- [Establish overlap for computation]
    noverlap = int(nfft_spec* overlap) #------------ [compute num. of samples in overlap]
    drange_db = 60 #-------------------------------- [dynamic range, dB]
    w_spec = get_window('hann', nfft_spec, fftbins=True) 

    # STFT and Compute PSDs
    freqs, times, Z = stft(x, fs, w_spec, nfft_spec, noverlap, return_onesided=False, boundary=None, padded=False)
    Zshft = np.fft.fftshift(Z, axes=0) #------------ [FFT & shift]
    freqs_shft = np.fft.fftshift(freqs) #----------- [FFT & shift]

    # ---- zoom around expected deviation (GFSK only) ----
    if title == "gfsk":
        f_dev = 150e3 #---------------------------- [Expected deviation]
        band = 6 * f_dev #------------------------- [show +/- 6*f_dev]
        mask = np.abs(freqs_shft) <= band #-------- [create mask for zoomed band]
        Zshft = Zshft[mask, :] #------------------- [Apply mask]
        freqs_shft = freqs_shft[mask] #------------- [Apply mask]
    U = np.sum(w_spec**2) #------------------------ [window power]
    Sxx = (np.abs(Zshft)**2) / (fs * U) #---------- [Compute PSD & normalize]
    Sxx_db = 10*np.log10(Sxx + 1e-18)


    # Color scaling
    vmax = np.percentile(Sxx_db, 98)
    vmin = vmax - drange_db

    plt.figure(figsize=(9,4.8))
    extent = [times[0], times[-1], freqs_shft[0]/1e6, freqs_shft[-1]/1e6]
    plt.imshow(Sxx_db, origin='lower', aspect='auto', extent=extent,
               vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar(label='PSD (dB/Hz)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency offset from fc (MHz)")
    plt.title(title + " — Spectrogram")
    plt.tight_layout()
    plt.savefig(title + "_SPEC_quicklook.png", dpi=150)
    plt.close()
    print(f"{title + "_SPEC_quicklook.png"} saved.\n")

def main():
    #----------------------------Define basic parameters-----------------------------------------------------
    fs = 20_000_000 #------------------------------------------------- [sample frequency, Hz]
    T = 0.25 #-------------------------------------------------------- [Snippet length, seconds]
    amps = (0.40, 0.40, 0.25) #-------------------------------------- [Signal Amplitude, (ofdm, gfsk, dsss)]
    noise_sigma = 0.10 #---------------------------------------------- [Normal Gaussian Noise, stddev]
    fc = 2412*1e6 #--------------------------------------------------- [WiFi Ch1, Center frequency, Hz]
    #--------------------------------------------------------------------------------------------------------
    x, signals = build_scene(fs, T, amps, noise_sigma, fc)
    print(f"Scene built.\n")
    
    titles = list(signals)
    quicklook(signals["ofdm"], fs, titles[0])
    quicklook(signals["gfsk"], fs, titles[1])
    quicklook(signals["uav"], fs, titles[2])
    quicklook(signals["noise"], fs, titles[3])
    quicklook(x, fs, title="Co-channel congestion @ 2.4 GHz")
    
    print(f"Center Frequency = {fc/1e6:.3f} MHz")
    return x, signals, fs, fc, T

if __name__ == "__main__":
    main()
