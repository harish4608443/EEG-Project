"""
UPDATE FIGURES — Supervisor feedback round 1
=============================================
1. fig_topomaps.png     → Grand-average difference topomaps at 8 time points,
                          with ~100ms dense sampling around peak (370–470ms).
2. fig_tfr_fullband.png → Crop time axis to 900ms (hide wavelet edge artefact).
3. fig_cluster_permutation_tfr.png → Same 900ms crop on all panels.
"""

import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR    = Path(r'd:\ds004347')
OUTPUT_DIR  = BASE_DIR / 'derivatives' / 'preprocessing_results'
FIGURES_DIR = BASE_DIR / 'report' / 'figures'
SUBJECTS    = [f'sub-{i:03d}' for i in range(1, 25)]

CROP_END_MS = 900   # crop TFR display here to hide wavelet edge artefact

# ============================================================================
# HELPER – load grand-average difference evoked
# ============================================================================

def load_grand_average_diff():
    """Load all subjects' evokeds, return grand-average symmetric, random, diff."""
    all_sym, all_rand = [], []
    for subj in SUBJECTS:
        ave_path = OUTPUT_DIR / f'{subj}_ours_ave.fif'
        if not ave_path.exists():
            continue
        evs = mne.read_evokeds(ave_path, verbose=False)
        for ev in evs:
            ev.info['bads'] = []
        # drop EXG/aux channels
        drop = [c for c in evs[0].ch_names
                if any(c.startswith(p) for p in ('EXG','GSR','Erg','Resp','Plet','Temp'))]
        for ev in evs:
            ev.drop_channels([c for c in drop if c in ev.ch_names])
        all_sym.append(evs[0])
        all_rand.append(evs[1])

    ga_sym  = mne.grand_average(all_sym)
    ga_rand = mne.grand_average(all_rand)
    ga_diff = mne.combine_evoked([ga_sym, ga_rand], weights=[1, -1])
    return ga_sym, ga_rand, ga_diff


# ============================================================================
# FIGURE 1 – Updated topomaps (grand-average, 8 time points around peak)
# ============================================================================

def update_topomaps():
    print("\n" + "="*65)
    print("UPDATING: fig_topomaps.png")
    print("="*65)

    ga_sym, ga_rand, ga_diff = load_grand_average_diff()

    # Find peak within analysis window
    oz_idx  = ga_diff.ch_names.index('Oz')
    t_ms    = ga_diff.times * 1000
    win_mask = (t_ms >= 300) & (t_ms <= 700)
    peak_ms  = t_ms[win_mask][np.argmax(np.abs(ga_diff.data[oz_idx, win_mask]))]
    print(f"  Grand-average peak: {peak_ms:.0f} ms")

    # 8 time points:  early context | ~100ms peak window | late context
    time_points_ms = [100, 250,
                      peak_ms - 50, peak_ms, peak_ms + 50,
                      550, 700, 850]
    time_points_ms = [float(t) for t in time_points_ms]

    # Global colour limit (98th pct of abs diff across peak window)
    win_data = ga_diff.data[:, win_mask]
    vlim = float(np.percentile(np.abs(win_data), 98)) * 1e6   # µV

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, t_ms_val in enumerate(time_points_ms):
        t_idx = np.argmin(np.abs(ga_diff.times * 1000 - t_ms_val))
        plot_data = ga_diff.data[:, t_idx] * 1e6  # µV

        im, _ = mne.viz.plot_topomap(
            plot_data,
            ga_diff.info,
            axes=axes[i],
            show=False,
            vlim=(-vlim, vlim),
            cmap='RdBu_r',
            contours=6
        )

        # Labels — highlight the peak-window panels
        actual_ms = ga_diff.times[t_idx] * 1000
        in_peak_win = (actual_ms >= peak_ms - 55) and (actual_ms <= peak_ms + 55)
        weight = 'heavy' if in_peak_win else 'normal'
        col    = 'darkred' if in_peak_win else 'black'
        # peak itself
        if abs(actual_ms - peak_ms) < 5:
            label = f'{actual_ms:.0f} ms  ← peak'
        else:
            label = f'{actual_ms:.0f} ms'
        axes[i].set_title(label, fontsize=11, fontweight=weight, color=col)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.75])
    plt.colorbar(im, cax=cbar_ax, label='Sym − Rand  (µV)')

    # Bracket annotation for peak window
    fig.text(0.5, 0.02,
             f'← Early context │  ≈100 ms peak window ({peak_ms-50:.0f}–{peak_ms+50:.0f} ms)  │  Late context →',
             ha='center', fontsize=10, color='#444444', style='italic')

    plt.suptitle(
        f'Grand-Average Topomap:  Symmetric − Random\n'
        f'Peak at {peak_ms:.0f} ms (Oz electrode);  colour scale ± {vlim:.2f} µV',
        fontsize=13, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0.05, 0.91, 0.96])

    outpath = FIGURES_DIR / 'fig_topomaps.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {outpath}")
    plt.close()


# ============================================================================
# FIGURE 2 – TFR full-band, cropped to CROP_END_MS
# ============================================================================

def update_tfr_fullband():
    print("\n" + "="*65)
    print("UPDATING: fig_tfr_fullband.png")
    print("="*65)

    all_sym, all_rand = [], []
    freqs     = np.logspace(np.log10(4), np.log10(40), 20)
    n_cycles  = freqs / 2.

    for subj in SUBJECTS:
        epo_path = OUTPUT_DIR / f'{subj}_ours_epo.fif'
        if not epo_path.exists():
            continue
        print(f"  {subj}...", end=' ', flush=True)
        epochs = mne.read_epochs(epo_path, verbose=False)
        epochs.pick_channels(['Oz'])
        ps = mne.time_frequency.tfr_morlet(
            epochs['Regular'], freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False)
        pr = mne.time_frequency.tfr_morlet(
            epochs['Random'],  freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False)
        all_sym.append(ps)
        all_rand.append(pr)
        print("done")

    ga_sym  = mne.grand_average(all_sym)
    ga_rand = mne.grand_average(all_rand)
    ga_diff = ga_sym.copy()
    ga_diff.data = ga_sym.data - ga_rand.data

    t_ms  = ga_sym.times * 1000
    # Crop indices for display
    t_start_ms, t_end_ms = -200, CROP_END_MS

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.32)

    extent = [t_ms[0], t_ms[-1], freqs[0], freqs[-1]]

    def _tfr_ax(ax, data, title, cmap='RdBu_r', vmin=None, vmax=None):
        im = ax.imshow(data, aspect='auto', origin='lower',
                       extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlim([t_start_ms, t_end_ms])
        ax.axvline(0, color='white', ls='--', lw=2)
        ax.axhline(8,  color='yellow', ls=':', lw=1, alpha=0.6)
        ax.axhline(12, color='yellow', ls=':', lw=1, alpha=0.6)
        ax.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        # shade artefact-free analysis window
        ax.axvspan(300, 700, alpha=0.08, color='lime')
        return im

    # Row 0 — per-condition and difference spectrograms
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = _tfr_ax(ax0, ga_sym.data[0], 'Symmetric – Power (Oz)')
    plt.colorbar(im0, ax=ax0, label='Power (dB)')

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = _tfr_ax(ax1, ga_rand.data[0], 'Random – Power (Oz)')
    plt.colorbar(im1, ax=ax1, label='Power (dB)')

    ax2 = fig.add_subplot(gs[0, 2])
    vmax = np.max(np.abs(ga_diff.data[0]))
    im2 = _tfr_ax(ax2, ga_diff.data[0], 'Difference (Sym − Rand)', vmin=-vmax, vmax=vmax)
    ax2.axvline(300, color='lime', ls=':', lw=2, alpha=0.8)
    ax2.axvline(700, color='lime', ls=':', lw=2, alpha=0.8)
    plt.colorbar(im2, ax=ax2, label='Δ Power (dB)')

    # Row 1 — per-band time courses
    bands = {
        'Theta (4–7 Hz)':  (4,  7, '#e41a1c'),
        'Alpha (8–12 Hz)': (8, 12, '#377eb8'),
        'Beta (13–30 Hz)':(13, 30, '#4daf4a'),
    }
    for col_i, (band_name, (flo, fhi, col)) in enumerate(bands.items()):
        ax = fig.add_subplot(gs[1, col_i])
        fmask   = (freqs >= flo) & (freqs <= fhi)
        sym_b   = ga_sym.data[0, fmask, :].mean(0)
        rand_b  = ga_rand.data[0, fmask, :].mean(0)
        diff_b  = sym_b - rand_b

        ax.plot(t_ms, sym_b, 'b',    lw=2, label='Symmetric', alpha=0.7)
        ax.plot(t_ms, rand_b, 'r',   lw=2, label='Random',    alpha=0.7)
        ax.plot(t_ms, diff_b, color=col, lw=2.5, label='Difference', alpha=0.9)
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(0, color='black', ls='--', lw=1)
        ax.axvspan(300, 700, alpha=0.1, color='green')
        ax.set_xlim([t_start_ms, t_end_ms])
        ax.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Power (dB)', fontsize=11, fontweight='bold')
        ax.set_title(band_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # Edge artefact annotation
    fig.text(0.98, 0.01,
             f'Note: display cropped at {CROP_END_MS} ms to avoid Morlet wavelet edge artefact.',
             ha='right', fontsize=8, color='#777777', style='italic')

    plt.suptitle('TIME-FREQUENCY ANALYSIS: Oscillatory Power Differences (Electrode Oz)',
                 fontsize=15, fontweight='bold', y=0.999)

    outpath = FIGURES_DIR / 'fig_tfr_fullband.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  Saved: {outpath}")
    plt.close()


# ============================================================================
# FIGURE 3 – Cluster permutation TFR, same 900ms crop
# ============================================================================

def _label_clusters_2d(mask):
    from scipy.ndimage import label as ndlabel
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, n = ndlabel(mask, structure=struct)
    return labeled, n


def update_cluster_permutation():
    print("\n" + "="*65)
    print("UPDATING: fig_cluster_permutation_tfr.png")
    print("="*65)

    freqs    = np.logspace(np.log10(4), np.log10(40), 20)
    n_cycles = freqs / 2.

    tfr_sym_all, tfr_rand_all = [], []

    for subj in SUBJECTS:
        epo_path = OUTPUT_DIR / f'{subj}_ours_epo.fif'
        if not epo_path.exists():
            continue
        print(f"  {subj}...", end=' ', flush=True)
        epochs = mne.read_epochs(epo_path, verbose=False)
        epochs.pick_channels(['Oz'])
        ts = mne.time_frequency.tfr_morlet(
            epochs['Regular'], freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False)
        tr = mne.time_frequency.tfr_morlet(
            epochs['Random'],  freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False)
        tfr_sym_all.append(ts.data[0])
        tfr_rand_all.append(tr.data[0])
        print("done")

    tfr_sym_all  = np.array(tfr_sym_all)
    tfr_rand_all = np.array(tfr_rand_all)
    diff_all     = tfr_sym_all - tfr_rand_all

    ga_sym_tfr = np.mean(tfr_sym_all, 0)
    ga_rand_tfr = np.mean(tfr_rand_all, 0)

    times_ms = ts.times * 1000

    # ---- Permutation test -----------------------------------------------
    print("\n  Running cluster permutation test (1000 permutations)...")
    n_subj, n_freqs, n_times = diff_all.shape
    t_obs, _ = stats.ttest_1samp(diff_all, 0, axis=0)

    rng = np.random.default_rng(42)
    cluster_ts_null = []
    for _ in range(1000):
        signs = rng.choice([-1, 1], size=n_subj)[:, None, None]
        t_p, _ = stats.ttest_1samp(diff_all * signs, 0, axis=0)
        above  = np.abs(t_p) > 2.0
        labeled, n_lab = _label_clusters_2d(above)
        if n_lab == 0:
            cluster_ts_null.append(0)
        else:
            masses = [np.abs(t_p[labeled == k]).sum() for k in range(1, n_lab+1)]
            cluster_ts_null.append(max(masses))

    cluster_ts_null = np.array(cluster_ts_null)
    thresh_95 = np.percentile(cluster_ts_null, 95)

    labeled_obs, n_lab_obs = _label_clusters_2d(np.abs(t_obs) > 2.0)
    sig_mask = np.zeros_like(t_obs, dtype=bool)
    for k in range(1, n_lab_obs + 1):
        mass = np.abs(t_obs[labeled_obs == k]).sum()
        if mass > thresh_95:
            sig_mask[labeled_obs == k] = True

    print(f"  Significant TF points: {sig_mask.sum()} (threshold={thresh_95:.2f})")

    # ---- Figure ----------------------------------------------------------
    t_start_ms, t_end_ms = -200, CROP_END_MS
    fig = plt.figure(figsize=(18, 13))
    gsfig = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    extent = [times_ms[0], times_ms[-1], freqs[0], freqs[-1]]

    def _tfr_panel(ax, data, title, cmap='RdBu_r', vmin=None, vmax=None):
        im = ax.imshow(data, aspect='auto', origin='lower',
                       extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlim([t_start_ms, t_end_ms])
        ax.axvline(0, color='w', lw=2, ls='--')
        ax.axvspan(300, 700, alpha=0.15, color='lime')
        ax.set_xlabel('Time (ms)', fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        return im

    # (a) Sym power
    ax0 = fig.add_subplot(gsfig[0, 0])
    im0 = _tfr_panel(ax0, ga_sym_tfr, '(a) Symmetric – Power')
    plt.colorbar(im0, ax=ax0, label='Power (dB)')

    # (b) Rand power
    ax1 = fig.add_subplot(gsfig[0, 1])
    im1 = _tfr_panel(ax1, ga_rand_tfr, '(b) Random – Power')
    plt.colorbar(im1, ax=ax1, label='Power (dB)')

    # (c) Difference + contour
    ax2 = fig.add_subplot(gsfig[0, 2])
    diff_ga = ga_sym_tfr - ga_rand_tfr
    vmax_d  = np.max(np.abs(diff_ga))
    im2 = _tfr_panel(ax2, diff_ga, '(c) Difference (Sym−Rand)\nBlack = significant cluster',
                     vmin=-vmax_d, vmax=vmax_d)
    if sig_mask.any():
        _t = np.linspace(times_ms[0], times_ms[-1], n_times)
        _f = np.linspace(freqs[0], freqs[-1], n_freqs)
        ax2.contour(_t, _f, sig_mask.astype(float), levels=[0.5],
                    colors='black', linewidths=2.5)
    plt.colorbar(im2, ax=ax2, label='Δ Power (dB)')

    # (d) t-map
    ax3 = fig.add_subplot(gsfig[1, 0])
    tv = np.max(np.abs(t_obs))
    im3 = _tfr_panel(ax3, t_obs, '(d) t-statistic map\nContour = cluster p < 0.05',
                     cmap='RdBu_r', vmin=-tv, vmax=tv)
    if sig_mask.any():
        ax3.contour(_t, _f, sig_mask.astype(float), levels=[0.5],
                    colors='black', linewidths=2.5)
    plt.colorbar(im3, ax=ax3, label='t-value')

    # (e) Null distribution
    ax4 = fig.add_subplot(gsfig[1, 1])
    ax4.hist(cluster_ts_null, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
    ax4.axvline(thresh_95, color='red', lw=3, ls='--', label=f'95th pct = {thresh_95:.1f}')
    obs_masses = [np.abs(t_obs[labeled_obs == k]).sum()
                  for k in range(1, n_lab_obs+1)
                  if np.abs(t_obs[labeled_obs == k]).sum() > thresh_95]
    if obs_masses:
        ax4.axvline(max(obs_masses), color='green', lw=3, label=f'Observed = {max(obs_masses):.1f}')
    ax4.set_xlabel('Max cluster t-sum', fontweight='bold')
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('(e) Permutation Null Distribution\n1000 permutations', fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    # (f) Band time courses
    ax5 = fig.add_subplot(gsfig[1, 2])
    bands = {'Theta (4–7 Hz)': (4, 7, '#e41a1c'),
             'Alpha (8–12 Hz)': (8, 12, '#377eb8'),
             'Beta (13–30 Hz)': (13, 30, '#4daf4a')}
    for lbl, (flo, fhi, col) in bands.items():
        fmask    = (freqs >= flo) & (freqs <= fhi)
        sym_band = ga_sym_tfr[fmask].mean(0)
        rnd_band = ga_rand_tfr[fmask].mean(0)
        diff_b   = sym_band - rnd_band
        ax5.plot(times_ms, diff_b, color=col, lw=2.5, label=lbl)
        sig_band = sig_mask[fmask].any(0)
        ax5.fill_between(times_ms, diff_b, where=sig_band, color=col, alpha=0.35)

    ax5.axhline(0, color='black', lw=1.5)
    ax5.axvline(0, color='black', lw=1, ls='--')
    ax5.axvspan(300, 700, alpha=0.1, color='green')
    ax5.set_xlim([t_start_ms, t_end_ms])
    ax5.set_xlabel('Time (ms)', fontweight='bold')
    ax5.set_ylabel('Δ Power (dB)', fontweight='bold')
    ax5.set_title('(f) Band time courses\nShaded = significant', fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # Edge artefact note
    fig.text(0.99, 0.005,
             f'Display cropped at {CROP_END_MS} ms to avoid Morlet wavelet edge artefact.',
             ha='right', fontsize=7.5, color='#777777', style='italic')

    plt.suptitle('Cluster-Based Permutation Tests on Time-Frequency Power\n'
                 '(Symmetric − Random, Electrode Oz)',
                 fontsize=15, fontweight='bold')

    outpath = FIGURES_DIR / 'fig_cluster_permutation_tfr.png'
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  Saved: {outpath}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import time
    t0 = time.time()

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   UPDATE FIGURES — Supervisor feedback                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    try:
        update_topomaps()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Topomaps: {e}")
        traceback.print_exc()

    try:
        update_tfr_fullband()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] TFR: {e}")
        traceback.print_exc()

    try:
        update_cluster_permutation()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Cluster permutation: {e}")
        traceback.print_exc()

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("\n✓ Done. Updated figures in d:\\ds004347\\report\\figures\\")
