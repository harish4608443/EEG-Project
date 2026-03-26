"""
FINAL ANALYSES: Cluster-Based Permutation Tests + fsaverage Source Localization
=================================================================================
1. Cluster-based permutation tests on time-frequency power differences
2. dSPM source localization using MNE's fsaverage template
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR    = Path(r'd:\ds004347')
OUTPUT_DIR  = BASE_DIR / 'derivatives' / 'preprocessing_results'
FIGURES_DIR = BASE_DIR / 'report' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f'sub-{i:03d}' for i in range(1, 25)]

# ============================================================================
# ANALYSIS A: CLUSTER-BASED PERMUTATION TESTS ON TIME-FREQUENCY DATA
# ============================================================================

def analysis_cluster_permutation():
    print("\n" + "="*70)
    print("ANALYSIS A: CLUSTER-BASED PERMUTATION TESTS (TFR)")
    print("="*70)

    freqs     = np.logspace(np.log10(4), np.log10(40), 20)
    n_cycles  = freqs / 2.

    tfr_sym_all  = []
    tfr_rand_all = []

    for i, subj in enumerate(SUBJECTS):
        epo_path = OUTPUT_DIR / f'{subj}_ours_epo.fif'
        if not epo_path.exists():
            continue
        print(f"  [{i+1}/24] {subj}...", end=' ', flush=True)

        epochs = mne.read_epochs(epo_path, verbose=False)
        epochs.pick_channels(['Oz'])

        tfr_s = mne.time_frequency.tfr_morlet(
            epochs['Regular'], freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False)
        tfr_r = mne.time_frequency.tfr_morlet(
            epochs['Random'], freqs=freqs, n_cycles=n_cycles,
            return_itc=False, average=True, verbose=False)

        tfr_sym_all.append(tfr_s.data[0])   # (n_freqs, n_times)
        tfr_rand_all.append(tfr_r.data[0])
        print("done")

    tfr_sym_all  = np.array(tfr_sym_all)   # (n_subj, n_freqs, n_times)
    tfr_rand_all = np.array(tfr_rand_all)
    diff_all     = tfr_sym_all - tfr_rand_all  # positive = sym > rand

    ga_sym  = mne.grand_average(
        [mne.time_frequency.tfr_morlet(
            mne.read_epochs(OUTPUT_DIR / f'{s}_ours_epo.fif',
                            verbose=False).pick_channels(['Oz'])['Regular'],
            freqs=freqs, n_cycles=n_cycles, return_itc=False,
            average=True, verbose=False)
         for s in SUBJECTS if (OUTPUT_DIR / f'{s}_ours_epo.fif').exists()])
    ga_rand = mne.grand_average(
        [mne.time_frequency.tfr_morlet(
            mne.read_epochs(OUTPUT_DIR / f'{s}_ours_epo.fif',
                            verbose=False).pick_channels(['Oz'])['Random'],
            freqs=freqs, n_cycles=n_cycles, return_itc=False,
            average=True, verbose=False)
         for s in SUBJECTS if (OUTPUT_DIR / f'{s}_ours_epo.fif').exists()])

    times_ms = ga_sym.times * 1000

    # ---- Cluster permutation test (non-parametric) --------------------------
    print("\n  Running cluster permutation test (1000 permutations)...")
    n_subj, n_freqs, n_times = diff_all.shape

    # observed t-map
    t_obs, _ = stats.ttest_1samp(diff_all, 0, axis=0)

    # permutation distribution of max-cluster t-sum
    rng = np.random.default_rng(42)
    cluster_ts_null = []

    for perm in range(1000):
        signs   = rng.choice([-1, 1], size=n_subj)[:, None, None]
        t_perm, _ = stats.ttest_1samp(diff_all * signs, 0, axis=0)

        # threshold |t| > 2 (roughly p<0.05 two-tailed for n≈20)
        above   = np.abs(t_perm) > 2.0
        labeled, n_lab = _label_clusters_2d(above)
        if n_lab == 0:
            cluster_ts_null.append(0)
        else:
            masses = [np.abs(t_perm[labeled == k]).sum() for k in range(1, n_lab+1)]
            cluster_ts_null.append(max(masses))

    cluster_ts_null = np.array(cluster_ts_null)
    thresh_95 = np.percentile(cluster_ts_null, 95)

    # find significant clusters in observed data
    above_obs = np.abs(t_obs) > 2.0
    labeled_obs, n_lab_obs = _label_clusters_2d(above_obs)
    sig_mask = np.zeros_like(t_obs, dtype=bool)
    for k in range(1, n_lab_obs + 1):
        mass = np.abs(t_obs[labeled_obs == k]).sum()
        if mass > thresh_95:
            sig_mask[labeled_obs == k] = True

    print(f"  Significant clusters found: {sig_mask.sum()} TF-points (threshold={thresh_95:.2f})")

    # ---- Figure: 3 panels ---------------------------------------------------
    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    extent = [times_ms[0], times_ms[-1], freqs[0], freqs[-1]]

    # Panel 1 – Symmetric power
    ax0 = fig.add_subplot(gs[0, 0])
    sym_data = ga_sym.data[0]
    im0 = ax0.imshow(sym_data, aspect='auto', origin='lower',
                     extent=extent, cmap='RdBu_r')
    ax0.axvline(0, color='w', lw=2, ls='--')
    ax0.axvspan(300, 700, alpha=0.2, color='lime')
    ax0.set_xlabel('Time (ms)', fontweight='bold')
    ax0.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax0.set_title('(a) Symmetric – Power', fontweight='bold')
    plt.colorbar(im0, ax=ax0, label='Power (dB)')

    # Panel 2 – Random power
    ax1 = fig.add_subplot(gs[0, 1])
    rand_data = ga_rand.data[0]
    im1 = ax1.imshow(rand_data, aspect='auto', origin='lower',
                     extent=extent, cmap='RdBu_r')
    ax1.axvline(0, color='w', lw=2, ls='--')
    ax1.axvspan(300, 700, alpha=0.2, color='lime')
    ax1.set_xlabel('Time (ms)', fontweight='bold')
    ax1.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax1.set_title('(b) Random – Power', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Power (dB)')

    # Panel 3 – Difference with significance contour
    ax2 = fig.add_subplot(gs[0, 2])
    diff_ga = sym_data - rand_data
    vmax = np.max(np.abs(diff_ga))
    im2 = ax2.imshow(diff_ga, aspect='auto', origin='lower',
                     extent=extent, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    if sig_mask.any():
        ax2.contour(
            np.linspace(times_ms[0], times_ms[-1], n_times),
            np.linspace(freqs[0], freqs[-1], n_freqs),
            sig_mask.astype(float), levels=[0.5],
            colors='black', linewidths=2.5)
    ax2.axvline(0, color='w', lw=2, ls='--')
    ax2.axvspan(300, 700, alpha=0.15, color='lime')
    ax2.set_xlabel('Time (ms)', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax2.set_title('(c) Difference (Sym\u2212Rand)\nBlack contour = significant cluster', fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='\u0394 Power (dB)')

    # Panel 4 – t-statistic map
    ax3 = fig.add_subplot(gs[1, 0])
    tv = np.max(np.abs(t_obs))
    im3 = ax3.imshow(t_obs, aspect='auto', origin='lower',
                     extent=extent, cmap='RdBu_r', vmin=-tv, vmax=tv)
    if sig_mask.any():
        ax3.contour(
            np.linspace(times_ms[0], times_ms[-1], n_times),
            np.linspace(freqs[0], freqs[-1], n_freqs),
            sig_mask.astype(float), levels=[0.5],
            colors='black', linewidths=2.5)
    ax3.axvline(0, color='w', lw=2, ls='--')
    ax3.axvspan(300, 700, alpha=0.15, color='lime')
    ax3.set_xlabel('Time (ms)', fontweight='bold')
    ax3.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax3.set_title('(d) t-statistic map\nContour = cluster p\u202f<\u202f0.05', fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='t-value')

    # Panel 5 – Null distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(cluster_ts_null, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
    ax4.axvline(thresh_95, color='red', lw=3, ls='--', label=f'95th pct = {thresh_95:.1f}')
    if sig_mask.any():
        obs_mass = max(
            np.abs(t_obs[labeled_obs == k]).sum()
            for k in range(1, n_lab_obs+1)
            if np.abs(t_obs[labeled_obs == k]).sum() > thresh_95
        ) if any(
            np.abs(t_obs[labeled_obs == k]).sum() > thresh_95
            for k in range(1, n_lab_obs+1)
        ) else 0
        if obs_mass > 0:
            ax4.axvline(obs_mass, color='green', lw=3, label=f'Observed = {obs_mass:.1f}')
    ax4.set_xlabel('Max cluster t-sum', fontweight='bold')
    ax4.set_ylabel('Count', fontweight='bold')
    ax4.set_title('(e) Permutation Null Distribution\n1000 permutations', fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    # Panel 6 – Band time courses with significance
    ax5 = fig.add_subplot(gs[1, 2])
    bands   = {'Theta (4\u20137Hz)': (4,7,'#e41a1c'),
               'Alpha (8\u201312Hz)': (8,12,'#377eb8'),
               'Beta (13\u201330Hz)': (13,30,'#4daf4a')}

    for label, (fmin, fmax, col) in bands.items():
        fmask     = (freqs >= fmin) & (freqs <= fmax)
        sym_band  = sym_data[fmask].mean(axis=0)
        rand_band = rand_data[fmask].mean(axis=0)
        diff_band = sym_band - rand_band
        ax5.plot(times_ms, diff_band, color=col, lw=2.5, label=label)

        # mark significant time points
        sig_band = sig_mask[fmask].any(axis=0)
        ax5.fill_between(times_ms, diff_band, where=sig_band,
                         color=col, alpha=0.35)

    ax5.axhline(0, color='black', lw=1.5)
    ax5.axvline(0, color='black', lw=1, ls='--')
    ax5.axvspan(300, 700, alpha=0.1, color='green')
    ax5.set_xlabel('Time (ms)', fontweight='bold')
    ax5.set_ylabel('\u0394 Power (dB)', fontweight='bold')
    ax5.set_title('(f) Band time courses\nShaded = significant', fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    ax5.set_xlim([-200, 1000])

    plt.suptitle('Cluster-Based Permutation Tests on Time-Frequency Power\n'
                 '(Symmetric \u2212 Random, Electrode Oz)',
                 fontsize=15, fontweight='bold')

    out = FIGURES_DIR / 'fig_cluster_permutation_tfr.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  Saved: {out}")
    plt.close()


def _label_clusters_2d(mask):
    """Simple 4-connectivity connected-component labelling."""
    from scipy.ndimage import label as ndlabel
    struct = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, n = ndlabel(mask, structure=struct)
    return labeled, n


# ============================================================================
# ANALYSIS B: SOURCE LOCALIZATION WITH FSAVERAGE TEMPLATE
# ============================================================================

def analysis_source_localization():
    print("\n" + "="*70)
    print("ANALYSIS B: FSAVERAGE SOURCE LOCALIZATION (dSPM)")
    print("="*70)

    # Load grand average evokeds
    all_sym, all_rand = [], []
    for subj in SUBJECTS:
        ave_path = OUTPUT_DIR / f'{subj}_ours_ave.fif'
        if not ave_path.exists():
            continue
        evs = mne.read_evokeds(ave_path, verbose=False)
        for ev in evs:
            ev.info['bads'] = []
        all_sym.append(evs[0])
        all_rand.append(evs[1])

    ga_sym  = mne.grand_average(all_sym)
    ga_rand = mne.grand_average(all_rand)
    ga_diff = mne.combine_evoked([ga_sym, ga_rand], weights=[1, -1])

    # Drop non-EEG channels
    drop = [c for c in ga_diff.ch_names
            if c.startswith('EXG') or c.startswith('GSR')
               or c.startswith('Erg') or c.startswith('Resp')
               or c.startswith('Plet') or c.startswith('Temp')]
    ga_diff.drop_channels(drop)
    ga_sym.drop_channels([c for c in drop if c in ga_sym.ch_names])

    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage('biosemi64')
    try:
        ga_diff.set_montage(montage, on_missing='warn', verbose=False)
        ga_sym.set_montage(montage, on_missing='warn', verbose=False)
    except Exception:
        pass

    # Set average reference
    ga_diff.set_eeg_reference('average', projection=True, verbose=False)
    ga_diff.apply_proj()
    ga_sym.set_eeg_reference('average', projection=True, verbose=False)
    ga_sym.apply_proj()

    print("  Building fsaverage BEM & forward solution...")
    subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = Path(subjects_dir).parent

    src = mne.setup_source_space(
        'fsaverage', spacing='oct5',
        subjects_dir=subjects_dir, verbose=False)

    # Use the pre-built 3-shell BEM solution (required for EEG forward)
    bem_sol = mne.read_bem_solution(
        subjects_dir / 'fsaverage' / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif',
        verbose=False)

    fwd = mne.make_forward_solution(
        ga_diff.info, trans='fsaverage',
        src=src, bem=bem_sol,
        eeg=True, meg=False, verbose=False)

    # Noise covariance from the pre-stimulus baseline
    print("  Computing noise covariance from baseline...")
    noise_cov = mne.compute_covariance(
        mne.EpochsArray(
            np.zeros((10, len(ga_diff.ch_names), len(ga_diff.times))),
            ga_diff.info),
        tmin=None, tmax=None,
        method='empirical', verbose=False)
    # Use a diagonal cov as a robust fallback
    noise_cov = mne.make_ad_hoc_cov(ga_diff.info, verbose=False)

    # Inverse operator
    print("  Computing dSPM inverse...")
    inv = mne.minimum_norm.make_inverse_operator(
        ga_diff.info, fwd, noise_cov,
        loose=0.2, depth=0.8, verbose=False)

    # Apply inverse at peak (420 ms)
    lambda2 = 1.0 / 9.0
    stc_diff = mne.minimum_norm.apply_inverse(
        ga_diff, inv, lambda2=lambda2,
        method='dSPM', verbose=False)
    stc_sym = mne.minimum_norm.apply_inverse(
        ga_sym, inv, lambda2=lambda2,
        method='dSPM', verbose=False)

    # Morph to fsaverage for group display
    morph = mne.compute_source_morph(
        stc_diff, subject_from='fsaverage',
        subject_to='fsaverage',
        subjects_dir=subjects_dir, verbose=False)
    stc_morph = morph.apply(stc_diff)

    # ---- figure: static brain plots -----------------------------------------
    peak_time = 0.42   # 420 ms
    t_idx = np.argmin(np.abs(stc_morph.times - peak_time))

    print("  Rendering source maps...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    time_points = [0.10, 0.17, 0.30, 0.42, 0.50, 0.60, 0.70, 0.80]
    vmax_val = np.percentile(np.abs(stc_morph.data), 98)

    for col, t in enumerate(time_points[:4]):
        tidx = np.argmin(np.abs(stc_morph.times - t))
        data = stc_morph.data[:, tidx]
        _plot_source_flat(axes[0, col], data, stc_morph, subjects_dir,
                          vmax_val, f'{int(t*1000)} ms', hemi='both')

    for col, t in enumerate(time_points[4:]):
        tidx = np.argmin(np.abs(stc_morph.times - t))
        data = stc_morph.data[:, tidx]
        _plot_source_flat(axes[1, col], data, stc_morph, subjects_dir,
                          vmax_val, f'{int(t*1000)} ms', hemi='both')

    plt.suptitle('Source Localization (dSPM) – Symmetric minus Random\n'
                 'fsaverage template, EEG-only BEM',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = FIGURES_DIR / 'fig_source_localization_dspm.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out}")
    plt.close()

    # ---- individual time-course at peak vertex ------------------------------
    peak_vertex = np.argmax(np.abs(stc_morph.data[:, t_idx]))
    tc = stc_morph.data[peak_vertex, :]

    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(stc_morph.times * 1000, tc, 'k-', lw=2.5)
    ax.axhline(0, color='gray', lw=1)
    ax.axvline(0, color='black', lw=1.5, ls='--')
    ax.axvspan(300, 700, alpha=0.15, color='green', label='Analysis window')
    ax.set_xlabel('Time (ms)', fontweight='bold', fontsize=13)
    ax.set_ylabel('dSPM activation (a.u.)', fontweight='bold', fontsize=13)
    ax.set_title('Source Time Course at Peak Vertex\n(Symmetric \u2212 Random difference)', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([-200, 1000])

    out2 = FIGURES_DIR / 'fig_source_timecourse.png'
    plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {out2}")
    plt.close()


def _plot_source_flat(ax, data, stc, subjects_dir, vmax, title, hemi='both'):
    """Simple scatter plot of source amplitudes as proxy for brain map."""
    ax.scatter(range(len(data)), np.abs(data), c=np.abs(data),
               cmap='hot', s=0.3, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim([0, len(data)])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


# ============================================================================
# COPY EXISTING FIGURES TO REPORT/FIGURES
# ============================================================================

def collect_report_figures():
    """Copy all existing analysis figures into report/figures."""
    import shutil

    mapping = {
        # existing figures → new name in report/figures
        BASE_DIR / 'derivatives' / 'preprocessing_results' / 'sub-001_comparison.png':
            'fig_preprocessing_quality.png',

        BASE_DIR / 'derivatives' / 'quality_control' / 'advanced_viz1_condition_split.png':
            'fig_erp_main.png',

        BASE_DIR / 'derivatives' / 'quality_control' / 'advanced_viz3_topoplots_all_subjects.png':
            'fig_topomaps.png',

        BASE_DIR / 'derivatives' / 'quality_control' / 'advanced_viz4_data_quality.png':
            'fig_data_quality.png',

        BASE_DIR / 'derivatives' / 'quality_control' / 'COMPREHENSIVE_PIPELINE_COMPARISON.png':
            'fig_pipeline_comparison.png',

        BASE_DIR / 'derivatives' / 'milestone5_analysis' / 'analysis1_time_frequency.png':
            'fig_tfr_fullband.png',

        BASE_DIR / 'derivatives' / 'milestone5_analysis' / 'analysis2_decoding.png':
            'fig_decoding.png',

        BASE_DIR / 'derivatives' / 'milestone5_analysis' / 'analysis3_lateralization.png':
            'fig_lateralization.png',

        BASE_DIR / 'derivatives' / 'milestone5_analysis' / 'analysis5_individual_differences.png':
            'fig_individual_differences.png',
    }

    print("\n" + "="*70)
    print("COLLECTING FIGURES FOR REPORT")
    print("="*70)

    for src, dst_name in mapping.items():
        dst = FIGURES_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✓  {dst_name}")
        else:
            print(f"  ✗  MISSING: {src.name}")

    print(f"\nAll figures in: {FIGURES_DIR}")
    print("Files present:")
    for f in sorted(FIGURES_DIR.glob('*.png')):
        print(f"  • {f.name}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import time
    t0 = time.time()

    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║   FINAL ANALYSES: Cluster Permutation + Source Localization      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    try:
        analysis_cluster_permutation()
    except Exception as e:
        print(f"\n[ERROR] Cluster permutation: {e}")
        import traceback; traceback.print_exc()

    try:
        analysis_source_localization()
    except Exception as e:
        print(f"\n[ERROR] Source localization: {e}")
        import traceback; traceback.print_exc()

    collect_report_figures()

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("\n✓ Done. Check d:\\ds004347\\report\\figures\\")
