"""
Quality Control Visualizations for EEG Preprocessing
Generates diagnostic plots to assess data quality and preprocessing effectiveness
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

# Paths
BASE_DIR = r'd:\ds004347'
SUBJECT = 'sub-001'
RAW_DATA_PATH = os.path.join(BASE_DIR, SUBJECT, 'eeg', f'{SUBJECT}_task-jacobsen_eeg.bdf')
OUTPUT_DIR = os.path.join(BASE_DIR, 'derivatives', 'quality_control')
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVENT_ID = {'Regular': 1, 'Random': 3}

print("="*80)
print("QUALITY CONTROL VISUALIZATIONS")
print("="*80)
print()

# ============================================================================
# LOAD RAW DATA
# ============================================================================
print("Loading raw data...")
raw_original = mne.io.read_raw_bdf(RAW_DATA_PATH, preload=True, verbose=False)
events = mne.find_events(raw_original, stim_channel='Status', verbose=False)
task_events = events[np.isin(events[:, 2], [1, 3])]

# Pick EEG channels
eeg_channels = mne.pick_types(raw_original.info, eeg=True, eog=True, exclude=[])
raw_original.pick(eeg_channels)

# Set montage
try:
    montage = mne.channels.make_standard_montage('biosemi64')
    raw_original.set_montage(montage, on_missing='ignore', verbose=False)
except:
    pass

# Create preprocessed copy
raw_preprocessed = raw_original.copy()
raw_preprocessed.filter(l_freq=0.1, h_freq=40.0, verbose=False)
raw_preprocessed.notch_filter(freqs=50, verbose=False)
raw_preprocessed.set_eeg_reference('average', projection=False, verbose=False)

print(f"  Loaded {len(raw_original.ch_names)} channels")
print(f"  Duration: {raw_original.times[-1]:.1f} seconds")
print()

# ============================================================================
# FIGURE 1: RAW vs PREPROCESSED DATA (10 seconds segment)
# ============================================================================
print("Creating Figure 1: Raw vs Preprocessed Comparison...")

# Select a few channels for clarity
channels_to_plot = ['Oz', 'Pz', 'Cz', 'Fz']
available_channels = [ch for ch in channels_to_plot if ch in raw_original.ch_names]

if not available_channels:
    # Fallback to first 4 channels
    available_channels = raw_original.ch_names[:4]

# Plot 10 seconds of data starting at 100s (avoid initial artifacts)
start_time = 100
duration = 10

# Create figure manually by extracting data
fig1, axes = plt.subplots(len(available_channels), 1, figsize=(15, 10), sharex=True)
fig1.suptitle('Raw vs Preprocessed Data Comparison (10 second segment)', fontsize=16, fontweight='bold')

# Get data segment
start_sample = int(start_time * raw_original.info['sfreq'])
n_samples = int(duration * raw_original.info['sfreq'])
times = np.arange(n_samples) / raw_original.info['sfreq'] + start_time

for idx, ch_name in enumerate(available_channels):
    ch_idx = raw_original.ch_names.index(ch_name)
    
    # Get data
    data_raw = raw_original.get_data(picks=[ch_idx])[0, start_sample:start_sample+n_samples] * 1e6  # Convert to µV
    data_prep = raw_preprocessed.get_data(picks=[ch_idx])[0, start_sample:start_sample+n_samples] * 1e6
    
    ax = axes[idx] if len(available_channels) > 1 else axes
    ax.plot(times, data_raw, color='red', alpha=0.7, linewidth=0.8, label='Raw')
    ax.plot(times, data_prep, color='blue', alpha=0.7, linewidth=0.8, label='Preprocessed')
    ax.set_ylabel(f'{ch_name}\n(µV)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper right')
    
axes[-1].set_xlabel('Time (s)', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{SUBJECT}_raw_vs_preprocessed.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {SUBJECT}_raw_vs_preprocessed.png")
plt.close()

# ============================================================================
# FIGURE 2: POWER SPECTRA
# ============================================================================
print("Creating Figure 2: Power Spectra...")

fig2, axes = plt.subplots(1, 2, figsize=(15, 5))
fig2.suptitle('Power Spectral Density: Raw vs Preprocessed', fontsize=16, fontweight='bold')

# Compute PSD for raw data
psd_raw = raw_original.compute_psd(fmax=100, verbose=False)
# Compute PSD for preprocessed data
psd_preprocessed = raw_preprocessed.compute_psd(fmax=100, verbose=False)

# Plot raw spectrum
ax1 = axes[0]
psd_raw.plot(picks='eeg', average=True, axes=ax1, show=False, spatial_colors=False)
ax1.set_title('RAW DATA Spectrum', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Power (µV²/Hz, dB)')
ax1.axvline(50, color='r', linestyle='--', linewidth=2, label='50 Hz (power line)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot preprocessed spectrum
ax2 = axes[1]
psd_preprocessed.plot(picks='eeg', average=True, axes=ax2, show=False, spatial_colors=False)
ax2.set_title('PREPROCESSED Spectrum', fontsize=14, fontweight='bold')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power (µV²/Hz, dB)')
ax2.axvline(50, color='r', linestyle='--', linewidth=2, alpha=0.3, label='50 Hz removed')
ax2.axvline(40, color='g', linestyle='--', linewidth=2, label='40 Hz cutoff')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{SUBJECT}_spectra.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {SUBJECT}_spectra.png")
plt.close()

# ============================================================================
# FIGURE 3: BUTTERFLY PLOTS (All channels overlaid)
# ============================================================================
print("Creating Figure 3: Butterfly Plots...")

# Create epochs for butterfly plot
epochs = mne.Epochs(
    raw_preprocessed, 
    task_events, 
    event_id=EVENT_ID,
    tmin=-0.2, 
    tmax=1.0, 
    baseline=(-0.2, 0),
    preload=True,
    verbose=False
)

# Average
evoked_regular = epochs['Regular'].average()
evoked_random = epochs['Random'].average()

fig3, axes = plt.subplots(1, 2, figsize=(15, 6))
fig3.suptitle('Butterfly Plots: All Channels Overlaid', fontsize=16, fontweight='bold')

# Regular condition
ax1 = axes[0]
times = evoked_regular.times * 1000  # Convert to ms
for ch_idx in range(len(evoked_regular.data)):
    ax1.plot(times, evoked_regular.data[ch_idx] * 1e6, alpha=0.3, linewidth=0.5, color='blue')
ax1.axvline(0, color='k', linestyle='--', linewidth=2)
ax1.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax1.set_xlabel('Time (ms)', fontsize=12)
ax1.set_ylabel('Amplitude (µV)', fontsize=12)
ax1.set_title(f'Regular Patterns (n={len(epochs["Regular"])} trials)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-200, 1000)

# Random condition
ax2 = axes[1]
for ch_idx in range(len(evoked_random.data)):
    ax2.plot(times, evoked_random.data[ch_idx] * 1e6, alpha=0.3, linewidth=0.5, color='red')
ax2.axvline(0, color='k', linestyle='--', linewidth=2)
ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Time (ms)', fontsize=12)
ax2.set_ylabel('Amplitude (µV)', fontsize=12)
ax2.set_title(f'Random Patterns (n={len(epochs["Random"])} trials)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-200, 1000)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{SUBJECT}_butterfly.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {SUBJECT}_butterfly.png")
plt.close()

# ============================================================================
# FIGURE 4: ICA COMPONENTS TOPOGRAPHIES + PROPERTIES
# ============================================================================
print("Creating Figure 4: ICA Component Topographies...")

# Fit ICA on high-passed data (exclude EOG channels with no positions)
raw_for_ica = raw_preprocessed.copy()
# Pick only EEG channels (exclude EXG/EOG channels)
eeg_channel_names = [ch for ch in raw_for_ica.ch_names if not ch.startswith('EXG')]
raw_for_ica.pick_channels(eeg_channel_names)

# Set fresh montage
raw_for_ica.set_montage(None)
montage = mne.channels.make_standard_montage('biosemi64')
raw_for_ica.set_montage(montage, on_missing='ignore', verbose=False)
raw_for_ica.filter(l_freq=1.0, h_freq=None, verbose=False)

ica = ICA(n_components=20, random_state=42, max_iter=800, method='fastica')
print("  Fitting ICA (this may take a minute)...")
ica.fit(raw_for_ica, verbose=False)
print(f"  ✓ ICA fitted with {ica.n_components_} components")

# Get ICA sources for analysis
sources = ica.get_sources(raw_for_ica)

# Calculate component properties
comp_variance = np.var(sources.get_data(), axis=1)
comp_kurtosis = np.array([np.mean((sources.get_data()[i] / np.std(sources.get_data()[i]))**4) - 3 
                          for i in range(ica.n_components_)])

# Identify likely artifacts (high variance or high kurtosis)
artifact_threshold_var = np.mean(comp_variance) + 2 * np.std(comp_variance)
artifact_threshold_kurt = 10  # High kurtosis indicates spiky artifacts

likely_artifacts = []
for i in range(ica.n_components_):
    if comp_variance[i] > artifact_threshold_var or comp_kurtosis[i] > artifact_threshold_kurt:
        likely_artifacts.append(i)

# Plot component topographies manually
n_components = min(20, ica.n_components_)
n_cols = 5
n_rows = int(np.ceil(n_components / n_cols))

fig4, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
fig4.suptitle('ICA Components - Spatial Topographies (Red borders = Likely Artifacts)', 
              fontsize=16, fontweight='bold')

axes = axes.flatten()

for idx in range(n_components):
    ax = axes[idx]
    
    # Get component topography (mixing matrix column)
    topo = ica.get_components()[:, idx]
    
    try:
        # Plot topography
        im, cn = mne.viz.plot_topomap(
            topo, 
            raw_for_ica.info, 
            axes=ax, 
            show=False,
            cmap='RdBu_r',
            contours=0,
            sensors=False
        )
        
        # Highlight if likely artifact
        if idx in likely_artifacts:
            ax.set_title(f'IC{idx:02d}*\n(Artifact)', fontsize=10, fontweight='bold', color='red')
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        else:
            ax.set_title(f'IC{idx:02d}', fontsize=10)
            
    except Exception as e:
        ax.text(0.5, 0.5, f'IC{idx:02d}\nNo plot', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

# Remove extra axes
for idx in range(n_components, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{SUBJECT}_ica_topographies.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {SUBJECT}_ica_topographies.png")
print(f"  ✓ Identified {len(likely_artifacts)} likely artifact components: {likely_artifacts}")
plt.close()

# ============================================================================
# FIGURE 5: ICA TIME COURSES + PROPERTIES (first 10 components)
# ============================================================================
print("Creating Figure 5: ICA Component Time Courses + Properties...")

# Get ICA sources
ica_sources = ica.get_sources(raw_for_ica)

fig5, axes = plt.subplots(10, 2, figsize=(18, 14))
fig5.suptitle('ICA Components - Time Courses + Power Spectra (10s segment)', 
              fontsize=16, fontweight='bold')

# Plot first 10 components
start_sample = int(100 * raw_for_ica.info['sfreq'])  # Start at 100s
duration_samples = int(10 * raw_for_ica.info['sfreq'])  # 10 seconds

for idx in range(10):
    # Time course
    ax_time = axes[idx, 0]
    data_segment = ica_sources.get_data()[idx, start_sample:start_sample+duration_samples]
    times_segment = np.arange(len(data_segment)) / raw_for_ica.info['sfreq']
    
    ax_time.plot(times_segment, data_segment, linewidth=0.5, color='black')
    
    # Add statistics
    var_text = f'Var: {comp_variance[idx]:.2e}'
    kurt_text = f'Kurt: {comp_kurtosis[idx]:.1f}'
    
    if idx in likely_artifacts:
        ax_time.set_ylabel(f'IC{idx:02d}*', fontsize=11, fontweight='bold', color='red')
        ax_time.set_facecolor('#ffeeee')  # Light red background
        ax_time.text(0.02, 0.95, f'ARTIFACT\n{var_text}\n{kurt_text}', 
                    transform=ax_time.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
    else:
        ax_time.set_ylabel(f'IC{idx:02d}', fontsize=11, fontweight='bold')
        ax_time.text(0.02, 0.95, f'{var_text}\n{kurt_text}', 
                    transform=ax_time.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xlim(0, 10)
    
    # Power spectrum
    ax_psd = axes[idx, 1]
    # Compute PSD for this component
    freqs = np.fft.rfftfreq(len(ica_sources.get_data()[idx]), 1/raw_for_ica.info['sfreq'])
    psd = np.abs(np.fft.rfft(ica_sources.get_data()[idx]))**2
    
    # Plot up to 50 Hz
    mask = freqs <= 50
    ax_psd.plot(freqs[mask], 10 * np.log10(psd[mask]), linewidth=1, color='blue')
    ax_psd.set_xlim(0, 50)
    ax_psd.grid(True, alpha=0.3)
    ax_psd.set_ylabel('Power (dB)', fontsize=9)
    
    # Highlight typical artifact frequencies
    ax_psd.axvline(50, color='r', linestyle='--', alpha=0.3, linewidth=1)  # Line noise
    ax_psd.axvspan(0, 4, alpha=0.1, color='orange')  # Eye movements
    
    if idx == 0:
        ax_time.set_title('Time Course (10s)', fontsize=12, fontweight='bold')
        ax_psd.set_title('Power Spectrum', fontsize=12, fontweight='bold')

axes[-1, 0].set_xlabel('Time (s)', fontsize=12)
axes[-1, 1].set_xlabel('Frequency (Hz)', fontsize=12)

# Add legend at bottom
fig5.text(0.5, 0.01, 'Red background = Likely artifact (high variance or kurtosis) | Orange shading = Eye movement range (0-4 Hz)', 
          ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.subplots_adjust(bottom=0.03)
plt.savefig(os.path.join(OUTPUT_DIR, f'{SUBJECT}_ica_timecourses.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {SUBJECT}_ica_timecourses.png")
plt.close()

# ============================================================================
# FIGURE 6: BAD CHANNELS DETECTION VISUALIZATION
# ============================================================================
print("Creating Figure 6: Bad Channels Detection...")

# Create temporary epochs for bad channel detection
epochs_temp = mne.Epochs(
    raw_preprocessed.copy(), 
    task_events, 
    event_id=EVENT_ID,
    tmin=-0.2, 
    tmax=1.0, 
    baseline=None,
    preload=True,
    reject=None,
    verbose=False
)

# Calculate variance per channel
channel_variances = np.var(epochs_temp.get_data(), axis=(0, 2))
z_scores = np.abs((channel_variances - np.mean(channel_variances)) / np.std(channel_variances))

fig6, axes = plt.subplots(2, 1, figsize=(15, 8))
fig6.suptitle('Bad Channel Detection - Variance Analysis', fontsize=16, fontweight='bold')

# Plot 1: Variance per channel
ax1 = axes[0]
channel_indices = np.arange(len(epochs_temp.ch_names))
colors = ['red' if z > 3 else 'blue' for z in z_scores]
ax1.bar(channel_indices, channel_variances, color=colors, alpha=0.6)
ax1.axhline(np.mean(channel_variances), color='green', linestyle='--', linewidth=2, label='Mean variance')
ax1.axhline(np.mean(channel_variances) + 3*np.std(channel_variances), color='red', 
            linestyle='--', linewidth=2, label='3 SD threshold')
ax1.set_xlabel('Channel Index', fontsize=12)
ax1.set_ylabel('Variance', fontsize=12)
ax1.set_title('Channel Variance (Red = Bad Channels)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Z-scores
ax2 = axes[1]
ax2.bar(channel_indices, z_scores, color=colors, alpha=0.6)
ax2.axhline(3, color='red', linestyle='--', linewidth=2, label='Threshold (z=3)')
ax2.set_xlabel('Channel Index', fontsize=12)
ax2.set_ylabel('Z-score', fontsize=12)
ax2.set_title('Z-scores of Channel Variance', fontsize=14, fontweight='bold')
ax2.set_xticks(channel_indices[::5])
ax2.set_xticklabels([epochs_temp.ch_names[i] for i in channel_indices[::5]], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Mark bad channels
bad_channels = [epochs_temp.ch_names[i] for i in np.where(z_scores > 3)[0]]
if bad_channels:
    fig6.text(0.5, 0.02, f'Bad Channels Detected: {", ".join(bad_channels)}', 
              ha='center', fontsize=12, color='red', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{SUBJECT}_bad_channels.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {SUBJECT}_bad_channels.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("="*80)
print("QUALITY CONTROL PLOTS COMPLETE!")
print("="*80)
print(f"\nAll plots saved to: {OUTPUT_DIR}\n")
print("Generated files:")
print(f"  1. {SUBJECT}_raw_vs_preprocessed.png - Compare raw vs cleaned data")
print(f"  2. {SUBJECT}_spectra.png - Frequency content analysis")
print(f"  3. {SUBJECT}_butterfly.png - All channels overlaid (ERP)")
print(f"  4. {SUBJECT}_ica_topographies.png - Spatial patterns of ICA components")
print(f"  5. {SUBJECT}_ica_timecourses.png - Time courses of ICA components")
print(f"  6. {SUBJECT}_bad_channels.png - Bad channel detection results")
print()
print("What to look for:")
print("  • Raw vs Preprocessed: Should see cleaner signal, less drift")
print("  • Spectra: 50 Hz peak should be gone, smooth dropoff at 40 Hz")
print("  • Butterfly: Should see clear ERP pattern, no extreme outliers")
print("  • ICA Topos: Look for eye (frontal), heart (frontal-left), muscle (temporal)")
print("  • ICA Time: High variance components = artifacts to remove")
print("  • Bad Channels: Red bars indicate noisy channels for interpolation")
print("="*80)
