"""
Quick Replication Check: Authors' 25Hz vs Our 40Hz Filtering
Can run on sub-001 immediately to demonstrate the concept
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BASE_DIR = Path(r"d:\ds004347")
OUTPUT_DIR = BASE_DIR / "derivatives" / "replication_checks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT = "sub-001"

print("="*80)
print("REPLICATION CHECK: Filtering Comparison (25 Hz vs 40 Hz)")
print("="*80)
print()

# Load raw data
raw_path = BASE_DIR / SUBJECT / "eeg" / f"{SUBJECT}_task-jacobsen_eeg.bdf"
print(f"Loading: {SUBJECT}")

raw = mne.io.read_raw_bdf(raw_path, preload=True, verbose=False)

# Extract events
events = mne.find_events(raw, stim_channel='Status', verbose=False)
task_events = events[np.isin(events[:, 2], [1, 3])]
event_id = {'Regular': 1, 'Random': 3}

print(f"Found {len(task_events)} events")

# Pick EEG channels
raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude=[])

# Set montage
montage = mne.channels.make_standard_montage('biosemi64')
raw.set_montage(montage, on_missing='ignore', verbose=False)

# ============================================================================
# Pipeline 1: Authors' 25 Hz filtering
# ============================================================================
print("\nProcessing Pipeline 1: Authors' (0.1-25 Hz)...")

raw_25hz = raw.copy()
raw_25hz.filter(l_freq=0.1, h_freq=25.0, verbose=False)
raw_25hz.set_eeg_reference('average', projection=False, verbose=False)

# Simple ICA
raw_for_ica_25 = raw_25hz.copy()
raw_for_ica_25.filter(l_freq=1.0, h_freq=None, verbose=False)

ica_25 = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter=800, method='fastica')
ica_25.fit(raw_for_ica_25, verbose=False)
ica_25.apply(raw_25hz, verbose=False)

# Epoch
epochs_25hz = mne.Epochs(
    raw_25hz, task_events, event_id=event_id,
    tmin=-1.0, tmax=1.0, baseline=(-0.2, 0.05), preload=True, verbose=False
)

evoked_regular_25 = epochs_25hz['Regular'].average()
evoked_random_25 = epochs_25hz['Random'].average()

print(f"  Epochs: {len(epochs_25hz)}")

# ============================================================================
# Pipeline 2: Our 40 Hz filtering
# ============================================================================
print("\nProcessing Pipeline 2: Ours (0.1-40 Hz + 50Hz notch)...")

raw_40hz = raw.copy()
raw_40hz.filter(l_freq=0.1, h_freq=40.0, verbose=False)
raw_40hz.notch_filter(freqs=50, verbose=False)
raw_40hz.set_eeg_reference('average', projection=False, verbose=False)

# Simple ICA
raw_for_ica_40 = raw_40hz.copy()
raw_for_ica_40.filter(l_freq=1.0, h_freq=None, verbose=False)

ica_40 = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter=800, method='fastica')
ica_40.fit(raw_for_ica_40, verbose=False)
ica_40.apply(raw_40hz, verbose=False)

# Epoch
epochs_40hz = mne.Epochs(
    raw_40hz, task_events, event_id=event_id,
    tmin=-0.2, tmax=1.0, baseline=(-0.2, 0.0), preload=True, verbose=False
)

# Median averaging
regular_data_40 = epochs_40hz['Regular'].get_data()
random_data_40 = epochs_40hz['Random'].get_data()

evoked_regular_40 = epochs_40hz['Regular'].average()
evoked_regular_40.data = np.median(regular_data_40, axis=0)

evoked_random_40 = epochs_40hz['Random'].average()
evoked_random_40.data = np.median(random_data_40, axis=0)

print(f"  Epochs: {len(epochs_40hz)}")

# ============================================================================
# Create Comparison Visualization
# ============================================================================
print("\nCreating replication check visualization...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Replication Check: Authors\' Pipeline (25 Hz) vs Our Pipeline (40 Hz)',
             fontsize=16, fontweight='bold')

# Find Oz electrode
oz_idx = evoked_regular_25.ch_names.index('Oz') if 'Oz' in evoked_regular_25.ch_names else 0
times_25 = evoked_regular_25.times
times_40 = evoked_regular_40.times

# Row 1: Authors' Pipeline ERPs
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(times_25, evoked_regular_25.data[oz_idx, :] * 1e6, 'b-', linewidth=2, label='Regular')
ax1.plot(times_25, evoked_random_25.data[oz_idx, :] * 1e6, 'r-', linewidth=2, label='Random')
ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude (μV)')
ax1.set_title('Authors\' Pipeline: 0.1-25 Hz\nMean Averaging', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Authors' difference wave
ax2 = fig.add_subplot(gs[0, 1])
diff_25 = (evoked_regular_25.data[oz_idx, :] - evoked_random_25.data[oz_idx, :]) * 1e6
ax2.plot(times_25, diff_25, 'purple', linewidth=2.5)
ax2.fill_between(times_25, 0, diff_25, alpha=0.3, color='purple')
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Difference (μV)')
ax2.set_title('Difference: Regular - Random', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Authors' topography at 400ms
ax3 = fig.add_subplot(gs[0, 2])
time_idx = np.argmin(np.abs(times_25 - 0.4))
try:
    evoked_diff_25 = evoked_regular_25.copy()
    evoked_diff_25.data = evoked_regular_25.data - evoked_random_25.data
    mne.viz.plot_topomap(evoked_diff_25.data[:, time_idx], evoked_diff_25.info,
                         axes=ax3, show=False, contours=6)
    ax3.set_title('Topography at 400ms', fontweight='bold')
except Exception as e:
    ax3.text(0.5, 0.5, f'Topomap\nError', ha='center', va='center')
    ax3.set_title('Topography at 400ms', fontweight='bold')

# Row 2: Our Pipeline ERPs
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(times_40, evoked_regular_40.data[oz_idx, :] * 1e6, 'b-', linewidth=2, label='Regular')
ax4.plot(times_40, evoked_random_40.data[oz_idx, :] * 1e6, 'r-', linewidth=2, label='Random')
ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
ax4.axvline(0, color='k', linestyle='--', alpha=0.3)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Amplitude (μV)')
ax4.set_title('Our Pipeline: 0.1-40 Hz + 50Hz Notch\nMedian Averaging', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Our difference wave
ax5 = fig.add_subplot(gs[1, 1])
diff_40 = (evoked_regular_40.data[oz_idx, :] - evoked_random_40.data[oz_idx, :]) * 1e6
ax5.plot(times_40, diff_40, 'green', linewidth=2.5)
ax5.fill_between(times_40, 0, diff_40, alpha=0.3, color='green')
ax5.axhline(0, color='k', linestyle='--', alpha=0.3)
ax5.axvline(0, color='k', linestyle='--', alpha=0.3)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Difference (μV)')
ax5.set_title('Difference: Regular - Random', fontweight='bold')
ax5.grid(True, alpha=0.3)

# Our topography at 400ms
ax6 = fig.add_subplot(gs[1, 2])
time_idx_40 = np.argmin(np.abs(times_40 - 0.4))
try:
    evoked_diff_40 = evoked_regular_40.copy()
    evoked_diff_40.data = evoked_regular_40.data - evoked_random_40.data
    mne.viz.plot_topomap(evoked_diff_40.data[:, time_idx_40], evoked_diff_40.info,
                         axes=ax6, show=False, contours=6)
    ax6.set_title('Topography at 400ms', fontweight='bold')
except Exception as e:
    ax6.text(0.5, 0.5, f'Topomap\nError', ha='center', va='center')
    ax6.set_title('Topography at 400ms', fontweight='bold')

# Row 3: Direct Comparisons
# Overlay both pipelines
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(times_25, evoked_regular_25.data[oz_idx, :] * 1e6, 'b-', 
         linewidth=2, label='Authors (25 Hz)', alpha=0.7)
ax7.plot(times_40, evoked_regular_40.data[oz_idx, :] * 1e6, 'b--', 
         linewidth=2, label='Ours (40 Hz)', alpha=0.7)
ax7.axhline(0, color='k', linestyle='--', alpha=0.3)
ax7.axvline(0, color='k', linestyle='--', alpha=0.3)
ax7.set_xlabel('Time (s)')
ax7.set_ylabel('Amplitude (μV)')
ax7.set_title('Regular Condition: Pipeline Overlay', fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Difference wave comparison
ax8 = fig.add_subplot(gs[2, 1])
# Interpolate to common timebase for fair comparison
from scipy.interpolate import interp1d
common_times = np.linspace(max(times_25[0], times_40[0]), 
                           min(times_25[-1], times_40[-1]), 500)
diff_25_interp = interp1d(times_25, diff_25, kind='cubic')(common_times)
diff_40_interp = interp1d(times_40, diff_40, kind='cubic')(common_times)

ax8.plot(common_times, diff_25_interp, 'purple', linewidth=2.5, 
         label='Authors (25 Hz)', alpha=0.7)
ax8.plot(common_times, diff_40_interp, 'green', linewidth=2.5,
         label='Ours (40 Hz)', alpha=0.7)
ax8.axhline(0, color='k', linestyle='--', alpha=0.3)
ax8.axvline(0, color='k', linestyle='--', alpha=0.3)
ax8.set_xlabel('Time (s)')
ax8.set_ylabel('Difference (μV)')
ax8.set_title('Difference Wave Comparison', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Correlation plot
ax9 = fig.add_subplot(gs[2, 2])
correlation = np.corrcoef(diff_25_interp, diff_40_interp)[0, 1]
ax9.scatter(diff_25_interp, diff_40_interp, alpha=0.5, s=20, c=common_times, cmap='viridis')
lim = max(abs(diff_25_interp.min()), abs(diff_25_interp.max()),
          abs(diff_40_interp.min()), abs(diff_40_interp.max()))
ax9.plot([-lim, lim], [-lim, lim], 'r--', linewidth=2, label='Identity')
ax9.set_xlabel('Authors\' Difference (μV)')
ax9.set_ylabel('Our Difference (μV)')
ax9.set_title(f'Pipeline Correlation\nr = {correlation:.3f}', fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)
ax9.axis('equal')

plt.savefig(OUTPUT_DIR / 'replication_check_25hz_vs_40hz.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {OUTPUT_DIR / 'replication_check_25hz_vs_40hz.png'}")

# Print summary statistics
print("\n" + "="*80)
print("REPLICATION CHECK SUMMARY")
print("="*80)
print(f"Subject: {SUBJECT}")
print(f"\nAuthors' Pipeline (25 Hz):")
print(f"  Epochs: {len(epochs_25hz)}")
print(f"  Peak difference: {diff_25.max():.2f} μV at {times_25[diff_25.argmax()]:.3f} s")
print(f"\nOur Pipeline (40 Hz):")
print(f"  Epochs: {len(epochs_40hz)}")
print(f"  Peak difference: {diff_40.max():.2f} μV at {times_40[diff_40.argmax()]:.3f} s")
print(f"\nPipeline Correlation: r = {correlation:.3f}")
print(f"\nConclusion: {'High' if correlation > 0.9 else 'Moderate' if correlation > 0.7 else 'Low'} agreement between pipelines")
print("="*80)
