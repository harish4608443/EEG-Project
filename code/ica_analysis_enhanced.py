"""
Enhanced ICA Component Visualization
Creates comprehensive plots to identify artifact components
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from scipy import stats

# Paths
BASE_DIR = r'd:\ds004347'
SUBJECT = 'sub-001'
RAW_DATA_PATH = os.path.join(BASE_DIR, SUBJECT, 'eeg', f'{SUBJECT}_task-jacobsen_eeg.bdf')
OUTPUT_DIR = os.path.join(BASE_DIR, 'derivatives', 'quality_control')

print("="*80)
print("ENHANCED ICA COMPONENT ANALYSIS")
print("="*80)
print()

# Load and prepare data
print("Loading data...")
raw = mne.io.read_raw_bdf(RAW_DATA_PATH, preload=True, verbose=False)
events = mne.find_events(raw, stim_channel='Status', verbose=False)

# Pick only EEG channels (no EOG/EXG)
eeg_channels = [ch for ch in raw.ch_names if not ch.startswith('EXG') and ch != 'Status']
raw.pick_channels(eeg_channels)

# Set montage
raw.set_montage(None)
montage = mne.channels.make_standard_montage('biosemi64')
raw.set_montage(montage, on_missing='ignore', verbose=False)

# Filter and reference
raw.filter(l_freq=0.1, h_freq=40.0, verbose=False)
raw.notch_filter(freqs=50, verbose=False)
raw.set_eeg_reference('average', projection=False, verbose=False)

# Prepare for ICA
raw_ica = raw.copy()
raw_ica.filter(l_freq=1.0, h_freq=None, verbose=False)

# Fit ICA
print("Fitting ICA with 15 components...")
ica = ICA(n_components=15, random_state=42, max_iter=800, method='fastica')
ica.fit(raw_ica, verbose=False)
print(f"✓ ICA completed\n")

# ============================================================================
# Analyze components to identify artifacts
# ============================================================================
print("Analyzing component properties...")

sources = ica.get_sources(raw_ica)
n_components = ica.n_components_

# Calculate properties for each component
properties = {
    'variance': np.var(sources.get_data(), axis=1),
    'kurtosis': stats.kurtosis(sources.get_data(), axis=1),
    'max_amp': np.max(np.abs(sources.get_data()), axis=1)
}

# Identify likely artifacts based on multiple criteria
artifact_components = []
brain_components = []

for i in range(n_components):
    is_artifact = False
    reasons = []
    
    # High variance (noisy/artifact)
    if properties['variance'][i] > np.mean(properties['variance']) + 2*np.std(properties['variance']):
        is_artifact = True
        reasons.append('High variance')
    
    # High kurtosis (spiky/blinks)
    if properties['kurtosis'][i] > 5:
        is_artifact = True
        reasons.append('High kurtosis (spiky)')
    
    # Very low variance (flat/bad)
    if properties['variance'][i] < np.mean(properties['variance']) - 2*np.std(properties['variance']):
        is_artifact = True
        reasons.append('Very low variance')
    
    if is_artifact:
        artifact_components.append((i, reasons))
    else:
        brain_components.append(i)

print(f"Identified {len(artifact_components)} likely artifact components:")
for comp, reasons in artifact_components:
    print(f"  IC{comp:02d}: {', '.join(reasons)}")
print()

# ============================================================================
# COMPREHENSIVE ICA FIGURE
# ============================================================================
print("Creating comprehensive ICA visualization...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(5, 5, hspace=0.4, wspace=0.3)

fig.suptitle('ICA Component Analysis - Quality Control', fontsize=18, fontweight='bold')

# TOPOGRAPHIES (Top 3 rows)
for idx in range(15):
    row = idx // 5
    col = idx % 5
    ax = fig.add_subplot(gs[row, col])
    
    # Get mixing matrix (topography)
    topo_data = ica.get_components()[:, idx]
    
    # Determine if artifact
    is_artifact = any(comp == idx for comp, _ in artifact_components)
    
    # Plot topography
    try:
        mne.viz.plot_topomap(
            topo_data,
            raw_ica.info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            contours=6,
            sensors=True,
            size=1
        )
        
        # Title with artifact indicator
        if is_artifact:
            reasons = [r for comp, r in artifact_components if comp == idx][0]
            title_color = 'red'
            title = f'IC{idx:02d} *ARTIFACT*'
            # Red border
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        else:
            title_color = 'green'
            title = f'IC{idx:02d} Brain'
            # Green border
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2)
        
        ax.set_title(title, fontsize=9, fontweight='bold', color=title_color)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'IC{idx:02d}\nError', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')

# PROPERTIES PLOTS (Bottom 2 rows)
# Variance
ax_var = fig.add_subplot(gs[3, :2])
bars = ax_var.bar(range(n_components), properties['variance'], 
                  color=['red' if any(c==i for c,_ in artifact_components) else 'green' 
                         for i in range(n_components)])
ax_var.axhline(np.mean(properties['variance']), color='blue', linestyle='--', label='Mean')
ax_var.axhline(np.mean(properties['variance']) + 2*np.std(properties['variance']), 
               color='red', linestyle='--', label='2 SD threshold')
ax_var.set_xlabel('Component', fontweight='bold')
ax_var.set_ylabel('Variance', fontweight='bold')
ax_var.set_title('Component Variance (Red = Artifact)', fontweight='bold')
ax_var.legend()
ax_var.grid(True, alpha=0.3)

# Kurtosis
ax_kurt = fig.add_subplot(gs[3, 2:])
ax_kurt.bar(range(n_components), properties['kurtosis'],
            color=['red' if any(c==i for c,_ in artifact_components) else 'green' 
                   for i in range(n_components)])
ax_kurt.axhline(5, color='red', linestyle='--', label='Artifact threshold')
ax_kurt.set_xlabel('Component', fontweight='bold')
ax_kurt.set_ylabel('Kurtosis', fontweight='bold')
ax_kurt.set_title('Component Kurtosis (High = Spiky/Blinks)', fontweight='bold')
ax_kurt.legend()
ax_kurt.grid(True, alpha=0.3)

# Time courses of top 3 artifact components
ax_tc = fig.add_subplot(gs[4, :])
if artifact_components:
    # Show first 10 seconds
    duration = 10
    start = int(100 * raw_ica.info['sfreq'])
    end = start + int(duration * raw_ica.info['sfreq'])
    times = np.arange(end - start) / raw_ica.info['sfreq']
    
    for i, (comp, reasons) in enumerate(artifact_components[:3]):
        data = sources.get_data()[comp, start:end]
        ax_tc.plot(times, data + i*5, label=f"IC{comp:02d}: {', '.join(reasons)}", linewidth=1)
    
    ax_tc.set_xlabel('Time (s)', fontweight='bold')
    ax_tc.set_ylabel('Amplitude (offset)', fontweight='bold')
    ax_tc.set_title('Time Courses of Top Artifact Components (10s segment)', fontweight='bold')
    ax_tc.legend(loc='upper right')
    ax_tc.grid(True, alpha=0.3)
else:
    ax_tc.text(0.5, 0.5, 'No artifact components identified', 
               ha='center', va='center', transform=ax_tc.transAxes, fontsize=14)
    ax_tc.axis('off')

# Summary text
summary_text = f"""
SUMMARY:
• Total components: {n_components}
• Artifact components: {len(artifact_components)} (marked with red border)
• Brain components: {len(brain_components)} (marked with green border)

INTERPRETATION GUIDE:
• Frontal topography + high kurtosis = Eye blinks
• Left frontal + rhythmic = Heart beat (ECG)
• Temporal/high freq = Muscle artifacts
• Central/posterior + low kurtosis = Brain signals

RECOMMENDATION:
Components to exclude: {[c for c, _ in artifact_components]}
"""

fig.text(0.02, 0.02, summary_text, fontsize=10, fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig(os.path.join(OUTPUT_DIR, f'{SUBJECT}_ica_comprehensive.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {SUBJECT}_ica_comprehensive.png")
plt.close()

# ============================================================================
# DETAILED COMPONENT REPORT
# ============================================================================
print("\nGenerating detailed component properties...")

fig2, axes = plt.subplots(n_components, 4, figsize=(16, 3*n_components))
fig2.suptitle('Detailed ICA Component Properties', fontsize=16, fontweight='bold')

for idx in range(n_components):
    is_artifact = any(comp == idx for comp, _ in artifact_components)
    bg_color = '#ffeeee' if is_artifact else '#eeffee'
    
    # Topography
    ax_topo = axes[idx, 0]
    topo_data = ica.get_components()[:, idx]
    try:
        mne.viz.plot_topomap(topo_data, raw_ica.info, axes=ax_topo, show=False, 
                            cmap='RdBu_r', contours=6, sensors=True)
        if is_artifact:
            reasons = [r for comp, r in artifact_components if comp == idx][0]
            ax_topo.set_title(f'IC{idx:02d} *ARTIFACT*\n{reasons[0] if reasons else ""}', 
                            fontsize=9, color='red', fontweight='bold')
        else:
            ax_topo.set_title(f'IC{idx:02d} Brain Signal', fontsize=9, color='green', fontweight='bold')
    except:
        ax_topo.text(0.5, 0.5, f'IC{idx:02d}', ha='center', va='center', transform=ax_topo.transAxes)
    ax_topo.set_facecolor(bg_color)
    
    # Time course (5 seconds)
    ax_time = axes[idx, 1]
    start = int(100 * raw_ica.info['sfreq'])
    end = start + int(5 * raw_ica.info['sfreq'])
    times = np.arange(end - start) / raw_ica.info['sfreq']
    data = sources.get_data()[idx, start:end]
    ax_time.plot(times, data, 'k-', linewidth=0.5)
    ax_time.set_ylabel('Amplitude', fontsize=8)
    ax_time.set_xlabel('Time (s)', fontsize=8)
    ax_time.grid(True, alpha=0.3)
    ax_time.set_facecolor(bg_color)
    
    # Power spectrum
    ax_psd = axes[idx, 2]
    freqs = np.fft.rfftfreq(sources.shape[1], 1/raw_ica.info['sfreq'])
    psd = np.abs(np.fft.rfft(sources.get_data()[idx]))**2
    mask = freqs <= 50
    ax_psd.semilogy(freqs[mask], psd[mask], 'b-', linewidth=1)
    ax_psd.set_xlabel('Frequency (Hz)', fontsize=8)
    ax_psd.set_ylabel('Power', fontsize=8)
    ax_psd.axvline(50, color='r', linestyle='--', alpha=0.3)
    ax_psd.grid(True, alpha=0.3)
    ax_psd.set_facecolor(bg_color)
    
    # Properties
    ax_props = axes[idx, 3]
    props_text = f"""
Variance: {properties['variance'][idx]:.2e}
Kurtosis: {properties['kurtosis'][idx]:.2f}
Max Amp: {properties['max_amp'][idx]:.2f}

Status: {'ARTIFACT' if is_artifact else 'Brain Signal'}
    """
    ax_props.text(0.1, 0.5, props_text, fontsize=8, va='center', fontfamily='monospace')
    ax_props.axis('off')
    ax_props.set_facecolor(bg_color)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{SUBJECT}_ica_detailed.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {SUBJECT}_ica_detailed.png")
plt.close()

print()
print("="*80)
print("ICA ANALYSIS COMPLETE")
print("="*80)
print(f"\nArtifact components identified: {[c for c, _ in artifact_components]}")
print(f"Brain components: {brain_components}")
print(f"\nRecommendation: Exclude components {[c for c, _ in artifact_components]} from analysis")
print("="*80)
