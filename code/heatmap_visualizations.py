"""
Proper Heatmap Visualizations - Clear patterns across 24 subjects
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configuration
BASE_DIR = Path(r'd:\ds004347')
OUTPUT_DIR = BASE_DIR / 'derivatives' / 'preprocessing_results'
QC_DIR = BASE_DIR / 'derivatives' / 'quality_control'
SUBJECTS = [f'sub-{i:03d}' for i in range(1, 25)]

sns.set_style('white')
plt.rcParams['font.size'] = 10

def load_all_data():
    """Load all evoked data"""
    print("Loading data from 24 subjects...")
    
    evokeds = {
        'regular': [],
        'random': []
    }
    
    for subject in SUBJECTS:
        try:
            evoked_o = mne.read_evokeds(OUTPUT_DIR / f'{subject}_ours_ave.fif', verbose=False)
            evokeds['regular'].append(evoked_o[0])
            evokeds['random'].append(evoked_o[1])
        except Exception as e:
            print(f"Error loading {subject}: {e}")
    
    print(f"Loaded {len(evokeds['regular'])} subjects\n")
    return evokeds

def heatmap1_time_subject_matrix(evokeds):
    """
    HEATMAP 1: Time x Subject matrix
    Shows: Effect strength over time for each subject
    """
    print("="*80)
    print("HEATMAP 1: Effect Over Time for Each Subject")
    print("="*80)
    
    # Extract Oz data for all subjects
    times = evokeds['regular'][0].times * 1000  # ms
    
    effect_matrix = []
    for reg, rand in zip(evokeds['regular'], evokeds['random']):
        oz_idx = reg.ch_names.index('Oz')
        reg_data = reg.data[oz_idx, :] * 1e6
        rand_data = rand.data[oz_idx, :] * 1e6
        effect = reg_data - rand_data
        effect_matrix.append(effect)
    
    effect_matrix = np.array(effect_matrix)  # 24 subjects x time points
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [4, 1]})
    
    # LEFT: Main heatmap
    im = axes[0].imshow(effect_matrix, aspect='auto', cmap='RdBu_r', 
                        extent=[times[0], times[-1], len(SUBJECTS), 0],
                        vmin=-5, vmax=5, interpolation='bilinear')
    
    axes[0].set_xlabel('Time (ms)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Subject', fontsize=13, fontweight='bold')
    axes[0].set_title('Brain Response Difference (Symmetric - Random) at Oz\nRed = Prefer Symmetric | Blue = Prefer Random', 
                     fontsize=14, fontweight='bold', pad=15)
    
    # Add vertical line at stimulus onset and key window
    axes[0].axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.8, label='Stimulus')
    axes[0].axvline(300, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].axvline(700, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Mark key window
    axes[0].axvspan(300, 700, alpha=0.15, color='green')
    axes[0].text(500, -1, 'Key Window', ha='center', fontsize=11, 
                fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Y-axis labels
    axes[0].set_yticks(np.arange(len(SUBJECTS)))
    axes[0].set_yticklabels(SUBJECTS, fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[0], orientation='horizontal', pad=0.08, aspect=50)
    cbar.set_label('Effect Size (μV)', fontsize=12, fontweight='bold')
    
    # RIGHT: Mean effect per subject (summary)
    mean_effects = np.mean(effect_matrix[:, (times >= 300) & (times <= 700)], axis=1)
    colors = ['red' if x < 0 else 'green' for x in mean_effects]
    
    axes[1].barh(np.arange(len(SUBJECTS)), mean_effects, color=colors, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='black', linewidth=2)
    axes[1].set_ylim(len(SUBJECTS), -1)
    axes[1].set_xlabel('Mean Effect\n(300-700ms)', fontsize=10, fontweight='bold')
    axes[1].set_title('Summary', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'heatmap1_time_subject_matrix.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'heatmap1_time_subject_matrix.png'}")
    print(f"Shows: How effect changes over time for each subject\n")
    plt.close()

def heatmap2_channel_subject_matrix(evokeds):
    """
    HEATMAP 2: Channel x Subject matrix
    Shows: Which channels show effects for which subjects
    """
    print("="*80)
    print("HEATMAP 2: Channel Activity Across Subjects")
    print("="*80)
    
    # Get posterior channels of interest
    posterior_channels = ['P7', 'P3', 'Pz', 'P4', 'P8', 
                         'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                         'O1', 'Oz', 'O2', 'Iz']
    
    # Extract data in key time window (300-700ms)
    effect_matrix = []
    
    for reg, rand in zip(evokeds['regular'], evokeds['random']):
        times = reg.times * 1000
        window_mask = (times >= 300) & (times <= 700)
        
        subject_effects = []
        for ch in posterior_channels:
            if ch in reg.ch_names:
                ch_idx = reg.ch_names.index(ch)
                reg_data = reg.data[ch_idx, window_mask] * 1e6
                rand_data = rand.data[ch_idx, window_mask] * 1e6
                mean_effect = np.mean(reg_data - rand_data)
                subject_effects.append(mean_effect)
            else:
                subject_effects.append(np.nan)
        
        effect_matrix.append(subject_effects)
    
    effect_matrix = np.array(effect_matrix)  # 24 subjects x channels
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [3, 1]})
    
    # LEFT: Main heatmap
    im = axes[0].imshow(effect_matrix.T, aspect='auto', cmap='RdBu_r',
                        vmin=-4, vmax=4, interpolation='nearest')
    
    axes[0].set_xlabel('Subject', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Electrode', fontsize=13, fontweight='bold')
    axes[0].set_title('Average Effect per Channel (300-700ms)\nRed = Prefer Symmetric | Blue = Prefer Random', 
                     fontsize=14, fontweight='bold', pad=15)
    
    # Axes
    axes[0].set_xticks(np.arange(len(SUBJECTS)))
    axes[0].set_xticklabels([s.replace('sub-', '') for s in SUBJECTS], fontsize=8, rotation=45)
    axes[0].set_yticks(np.arange(len(posterior_channels)))
    axes[0].set_yticklabels(posterior_channels, fontsize=10, fontweight='bold')
    
    # Add grid
    for i in range(len(posterior_channels) + 1):
        axes[0].axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
    for i in range(len(SUBJECTS) + 1):
        axes[0].axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[0], orientation='horizontal', pad=0.12, aspect=50)
    cbar.set_label('Effect Size (μV)', fontsize=12, fontweight='bold')
    
    # RIGHT: Mean effect per channel (across subjects)
    mean_per_channel = np.nanmean(effect_matrix, axis=0)
    colors = ['red' if x < 0 else 'green' for x in mean_per_channel]
    
    axes[1].barh(np.arange(len(posterior_channels)), mean_per_channel, 
                color=colors, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='black', linewidth=2)
    axes[1].set_ylim(len(posterior_channels) - 0.5, -0.5)
    axes[1].set_xlabel('Mean Effect\n(across subjects)', fontsize=10, fontweight='bold')
    axes[1].set_title('Channel\nSummary', fontsize=11, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].set_yticks([])
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'heatmap2_channel_subject_matrix.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'heatmap2_channel_subject_matrix.png'}")
    print(f"Shows: Which brain regions (channels) show effects for which subjects\n")
    plt.close()

def heatmap3_quality_metrics(evokeds):
    """
    HEATMAP 3: Quality metrics across subjects
    Shows: Data quality patterns
    """
    print("="*80)
    print("HEATMAP 3: Quality Metrics Heatmap")
    print("="*80)
    
    # Load metrics
    df = pd.read_csv(OUTPUT_DIR / 'preprocessing_metrics_all_subjects.csv')
    
    # Select metrics
    metrics = {
        'Events': 'n_events_total',
        'Bad Channels': 'ours_bad_channels',
        'ICA Excluded': 'ours_ica_excluded',
        'Epochs Kept': 'ours_epochs_total',
        'SNR': 'ours_snr_estimate',
        'Peak Amplitude': 'ours_erp_peak_amplitude'
    }
    
    # Create matrix
    metric_matrix = []
    for metric_name, col_name in metrics.items():
        metric_matrix.append(df[col_name].values)
    
    metric_matrix = np.array(metric_matrix)
    
    # Normalize for visualization (z-score per metric)
    metric_matrix_normalized = np.zeros_like(metric_matrix, dtype=float)
    for i in range(len(metrics)):
        mean = np.mean(metric_matrix[i])
        std = np.std(metric_matrix[i])
        if std > 0:
            metric_matrix_normalized[i] = (metric_matrix[i] - mean) / std
        else:
            metric_matrix_normalized[i] = 0
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # TOP: Heatmap with annotations
    im = axes[0].imshow(metric_matrix_normalized, aspect='auto', cmap='RdYlGn',
                        vmin=-2, vmax=2, interpolation='nearest')
    
    # Annotate with actual values
    for i in range(len(metrics)):
        for j in range(len(SUBJECTS)):
            text_color = 'white' if abs(metric_matrix_normalized[i, j]) > 1 else 'black'
            axes[0].text(j, i, f'{metric_matrix[i, j]:.1f}',
                        ha='center', va='center', fontsize=8, 
                        color=text_color, fontweight='bold')
    
    axes[0].set_xlabel('Subject', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Quality Metric', fontsize=13, fontweight='bold')
    axes[0].set_title('Quality Metrics Across All Subjects\nGreen = Better than average | Red = Worse than average', 
                     fontsize=14, fontweight='bold', pad=15)
    
    # Axes
    axes[0].set_xticks(np.arange(len(SUBJECTS)))
    axes[0].set_xticklabels([s.replace('sub-', '') for s in SUBJECTS], fontsize=8, rotation=45)
    axes[0].set_yticks(np.arange(len(metrics)))
    axes[0].set_yticklabels(list(metrics.keys()), fontsize=11, fontweight='bold')
    
    # Grid
    for i in range(len(metrics) + 1):
        axes[0].axhline(i - 0.5, color='black', linewidth=1, alpha=0.3)
    for i in range(len(SUBJECTS) + 1):
        axes[0].axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.2)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[0], orientation='horizontal', pad=0.08, aspect=50)
    cbar.set_label('Z-Score (standard deviations from mean)', fontsize=11, fontweight='bold')
    
    # BOTTOM: Summary statistics
    axes[1].axis('off')
    
    summary_text = "SUMMARY STATISTICS:\n\n"
    for display_name, col_name in metrics.items():
        values = df[col_name].values
        summary_text += f"{display_name:20s}: Mean={np.mean(values):7.2f}  SD={np.std(values):6.2f}  "
        summary_text += f"Range=[{np.min(values):6.2f}, {np.max(values):6.2f}]\n"
    
    axes[1].text(0.05, 0.9, summary_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'heatmap3_quality_metrics.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'heatmap3_quality_metrics.png'}")
    print(f"Shows: Quality metrics with actual values annotated\n")
    plt.close()

def heatmap4_correlation_matrix(evokeds):
    """
    HEATMAP 4: Subject similarity matrix
    Shows: Which subjects have similar response patterns
    """
    print("="*80)
    print("HEATMAP 4: Subject Similarity Matrix")
    print("="*80)
    
    # Extract Oz data in key window for all subjects
    times = evokeds['regular'][0].times * 1000
    window_mask = (times >= 300) & (times <= 700)
    
    effect_vectors = []
    for reg, rand in zip(evokeds['regular'], evokeds['random']):
        oz_idx = reg.ch_names.index('Oz')
        reg_data = reg.data[oz_idx, window_mask] * 1e6
        rand_data = rand.data[oz_idx, window_mask] * 1e6
        effect = reg_data - rand_data
        effect_vectors.append(effect)
    
    effect_vectors = np.array(effect_vectors)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(effect_vectors)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})
    
    # LEFT: Correlation matrix
    im = axes[0].imshow(corr_matrix, aspect='auto', cmap='coolwarm',
                        vmin=-1, vmax=1, interpolation='nearest')
    
    axes[0].set_xlabel('Subject', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Subject', fontsize=13, fontweight='bold')
    axes[0].set_title('Subject Response Similarity (300-700ms at Oz)\nRed = Similar patterns | Blue = Opposite patterns', 
                     fontsize=14, fontweight='bold', pad=15)
    
    # Axes
    axes[0].set_xticks(np.arange(len(SUBJECTS)))
    axes[0].set_xticklabels([s.replace('sub-', '') for s in SUBJECTS], fontsize=8, rotation=45)
    axes[0].set_yticks(np.arange(len(SUBJECTS)))
    axes[0].set_yticklabels([s.replace('sub-', '') for s in SUBJECTS], fontsize=8)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=axes[0], orientation='horizontal', pad=0.12, aspect=50)
    cbar.set_label('Correlation Coefficient', fontsize=12, fontweight='bold')
    
    # RIGHT: Distribution of correlations
    axes[1].axis('off')
    
    # Get upper triangle (exclude diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    correlations = corr_matrix[mask]
    
    # Create mini histogram
    mini_ax = fig.add_axes([0.72, 0.3, 0.2, 0.4])
    mini_ax.hist(correlations, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    mini_ax.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(correlations):.3f}')
    mini_ax.set_xlabel('Correlation', fontsize=10, fontweight='bold')
    mini_ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    mini_ax.set_title('Distribution', fontsize=11, fontweight='bold')
    mini_ax.legend(fontsize=9)
    mini_ax.grid(alpha=0.3)
    
    # Statistics text
    stats_text = f"""
    INTERPRETATION:
    
    • High correlation (red):
      Subjects show similar
      response patterns
    
    • Low correlation (blue):
      Subjects show opposite
      response patterns
    
    STATISTICS:
    
    Mean correlation: {np.mean(correlations):.3f}
    Median: {np.median(correlations):.3f}
    SD: {np.std(correlations):.3f}
    
    Min: {np.min(correlations):.3f}
    Max: {np.max(correlations):.3f}
    
    High similarity suggests
    consistent effect across
    subjects
    """
    
    axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'heatmap4_correlation_matrix.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'heatmap4_correlation_matrix.png'}")
    print(f"Shows: Which subjects have similar response patterns\n")
    plt.close()

def heatmap5_grand_average_channels_time(evokeds):
    """
    HEATMAP 5: Channels x Time grand average heatmap
    Shows: Spatiotemporal evolution of the effect
    """
    print("="*80)
    print("HEATMAP 5: Spatiotemporal Evolution (All Channels)")
    print("="*80)
    
    # Compute grand average difference
    def compute_grand_diff(evoked_list_reg, evoked_list_rand):
        all_reg = []
        all_rand = []
        for reg, rand in zip(evoked_list_reg, evoked_list_rand):
            # Drop EXG channels
            reg_clean = reg.copy()
            rand_clean = rand.copy()
            exg_ch = [ch for ch in reg_clean.ch_names if 'EXG' in ch]
            if exg_ch:
                reg_clean = reg_clean.drop_channels(exg_ch)
                rand_clean = rand_clean.drop_channels(exg_ch)
            all_reg.append(reg_clean.data * 1e6)
            all_rand.append(rand_clean.data * 1e6)
        
        mean_reg = np.mean(all_reg, axis=0)
        mean_rand = np.mean(all_rand, axis=0)
        return mean_reg - mean_rand, reg_clean
    
    diff_data, template = compute_grand_diff(evokeds['regular'], evokeds['random'])
    times = template.times * 1000
    
    # Select posterior channels for better visualization
    posterior_channels = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                         'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                         'O1', 'Oz', 'O2', 'Iz']
    
    channel_indices = [i for i, ch in enumerate(template.ch_names) if ch in posterior_channels]
    channel_names = [template.ch_names[i] for i in channel_indices]
    
    diff_data_subset = diff_data[channel_indices, :]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Main heatmap
    im = ax.imshow(diff_data_subset, aspect='auto', cmap='RdBu_r',
                   extent=[times[0], times[-1], len(channel_names), 0],
                   vmin=-3, vmax=3, interpolation='bilinear')
    
    ax.set_xlabel('Time (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Electrode (Posterior Region)', fontsize=13, fontweight='bold')
    ax.set_title('Spatiotemporal Evolution: Grand Average Difference (Symmetric - Random)\nRed = Prefer Symmetric | Blue = Prefer Random', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Add markers
    ax.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.9, label='Stimulus')
    ax.axvline(300, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(700, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvspan(300, 700, alpha=0.1, color='green')
    
    # Y-axis
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_yticklabels(channel_names, fontsize=10, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, aspect=50)
    cbar.set_label('Effect Size (μV)', fontsize=12, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'heatmap5_grand_average_channels_time.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'heatmap5_grand_average_channels_time.png'}")
    print(f"Shows: How the effect evolves over time across brain regions\n")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("PROPER HEATMAP VISUALIZATIONS")
    print("Clear patterns and relationships from 24 subjects")
    print("="*80 + "\n")
    
    # Load all data
    evokeds = load_all_data()
    
    # Generate heatmaps
    print("Generating 5 informative heatmaps...\n")
    
    heatmap1_time_subject_matrix(evokeds)
    heatmap2_channel_subject_matrix(evokeds)
    heatmap3_quality_metrics(evokeds)
    heatmap4_correlation_matrix(evokeds)
    heatmap5_grand_average_channels_time(evokeds)
    
    print("\n" + "="*80)
    print("ALL HEATMAPS COMPLETE!")
    print("="*80)
    print(f"\nGenerated 5 heatmaps in: {QC_DIR}")
    print("\nWhat each heatmap shows:")
    print("  1. Time x Subject - Effect progression over time for each person")
    print("  2. Channel x Subject - Which brain regions show effects per person")
    print("  3. Quality Metrics - Data quality patterns with actual values")
    print("  4. Subject Similarity - Who has similar/opposite response patterns")
    print("  5. Spatiotemporal Evolution - How effect spreads across brain over time")
    print("\nThese heatmaps reveal patterns you couldn't see in line plots!")
    print("="*80 + "\n")
