"""
Pipeline Comparison Visualizations
Shows differences between Authors' pipeline vs Our pipeline
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

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10

def load_all_data():
    """Load all evoked data from both pipelines"""
    print("Loading data from both pipelines (24 subjects)...")
    
    data = {
        'authors_regular': [],
        'authors_random': [],
        'ours_regular': [],
        'ours_random': []
    }
    
    for subject in SUBJECTS:
        try:
            # Authors
            evoked_a = mne.read_evokeds(OUTPUT_DIR / f'{subject}_authors_ave.fif', verbose=False)
            data['authors_regular'].append(evoked_a[0])
            data['authors_random'].append(evoked_a[1])
            
            # Ours
            evoked_o = mne.read_evokeds(OUTPUT_DIR / f'{subject}_ours_ave.fif', verbose=False)
            data['ours_regular'].append(evoked_o[0])
            data['ours_random'].append(evoked_o[1])
        except Exception as e:
            print(f"Error loading {subject}: {e}")
    
    print(f"Loaded {len(data['ours_regular'])} subjects\n")
    return data

def comparison_viz1_pipeline_flowchart():
    """
    VIZ 1: Visual flowchart showing pipeline differences
    """
    print("="*80)
    print("COMPARISON VIZ 1: Pipeline Steps Comparison")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # LEFT: Authors' pipeline
    axes[0].axis('off')
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 14)
    
    steps_authors = [
        ("1. Load Raw Data", "BioSemi 64ch EEG", 'lightblue'),
        ("2. Set Montage", "Standard 10-20", 'lightblue'),
        ("3. Filter", "0.1-30 Hz", 'lightgreen'),
        ("4. Bad Channels", "Manual rejection", 'yellow'),
        ("5. Re-reference", "Average reference", 'lightcoral'),
        ("6. Epoch", "-200 to 1000ms", 'lightyellow'),
        ("7. ICA", "Remove artifacts", 'lightgreen'),
        ("8. Reject Epochs", "Amplitude threshold", 'yellow'),
        ("9. Baseline", "-200 to 0ms", 'lightblue'),
        ("10. Average", "Create ERPs", 'lightcoral')
    ]
    
    y_start = 13
    for i, (step, detail, color) in enumerate(steps_authors):
        y_pos = y_start - i * 1.3
        # Box
        rect = plt.Rectangle((1, y_pos-0.5), 8, 1, facecolor=color, edgecolor='black', linewidth=2)
        axes[0].add_patch(rect)
        # Text
        axes[0].text(5, y_pos, f"{step}\n{detail}", ha='center', va='center', 
                    fontsize=10, fontweight='bold')
        # Arrow
        if i < len(steps_authors) - 1:
            axes[0].arrow(5, y_pos-0.6, 0, -0.5, head_width=0.3, head_length=0.1, 
                         fc='black', ec='black', linewidth=2)
    
    axes[0].set_title("AUTHORS' PIPELINE", fontsize=16, fontweight='bold', pad=20)
    
    # RIGHT: Our pipeline
    axes[1].axis('off')
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 14)
    
    steps_ours = [
        ("1. Load Raw Data", "BioSemi 64ch EEG", 'lightblue'),
        ("2. Set Montage", "Standard 10-20", 'lightblue'),
        ("3. Filter", "0.1-40 Hz ★", 'orange'),  # DIFFERENT
        ("4. Bad Channels", "Automated detection ★", 'orange'),  # DIFFERENT
        ("5. Re-reference", "Average reference", 'lightcoral'),
        ("6. Epoch", "-200 to 1000ms", 'lightyellow'),
        ("7. ICA", "Automated selection ★", 'orange'),  # DIFFERENT
        ("8. Reject Epochs", "Autoreject ★", 'orange'),  # DIFFERENT
        ("9. Baseline", "-200 to 0ms", 'lightblue'),
        ("10. Average", "Create ERPs", 'lightcoral')
    ]
    
    y_start = 13
    for i, (step, detail, color) in enumerate(steps_ours):
        y_pos = y_start - i * 1.3
        # Box
        rect = plt.Rectangle((1, y_pos-0.5), 8, 1, facecolor=color, edgecolor='black', linewidth=2)
        axes[1].add_patch(rect)
        # Text
        axes[1].text(5, y_pos, f"{step}\n{detail}", ha='center', va='center', 
                    fontsize=10, fontweight='bold')
        # Arrow
        if i < len(steps_ours) - 1:
            axes[1].arrow(5, y_pos-0.6, 0, -0.5, head_width=0.3, head_length=0.1, 
                         fc='black', ec='black', linewidth=2)
    
    axes[1].set_title("OUR PIPELINE", fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    fig.text(0.5, 0.02, '★ = Key Differences | Orange = Modified Steps', 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))
    
    plt.suptitle('Preprocessing Pipeline Comparison', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(QC_DIR / 'comparison_viz1_pipeline_flowchart.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'comparison_viz1_pipeline_flowchart.png'}\n")
    plt.close()

def comparison_viz2_erp_side_by_side(data):
    """
    VIZ 2: Side-by-side ERP comparison at Oz
    """
    print("="*80)
    print("COMPARISON VIZ 2: ERP Waveforms Side-by-Side")
    print("="*80)
    
    # Compute grand averages - handle different time dimensions
    def compute_grand_avg(evoked_list):
        all_data = []
        for evoked in evoked_list:
            oz_idx = evoked.ch_names.index('Oz')
            all_data.append(evoked.data[oz_idx, :] * 1e6)
        return np.mean(all_data, axis=0), np.std(all_data, axis=0), evoked_list[0].times * 1000
    
    authors_reg, authors_reg_std, times_authors = compute_grand_avg(data['authors_regular'])
    authors_rand, authors_rand_std, _ = compute_grand_avg(data['authors_random'])
    ours_reg, ours_reg_std, times_ours = compute_grand_avg(data['ours_regular'])
    ours_rand, ours_rand_std, _ = compute_grand_avg(data['ours_random'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # TOP LEFT: Authors - Regular
    axes[0, 0].plot(times_authors, authors_reg, 'b-', linewidth=3, label='Regular (Symmetric)')
    axes[0, 0].fill_between(times_authors, authors_reg - authors_reg_std, authors_reg + authors_reg_std, 
                            alpha=0.2, color='blue')
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].axvspan(300, 700, alpha=0.1, color='green')
    axes[0, 0].set_ylabel('Amplitude (μV)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Authors Pipeline - Regular', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xlim([-200, 1000])
    
    # TOP RIGHT: Authors - Random
    axes[0, 1].plot(times_authors, authors_rand, 'r-', linewidth=3, label='Random')
    axes[0, 1].fill_between(times_authors, authors_rand - authors_rand_std, authors_rand + authors_rand_std, 
                            alpha=0.2, color='red')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0, 1].axvspan(300, 700, alpha=0.1, color='green')
    axes[0, 1].set_title('Authors Pipeline - Random', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xlim([-200, 1000])
    
    # BOTTOM LEFT: Ours - Regular
    axes[1, 0].plot(times_ours, ours_reg, 'b-', linewidth=3, label='Regular (Symmetric)')
    axes[1, 0].fill_between(times_ours, ours_reg - ours_reg_std, ours_reg + ours_reg_std, 
                            alpha=0.2, color='blue')
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].axvspan(300, 700, alpha=0.1, color='green')
    axes[1, 0].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Amplitude (μV)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Our Pipeline - Regular', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlim([-200, 1000])
    
    # BOTTOM RIGHT: Ours - Random
    axes[1, 1].plot(times_ours, ours_rand, 'r-', linewidth=3, label='Random')
    axes[1, 1].fill_between(times_ours, ours_rand - ours_rand_std, ours_rand + ours_rand_std, 
                            alpha=0.2, color='red')
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].axvspan(300, 700, alpha=0.1, color='green')
    axes[1, 1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Our Pipeline - Random', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim([-200, 1000])
    
    plt.suptitle('ERP Comparison: Authors vs Our Pipeline (Grand Average at Oz, n=24)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(QC_DIR / 'comparison_viz2_erp_side_by_side.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'comparison_viz2_erp_side_by_side.png'}\n")
    plt.close()

def comparison_viz3_difference_waves(data):
    """
    VIZ 3: Direct comparison of difference waves
    """
    print("="*80)
    print("COMPARISON VIZ 3: Difference Waves Comparison")
    print("="*80)
    
    # Compute grand averages and differences
    def compute_grand_avg(evoked_list):
        all_data = []
        for evoked in evoked_list:
            oz_idx = evoked.ch_names.index('Oz')
            all_data.append(evoked.data[oz_idx, :] * 1e6)
        return np.mean(all_data, axis=0), evoked_list[0].times * 1000
    
    authors_reg, times_authors = compute_grand_avg(data['authors_regular'])
    authors_rand, _ = compute_grand_avg(data['authors_random'])
    ours_reg, times_ours = compute_grand_avg(data['ours_regular'])
    ours_rand, _ = compute_grand_avg(data['ours_random'])
    
    # Calculate differences
    authors_diff = authors_reg - authors_rand
    ours_diff = ours_reg - ours_rand
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # TOP: Overlaid difference waves
    axes[0].plot(times_authors, authors_diff, 'purple', linewidth=3, label="Authors' Pipeline", alpha=0.8)
    axes[0].plot(times_ours, ours_diff, 'orange', linewidth=3, label='Our Pipeline', alpha=0.8)
    axes[0].axhline(0, color='black', linestyle='-', linewidth=1.5)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].axvspan(300, 700, alpha=0.1, color='green', label='Analysis Window')
    axes[0].fill_between(times_authors, authors_diff, alpha=0.2, color='purple')
    axes[0].fill_between(times_ours, ours_diff, alpha=0.2, color='orange')
    
    # Mark peaks
    window_mask_a = (times_authors >= 300) & (times_authors <= 700)
    window_mask_o = (times_ours >= 300) & (times_ours <= 700)
    authors_peak = np.max(authors_diff[window_mask_a])
    ours_peak = np.max(ours_diff[window_mask_o])
    authors_peak_time = times_authors[window_mask_a][np.argmax(authors_diff[window_mask_a])]
    ours_peak_time = times_ours[window_mask_o][np.argmax(ours_diff[window_mask_o])]
    
    axes[0].plot(authors_peak_time, authors_peak, 'o', color='purple', markersize=12)
    axes[0].plot(ours_peak_time, ours_peak, 'o', color='orange', markersize=12)
    
    axes[0].set_ylabel('Amplitude (μV)', fontsize=13, fontweight='bold')
    axes[0].set_title('Difference Waves: Regular - Random (Grand Average at Oz)', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=12, loc='upper right')
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([-200, 1000])
    
    # Add comparison text
    comparison_text = f"""
    Authors' Peak: {authors_peak:.2f} μV @ {authors_peak_time:.0f}ms
    Our Peak: {ours_peak:.2f} μV @ {ours_peak_time:.0f}ms
    
    Difference: {ours_peak - authors_peak:.2f} μV
    """
    axes[0].text(0.02, 0.98, comparison_text, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2),
                family='monospace')
    
    # BOTTOM: Show both pipelines with common time axis (interpolate to common grid)
    # Use the shorter time range
    common_time_start = max(times_authors[0], times_ours[0])
    common_time_end = min(times_authors[-1], times_ours[-1])
    common_times = np.linspace(common_time_start, common_time_end, 500)
    
    # Interpolate both to common time grid
    authors_diff_interp = np.interp(common_times, times_authors, authors_diff)
    ours_diff_interp = np.interp(common_times, times_ours, ours_diff)
    
    pipeline_diff = ours_diff_interp - authors_diff_interp
    axes[1].plot(common_times, pipeline_diff, 'green', linewidth=3)
    axes[1].fill_between(common_times, 0, pipeline_diff, where=(pipeline_diff > 0), 
                        alpha=0.3, color='green', label='Our > Authors')
    axes[1].fill_between(common_times, 0, pipeline_diff, where=(pipeline_diff < 0), 
                        alpha=0.3, color='red', label='Authors > Our')
    axes[1].axhline(0, color='black', linestyle='-', linewidth=1.5)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].axvspan(300, 700, alpha=0.1, color='green')
    
    axes[1].set_xlabel('Time (ms)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Amplitude Difference (μV)', fontsize=13, fontweight='bold')
    axes[1].set_title('Pipeline Difference: Our - Authors', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([-200, 1000])
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'comparison_viz3_difference_waves.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'comparison_viz3_difference_waves.png'}\n")
    plt.close()

def comparison_viz4_quality_metrics():
    """
    VIZ 4: Quality metrics comparison
    """
    print("="*80)
    print("COMPARISON VIZ 4: Quality Metrics Comparison")
    print("="*80)
    
    # Load metrics
    df = pd.read_csv(OUTPUT_DIR / 'preprocessing_metrics_all_subjects.csv')
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    comparisons = [
        ('ICA Components Removed', 'authors_ica_excluded', 'ours_ica_excluded', 'Count'),
        ('Epochs Kept', 'authors_epochs_total', 'ours_epochs_total', 'Count'),
        ('Regular Condition Epochs', 'authors_epochs_regular', 'ours_epochs_regular', 'Count'),
        ('SNR Estimate', 'authors_snr_estimate', 'ours_snr_estimate', 'dB'),
        ('ERP Peak Amplitude', 'authors_erp_peak_amplitude', 'ours_erp_peak_amplitude', 'μV'),
        ('Processing Quality', 'authors_epochs_total', 'ours_epochs_total', 'Score')
    ]
    
    for idx, (title, authors_col, ours_col, unit) in enumerate(comparisons):
        ax = axes[idx]
        
        if idx < 5:  # Real comparisons
            authors_vals = df[authors_col].values
            ours_vals = df[ours_col].values
            
            # Box plots
            bp = ax.boxplot([authors_vals, ours_vals], 
                           labels=["Authors", "Our"],
                           patch_artist=True,
                           widths=0.6)
            
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            # Add scatter points
            x1 = np.random.normal(1, 0.04, len(authors_vals))
            x2 = np.random.normal(2, 0.04, len(ours_vals))
            ax.scatter(x1, authors_vals, alpha=0.4, s=30, color='darkblue')
            ax.scatter(x2, ours_vals, alpha=0.4, s=30, color='darkred')
            
            # Statistical test
            t_stat, p_val = stats.ttest_rel(authors_vals, ours_vals)
            sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            
            ax.set_ylabel(unit, fontsize=11, fontweight='bold')
            ax.set_title(f'{title}\n{sig_marker} (p={p_val:.4f})', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add mean lines
            ax.axhline(np.mean(authors_vals), color='blue', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(np.mean(ours_vals), color='red', linestyle='--', linewidth=1, alpha=0.5)
        else:
            # Summary statistics table
            ax.axis('off')
            
            summary_data = []
            for metric, auth_col, our_col, _ in comparisons[:5]:
                auth_mean = df[auth_col].mean()
                our_mean = df[our_col].mean()
                diff = our_mean - auth_mean
                pct_change = (diff / auth_mean * 100) if auth_mean != 0 else 0
                summary_data.append([metric, f'{auth_mean:.2f}', f'{our_mean:.2f}', 
                                   f'{diff:+.2f}', f'{pct_change:+.1f}%'])
            
            # Create table
            table_text = "SUMMARY:\n\n"
            table_text += f"{'Metric':<20} {'Authors':<10} {'Ours':<10} {'Diff':<10} {'%Change':<10}\n"
            table_text += "="*70 + "\n"
            for row in summary_data:
                table_text += f"{row[0]:<20} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10}\n"
            
            ax.text(0.1, 0.9, table_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                            edgecolor='black', linewidth=2))
    
    plt.suptitle('Quality Metrics Comparison: Authors vs Our Pipeline (n=24 subjects)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(QC_DIR / 'comparison_viz4_quality_metrics.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'comparison_viz4_quality_metrics.png'}\n")
    plt.close()

def comparison_viz5_correlation_plot(data):
    """
    VIZ 5: Correlation between pipelines
    """
    print("="*80)
    print("COMPARISON VIZ 5: Pipeline Correlation Analysis")
    print("="*80)
    
    # Extract effect sizes from both pipelines
    def get_effects(regular_list, random_list):
        effects = []
        for reg, rand in zip(regular_list, random_list):
            oz_idx = reg.ch_names.index('Oz')
            times = reg.times * 1000
            window_mask = (times >= 300) & (times <= 700)
            
            reg_data = reg.data[oz_idx, window_mask] * 1e6
            rand_data = rand.data[oz_idx, window_mask] * 1e6
            effect = np.mean(reg_data - rand_data)
            effects.append(effect)
        return np.array(effects)
    
    authors_effects = get_effects(data['authors_regular'], data['authors_random'])
    ours_effects = get_effects(data['ours_regular'], data['ours_random'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # LEFT: Scatter plot
    axes[0].scatter(authors_effects, ours_effects, s=100, alpha=0.6, 
                   c=range(len(authors_effects)), cmap='viridis', edgecolors='black', linewidth=1)
    
    # Fit line
    z = np.polyfit(authors_effects, ours_effects, 1)
    p = np.poly1d(z)
    x_line = np.linspace(authors_effects.min(), authors_effects.max(), 100)
    axes[0].plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Identity line
    min_val = min(authors_effects.min(), ours_effects.min())
    max_val = max(authors_effects.max(), ours_effects.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, 
                alpha=0.5, label='Identity (y=x)')
    
    # Correlation
    corr = np.corrcoef(authors_effects, ours_effects)[0, 1]
    
    axes[0].set_xlabel("Authors' Pipeline Effect Size (μV)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Our Pipeline Effect Size (μV)", fontsize=12, fontweight='bold')
    axes[0].set_title(f'Pipeline Correlation (r = {corr:.3f})', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.5)
    
    # RIGHT: Bland-Altman plot
    mean_effects = (authors_effects + ours_effects) / 2
    diff_effects = ours_effects - authors_effects
    
    axes[1].scatter(mean_effects, diff_effects, s=100, alpha=0.6, 
                   c=range(len(mean_effects)), cmap='viridis', edgecolors='black', linewidth=1)
    
    mean_diff = np.mean(diff_effects)
    std_diff = np.std(diff_effects)
    
    axes[1].axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.2f} μV')
    axes[1].axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', linewidth=1, 
                   label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
    axes[1].axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', linewidth=1,
                   label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
    axes[1].axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    axes[1].set_xlabel('Mean of Two Pipelines (μV)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Difference (Our - Authors) (μV)', fontsize=12, fontweight='bold')
    axes[1].set_title('Bland-Altman Plot: Agreement Analysis', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Pipeline Agreement Analysis (Effect Size at Oz, 300-700ms)', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(QC_DIR / 'comparison_viz5_correlation_plot.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'comparison_viz5_correlation_plot.png'}")
    print(f"Correlation: r = {corr:.3f}")
    print(f"Mean difference: {mean_diff:.2f} μV\n")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("PIPELINE COMPARISON VISUALIZATIONS")
    print("Showing differences between Authors' vs Our preprocessing")
    print("="*80 + "\n")
    
    # Generate comparison visualizations
    print("Generating 5 comparison visualizations...\n")
    
    comparison_viz1_pipeline_flowchart()
    
    # Load data for remaining visualizations
    data = load_all_data()
    
    comparison_viz2_erp_side_by_side(data)
    comparison_viz3_difference_waves(data)
    comparison_viz4_quality_metrics()
    comparison_viz5_correlation_plot(data)
    
    print("\n" + "="*80)
    print("ALL COMPARISON VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nGenerated 5 comparison figures in: {QC_DIR}")
    print("\nWhat each comparison shows:")
    print("  1. Pipeline Flowchart - Visual comparison of preprocessing steps")
    print("  2. ERP Side-by-Side - Direct waveform comparison")
    print("  3. Difference Waves - Effect comparison (Regular - Random)")
    print("  4. Quality Metrics - Statistical comparison of data quality")
    print("  5. Correlation Plot - Agreement between pipelines")
    print("\nThese clearly show what changed and how it affects results!")
    print("="*80 + "\n")
