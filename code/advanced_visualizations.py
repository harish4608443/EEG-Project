"""
Advanced Visualizations - Show meaningful patterns across 24 subjects
These are publication-ready figures showing actual scientific insights
"""

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import matplotlib.gridspec as gridspec

# Configuration
BASE_DIR = Path(r'd:\ds004347')
OUTPUT_DIR = BASE_DIR / 'derivatives' / 'preprocessing_results'
QC_DIR = BASE_DIR / 'derivatives' / 'quality_control'
SUBJECTS = [f'sub-{i:03d}' for i in range(1, 25)]

sns.set_style('white')
plt.rcParams['font.size'] = 10

def load_all_evoked_data():
    """Load all evoked data for analysis"""
    print("Loading evoked data from all 24 subjects...")
    
    evokeds_dict = {
        'authors_regular': [],
        'authors_random': [],
        'ours_regular': [],
        'ours_random': []
    }
    
    for subject in SUBJECTS:
        try:
            # Authors
            evoked_a = mne.read_evokeds(OUTPUT_DIR / f'{subject}_authors_ave.fif', verbose=False)
            evokeds_dict['authors_regular'].append(evoked_a[0])
            evokeds_dict['authors_random'].append(evoked_a[1])
            
            # Ours
            evoked_o = mne.read_evokeds(OUTPUT_DIR / f'{subject}_ours_ave.fif', verbose=False)
            evokeds_dict['ours_regular'].append(evoked_o[0])
            evokeds_dict['ours_random'].append(evoked_o[1])
        except Exception as e:
            print(f"Error loading {subject}: {e}")
    
    print(f"Loaded {len(evokeds_dict['ours_regular'])} subjects")
    return evokeds_dict

def viz1_individual_subject_erps(evokeds_dict):
    """
    VIZ 1: All 24 subjects' ERPs in a grid
    Shows: Individual differences, who has strong/weak effects
    """
    print("\n" + "="*80)
    print("VISUALIZATION 1: Individual Subject ERPs (Our Pipeline)")
    print("Purpose: See individual differences and effect consistency")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(6, 4, hspace=0.4, wspace=0.3)
    
    for i, (reg, rand) in enumerate(zip(evokeds_dict['ours_regular'], 
                                         evokeds_dict['ours_random'])):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        
        # Get Oz channel
        oz_idx = reg.ch_names.index('Oz')
        times = reg.times * 1000
        
        # Plot both conditions
        ax.plot(times, reg.data[oz_idx, :] * 1e6, 'b-', linewidth=1.5, label='Regular', alpha=0.8)
        ax.plot(times, rand.data[oz_idx, :] * 1e6, 'r-', linewidth=1.5, label='Random', alpha=0.8)
        
        # Calculate difference (effect size)
        diff = (reg.data[oz_idx, :] - rand.data[oz_idx, :]) * 1e6
        peak_diff = np.max(diff[(times >= 300) & (times <= 700)])
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvspan(300, 700, alpha=0.05, color='green')
        
        ax.set_xlim([-200, 1000])
        ax.set_ylim([-8, 8])
        ax.set_title(f'{SUBJECTS[i]} | Peak: {peak_diff:.2f} μV', fontsize=10, fontweight='bold')
        
        if i % 4 == 0:
            ax.set_ylabel('Amplitude (μV)', fontsize=9)
        if i >= 20:
            ax.set_xlabel('Time (ms)', fontsize=9)
        
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')
        
        ax.grid(alpha=0.2)
    
    fig.suptitle('Individual Subject ERPs at Oz: Regular vs Random (Our Pipeline)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(QC_DIR / 'viz1_individual_subject_erps.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'viz1_individual_subject_erps.png'}")
    print("What this shows: Each subject's ERP response + their effect size")
    plt.close()

def viz2_effect_size_ranking(evokeds_dict):
    """
    VIZ 2: Effect size (Regular - Random) ranked across subjects
    Shows: Who shows strongest/weakest symmetry effect
    """
    print("\n" + "="*80)
    print("VISUALIZATION 2: Effect Size Ranking")
    print("Purpose: Which subjects show strongest symmetry preference?")
    print("="*80)
    
    effect_sizes = []
    
    for i, (reg, rand) in enumerate(zip(evokeds_dict['ours_regular'], 
                                         evokeds_dict['ours_random'])):
        oz_idx = reg.ch_names.index('Oz')
        times = reg.times * 1000
        
        # Calculate effect in 300-700ms window
        time_mask = (times >= 300) & (times <= 700)
        diff = (reg.data[oz_idx, time_mask] - rand.data[oz_idx, time_mask]) * 1e6
        
        effect_sizes.append({
            'subject': SUBJECTS[i],
            'peak_effect': np.max(diff),
            'mean_effect': np.mean(diff),
            'latency': times[time_mask][np.argmax(diff)]
        })
    
    df_effects = pd.DataFrame(effect_sizes)
    df_effects = df_effects.sort_values('peak_effect', ascending=False)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # LEFT: Ranked bar chart
    colors = ['green' if x > 0 else 'red' for x in df_effects['peak_effect']]
    bars = axes[0].barh(range(len(df_effects)), df_effects['peak_effect'], color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(df_effects)))
    axes[0].set_yticklabels(df_effects['subject'], fontsize=9)
    axes[0].set_xlabel('Peak Effect Size (μV)', fontsize=12, fontweight='bold')
    axes[0].set_title('Subjects Ranked by Symmetry Effect\n(Regular - Random at Oz, 300-700ms)', 
                     fontsize=13, fontweight='bold')
    axes[0].axvline(0, color='black', linewidth=1.5)
    axes[0].grid(axis='x', alpha=0.3)
    axes[0].invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(df_effects.iterrows()):
        axes[0].text(row['peak_effect'] + 0.1, i, f"{row['peak_effect']:.2f}", 
                    va='center', fontsize=8, fontweight='bold')
    
    # RIGHT: Distribution and statistics
    axes[1].hist(df_effects['peak_effect'], bins=15, color='steelblue', 
                alpha=0.7, edgecolor='black')
    axes[1].axvline(df_effects['peak_effect'].mean(), color='red', 
                   linestyle='--', linewidth=2.5, label=f'Mean: {df_effects["peak_effect"].mean():.2f} μV')
    axes[1].axvline(df_effects['peak_effect'].median(), color='orange', 
                   linestyle='--', linewidth=2.5, label=f'Median: {df_effects["peak_effect"].median():.2f} μV')
    
    axes[1].set_xlabel('Peak Effect Size (μV)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribution of Effect Sizes', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    
    # Add statistics box
    n_positive = (df_effects['peak_effect'] > 0).sum()
    n_negative = (df_effects['peak_effect'] < 0).sum()
    
    stats_text = f"""
    STATISTICS:
    • Positive effects: {n_positive}/24 subjects
    • Negative effects: {n_negative}/24 subjects
    • Mean: {df_effects['peak_effect'].mean():.2f} μV
    • SD: {df_effects['peak_effect'].std():.2f} μV
    • Range: {df_effects['peak_effect'].min():.2f} to {df_effects['peak_effect'].max():.2f}
    
    Interpretation:
    • Positive = Prefer symmetry (expected)
    • Negative = Prefer randomness (unexpected)
    """
    
    axes[1].text(0.98, 0.97, stats_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'viz2_effect_size_ranking.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'viz2_effect_size_ranking.png'}")
    print(f"Mean effect: {df_effects['peak_effect'].mean():.2f} μV")
    print(f"{n_positive}/24 subjects show positive effect (prefer symmetry)")
    plt.close()
    
    return df_effects

def viz3_butterfly_plot(evokeds_dict):
    """
    VIZ 3: Butterfly plot - all subjects overlaid
    Shows: Variability across subjects, grand average pattern
    """
    print("\n" + "="*80)
    print("VISUALIZATION 3: Butterfly Plot (All Subjects Overlaid)")
    print("Purpose: See variability and overall pattern")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Butterfly Plot: All 24 Subjects at Oz Electrode', fontsize=16, fontweight='bold')
    
    # Manually compute grand averages (avoid interpolation issues)
    def compute_grand_avg_oz(evoked_list):
        """Compute grand average at Oz channel only"""
        all_data = []
        for evoked in evoked_list:
            oz_idx = evoked.ch_names.index('Oz')
            all_data.append(evoked.data[oz_idx, :] * 1e6)
        return np.mean(all_data, axis=0), evoked_list[0].times * 1000
    
    # Authors - Regular
    for i, evoked in enumerate(evokeds_dict['authors_regular']):
        oz_idx = evoked.ch_names.index('Oz')
        axes[0, 0].plot(evoked.times * 1000, evoked.data[oz_idx, :] * 1e6, 
                       color='blue', alpha=0.15, linewidth=0.8)
    
    grand_avg_data, times = compute_grand_avg_oz(evokeds_dict['authors_regular'])
    axes[0, 0].plot(times, grand_avg_data, color='darkblue', linewidth=3, label='Grand Average')
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0, 0].set_title('Authors - Regular', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude (μV)', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xlim([-200, 1000])
    
    # Authors - Random
    for i, evoked in enumerate(evokeds_dict['authors_random']):
        oz_idx = evoked.ch_names.index('Oz')
        axes[0, 1].plot(evoked.times * 1000, evoked.data[oz_idx, :] * 1e6,
                       color='red', alpha=0.15, linewidth=0.8)
    
    grand_avg_data, times = compute_grand_avg_oz(evokeds_dict['authors_random'])
    axes[0, 1].plot(times, grand_avg_data, color='darkred', linewidth=3, label='Grand Average')
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0, 1].set_title('Authors - Random', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xlim([-200, 1000])
    
    # Ours - Regular
    for i, evoked in enumerate(evokeds_dict['ours_regular']):
        oz_idx = evoked.ch_names.index('Oz')
        axes[1, 0].plot(evoked.times * 1000, evoked.data[oz_idx, :] * 1e6,
                       color='blue', alpha=0.15, linewidth=0.8)
    
    grand_avg_data, times = compute_grand_avg_oz(evokeds_dict['ours_regular'])
    axes[1, 0].plot(times, grand_avg_data, color='darkblue', linewidth=3, label='Grand Average')
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1, 0].set_title('Ours - Regular', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (ms)', fontsize=11)
    axes[1, 0].set_ylabel('Amplitude (μV)', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_xlim([-200, 1000])
    
    # Ours - Random
    for i, evoked in enumerate(evokeds_dict['ours_random']):
        oz_idx = evoked.ch_names.index('Oz')
        axes[1, 1].plot(evoked.times * 1000, evoked.data[oz_idx, :] * 1e6,
                       color='red', alpha=0.15, linewidth=0.8)
    
    grand_avg_data, times = compute_grand_avg_oz(evokeds_dict['ours_random'])
    axes[1, 1].plot(times, grand_avg_data, color='darkred', linewidth=3, label='Grand Average')
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1, 1].set_title('Ours - Random', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time (ms)', fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim([-200, 1000])
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'viz3_butterfly_plot.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'viz3_butterfly_plot.png'}")
    print("What this shows: Individual variability (thin lines) vs grand average (thick line)")
    plt.close()

def viz4_time_course_analysis(evokeds_dict):
    """
    VIZ 4: When does the effect emerge? Time-course analysis
    Shows: At what time point do conditions diverge
    """
    print("\n" + "="*80)
    print("VISUALIZATION 4: Time-Course of the Effect")
    print("Purpose: When do Regular and Random patterns diverge?")
    print("="*80)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Compute grand averages manually
    def compute_grand_avg_oz(evoked_list):
        all_data = []
        for evoked in evoked_list:
            oz_idx = evoked.ch_names.index('Oz')
            all_data.append(evoked.data[oz_idx, :] * 1e6)
        return np.mean(all_data, axis=0), evoked_list[0].times * 1000
    
    reg_data, times = compute_grand_avg_oz(evokeds_dict['ours_regular'])
    rand_data, _ = compute_grand_avg_oz(evokeds_dict['ours_random'])
    diff_data = reg_data - rand_data
    
    # TOP: Waveforms with shaded difference
    axes[0].plot(times, reg_data, 'b-', linewidth=2.5, label='Regular (Symmetric)')
    axes[0].plot(times, rand_data, 'r-', linewidth=2.5, label='Random')
    axes[0].fill_between(times, reg_data, rand_data, alpha=0.2, color='green')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Amplitude (μV)', fontsize=12, fontweight='bold')
    axes[0].set_title('Grand Average ERPs: Regular vs Random', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11, loc='upper right')
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([-200, 1000])
    
    # BOTTOM: Difference wave with statistics
    axes[1].plot(times, diff_data, 'purple', linewidth=3, label='Difference (Regular - Random)')
    axes[1].fill_between(times, 0, diff_data, where=(diff_data > 0), alpha=0.3, color='green', label='Positive')
    axes[1].fill_between(times, 0, diff_data, where=(diff_data < 0), alpha=0.3, color='red', label='Negative')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.2)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    
    # Mark analysis window
    axes[1].axvspan(300, 700, alpha=0.1, color='orange', label='Analysis window (300-700ms)')
    
    # Find and mark peak
    analysis_mask = (times >= 300) & (times <= 700)
    peak_time = times[analysis_mask][np.argmax(diff_data[analysis_mask])]
    peak_amp = np.max(diff_data[analysis_mask])
    
    axes[1].plot(peak_time, peak_amp, 'ro', markersize=12, label=f'Peak: {peak_amp:.2f} μV @ {peak_time:.0f}ms')
    axes[1].annotate(f'Peak\n{peak_amp:.2f} μV\n{peak_time:.0f} ms', 
                    xy=(peak_time, peak_amp), xytext=(peak_time + 150, peak_amp + 1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    axes[1].set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Difference (μV)', fontsize=12, fontweight='bold')
    axes[1].set_title('Effect Time Course: When Does Symmetry Preference Emerge?', 
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([-200, 1000])
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'viz4_time_course_analysis.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'viz4_time_course_analysis.png'}")
    print(f"Peak effect: {peak_amp:.2f} μV at {peak_time:.0f} ms")
    plt.close()

def viz5_quality_heatmap():
    """
    VIZ 5: Quality metrics heatmap for all subjects
    Shows: Overall quality pattern, problematic subjects
    """
    print("\n" + "="*80)
    print("VISUALIZATION 5: Quality Metrics Heatmap")
    print("Purpose: See all quality metrics at a glance")
    print("="*80)
    
    # Load metrics
    df = pd.read_csv(OUTPUT_DIR / 'preprocessing_metrics_all_subjects.csv')
    
    # Select key metrics for visualization
    metrics_to_show = [
        'n_events_total',
        'ours_bad_channels',
        'ours_epochs_total',
        'ours_ica_excluded',
        'ours_snr_estimate',
        'ours_erp_peak_amplitude'
    ]
    
    # Create normalized version for heatmap
    df_metrics = df[['subject'] + metrics_to_show].set_index('subject')
    df_normalized = (df_metrics - df_metrics.mean()) / df_metrics.std()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # LEFT: Heatmap
    sns.heatmap(df_normalized.T, cmap='RdYlGn', center=0, annot=df_metrics.T, fmt='.1f',
                linewidths=0.5, cbar_kws={'label': 'Z-score'}, ax=axes[0])
    axes[0].set_title('Quality Metrics Heatmap (All 24 Subjects)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Subject', fontsize=11)
    axes[0].set_ylabel('Metric', fontsize=11)
    
    # Rename for readability
    metric_names = {
        'n_events_total': 'Total Events',
        'ours_bad_channels': 'Bad Channels',
        'ours_epochs_total': 'Epochs Kept',
        'ours_ica_excluded': 'ICA Excluded',
        'ours_snr_estimate': 'SNR',
        'ours_erp_peak_amplitude': 'Peak Amplitude (μV)'
    }
    axes[0].set_yticklabels([metric_names.get(m, m) for m in metrics_to_show])
    
    # RIGHT: Summary statistics
    axes[1].axis('off')
    
    summary_text = "QUALITY SUMMARY (Our Pipeline)\n\n"
    for metric in metrics_to_show:
        mean_val = df_metrics[metric].mean()
        std_val = df_metrics[metric].std()
        min_val = df_metrics[metric].min()
        max_val = df_metrics[metric].max()
        
        display_name = metric_names.get(metric, metric)
        summary_text += f"{display_name}:\n"
        summary_text += f"  Mean: {mean_val:.2f} ± {std_val:.2f}\n"
        summary_text += f"  Range: {min_val:.2f} - {max_val:.2f}\n\n"
    
    axes[1].text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'viz5_quality_heatmap.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'viz5_quality_heatmap.png'}")
    print("What this shows: Green=good, Red=problematic for each metric")
    plt.close()

def viz6_consistency_analysis(df_effects):
    """
    VIZ 6: Effect consistency across subjects
    Shows: Statistical reliability, outliers
    """
    print("\n" + "="*80)
    print("VISUALIZATION 6: Effect Consistency & Reliability")
    print("Purpose: How consistent is the effect across subjects?")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Effect Consistency Analysis (24 Subjects)', fontsize=16, fontweight='bold')
    
    # Load metrics for correlation analysis
    df_metrics = pd.read_csv(OUTPUT_DIR / 'preprocessing_metrics_all_subjects.csv')
    
    # Merge with effect sizes
    df_full = df_metrics.merge(df_effects, left_on='subject', right_on='subject')
    
    # TOP LEFT: Effect size vs SNR
    axes[0, 0].scatter(df_full['ours_snr_estimate'], df_full['peak_effect'], 
                      s=100, alpha=0.6, c=df_full['ours_bad_channels'], cmap='YlOrRd')
    
    # Fit line
    z = np.polyfit(df_full['ours_snr_estimate'], df_full['peak_effect'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_full['ours_snr_estimate'].min(), df_full['ours_snr_estimate'].max(), 100)
    axes[0, 0].plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit line')
    
    corr = np.corrcoef(df_full['ours_snr_estimate'], df_full['peak_effect'])[0, 1]
    axes[0, 0].set_xlabel('SNR Estimate', fontsize=11)
    axes[0, 0].set_ylabel('Peak Effect Size (μV)', fontsize=11)
    axes[0, 0].set_title(f'Effect Size vs Signal Quality\nr = {corr:.3f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    
    # TOP RIGHT: Effect size vs Epochs kept
    axes[0, 1].scatter(df_full['ours_epochs_total'], df_full['peak_effect'],
                      s=100, alpha=0.6, color='steelblue')
    
    corr2 = np.corrcoef(df_full['ours_epochs_total'], df_full['peak_effect'])[0, 1]
    axes[0, 1].set_xlabel('Epochs Kept', fontsize=11)
    axes[0, 1].set_ylabel('Peak Effect Size (μV)', fontsize=11)
    axes[0, 1].set_title(f'Effect Size vs Data Quality\nr = {corr2:.3f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # BOTTOM LEFT: One-sample t-test visualization
    effect_values = df_effects['peak_effect'].values
    t_stat, p_val = stats.ttest_1samp(effect_values, 0)
    
    axes[1, 0].violinplot([effect_values], positions=[1], showmeans=True, showmedians=True)
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2, label='Null hypothesis (0)')
    axes[1, 0].scatter([1]*len(effect_values), effect_values, alpha=0.4, s=50, color='blue')
    axes[1, 0].set_xticks([1])
    axes[1, 0].set_xticklabels(['All Subjects'])
    axes[1, 0].set_ylabel('Peak Effect Size (μV)', fontsize=11)
    axes[1, 0].set_title(f'Effect vs Null Hypothesis\nt={t_stat:.2f}, p={p_val:.6f}',
                        fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # BOTTOM RIGHT: Summary table
    axes[1, 1].axis('off')
    
    # Calculate confidence interval
    ci = stats.t.interval(0.95, len(effect_values)-1,
                         loc=np.mean(effect_values),
                         scale=stats.sem(effect_values))
    
    # Cohen's d effect size
    cohens_d = np.mean(effect_values) / np.std(effect_values)
    
    summary = f"""
    STATISTICAL SUMMARY
    
    Sample: n = {len(effect_values)} subjects
    
    Effect Size (Regular - Random):
    • Mean: {np.mean(effect_values):.3f} μV
    • SD: {np.std(effect_values):.3f} μV
    • 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]
    
    One-Sample t-test (vs 0):
    • t-statistic: {t_stat:.3f}
    • p-value: {p_val:.6f}
    • Cohen's d: {cohens_d:.3f}
    
    Interpretation:
    • {len(effect_values[effect_values > 0])}/24 subjects show positive effect
    • {len(effect_values[effect_values < 0])}/24 subjects show negative effect
    
    Reliability:
    • Effect is {"SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"}
    • Effect size is {"LARGE" if abs(cohens_d) > 0.8 else "MEDIUM" if abs(cohens_d) > 0.5 else "SMALL"}
    """
    
    axes[1, 1].text(0.1, 0.9, summary, fontsize=10, verticalalignment='top',
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'viz6_consistency_analysis.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'viz6_consistency_analysis.png'}")
    print(f"Statistical test: t={t_stat:.2f}, p={p_val:.6f}")
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("ADVANCED VISUALIZATIONS - Publication Quality Figures")
    print("="*80 + "\n")
    
    # Load all data
    evokeds_dict = load_all_evoked_data()
    
    # Generate all visualizations
    print("\nGenerating 6 advanced visualizations...\n")
    
    viz1_individual_subject_erps(evokeds_dict)
    df_effects = viz2_effect_size_ranking(evokeds_dict)
    viz3_butterfly_plot(evokeds_dict)
    viz4_time_course_analysis(evokeds_dict)
    viz5_quality_heatmap()
    viz6_consistency_analysis(df_effects)
    
    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nGenerated 6 publication-quality figures in: {QC_DIR}")
    print("\nWhat each figure shows:")
    print("  1. Individual ERPs - See each subject's response")
    print("  2. Effect Size Ranking - Who shows strongest/weakest effects")
    print("  3. Butterfly Plot - Overall variability and grand average")
    print("  4. Time-Course - When does the effect emerge")
    print("  5. Quality Heatmap - All metrics at a glance")
    print("  6. Consistency Analysis - Statistical reliability")
    print("\nThese figures are ready for your presentation/publication!")
    print("="*80 + "\n")
