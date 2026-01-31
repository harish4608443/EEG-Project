"""
Sanity Checks and Quality Control - Milestone 4
Analyze all subjects for outliers and problems
Compare to author results
"""

import os
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

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)

def load_metrics():
    """Load preprocessing metrics from batch processing"""
    metrics_path = OUTPUT_DIR / 'preprocessing_metrics_all_subjects.csv'
    if not metrics_path.exists():
        print("ERROR: Metrics file not found. Run batch_preprocessing.py first!")
        return None
    return pd.read_csv(metrics_path)

def sanity_check_1_event_counts(df):
    """Check if all subjects have similar number of events"""
    print("\n" + "="*80)
    print("SANITY CHECK 1: Event Counts")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Sanity Check 1: Event Counts Across Subjects', fontsize=16, fontweight='bold')
    
    # Total events
    axes[0, 0].bar(range(len(df)), df['n_events_total'])
    axes[0, 0].axhline(df['n_events_total'].mean(), color='r', linestyle='--', label=f'Mean: {df["n_events_total"].mean():.1f}')
    axes[0, 0].set_xlabel('Subject Index')
    axes[0, 0].set_ylabel('Number of Events')
    axes[0, 0].set_title('Total Events per Subject')
    axes[0, 0].legend()
    
    # Regular vs Random
    x = np.arange(len(df))
    width = 0.35
    axes[0, 1].bar(x - width/2, df['n_events_regular'], width, label='Regular', alpha=0.8)
    axes[0, 1].bar(x + width/2, df['n_events_random'], width, label='Random', alpha=0.8)
    axes[0, 1].set_xlabel('Subject Index')
    axes[0, 1].set_ylabel('Number of Events')
    axes[0, 1].set_title('Regular vs Random Events')
    axes[0, 1].legend()
    
    # Distribution
    axes[1, 0].hist(df['n_events_total'], bins=15, edgecolor='black')
    axes[1, 0].axvline(df['n_events_total'].mean(), color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Number of Events')
    axes[1, 0].set_ylabel('Number of Subjects')
    axes[1, 0].set_title('Distribution of Total Events')
    
    # Summary statistics
    stats_text = f"""Summary Statistics:
    
Total Events:
  Mean: {df['n_events_total'].mean():.1f}
  Std: {df['n_events_total'].std():.1f}
  Min: {df['n_events_total'].min():.0f}
  Max: {df['n_events_total'].max():.0f}
  
Regular Events:
  Mean: {df['n_events_regular'].mean():.1f}
  Range: {df['n_events_regular'].min():.0f}-{df['n_events_regular'].max():.0f}
  
Random Events:
  Mean: {df['n_events_random'].mean():.1f}
  Range: {df['n_events_random'].min():.0f}-{df['n_events_random'].max():.0f}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'sanity_check_1_event_counts.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {QC_DIR / 'sanity_check_1_event_counts.png'}")
    
    # Identify outliers
    mean_events = df['n_events_total'].mean()
    std_events = df['n_events_total'].std()
    outliers = df[np.abs(df['n_events_total'] - mean_events) > 2 * std_events]
    
    if len(outliers) > 0:
        print(f"\n[WARNING] WARNING: {len(outliers)} subject(s) with unusual event counts:")
        for _, row in outliers.iterrows():
            print(f"  {row['subject']}: {row['n_events_total']} events (expected ~{mean_events:.0f})")
    else:
        print("\n[OK] All subjects have normal event counts")
    
    plt.close()

def sanity_check_2_bad_channels(df):
    """Check bad channel detection across subjects"""
    print("\n" + "="*80)
    print("SANITY CHECK 2: Bad Channel Detection")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Sanity Check 2: Bad Channel Detection', fontsize=16, fontweight='bold')
    
    # Number of bad channels per subject
    axes[0, 0].bar(range(len(df)), df['ours_bad_channels'], color='coral')
    axes[0, 0].axhline(df['ours_bad_channels'].mean(), color='r', linestyle='--', 
                       label=f'Mean: {df["ours_bad_channels"].mean():.2f}')
    axes[0, 0].set_xlabel('Subject Index')
    axes[0, 0].set_ylabel('Number of Bad Channels')
    axes[0, 0].set_title('Bad Channels per Subject (Our Pipeline)')
    axes[0, 0].legend()
    
    # Distribution
    axes[0, 1].hist(df['ours_bad_channels'], bins=range(0, int(df['ours_bad_channels'].max())+2), 
                    edgecolor='black', color='coral')
    axes[0, 1].set_xlabel('Number of Bad Channels')
    axes[0, 1].set_ylabel('Number of Subjects')
    axes[0, 1].set_title('Distribution of Bad Channels')
    
    # ICA components excluded
    axes[1, 0].bar(range(len(df)), df['ours_ica_excluded'], color='skyblue', label='Ours (Auto)')
    axes[1, 0].bar(range(len(df)), df['authors_ica_excluded'], color='lightcoral', 
                   alpha=0.6, label='Authors (Manual)')
    axes[1, 0].set_xlabel('Subject Index')
    axes[1, 0].set_ylabel('ICA Components Excluded')
    axes[1, 0].set_title('ICA Components Excluded per Subject')
    axes[1, 0].legend()
    
    # Summary
    stats_text = f"""Summary Statistics:
    
Bad Channels (Our Pipeline):
  Mean: {df['ours_bad_channels'].mean():.2f}
  Median: {df['ours_bad_channels'].median():.1f}
  Max: {df['ours_bad_channels'].max():.0f}
  Subjects with 0 bad: {(df['ours_bad_channels'] == 0).sum()}
  Subjects with >2 bad: {(df['ours_bad_channels'] > 2).sum()}
  
ICA Components Excluded:
  Our Pipeline: {df['ours_ica_excluded'].mean():.2f} (mean)
  Authors: {df['authors_ica_excluded'].mean():.2f} (mean)
  
Interpolation Success:
  Successful: {df['ours_interpolation_success'].sum()}
  Failed: {(~df['ours_interpolation_success']).sum()}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'sanity_check_2_bad_channels.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {QC_DIR / 'sanity_check_2_bad_channels.png'}")
    
    # Identify subjects with many bad channels
    if df['ours_bad_channels'].max() > 3:
        problem_subjects = df[df['ours_bad_channels'] > 3]
        print(f"\n[WARNING] WARNING: {len(problem_subjects)} subject(s) with >3 bad channels:")
        for _, row in problem_subjects.iterrows():
            print(f"  {row['subject']}: {row['ours_bad_channels']} bad channels - {row['ours_bad_channels_list']}")
    else:
        print("\n[OK] All subjects have acceptable number of bad channels (<=3)")
    
    plt.close()

def sanity_check_3_epoch_counts(df):
    """Check epoch rejection and trial counts"""
    print("\n" + "="*80)
    print("SANITY CHECK 3: Epoch Counts and Trial Retention")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Sanity Check 3: Epoch Counts & Retention (NOT SNR)', fontsize=16, fontweight='bold')
    
    # Calculate retention rate
    df['authors_retention'] = (df['authors_epochs_total'] / df['n_events_total']) * 100
    df['ours_retention'] = (df['ours_epochs_total'] / df['n_events_total']) * 100
    
    # Plot 1: Absolute epoch counts per subject (not retention %)
    x = np.arange(len(df))
    width = 0.35
    axes[0, 0].bar(x - width/2, df['authors_epochs_total'], width, label='Authors', alpha=0.8, color='purple')
    axes[0, 0].bar(x + width/2, df['ours_epochs_total'], width, label='Ours', alpha=0.8, color='green')
    axes[0, 0].set_xlabel('Subject Index')
    axes[0, 0].set_ylabel('Number of Epochs Kept')
    axes[0, 0].set_title('Total Epochs Retained per Subject')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([140, 165])
    
    # Plot 2: Rejection rate (inverse of retention)
    df['authors_rejection'] = 100 - df['authors_retention']
    df['ours_rejection'] = 100 - df['ours_retention']
    
    axes[0, 1].bar(x - width/2, df['authors_rejection'], width, label='Authors', alpha=0.8, color='purple')
    axes[0, 1].bar(x + width/2, df['ours_rejection'], width, label='Ours', alpha=0.8, color='green')
    axes[0, 1].axhline(5, color='r', linestyle='--', linewidth=1, label='5% threshold')
    axes[0, 1].set_xlabel('Subject Index')
    axes[0, 1].set_ylabel('Rejection Rate (%)')
    axes[0, 1].set_title('Epoch Rejection Rate per Subject')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 15])
    
    # Plot 3: Condition balance (Regular vs Random)
    axes[1, 0].scatter(df['ours_epochs_regular'], df['ours_epochs_random'], alpha=0.6, s=100, color='green')
    min_val = min(df['ours_epochs_regular'].min(), df['ours_epochs_random'].min())
    max_val = max(df['ours_epochs_regular'].max(), df['ours_epochs_random'].max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect balance', linewidth=2)
    axes[1, 0].set_xlabel('Regular Epochs Retained')
    axes[1, 0].set_ylabel('Random Epochs Retained')
    axes[1, 0].set_title('Condition Balance Check')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    # Plot 4: Summary statistics
    balance_diff = abs(df['ours_epochs_regular'] - df['ours_epochs_random'])
    stats_text = f"""Summary Statistics:
    
Epochs Kept (Authors):
  Mean: {df['authors_epochs_total'].mean():.1f} epochs
  Range: {df['authors_epochs_total'].min():.0f}-{df['authors_epochs_total'].max():.0f}
  
Epochs Kept (Ours):
  Mean: {df['ours_epochs_total'].mean():.1f} epochs
  Range: {df['ours_epochs_total'].min():.0f}-{df['ours_epochs_total'].max():.0f}
  
Rejection Rate:
  Authors: {df['authors_rejection'].mean():.2f}% (mean)
  Ours: {df['ours_rejection'].mean():.2f}% (mean)
  
Condition Balance (Ours):
  Regular: {df['ours_epochs_regular'].mean():.1f}
  Random: {df['ours_epochs_random'].mean():.1f}
  Max imbalance: {balance_diff.max():.0f} epochs
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'sanity_check_3_epoch_retention.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {QC_DIR / 'sanity_check_3_epoch_retention.png'}")
    
    # Check for low retention
    low_retention = df[df['ours_retention'] < 90]
    if len(low_retention) > 0:
        print(f"\n[WARNING] WARNING: {len(low_retention)} subject(s) with <90% retention:")
        for _, row in low_retention.iterrows():
            print(f"  {row['subject']}: {row['ours_retention']:.1f}% retention")
    else:
        print("\n[OK] All subjects have good epoch retention (>=90%)")
    
    plt.close()
    return df

def sanity_check_4_snr_quality(df):
    """Check signal-to-noise ratio estimates"""
    print("\n" + "="*80)
    print("SANITY CHECK 4: Signal Quality (SNR Estimates)")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Sanity Check 4: Signal Quality & Peak Amplitudes', fontsize=16, fontweight='bold')
    
    # Plot 1: SNR per subject (NOT bar comparison - use scatter)
    x = np.arange(len(df))
    axes[0, 0].scatter(x, df['authors_snr_estimate'], label='Authors (Mean)', alpha=0.7, s=80, color='red', marker='o')
    axes[0, 0].scatter(x, df['ours_snr_estimate'], label='Ours (Median)', alpha=0.7, s=80, color='blue', marker='s')
    axes[0, 0].plot(x, df['authors_snr_estimate'], 'r-', alpha=0.3)
    axes[0, 0].plot(x, df['ours_snr_estimate'], 'b-', alpha=0.3)
    axes[0, 0].set_xlabel('Subject Index')
    axes[0, 0].set_ylabel('SNR Estimate (ratio)')
    axes[0, 0].set_title('SNR Evolution Across Subjects')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: SNR Distribution with violin plot
    snr_data = [df['authors_snr_estimate'].values, df['ours_snr_estimate'].values]
    parts = axes[0, 1].violinplot(snr_data, positions=[1, 2], showmeans=True, showmedians=True)
    axes[0, 1].set_xticks([1, 2])
    axes[0, 1].set_xticklabels(['Authors\n(Mean Avg)', 'Ours\n(Median Avg)'])
    axes[0, 1].set_ylabel('SNR Estimate')
    axes[0, 1].set_title('SNR Distribution Comparison')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Peak amplitude correlation (multiply by 1e6 for μV)
    axes[1, 0].scatter(df['authors_erp_peak_amplitude'] * 1e6, 
                      df['ours_erp_peak_amplitude'] * 1e6, 
                      alpha=0.6, s=100, c=np.arange(len(df)), cmap='viridis')
    
    # Add correlation line
    from scipy.stats import linregress
    slope, intercept, r_value, _, _ = linregress(df['authors_erp_peak_amplitude'] * 1e6, 
                                                   df['ours_erp_peak_amplitude'] * 1e6)
    x_line = np.array([df['authors_erp_peak_amplitude'].min() * 1e6, 
                       df['authors_erp_peak_amplitude'].max() * 1e6])
    axes[1, 0].plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, label=f'Fit: r²={r_value**2:.3f}')
    
    axes[1, 0].set_xlabel('Authors Peak Amplitude (μV)')
    axes[1, 0].set_ylabel('Our Peak Amplitude (μV)')
    axes[1, 0].set_title('Peak ERP Amplitude Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary with boxplot
    snr_corr = np.corrcoef(df['authors_snr_estimate'], df['ours_snr_estimate'])[0, 1]
    amp_corr = np.corrcoef(df['authors_erp_peak_amplitude'], df['ours_erp_peak_amplitude'])[0, 1]
    
    stats_text = f"""Summary Statistics:
    
SNR Estimates:
  Authors (Mean Avg):
    Mean: {df['authors_snr_estimate'].mean():.2f} ± {df['authors_snr_estimate'].std():.2f}
    Median: {df['authors_snr_estimate'].median():.2f}
  
  Ours (Median Avg):
    Mean: {df['ours_snr_estimate'].mean():.2f} ± {df['ours_snr_estimate'].std():.2f}
    Median: {df['ours_snr_estimate'].median():.2f}
  
  Correlation: r = {snr_corr:.3f}
  Difference: {(df['authors_snr_estimate'] - df['ours_snr_estimate']).mean():.2f}
  
Peak Amplitudes:
  Correlation: r = {amp_corr:.3f}
  Our > Authors: {(df['ours_erp_peak_amplitude'] > df['authors_erp_peak_amplitude']).sum()}/{len(df)} subjects
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'sanity_check_4_snr_quality.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {QC_DIR / 'sanity_check_4_snr_quality.png'}")
    
    # Statistical test
    t_stat, p_val = stats.ttest_rel(df['ours_snr_estimate'], df['authors_snr_estimate'])
    if p_val < 0.05:
        if df['ours_snr_estimate'].mean() > df['authors_snr_estimate'].mean():
            print(f"\n[OK] Our pipeline has significantly BETTER SNR (p={p_val:.4f})")
        else:
            print(f"\n[WARNING] Authors' pipeline has significantly HIGHER SNR (p={p_val:.4f})")
            print("   Note: This is EXPECTED - mean averaging inflates SNR vs median")
    else:
        print(f"\n[INFO] No significant difference in SNR between pipelines (p={p_val:.4f})")
    
    plt.close()

def sanity_check_5_grand_average_erps():
    """Check grand average ERPs with baseline"""
    print("\n" + "="*80)
    print("SANITY CHECK 5: Grand Average ERPs")
    print("="*80)
    
    # Load all evoked data
    evokeds_authors_regular = []
    evokeds_authors_random = []
    evokeds_ours_regular = []
    evokeds_ours_random = []
    
    for subject in SUBJECTS:
        try:
            # Authors
            evoked_a = mne.read_evokeds(OUTPUT_DIR / f'{subject}_authors_ave.fif')
            evokeds_authors_regular.append(evoked_a[0])
            evokeds_authors_random.append(evoked_a[1])
            
            # Ours
            evoked_o = mne.read_evokeds(OUTPUT_DIR / f'{subject}_ours_ave.fif')
            evokeds_ours_regular.append(evoked_o[0])
            evokeds_ours_random.append(evoked_o[1])
        except:
            print(f"  Warning: Could not load {subject}")
    
    print(f"Loaded {len(evokeds_authors_regular)} subjects for grand averaging")
    
    # Compute grand averages WITHOUT interpolation to avoid NaN/Inf errors
    # Just exclude channels that are bad in ANY subject
    all_bads_authors = set()
    all_bads_ours = set()
    for evk_a, evk_o in zip(evokeds_authors_regular, evokeds_ours_regular):
        all_bads_authors.update(evk_a.info['bads'])
        all_bads_ours.update(evk_o.info['bads'])
    
    print(f"  Authors: {len(all_bads_authors)} channels bad in at least one subject: {sorted(all_bads_authors)}")
    print(f"  Ours: {len(all_bads_ours)} channels bad in at least one subject: {sorted(all_bads_ours)}")
    
    # Drop bad channels instead of interpolating
    for evk in evokeds_authors_regular + evokeds_authors_random:
        evk.info['bads'] = list(all_bads_authors)
        if len(all_bads_authors) > 0:
            evk.drop_channels(list(all_bads_authors))
    
    for evk in evokeds_ours_regular + evokeds_ours_random:
        evk.info['bads'] = list(all_bads_ours)
        if len(all_bads_ours) > 0:
            evk.drop_channels(list(all_bads_ours))
    
    # Compute grand averages
    grand_avg_authors_regular = mne.grand_average(evokeds_authors_regular, interpolate_bads=False, drop_bads=False)
    grand_avg_authors_random = mne.grand_average(evokeds_authors_random, interpolate_bads=False, drop_bads=False)
    grand_avg_ours_regular = mne.grand_average(evokeds_ours_regular, interpolate_bads=False, drop_bads=False)
    grand_avg_ours_random = mne.grand_average(evokeds_ours_random, interpolate_bads=False, drop_bads=False)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Sanity Check 5: Grand Average ERPs at Oz (Occipital Midline)', fontsize=16, fontweight='bold')
    
    # Authors - Regular
    axes[0, 0].plot(grand_avg_authors_regular.times * 1000, 
                    grand_avg_authors_regular.data[grand_avg_authors_regular.ch_names.index('Oz'), :] * 1e6,
                    color='blue', linewidth=2, label='Regular')
    axes[0, 0].plot(grand_avg_authors_random.times * 1000,
                    grand_avg_authors_random.data[grand_avg_authors_random.ch_names.index('Oz'), :] * 1e6,
                    color='red', linewidth=2, label='Random')
    axes[0, 0].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 0].axvline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 0].axvspan(-200, 50, alpha=0.2, color='yellow', label='Authors Baseline')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].set_title('Authors Pipeline (0.1-25 Hz)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Ours - Regular  
    axes[0, 1].plot(grand_avg_ours_regular.times * 1000,
                    grand_avg_ours_regular.data[grand_avg_ours_regular.ch_names.index('Oz'), :] * 1e6,
                    color='blue', linewidth=2, label='Regular')
    axes[0, 1].plot(grand_avg_ours_random.times * 1000,
                    grand_avg_ours_random.data[grand_avg_ours_random.ch_names.index('Oz'), :] * 1e6,
                    color='red', linewidth=2, label='Random')
    axes[0, 1].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 1].axvline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 1].axvspan(-200, 0, alpha=0.2, color='cyan', label='Our Baseline')
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Amplitude (μV)')
    axes[0, 1].set_title('Our Pipeline (0.1-40 Hz + 50Hz notch)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Difference waves
    diff_authors = (grand_avg_authors_regular.data[grand_avg_authors_regular.ch_names.index('Oz'), :] - 
                    grand_avg_authors_random.data[grand_avg_authors_random.ch_names.index('Oz'), :]) * 1e6
    diff_ours = (grand_avg_ours_regular.data[grand_avg_ours_regular.ch_names.index('Oz'), :] - 
                 grand_avg_ours_random.data[grand_avg_ours_random.ch_names.index('Oz'), :]) * 1e6
    
    axes[1, 0].plot(grand_avg_authors_regular.times * 1000, diff_authors, 
                    color='purple', linewidth=2.5, label='Authors (Regular - Random)')
    axes[1, 0].plot(grand_avg_ours_regular.times * 1000, diff_ours,
                    color='green', linewidth=2.5, label='Ours (Regular - Random)')
    axes[1, 0].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 0].axvline(0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 0].fill_between(grand_avg_authors_regular.times * 1000, diff_authors, alpha=0.3, color='purple')
    axes[1, 0].fill_between(grand_avg_ours_regular.times * 1000, diff_ours, alpha=0.3, color='green')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Amplitude Difference (μV)')
    axes[1, 0].set_title('Difference Waves: Regular - Random')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Peak latency comparison
    # Find peaks in 300-700ms window
    time_mask_a = (grand_avg_authors_regular.times >= 0.3) & (grand_avg_authors_regular.times <= 0.7)
    time_mask_o = (grand_avg_ours_regular.times >= 0.3) & (grand_avg_ours_regular.times <= 0.7)
    
    # Get the difference wave values in the time window
    diff_authors_window = diff_authors[time_mask_a]
    diff_ours_window = diff_ours[time_mask_o]
    
    peak_idx_a = np.argmax(diff_authors_window)
    peak_idx_o = np.argmax(diff_ours_window)
    
    peak_time_a = grand_avg_authors_regular.times[time_mask_a][peak_idx_a] * 1000
    peak_time_o = grand_avg_ours_regular.times[time_mask_o][peak_idx_o] * 1000
    peak_amp_a = diff_authors_window[peak_idx_a]
    peak_amp_o = diff_ours_window[peak_idx_o]
    
    stats_text = f"""Grand Average Statistics:
    
Number of subjects: {len(evokeds_authors_regular)}
    
Peak Difference (300-700ms):
  Authors Pipeline:
    Latency: {peak_time_a:.0f} ms
    Amplitude: {peak_amp_a:.2f} μV
    
  Our Pipeline:
    Latency: {peak_time_o:.0f} ms
    Amplitude: {peak_amp_o:.2f} μV
    
Baseline Periods:
  Authors: -200 to +50 ms
  Ours: -200 to 0 ms
  
Effect Direction:
  {"[OK] Regular > Random (expected)" if peak_amp_a > 0 else "[ERROR] Random > Regular"}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'sanity_check_5_grand_average_erps.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {QC_DIR / 'sanity_check_5_grand_average_erps.png'}")
    
    # Save grand averages
    mne.write_evokeds(OUTPUT_DIR / 'grand_average_authors.fif',
                     [grand_avg_authors_regular, grand_avg_authors_random], overwrite=True)
    mne.write_evokeds(OUTPUT_DIR / 'grand_average_ours.fif',
                     [grand_avg_ours_regular, grand_avg_ours_random], overwrite=True)
    print(f"[OK] Saved grand averages to {OUTPUT_DIR}")
    
    plt.close()

def sanity_check_6_filter_comparison():
    """Replication check: 25 Hz vs 40 Hz filter comparison"""
    print("\n" + "="*80)
    print("SANITY CHECK 6: Filter Comparison (25 Hz vs 40 Hz)")
    print("="*80)
    
    # Load grand averages
    grand_avg_25hz = mne.read_evokeds(OUTPUT_DIR / 'grand_average_authors.fif')  # 25 Hz
    grand_avg_40hz = mne.read_evokeds(OUTPUT_DIR / 'grand_average_ours.fif')  # 40 Hz
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Replication Check: Authors (25 Hz) vs Our Baseline (40 Hz)', fontsize=16, fontweight='bold')
    
    # Get Oz channel data
    oz_25_reg = grand_avg_25hz[0].data[grand_avg_25hz[0].ch_names.index('Oz'), :] * 1e6
    oz_25_rand = grand_avg_25hz[1].data[grand_avg_25hz[1].ch_names.index('Oz'), :] * 1e6
    oz_40_reg = grand_avg_40hz[0].data[grand_avg_40hz[0].ch_names.index('Oz'), :] * 1e6
    oz_40_rand = grand_avg_40hz[1].data[grand_avg_40hz[1].ch_names.index('Oz'), :] * 1e6
    
    times_25 = grand_avg_25hz[0].times * 1000
    times_40 = grand_avg_40hz[0].times * 1000
    
    # Regular condition comparison
    axes[0, 0].plot(times_25, oz_25_reg, color='blue', linewidth=2, label='25 Hz (Authors)')
    axes[0, 0].plot(times_40, oz_40_reg, color='cyan', linewidth=2, label='40 Hz (Ours)', linestyle='--')
    axes[0, 0].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 0].axvline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].set_title('Regular Condition: Filter Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Random condition comparison
    axes[0, 1].plot(times_25, oz_25_rand, color='red', linewidth=2, label='25 Hz (Authors)')
    axes[0, 1].plot(times_40, oz_40_rand, color='orange', linewidth=2, label='40 Hz (Ours)', linestyle='--')
    axes[0, 1].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 1].axvline(0, color='k', linestyle='--', linewidth=0.5)
    axes[0, 1].set_xlabel('Time (ms)')
    axes[0, 1].set_ylabel('Amplitude (μV)')
    axes[0, 1].set_title('Random Condition: Filter Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Difference waves
    diff_25 = oz_25_reg - oz_25_rand
    diff_40 = oz_40_reg - oz_40_rand
    
    axes[1, 0].plot(times_25, diff_25, color='purple', linewidth=2.5, label='25 Hz (Authors)')
    axes[1, 0].plot(times_40, diff_40, color='green', linewidth=2.5, label='40 Hz (Ours)', linestyle='--')
    axes[1, 0].axhline(0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 0].axvline(0, color='k', linestyle='--', linewidth=0.5)
    axes[1, 0].fill_between(times_25, diff_25, alpha=0.2, color='purple')
    axes[1, 0].fill_between(times_40, diff_40, alpha=0.2, color='green')
    axes[1, 0].set_xlabel('Time (ms)')
    axes[1, 0].set_ylabel('Amplitude Difference (μV)')
    axes[1, 0].set_title('Effect Size: Regular - Random')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation in overlapping time window
    # Resample to same time points for correlation
    from scipy.interpolate import interp1d
    f_25 = interp1d(times_25, diff_25, bounds_error=False, fill_value=0)
    diff_25_resampled = f_25(times_40)
    
    corr = np.corrcoef(diff_25_resampled[~np.isnan(diff_25_resampled)], 
                       diff_40[~np.isnan(diff_25_resampled)])[0, 1]
    
    # RMSE
    rmse = np.sqrt(np.mean((diff_25_resampled[~np.isnan(diff_25_resampled)] - 
                           diff_40[~np.isnan(diff_25_resampled)])**2))
    
    stats_text = f"""Replication Statistics:
    
Filter Settings:
  Authors: 0.1-25 Hz bandpass
  Ours: 0.1-40 Hz bandpass + 50 Hz notch
  
Correlation:
  Difference waves: r = {corr:.3f}
  Interpretation: {"High replication" if corr > 0.9 else "Moderate replication" if corr > 0.7 else "Low replication"}
  
RMSE:
  {rmse:.3f} μV
  
Peak Amplitudes (Regular - Random):
  25 Hz: {np.max(diff_25):.2f} μV
  40 Hz: {np.max(diff_40):.2f} μV
  Difference: {np.max(diff_40) - np.max(diff_25):.2f} μV
  
Interpretation:
  {"[OK] Both filters capture the symmetry effect" if corr > 0.7 else "[WARNING] Different results between filters"}
  {"[OK] 40 Hz preserves higher frequencies" if np.max(diff_40) > np.max(diff_25) else "[INFO] Similar amplitude"}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'sanity_check_6_filter_comparison.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: {QC_DIR / 'sanity_check_6_filter_comparison.png'}")
    
    print(f"\n[INFO] Replication correlation: {corr:.3f}")
    if corr > 0.9:
        print("[OK] Excellent replication - both filters produce highly similar results")
    elif corr > 0.7:
        print("[OK] Good replication - filters produce similar patterns")
    else:
        print("[WARNING] Filters produce notably different results - investigate further")
    
    plt.close()


# ============================================================================
# MAIN SANITY CHECK EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("MILESTONE 4: SANITY CHECKS AND QUALITY CONTROL")
    print("="*80 + "\n")
    
    # Load metrics
    df = load_metrics()
    if df is None:
        print("\nERROR: Run batch_preprocessing.py first!")
        exit(1)
    
    print(f"Loaded metrics for {len(df)} subjects\n")
    
    # Run all sanity checks
    sanity_check_1_event_counts(df)
    sanity_check_2_bad_channels(df)
    df = sanity_check_3_epoch_counts(df)
    sanity_check_4_snr_quality(df)
    sanity_check_5_grand_average_erps()
    sanity_check_6_filter_comparison()
    
    print("\n" + "="*80)
    print("SANITY CHECKS COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {QC_DIR}")
    print("\nReview the generated plots to identify any problematic subjects.")
    print("="*80 + "\n")
