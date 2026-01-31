"""
Easy-to-Understand Visualizations - Clear insights from 24 subjects
Focus: Simple, intuitive plots that anyone can interpret
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

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11

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

def easy_viz1_simple_comparison(evokeds):
    """
    EASY VIZ 1: Dead simple - Regular vs Random (Grand Average)
    ONE MESSAGE: Do people prefer symmetry? YES/NO
    """
    print("="*80)
    print("EASY VIZ 1: Do People Prefer Symmetric Patterns?")
    print("="*80)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Compute grand averages
    def compute_grand_avg(evoked_list):
        all_data = []
        for evoked in evoked_list:
            oz_idx = evoked.ch_names.index('Oz')
            all_data.append(evoked.data[oz_idx, :] * 1e6)
        return np.mean(all_data, axis=0), np.std(all_data, axis=0), evoked_list[0].times * 1000
    
    reg_mean, reg_std, times = compute_grand_avg(evokeds['regular'])
    rand_mean, rand_std, _ = compute_grand_avg(evokeds['random'])
    
    # Plot with confidence bands
    ax.plot(times, reg_mean, 'b-', linewidth=4, label='SYMMETRIC patterns', alpha=0.9)
    ax.fill_between(times, reg_mean - reg_std, reg_mean + reg_std, alpha=0.2, color='blue')
    
    ax.plot(times, rand_mean, 'r-', linewidth=4, label='RANDOM patterns', alpha=0.9)
    ax.fill_between(times, rand_mean - rand_std, rand_mean + rand_std, alpha=0.2, color='red')
    
    # Highlight key window
    ax.axvspan(300, 700, alpha=0.08, color='green', label='Key time window (300-700ms)')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Calculate mean difference in window
    window_mask = (times >= 300) & (times <= 700)
    mean_diff = np.mean(reg_mean[window_mask] - rand_mean[window_mask])
    
    # Add clear verdict - positioned to not overlap with plot
    verdict_color = 'green' if mean_diff > 0 else 'red'
    verdict_text = 'YES! Brain prefers SYMMETRIC patterns' if mean_diff > 0 else 'NO! Brain prefers RANDOM patterns'
    
    # Place verdict at top with better contrast
    ax.text(0.5, 1.12, verdict_text, 
            transform=ax.transAxes, fontsize=18, fontweight='bold',
            ha='center', va='top', color='white',
            bbox=dict(boxstyle='round,pad=1', facecolor=verdict_color, alpha=0.95, edgecolor='black', linewidth=3))
    
    # Place explanation below verdict
    ax.text(0.5, 1.04, f'Average difference: {mean_diff:.2f} μV\n(Blue line higher = prefer symmetry)', 
            transform=ax.transAxes, fontsize=11,
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95, edgecolor='black', linewidth=1.5))
    
    ax.set_xlabel('Time after stimulus (milliseconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Brain Response (μV)', fontsize=14, fontweight='bold')
    ax.set_title('Brain Activity: Symmetric vs Random Patterns (24 Subjects Average)', 
                 fontsize=16, fontweight='bold', pad=80)  # More padding for verdict boxes
    ax.legend(fontsize=12, loc='lower right', framealpha=0.95, edgecolor='black', fancybox=True)
    ax.grid(alpha=0.3)
    ax.set_xlim([-200, 1000])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for verdict boxes
    plt.savefig(QC_DIR / 'easy_viz1_simple_comparison.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'easy_viz1_simple_comparison.png'}")
    print(f"ANSWER: {verdict_text}")
    print(f"Effect size: {mean_diff:.2f} μV\n")
    plt.close()

def easy_viz2_who_shows_effect(evokeds):
    """
    EASY VIZ 2: Subject scoreboard - Who shows the effect?
    ONE MESSAGE: How many subjects show symmetry preference?
    """
    print("="*80)
    print("EASY VIZ 2: Subject-by-Subject Scoreboard")
    print("="*80)
    
    # Calculate effect for each subject
    effects = []
    for i, (reg, rand) in enumerate(zip(evokeds['regular'], evokeds['random'])):
        oz_idx = reg.ch_names.index('Oz')
        times = reg.times * 1000
        
        window_mask = (times >= 300) & (times <= 700)
        reg_window = reg.data[oz_idx, window_mask] * 1e6
        rand_window = rand.data[oz_idx, window_mask] * 1e6
        
        mean_effect = np.mean(reg_window - rand_window)
        effects.append({
            'subject': SUBJECTS[i],
            'effect': mean_effect,
            'prefers': 'SYMMETRIC' if mean_effect > 0 else 'RANDOM'
        })
    
    df = pd.DataFrame(effects).sort_values('effect', ascending=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    
    # Color code: green for positive (prefer symmetry), red for negative
    colors = ['green' if x > 0 else 'red' for x in df['effect']]
    
    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['effect'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['subject'], fontsize=10)
    ax.set_xlabel('Effect Size (μV)', fontsize=13, fontweight='bold')
    ax.set_title('Subject Scoreboard: Who Prefers Symmetry?', fontsize=15, fontweight='bold', pad=15)
    ax.axvline(0, color='black', linewidth=2)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars - positioned to not overlap
    for i, (idx, row) in enumerate(df.iterrows()):
        # Place labels with better offset
        if abs(row['effect']) < 0.5:
            # For small bars, place outside
            label_x = row['effect'] + (0.3 if row['effect'] > 0 else -0.3)
            ha = 'left' if row['effect'] > 0 else 'right'
        else:
            # For larger bars, place outside end
            label_x = row['effect'] + (0.2 if row['effect'] > 0 else -0.2)
            ha = 'left' if row['effect'] > 0 else 'right'
        ax.text(label_x, i, f"{row['effect']:.2f}", va='center', ha=ha, 
                fontsize=8, fontweight='bold')
    
    # Summary box - positioned in BOTTOM right, clear of all subjects
    n_positive = (df['effect'] > 0).sum()
    n_negative = (df['effect'] < 0).sum()
    
    summary = f"""
  SUMMARY:
  
  {n_positive} subjects prefer SYMMETRIC
  {n_negative} subjects prefer RANDOM
  
  Green = prefer symmetry
  Red = prefer randomness
  
  Longer = stronger
    """
    
    ax.text(0.97, 0.05, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.98, edgecolor='black', linewidth=2.5),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'easy_viz2_who_shows_effect.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'easy_viz2_who_shows_effect.png'}")
    print(f"Result: {n_positive}/24 subjects prefer symmetry\n")
    plt.close()
    
    return df

def easy_viz3_when_does_it_happen(evokeds):
    """
    EASY VIZ 3: Timeline - When does the brain "notice" symmetry?
    ONE MESSAGE: At what moment does the difference appear?
    """
    print("="*80)
    print("EASY VIZ 3: Timeline - When Does Brain Notice Symmetry?")
    print("="*80)
    
    # Compute grand average difference
    def compute_grand_avg(evoked_list):
        all_data = []
        for evoked in evoked_list:
            oz_idx = evoked.ch_names.index('Oz')
            all_data.append(evoked.data[oz_idx, :] * 1e6)
        return np.mean(all_data, axis=0), evoked_list[0].times * 1000
    
    reg_mean, times = compute_grand_avg(evokeds['regular'])
    rand_mean, _ = compute_grand_avg(evokeds['random'])
    diff = reg_mean - rand_mean
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    # TOP: Difference wave with timeline markers
    axes[0].plot(times, diff, 'purple', linewidth=4)
    axes[0].fill_between(times, 0, diff, where=(diff > 0), alpha=0.3, color='green', label='Prefers Symmetry')
    axes[0].fill_between(times, 0, diff, where=(diff < 0), alpha=0.3, color='red', label='Prefers Random')
    axes[0].axhline(0, color='black', linestyle='-', linewidth=2)
    axes[0].axvline(0, color='blue', linestyle='--', linewidth=2, label='Stimulus shown', alpha=0.7)
    
    # Find key moments
    # 1. First significant positive deflection
    first_pos_idx = np.where(diff > 0.5)[0]
    if len(first_pos_idx) > 0:
        first_pos_time = times[first_pos_idx[0]]
        axes[0].axvline(first_pos_time, color='green', linestyle='--', linewidth=2, alpha=0.7)
        axes[0].text(first_pos_time, axes[0].get_ylim()[1] * 0.9, 
                    f'First response\n{first_pos_time:.0f}ms', 
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 2. Peak difference
    window_mask = (times >= 300) & (times <= 700)
    peak_idx = window_mask.nonzero()[0][np.argmax(diff[window_mask])]
    peak_time = times[peak_idx]
    peak_amp = diff[peak_idx]
    
    axes[0].plot(peak_time, peak_amp, 'ro', markersize=15, zorder=10)
    axes[0].text(peak_time, peak_amp + 0.5, 
                f'PEAK\n{peak_time:.0f}ms\n{peak_amp:.2f}μV', 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2))
    
    axes[0].set_ylabel('Difference in Brain Response (μV)', fontsize=13, fontweight='bold')
    axes[0].set_title('Timeline: When Does the Brain Notice Symmetry?', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=11, loc='lower left')
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([-200, 1000])
    
    # BOTTOM: Timeline with phases
    axes[1].set_ylim([0, 1])
    axes[1].set_xlim([-200, 1000])
    
    # Phase boxes with better text visibility
    phases = [
        {'start': -200, 'end': 0, 'color': 'lightgray', 'label': 'Before\nstimulus'},
        {'start': 0, 'end': 200, 'color': 'lightblue', 'label': 'Early\nprocessing'},
        {'start': 200, 'end': 400, 'color': 'lightcoral', 'label': 'Pattern\ndetection'},
        {'start': 400, 'end': 700, 'color': 'lightgreen', 'label': 'Symmetry\nevaluation'},
        {'start': 700, 'end': 1000, 'color': 'lightyellow', 'label': 'Late\nprocessing'}
    ]
    
    for phase in phases:
        axes[1].axvspan(phase['start'], phase['end'], alpha=0.6, color=phase['color'], edgecolor='black', linewidth=1.5)
        mid_point = (phase['start'] + phase['end']) / 2
        # Add white background box for better text visibility
        axes[1].text(mid_point, 0.5, phase['label'], ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1))
    
    axes[1].set_xlabel('Time after stimulus (milliseconds)', fontsize=13, fontweight='bold')
    axes[1].set_yticks([])
    axes[1].set_title('Brain Processing Phases', fontsize=13, fontweight='bold')
    axes[1].axvline(0, color='blue', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'easy_viz3_when_does_it_happen.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'easy_viz3_when_does_it_happen.png'}")
    print(f"Peak response at {peak_time:.0f}ms with {peak_amp:.2f}μV difference\n")
    plt.close()

def easy_viz4_how_strong(df_effects):
    """
    EASY VIZ 4: Strength meter - How strong is the effect?
    ONE MESSAGE: Weak, medium, or strong effect?
    """
    print("="*80)
    print("EASY VIZ 4: Effect Strength Meter")
    print("="*80)
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])
    
    # TOP LEFT: Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    
    effects = df_effects['effect'].values
    
    # Create histogram
    n, bins, patches = ax1.hist(effects, bins=15, edgecolor='black', linewidth=1.5, alpha=0.7)
    
    # Color code bins
    for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
        if bin_val > 0:
            patch.set_facecolor('green')
        else:
            patch.set_facecolor('red')
    
    # Add mean line
    mean_val = np.mean(effects)
    ax1.axvline(mean_val, color='blue', linestyle='--', linewidth=3, label=f'Average: {mean_val:.2f}μV')
    ax1.axvline(0, color='black', linestyle='-', linewidth=2)
    
    ax1.set_xlabel('Effect Size (μV)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Effect Sizes', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # TOP RIGHT: Strength meter gauge
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    # Calculate statistics
    t_stat, p_val = stats.ttest_1samp(effects, 0)
    cohens_d = mean_val / np.std(effects)
    
    # Determine strength category
    if abs(cohens_d) < 0.2:
        strength = "VERY WEAK"
        strength_color = 'red'
    elif abs(cohens_d) < 0.5:
        strength = "WEAK"
        strength_color = 'orange'
    elif abs(cohens_d) < 0.8:
        strength = "MEDIUM"
        strength_color = 'gold'
    else:
        strength = "STRONG"
        strength_color = 'green'
    
    # Significance
    if p_val < 0.001:
        sig_text = "HIGHLY SIGNIFICANT ✓✓✓"
        sig_color = 'darkgreen'
    elif p_val < 0.01:
        sig_text = "VERY SIGNIFICANT ✓✓"
        sig_color = 'green'
    elif p_val < 0.05:
        sig_text = "SIGNIFICANT ✓"
        sig_color = 'lightgreen'
    else:
        sig_text = "NOT SIGNIFICANT ✗"
        sig_color = 'red'
    
    gauge_text = f"""
EFFECT STRENGTH

{'='*28}

Strength: {strength}

Significance:
{sig_text}

{'='*28}

TECHNICAL:

Mean: {mean_val:.3f} μV
Cohen's d: {cohens_d:.3f}
t-stat: {t_stat:.3f}
p-value: {p_val:.6f}
n = {len(effects)} subjects

{'='*28}

INTERPRETATION:
Cohen's d scale:
0.2 = small
0.5 = medium
0.8 = large

Your effect:
{cohens_d:.2f} = {strength}
    """
    
    ax2.text(0.5, 0.5, gauge_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=1', facecolor=strength_color, alpha=0.3, 
                     edgecolor='black', linewidth=3),
            family='monospace')
    
    # BOTTOM: Simple bar comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    categories = ['Positive\nEffect', 'Negative\nEffect', 'Overall\nStrength']
    values = [
        (effects > 0).sum(),
        (effects < 0).sum(),
        abs(mean_val) * 10  # Scale for visualization
    ]
    colors_bar = ['green', 'red', strength_color]
    
    bars = ax3.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels with better spacing
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if val == values[2]:  # Strength bar
            label = f"{abs(mean_val):.2f}μV"
        else:
            label = f"{int(val)} subjects"
        # Place label above bar with margin
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.03,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'))
    
    ax3.set_ylabel('Count / Scaled Value', fontsize=12, fontweight='bold')
    ax3.set_title('Summary Statistics', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(QC_DIR / 'easy_viz4_how_strong.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'easy_viz4_how_strong.png'}")
    print(f"Effect strength: {strength} (Cohen's d = {cohens_d:.3f})")
    print(f"Significance: {sig_text} (p = {p_val:.6f})\n")
    plt.close()

def easy_viz5_brain_map(evokeds):
    """
    EASY VIZ 5: Brain topography - Where in the brain?
    ONE MESSAGE: Which brain regions show the effect?
    """
    print("="*80)
    print("EASY VIZ 5: Brain Topography Map")
    print("="*80)
    
    # Compute grand averages for EEG channels only
    def compute_grand_avg_eeg_only(evoked_list):
        # First pick EEG from template, explicitly drop EXG channels
        template = evoked_list[0].copy()
        exg_channels = [ch for ch in template.ch_names if 'EXG' in ch]
        if exg_channels:
            template = template.drop_channels(exg_channels)
        
        all_data = []
        for evoked in evoked_list:
            evoked_clean = evoked.copy()
            exg_ch = [ch for ch in evoked_clean.ch_names if 'EXG' in ch]
            if exg_ch:
                evoked_clean = evoked_clean.drop_channels(exg_ch)
            all_data.append(evoked_clean.data * 1e6)
        return np.mean(all_data, axis=0), template
    
    reg_mean, template_evoked = compute_grand_avg_eeg_only(evokeds['regular'])
    rand_mean, _ = compute_grand_avg_eeg_only(evokeds['random'])
    
    # Create difference evoked object (already EEG only)
    diff_evoked = template_evoked.copy()
    diff_evoked.data = (reg_mean - rand_mean) * 1e-6  # Back to V
    
    # Create figure with time points
    times_to_plot = [0, 200, 400, 600, 800]
    
    fig, axes = plt.subplots(1, len(times_to_plot), figsize=(18, 4))
    
    # Calculate mean difference in analysis window
    times_ms = diff_evoked.times * 1000
    analysis_mask = (times_ms >= 300) & (times_ms <= 700)
    mean_effect = np.mean(diff_evoked.data[:, analysis_mask] * 1e6)
    
    # Determine what the data actually shows
    if mean_effect > 0:
        title_text = 'Brain Activity Map: Symmetric Patterns Preferred (Red regions)'
    else:
        title_text = 'Brain Activity Map: Random Patterns Preferred (Blue regions)'
    
    fig.suptitle(title_text, fontsize=15, fontweight='bold', y=1.05)
    
    # Set color scale to match actual data range for better visibility
    # Using symmetric scale around zero
    vlim = (-2.5, 2.5)  # Adjusted to show actual effects better
    
    for i, t in enumerate(times_to_plot):
        time_idx = np.argmin(np.abs(diff_evoked.times - t/1000))
        
        # Plot topography
        im, _ = mne.viz.plot_topomap(
            diff_evoked.data[:, time_idx] * 1e6,
            diff_evoked.info,
            axes=axes[i],
            show=False,
            vlim=vlim,
            cmap='RdBu_r',
            contours=6
        )
        
        # Add time label and value
        diff_val = np.mean(diff_evoked.data[:, time_idx] * 1e6)
        axes[i].set_title(f'{t}ms\n({diff_val:+.1f}μV)', fontsize=12, fontweight='bold')
    
    # Add colorbar BELOW the visualization
    cbar = plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.15, shrink=0.6, aspect=30)
    cbar.set_label('Difference (Symmetric - Random) in μV', fontsize=12, fontweight='bold')
    
    # Add legend with correct interpretation
    legend_text = 'Red = Brain prefers Symmetric | Blue = Brain prefers Random'
    if mean_effect < 0:
        legend_text = 'Blue = Brain prefers Random (OPPOSITE of expected!) | Red = Brain prefers Symmetric'
    
    fig.text(0.5, -0.02, legend_text, 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow' if mean_effect < 0 else 'white', 
                     alpha=0.8, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for colorbar
    plt.savefig(QC_DIR / 'easy_viz5_brain_map.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'easy_viz5_brain_map.png'}")
    print("Shows which brain regions respond differently to symmetric vs random\n")
    plt.close()

def easy_viz6_reliability_check(evokeds):
    """
    EASY VIZ 6: Reliability - Is this effect trustworthy?
    ONE MESSAGE: Can we trust these results?
    """
    print("="*80)
    print("EASY VIZ 6: Reliability Check")
    print("="*80)
    
    # Calculate effect for each subject
    effects = []
    snrs = []
    
    df_metrics = pd.read_csv(OUTPUT_DIR / 'preprocessing_metrics_all_subjects.csv')
    
    for i, (reg, rand) in enumerate(zip(evokeds['regular'], evokeds['random'])):
        oz_idx = reg.ch_names.index('Oz')
        times = reg.times * 1000
        
        window_mask = (times >= 300) & (times <= 700)
        reg_window = reg.data[oz_idx, window_mask] * 1e6
        rand_window = rand.data[oz_idx, window_mask] * 1e6
        
        effect = np.mean(reg_window - rand_window)
        effects.append(effect)
        
        # Get SNR for this subject
        snr = df_metrics[df_metrics['subject'] == SUBJECTS[i]]['ours_snr_estimate'].values[0]
        snrs.append(snr)
    
    effects = np.array(effects)
    snrs = np.array(snrs)
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # TOP LEFT: Effect consistency across subjects
    ax1 = fig.add_subplot(gs[0, 0])
    
    x_pos = np.arange(len(effects))
    colors = ['green' if e > 0 else 'red' for e in effects]
    ax1.bar(x_pos, effects, color=colors, alpha=0.6, edgecolor='black')
    ax1.axhline(0, color='black', linewidth=2)
    ax1.axhline(np.mean(effects), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(effects):.2f}μV')
    
    # Add confidence interval
    ci_lower, ci_upper = stats.t.interval(0.95, len(effects)-1, 
                                          loc=np.mean(effects), 
                                          scale=stats.sem(effects))
    ax1.axhspan(ci_lower, ci_upper, alpha=0.2, color='blue', label='95% Confidence')
    
    ax1.set_xlabel('Subject Number', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Effect Size (μV)', fontsize=11, fontweight='bold')
    ax1.set_title('Consistency: Effect in Each Subject', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # TOP RIGHT: Variability
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Box plot
    bp = ax2.boxplot([effects], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=3),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2))
    
    # Add individual points
    ax2.scatter([1]*len(effects), effects, alpha=0.4, s=100, color='darkblue', zorder=10)
    ax2.axhline(0, color='black', linestyle='--', linewidth=2)
    
    ax2.set_ylabel('Effect Size (μV)', fontsize=11, fontweight='bold')
    ax2.set_title('Variability Across Subjects', fontsize=12, fontweight='bold')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['All Subjects'])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
    Mean: {np.mean(effects):.2f}
    Median: {np.median(effects):.2f}
    SD: {np.std(effects):.2f}
    Range: {np.min(effects):.2f} to {np.max(effects):.2f}
    """
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    # BOTTOM LEFT: Correlation with data quality
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.scatter(snrs, effects, s=100, alpha=0.6, c=effects, cmap='RdYlGn', 
               edgecolors='black', linewidth=1)
    
    # Fit line
    z = np.polyfit(snrs, effects, 1)
    p = np.poly1d(z)
    x_line = np.linspace(snrs.min(), snrs.max(), 100)
    ax3.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7)
    
    corr = np.corrcoef(snrs, effects)[0, 1]
    ax3.set_xlabel('Data Quality (SNR)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Effect Size (μV)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Effect vs Data Quality (r={corr:.3f})', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # BOTTOM RIGHT: Reliability verdict
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate reliability metrics
    t_stat, p_val = stats.ttest_1samp(effects, 0)
    n_positive = (effects > 0).sum()
    percentage_positive = (n_positive / len(effects)) * 100
    
    # Determine reliability
    reliability_score = 0
    if p_val < 0.001:
        reliability_score += 3
    elif p_val < 0.01:
        reliability_score += 2
    elif p_val < 0.05:
        reliability_score += 1
    
    if percentage_positive > 80:
        reliability_score += 2
    elif percentage_positive > 70:
        reliability_score += 1
    
    if np.std(effects) < 2:
        reliability_score += 1
    
    # Verdict
    if reliability_score >= 5:
        verdict = "HIGHLY RELIABLE ✓✓✓"
        verdict_color = 'darkgreen'
        trust = "You can TRUST these results"
    elif reliability_score >= 3:
        verdict = "RELIABLE ✓✓"
        verdict_color = 'green'
        trust = "Results are trustworthy"
    elif reliability_score >= 2:
        verdict = "MODERATELY RELIABLE ✓"
        verdict_color = 'orange'
        trust = "Results are somewhat trustworthy"
    else:
        verdict = "NOT RELIABLE ✗"
        verdict_color = 'red'
        trust = "Results need more investigation"
    
    verdict_text = f"""
    RELIABILITY VERDICT
    
    {'='*40}
    
    {verdict}
    
    {trust}
    
    {'='*40}
    
    EVIDENCE:
    
    ✓ Statistical significance: p={p_val:.6f}
      {'Very strong' if p_val < 0.001 else 'Strong' if p_val < 0.01 else 'Moderate'}
    
    ✓ Consistency: {n_positive}/{len(effects)} subjects ({percentage_positive:.0f}%)
      show positive effect
    
    ✓ Variability: SD = {np.std(effects):.2f} μV
      {'Low (good)' if np.std(effects) < 2 else 'Moderate' if np.std(effects) < 3 else 'High'}
    
    ✓ Quality correlation: r = {corr:.3f}
      {'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'}
    
    {'='*40}
    
    RELIABILITY SCORE: {reliability_score}/6
    """
    
    ax4.text(0.5, 0.5, verdict_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=verdict_color, alpha=0.3,
                     edgecolor=verdict_color, linewidth=4),
            family='monospace')
    
    plt.suptitle('Reliability Analysis: Can We Trust These Results?', 
                fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(QC_DIR / 'easy_viz6_reliability_check.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {QC_DIR / 'easy_viz6_reliability_check.png'}")
    print(f"Reliability: {verdict}")
    print(f"Verdict: {trust}\n")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("EASY-TO-UNDERSTAND VISUALIZATIONS")
    print("Clear, Simple, Meaningful Insights from 24 Subjects")
    print("="*80 + "\n")
    
    # Load all data
    evokeds = load_all_data()
    
    # Generate visualizations
    print("\nGenerating 6 easy-to-understand visualizations...\n")
    
    easy_viz1_simple_comparison(evokeds)
    df_effects = easy_viz2_who_shows_effect(evokeds)
    easy_viz3_when_does_it_happen(evokeds)
    easy_viz4_how_strong(df_effects)
    easy_viz5_brain_map(evokeds)
    easy_viz6_reliability_check(evokeds)
    
    print("\n" + "="*80)
    print("ALL EASY VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nGenerated 6 easy-to-understand figures in: {QC_DIR}")
    print("\nWhat each figure answers:")
    print("  1. Do people prefer symmetry? → YES/NO answer")
    print("  2. How many subjects show the effect? → Subject scoreboard")
    print("  3. When does it happen? → Timeline with phases")
    print("  4. How strong is it? → Strength meter with verdict")
    print("  5. Where in the brain? → Topography maps over time")
    print("  6. Can we trust it? → Reliability check with score")
    print("\nThese are the EASIEST figures to understand and present!")
    print("="*80 + "\n")
