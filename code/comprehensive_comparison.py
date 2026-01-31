"""
Comprehensive Pipeline Comparison - Single Big Visualization
Shows ALL differences between Authors' and Our pipeline in one figure
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
plt.rcParams['font.size'] = 9

def load_all_data():
    """Load all data from both pipelines"""
    print("Loading data from both pipelines (24 subjects)...")
    
    data = {
        'authors_regular': [],
        'authors_random': [],
        'ours_regular': [],
        'ours_random': []
    }
    
    for subject in SUBJECTS:
        try:
            evoked_a = mne.read_evokeds(OUTPUT_DIR / f'{subject}_authors_ave.fif', verbose=False)
            data['authors_regular'].append(evoked_a[0])
            data['authors_random'].append(evoked_a[1])
            
            evoked_o = mne.read_evokeds(OUTPUT_DIR / f'{subject}_ours_ave.fif', verbose=False)
            data['ours_regular'].append(evoked_o[0])
            data['ours_random'].append(evoked_o[1])
        except Exception as e:
            print(f"Error loading {subject}: {e}")
    
    print(f"Loaded {len(data['ours_regular'])} subjects\n")
    return data

def create_comprehensive_comparison():
    """
    Create ONE BIG visualization showing everything
    """
    print("="*80)
    print("CREATING COMPREHENSIVE PIPELINE COMPARISON")
    print("="*80 + "\n")
    
    # Load data
    data = load_all_data()
    df_metrics = pd.read_csv(OUTPUT_DIR / 'preprocessing_metrics_all_subjects.csv')
    
    # Create figure with custom grid
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)
    
    # ========================================================================
    # PANEL 1: Pipeline Steps (Top Left - spans 2 columns)
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 11)
    ax1.set_title('PIPELINE DIFFERENCES', fontsize=14, fontweight='bold', pad=10)
    
    steps_comparison = [
        ("Step", "Authors", "Ours", "Impact"),
        ("Filter", "0.1-30 Hz", "0.1-40 Hz", "HIGH"),
        ("Bad Channels", "Manual", "Automated", "HIGH"),
        ("ICA", "Manual", "Automated", "MEDIUM"),
        ("Epoch Rejection", "Threshold", "Autoreject", "HIGH"),
    ]
    
    y_pos = 9.5
    # Header
    ax1.text(1, y_pos, steps_comparison[0][0], fontsize=11, fontweight='bold', ha='left')
    ax1.text(3, y_pos, steps_comparison[0][1], fontsize=11, fontweight='bold', ha='center')
    ax1.text(5.5, y_pos, steps_comparison[0][2], fontsize=11, fontweight='bold', ha='center')
    ax1.text(8, y_pos, steps_comparison[0][3], fontsize=11, fontweight='bold', ha='center')
    
    y_pos -= 0.8
    ax1.plot([0.5, 9.5], [y_pos+0.3, y_pos+0.3], 'k-', linewidth=2)
    
    # Rows
    for i, (step, auth, ours, impact) in enumerate(steps_comparison[1:], 1):
        y_pos -= 1.5
        
        # Step name
        ax1.text(1, y_pos, step, fontsize=10, fontweight='bold', ha='left')
        
        # Authors box
        rect1 = plt.Rectangle((2.2, y_pos-0.4), 1.6, 0.8, facecolor='lightblue', 
                              edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect1)
        ax1.text(3, y_pos, auth, fontsize=9, ha='center', va='center')
        
        # Arrow
        ax1.arrow(4, y_pos, 1, 0, head_width=0.2, head_length=0.2, fc='gray', ec='gray', linewidth=2)
        
        # Ours box
        rect2 = plt.Rectangle((5.2, y_pos-0.4), 1.6, 0.8, facecolor='lightcoral', 
                              edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect2)
        ax1.text(6, y_pos, ours, fontsize=9, ha='center', va='center')
        
        # Impact
        impact_color = 'red' if impact == 'HIGH' else 'orange'
        circle = plt.Circle((8, y_pos), 0.3, color=impact_color, alpha=0.7)
        ax1.add_patch(circle)
        ax1.text(8, y_pos, impact[0], fontsize=10, fontweight='bold', ha='center', va='center', color='white')
    
    # ========================================================================
    # PANEL 2: Key Statistics (Top Right - spans 2 columns)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.axis('off')
    ax2.set_title('KEY STATISTICS', fontsize=14, fontweight='bold', pad=10)
    
    # Calculate key stats
    authors_ica = df_metrics['authors_ica_excluded'].mean()
    ours_ica = df_metrics['ours_ica_excluded'].mean()
    authors_snr = df_metrics['authors_snr_estimate'].mean()
    ours_snr = df_metrics['ours_snr_estimate'].mean()
    authors_erp = df_metrics['authors_erp_peak_amplitude'].mean()
    ours_erp = df_metrics['ours_erp_peak_amplitude'].mean()
    
    stats_text = f"""
    METRIC                    AUTHORS        OURS         CHANGE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    ICA Components Removed    {authors_ica:>6.1f}     {ours_ica:>6.1f}     {ours_ica-authors_ica:+6.1f}
    
    SNR (dB)                  {authors_snr:>6.2f}     {ours_snr:>6.2f}     {ours_snr-authors_snr:+6.2f}
    
    ERP Peak (μV)             {authors_erp:>6.2f}     {ours_erp:>6.2f}     {ours_erp-authors_erp:+6.2f}
    
    Bad Channels (avg)          N/A       {df_metrics['ours_bad_channels'].mean():>6.1f}        NEW
    """
    
    ax2.text(0.5, 0.5, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='center', ha='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                     edgecolor='black', linewidth=3))
    
    # ========================================================================
    # PANEL 3: ERP Grand Averages (Row 2, Left 2 columns)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, :2])
    
    def compute_grand_avg(evoked_list):
        all_data = []
        for evoked in evoked_list:
            oz_idx = evoked.ch_names.index('Oz')
            all_data.append(evoked.data[oz_idx, :] * 1e6)
        return np.mean(all_data, axis=0), evoked_list[0].times * 1000
    
    authors_reg, times_a = compute_grand_avg(data['authors_regular'])
    authors_rand, _ = compute_grand_avg(data['authors_random'])
    ours_reg, times_o = compute_grand_avg(data['ours_regular'])
    ours_rand, _ = compute_grand_avg(data['ours_random'])
    
    ax3.plot(times_a, authors_reg, 'b-', linewidth=2.5, label='Authors: Regular', alpha=0.7)
    ax3.plot(times_a, authors_rand, 'r-', linewidth=2.5, label='Authors: Random', alpha=0.7)
    ax3.plot(times_o, ours_reg, 'b--', linewidth=2.5, label='Ours: Regular', alpha=0.7)
    ax3.plot(times_o, ours_rand, 'r--', linewidth=2.5, label='Ours: Random', alpha=0.7)
    
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(0, color='black', linestyle='--', linewidth=1)
    ax3.axvspan(300, 700, alpha=0.15, color='green', label='Analysis Window')
    
    ax3.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Amplitude (μV)', fontsize=11, fontweight='bold')
    ax3.set_title('GRAND AVERAGE ERPs AT Oz', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, ncol=3)
    ax3.grid(alpha=0.3)
    ax3.set_xlim([-200, 1000])
    
    # ========================================================================
    # PANEL 4: Difference Waves (Row 2, Right 2 columns)
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 2:])
    
    authors_diff = authors_reg - authors_rand
    ours_diff = ours_reg - ours_rand
    
    ax4.plot(times_a, authors_diff, 'purple', linewidth=3, label='Authors: Reg-Rand', alpha=0.8)
    ax4.plot(times_o, ours_diff, 'orange', linewidth=3, label='Ours: Reg-Rand', alpha=0.8)
    ax4.fill_between(times_a, authors_diff, alpha=0.2, color='purple')
    ax4.fill_between(times_o, ours_diff, alpha=0.2, color='orange')
    
    ax4.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax4.axvline(0, color='black', linestyle='--', linewidth=1)
    ax4.axvspan(300, 700, alpha=0.15, color='green')
    
    # Mark peaks
    window_mask_a = (times_a >= 300) & (times_a <= 700)
    window_mask_o = (times_o >= 300) & (times_o <= 700)
    authors_peak = np.mean(authors_diff[window_mask_a])
    ours_peak = np.mean(ours_diff[window_mask_o])
    
    ax4.set_xlabel('Time (ms)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Amplitude (μV)', fontsize=11, fontweight='bold')
    ax4.set_title('DIFFERENCE WAVES (Regular - Random)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)
    ax4.set_xlim([-200, 1000])
    
    # ========================================================================
    # PANEL 5: Time-Course Comparison (Row 3, spans all 4 columns)
    # ========================================================================
    # Show how the effect evolves over time for both pipelines
    
    # Calculate sliding window averages
    window_size = 100  # ms
    window_step = 20   # ms
    window_centers = np.arange(0, 1000, window_step)
    
    authors_time_course = []
    ours_time_course = []
    
    for t_center in window_centers:
        t_start = t_center - window_size/2
        t_end = t_center + window_size/2
        
        # Authors
        mask_a = (times_a >= t_start) & (times_a <= t_end)
        if np.any(mask_a):
            authors_time_course.append(np.mean(authors_diff[mask_a]))
        else:
            authors_time_course.append(np.nan)
        
        # Ours
        mask_o = (times_o >= t_start) & (times_o <= t_end)
        if np.any(mask_o):
            ours_time_course.append(np.mean(ours_diff[mask_o]))
        else:
            ours_time_course.append(np.nan)
    
    authors_time_course = np.array(authors_time_course)
    ours_time_course = np.array(ours_time_course)
    
    ax5 = fig.add_subplot(gs[2, :])
    
    # Plot time courses
    ax5.plot(window_centers, authors_time_course, 'purple', linewidth=3, 
            label="Authors' Pipeline", alpha=0.8, marker='o', markersize=4)
    ax5.plot(window_centers, ours_time_course, 'orange', linewidth=3, 
            label='Our Pipeline', alpha=0.8, marker='s', markersize=4)
    
    # Highlight analysis window
    ax5.axvspan(300, 700, alpha=0.15, color='green', label='Analysis Window (300-700ms)')
    ax5.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax5.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # Add shading for standard regions
    ax5.axvspan(-200, 0, alpha=0.1, color='gray')
    ax5.text(-100, ax5.get_ylim()[1]*0.9, 'Baseline', ha='center', fontsize=10, 
            fontweight='bold', alpha=0.7)
    
    ax5.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Effect Size (μV)\n100ms sliding window', fontsize=12, fontweight='bold')
    ax5.set_title('HOW EFFECT EVOLVES OVER TIME (Both Pipelines)', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=11, loc='upper right')
    ax5.grid(alpha=0.3)
    ax5.set_xlim([-200, 1000])
    
    # ========================================================================
    # PANEL 6: Correlation Analysis (Row 4, Left)
    # ========================================================================
    ax6 = fig.add_subplot(gs[3, :2])
    
    # Get effect sizes
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
    
    ax6.scatter(authors_effects, ours_effects, s=120, alpha=0.6, 
               c=range(len(authors_effects)), cmap='viridis', 
               edgecolors='black', linewidth=1.5)
    
    # Fit line
    z = np.polyfit(authors_effects, ours_effects, 1)
    p = np.poly1d(z)
    x_line = np.linspace(authors_effects.min(), authors_effects.max(), 100)
    ax6.plot(x_line, p(x_line), "r--", linewidth=3, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Identity line
    min_val = min(authors_effects.min(), ours_effects.min())
    max_val = max(authors_effects.max(), ours_effects.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
            alpha=0.5, label='Identity')
    
    corr = np.corrcoef(authors_effects, ours_effects)[0, 1]
    
    ax6.set_xlabel("Authors' Effect (μV)", fontsize=11, fontweight='bold')
    ax6.set_ylabel("Ours Effect (μV)", fontsize=11, fontweight='bold')
    ax6.set_title(f'PIPELINE CORRELATION (r={corr:.3f})', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3)
    ax6.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax6.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    
    # ========================================================================
    # PANEL 7: Subject-by-Subject Agreement (Row 4, Right)
    # ========================================================================
    ax7 = fig.add_subplot(gs[3, 2:])
    
    subject_labels = [f'{i}' for i in range(1, 25)]
    x_pos = np.arange(len(subject_labels))
    
    differences = ours_effects - authors_effects
    colors = ['green' if d > 0 else 'red' for d in differences]
    
    bars = ax7.bar(x_pos, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax7.axhline(0, color='black', linestyle='-', linewidth=2)
    ax7.set_xlabel('Subject', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Difference (Ours - Authors) μV', fontsize=11, fontweight='bold')
    ax7.set_title('SUBJECT-BY-SUBJECT DIFFERENCES', fontsize=13, fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(subject_labels, fontsize=8)
    ax7.grid(axis='y', alpha=0.3)
    
    # ========================================================================
    # MAIN TITLE
    # ========================================================================
    fig.suptitle('COMPREHENSIVE PIPELINE COMPARISON: Authors vs Our Preprocessing',
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.95, bottom=0.05)
    
    output_path = QC_DIR / 'COMPREHENSIVE_PIPELINE_COMPARISON.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Calculate mean difference for reporting
    mean_diff = np.mean(differences)
    
    print(f"\nSaved: {output_path}")
    print(f"\nFigure size: 24x16 inches at 300 DPI")
    print(f"Correlation: r = {corr:.3f}")
    print(f"Mean difference: {mean_diff:+.2f} μV")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("COMPREHENSIVE PIPELINE COMPARISON")
    print("One visualization showing everything!")
    print("="*80 + "\n")
    
    create_comprehensive_comparison()
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("\nOne comprehensive figure showing:")
    print("  1. Pipeline step differences")
    print("  2. Key statistics comparison")
    print("  3. Grand average ERPs")
    print("  4. Difference waves")
    print("  5. Time-course evolution (easier to understand)")
    print("  6. Correlation analysis")
    print("  7. Subject-by-subject differences")
    print("\nEverything in ONE BIG visualization!")
    print("="*80 + "\n")
