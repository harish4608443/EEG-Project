"""
Batch Preprocessing Pipeline - Milestone 4
Run preprocessing for all 24 subjects and collect metrics
"""

import os
import mne
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Import preprocessing functions
import sys
sys.path.append(os.path.dirname(__file__))

# Configuration
BASE_DIR = Path(r'd:\ds004347')
SUBJECTS = [f'sub-{i:03d}' for i in range(1, 25)]  # sub-001 to sub-024
OUTPUT_DIR = BASE_DIR / 'derivatives' / 'preprocessing_results'
QC_DIR = BASE_DIR / 'derivatives' / 'quality_control'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

# Event codes
EVENT_ID = {'Regular': 1, 'Random': 3}

# Storage for metrics across all subjects
metrics_all = []

def preprocess_subject(subject_id):
    """
    Run preprocessing pipeline for a single subject
    Returns: dict of metrics for sanity checking
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {subject_id}")
    print(f"{'='*80}\n")
    
    # Paths
    raw_path = BASE_DIR / subject_id / 'eeg' / f'{subject_id}_task-jacobsen_eeg.bdf'
    
    if not raw_path.exists():
        print(f"WARNING: Data file not found for {subject_id}")
        return None
    
    metrics = {
        'subject': subject_id,
        'timestamp': datetime.now().isoformat(),
    }
    
    try:
        # Load data
        print(f"Loading {subject_id}...")
        raw = mne.io.read_raw_bdf(raw_path, preload=True, verbose=False)
        metrics['n_channels_original'] = len(raw.ch_names)
        metrics['sfreq'] = raw.info['sfreq']
        metrics['duration_sec'] = raw.times[-1]
        
        # Extract events BEFORE picking channels
        events = mne.find_events(raw, stim_channel='Status', verbose=False)
        task_events = events[np.isin(events[:, 2], list(EVENT_ID.values()))]
        metrics['n_events_total'] = len(task_events)
        metrics['n_events_regular'] = np.sum(task_events[:, 2] == EVENT_ID['Regular'])
        metrics['n_events_random'] = np.sum(task_events[:, 2] == EVENT_ID['Random'])
        
        # Pick EEG channels only
        raw.pick_types(eeg=True, exclude=[])
        
        # ====================================================================
        # AUTHORS' PIPELINE
        # ====================================================================
        print(f"\n--- Authors' Pipeline ---")
        raw_authors = raw.copy()
        
        # Filter 0.1-25 Hz
        raw_authors.filter(l_freq=0.1, h_freq=25.0, verbose=False)
        
        # Average reference
        raw_authors.set_eeg_reference('average', projection=False, verbose=False)
        
        # ICA
        raw_for_ica_a = raw_authors.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        ica_authors = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter=800)
        ica_authors.fit(raw_for_ica_a, verbose=False)
        ica_authors.exclude = []  # Manual inspection in original - we use none for baseline
        raw_authors_clean = raw_authors.copy()
        ica_authors.apply(raw_authors_clean, verbose=False)
        
        metrics['authors_ica_excluded'] = len(ica_authors.exclude)
        
        # Epoch
        epochs_authors = mne.Epochs(
            raw_authors_clean, task_events, event_id=EVENT_ID,
            tmin=-1.0, tmax=1.0, baseline=None, preload=True,
            reject=None, verbose=False
        )
        epochs_authors.apply_baseline(baseline=(-0.2, 0.05), verbose=False)
        
        metrics['authors_epochs_total'] = len(epochs_authors)
        metrics['authors_epochs_regular'] = len(epochs_authors['Regular'])
        metrics['authors_epochs_random'] = len(epochs_authors['Random'])
        
        # Calculate ERPs
        evoked_regular_authors = epochs_authors['Regular'].average()
        evoked_random_authors = epochs_authors['Random'].average()
        
        # ====================================================================
        # OUR PIPELINE
        # ====================================================================
        print(f"--- Our Pipeline ---")
        raw_ours = raw.copy()
        
        # Filter 0.1-40 Hz + notch 50 Hz
        raw_ours.filter(l_freq=0.1, h_freq=40.0, verbose=False)
        raw_ours.notch_filter(freqs=50, verbose=False)
        
        # Average reference
        raw_ours.set_eeg_reference('average', projection=False, verbose=False)
        
        # Bad channel detection
        epochs_temp = mne.Epochs(
            raw_ours, task_events, event_id=EVENT_ID,
            tmin=-0.2, tmax=1.0, baseline=None, preload=True,
            reject=None, verbose=False
        )
        channel_variances = np.var(epochs_temp.get_data(), axis=(0, 2))
        z_scores = np.abs((channel_variances - np.mean(channel_variances)) / np.std(channel_variances))
        bad_channels = [epochs_temp.ch_names[i] for i in np.where(z_scores > 3)[0]]
        
        raw_ours.info['bads'] = bad_channels
        metrics['ours_bad_channels'] = len(bad_channels)
        metrics['ours_bad_channels_list'] = bad_channels
        
        # Interpolate bad channels
        if bad_channels:
            try:
                montage = mne.channels.make_standard_montage('biosemi64')
                raw_ours.set_montage(montage, on_missing='ignore', verbose=False)
                raw_ours.interpolate_bads(reset_bads=True, verbose=False)
                metrics['ours_interpolation_success'] = True
            except Exception as e:
                metrics['ours_interpolation_success'] = False
                metrics['ours_interpolation_error'] = str(e)
        
        # ICA with automatic detection
        raw_for_ica_ours = raw_ours.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        ica_ours = mne.preprocessing.ICA(n_components=20, random_state=42, max_iter=800, method='fastica')
        ica_ours.fit(raw_for_ica_ours, verbose=False)
        
        # Automatic component exclusion
        component_vars = np.var(ica_ours.get_sources(raw_for_ica_ours).get_data(), axis=1)
        z_scores_ica = np.abs((component_vars - np.mean(component_vars)) / np.std(component_vars))
        ica_ours.exclude = [i for i in range(len(z_scores_ica)) if z_scores_ica[i] > 2.5][:5]
        
        raw_ours_clean = raw_ours.copy()
        ica_ours.apply(raw_ours_clean, verbose=False)
        
        metrics['ours_ica_excluded'] = len(ica_ours.exclude)
        metrics['ours_ica_excluded_list'] = ica_ours.exclude
        
        # Epoch
        epochs_ours = mne.Epochs(
            raw_ours_clean, task_events, event_id=EVENT_ID,
            tmin=-0.2, tmax=1.0, baseline=None, preload=True,
            reject=None, verbose=False
        )
        epochs_ours.apply_baseline(baseline=(-0.2, 0.0), verbose=False)
        
        metrics['ours_epochs_total'] = len(epochs_ours)
        metrics['ours_epochs_regular'] = len(epochs_ours['Regular'])
        metrics['ours_epochs_random'] = len(epochs_ours['Random'])
        
        # Calculate median ERPs
        regular_data = epochs_ours['Regular'].get_data()
        random_data = epochs_ours['Random'].get_data()
        
        evoked_regular_ours = epochs_ours['Regular'].average()
        evoked_regular_ours.data = np.median(regular_data, axis=0)
        
        evoked_random_ours = epochs_ours['Random'].average()
        evoked_random_ours.data = np.median(random_data, axis=0)
        
        # ====================================================================
        # SAVE RESULTS
        # ====================================================================
        print(f"Saving results for {subject_id}...")
        
        # Save epochs
        epochs_authors.save(OUTPUT_DIR / f'{subject_id}_authors_epo.fif', overwrite=True)
        epochs_ours.save(OUTPUT_DIR / f'{subject_id}_ours_epo.fif', overwrite=True)
        
        # Save evoked
        mne.write_evokeds(OUTPUT_DIR / f'{subject_id}_authors_ave.fif', 
                         [evoked_regular_authors, evoked_random_authors], overwrite=True)
        mne.write_evokeds(OUTPUT_DIR / f'{subject_id}_ours_ave.fif',
                         [evoked_regular_ours, evoked_random_ours], overwrite=True)
        
        # Save ICA
        ica_authors.save(OUTPUT_DIR / f'{subject_id}_authors_ica.fif', overwrite=True)
        ica_ours.save(OUTPUT_DIR / f'{subject_id}_ours_ica.fif', overwrite=True)
        
        # Calculate quality metrics
        metrics['authors_erp_peak_amplitude'] = np.max(np.abs(evoked_regular_authors.data))
        metrics['ours_erp_peak_amplitude'] = np.max(np.abs(evoked_regular_ours.data))
        
        # Signal-to-noise ratio estimate (peak / baseline std)
        baseline_std_a = np.std(evoked_regular_authors.copy().crop(tmin=-0.2, tmax=0).data)
        baseline_std_o = np.std(evoked_regular_ours.copy().crop(tmin=-0.2, tmax=0).data)
        metrics['authors_snr_estimate'] = metrics['authors_erp_peak_amplitude'] / baseline_std_a
        metrics['ours_snr_estimate'] = metrics['ours_erp_peak_amplitude'] / baseline_std_o
        
        metrics['processing_success'] = True
        print(f"✓ {subject_id} completed successfully")
        
    except Exception as e:
        print(f"✗ ERROR processing {subject_id}: {str(e)}")
        metrics['processing_success'] = False
        metrics['error_message'] = str(e)
        import traceback
        metrics['error_traceback'] = traceback.format_exc()
    
    return metrics


# ============================================================================
# MAIN BATCH PROCESSING
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("BATCH PREPROCESSING - MILESTONE 4")
    print("Processing all 24 subjects")
    print("="*80 + "\n")
    
    start_time = datetime.now()
    
    for subject_id in SUBJECTS:
        metrics = preprocess_subject(subject_id)
        if metrics:
            metrics_all.append(metrics)
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_all)
    metrics_path = OUTPUT_DIR / 'preprocessing_metrics_all_subjects.csv'
    metrics_df.to_csv(metrics_path, index=False)
    
    # Summary statistics
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total subjects: {len(SUBJECTS)}")
    print(f"Successful: {metrics_df['processing_success'].sum()}")
    print(f"Failed: {(~metrics_df['processing_success']).sum()}")
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Processing time: {datetime.now() - start_time}")
    print("="*80 + "\n")
