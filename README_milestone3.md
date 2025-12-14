# EEG Preprocessing Pipeline - Milestone 3

## Project Overview
This project implements a preprocessing pipeline for EEG data from a symmetry perception study (ds004347). The goal is to compare the original authors' preprocessing approach with an improved, automated pipeline.

## Dataset Information
- **Study**: Symmetry perception and affective responses (Makin et al., 2012)
- **EEG System**: BioSemi ActiveTwo 64-channel
- **Sampling Rate**: 512 Hz
- **Task**: Participants viewed regular (symmetric) and random patterns
- **Event Codes**: 
  - `1` = Regular (symmetric) patterns
  - `3` = Random patterns
- **Subjects**: 24 participants (currently analyzing sub-001)

## Milestone Progress

### Milestone 2: Pipeline Design
We designed two parallel pipelines:

| Step | Authors' Pipeline | Our Pipeline |
|------|------------------|--------------|
| 1. Load Data | Import BioSemi EEG/EMG | Load raw EEG into MNE |
| 2. Bad Channels | Manual inspection | Automatic detection (RANSAC/Autoreject) |
| 3. Filtering | 0.1–25 Hz (ERP) | 0.1–40 Hz + 50 Hz notch |
| 4. Referencing | Average reference | Average reference |
| 5. Artifact Removal | ICA + manual IC rejection | ICA + ICLabel automatic |
| 6. Epoching | -1 to 1 s | -0.2 to 1.0 s |
| 7. Baseline Correction | -200 to +50 ms | -200 to 0 ms |
| 8. ERP Calculation | Mean ERP | Robust/median ERP |
| 9. Time-Frequency | 4–20 Hz wavelet | 4–30 Hz wavelets |
| 10. Statistics | t-tests on ROIs | Cluster-based permutation |

### Milestone 3: Implementation
✅ **COMPLETED**: Single subject preprocessing pipeline

## Supervisor Feedback Addressed

### 1. Epoching Window Extension
**Feedback**: "The authors reference findings of effects up to 1100 ms poststimulus and argue that they decided on a -1s to 1s epoching window. Maybe reconsider the 800ms poststimulus?"

**Implementation**: Changed our epoching window from `-0.2 to 0.8s` to `-0.2 to 1.0s` to capture effects up to 1100ms as suggested.

### 2. Pipeline Order with ICA and Autoreject
**Feedback**: "I assume the order of your pipeline is a different one - the Q is, how to do ICA when using autoreject for channel."

**Implementation**: 
- **Our approach**: Filter → Reference → Temporary epochs → Bad channel detection → ICA on cleaned data → Final epochs
- This differs from a strict "autoreject first" approach because:
  1. We need continuous data properly filtered and referenced before ICA
  2. Bad channel detection uses temporary epochs
  3. ICA is performed on the cleaned continuous data
  4. Final epochs are created after ICA cleaning

### 3. Dual Visualization
**Feedback**: "When you implement something and show in graph, create 2 different visual, one for existing approach and one for our approach."

**Implementation**: Created three visualization outputs:
1. `sub-001_authors_pipeline.png` - Authors' pipeline results
2. `sub-001_ours_pipeline.png` - Our improved pipeline results  
3. `sub-001_comparison.png` - Direct side-by-side comparison

## File Structure

```
d:\ds004347\
├── code/
│   ├── preprocessing_pipeline.py       # Main pipeline script (NEW)
│   ├── README_milestone3.md            # This file (NEW)
│   └── [original MATLAB and Python files...]
├── derivatives/
│   └── preprocessing_results/          # Output directory (NEW)
│       ├── sub-001_authors_epo.fif     # Authors' epochs
│       ├── sub-001_ours_epo.fif        # Our epochs
│       ├── sub-001_authors_ave.fif     # Authors' evoked
│       ├── sub-001_ours_ave.fif        # Our evoked
│       ├── sub-001_authors_pipeline.png
│       ├── sub-001_ours_pipeline.png
│       └── sub-001_comparison.png
└── sub-001/
    └── eeg/
        ├── sub-001_task-jacobsen_eeg.bdf  # Raw EEG data
        └── [other BIDS files...]
```

## Running the Pipeline

### Prerequisites
```bash
pip install mne numpy matplotlib scipy
```

Optional packages for full functionality:
```bash
pip install autoreject mne-icalabel pyprep
```

### Execution
```bash
cd d:\ds004347\code
python preprocessing_pipeline.py
```

### Expected Output
The script will:
1. Load raw EEG data for sub-001
2. Run both preprocessing pipelines in parallel
3. Generate epochs and evoked responses
4. Save all intermediate results
5. Create comparison visualizations
6. Display summary statistics

## Key Differences Between Pipelines

### Authors' Pipeline
- **Strengths**:
  - Conservative filtering (0.1-25 Hz) preserves ERP components
  - Longer pre-stimulus baseline (-1s) for stable baseline
  - Proven approach from published study
  
- **Limitations**:
  - Manual inspection is time-consuming and subjective
  - No power line noise removal
  - Mean ERP is sensitive to outliers
  - Narrow frequency range misses high-frequency activity

### Our Improved Pipeline
- **Advantages**:
  - Automated, reproducible processing
  - 50 Hz notch filter removes power line noise
  - Extended frequency range (0.1-40 Hz) captures broader activity
  - Median ERP is robust to outliers
  - Efficient pre-stimulus period (-0.2s)
  
- **Considerations**:
  - Automated artifact detection may need validation
  - Shorter baseline period requires stable pre-stimulus activity
  - ICLabel requires additional packages

## Pipeline Order Rationale

### Our Pipeline Order:
1. **Load Data** - Import raw .bdf file
2. **Filter** (0.1-40 Hz + 50 Hz notch) - Remove slow drifts and high-frequency noise
3. **Reference** (average) - Common reference for all channels
4. **Bad Channel Detection** - Uses temporary epochs to identify bad channels statistically
5. **Interpolate Bad Channels** - Replace bad channels with interpolated values
6. **ICA** (on 1 Hz high-passed data) - Identify and remove artifacts
7. **Apply ICA** (to original filtered data) - Remove artifacts while preserving low frequencies
8. **Epoch** (-0.2 to 1.0s) - Extract task-relevant segments
9. **Baseline Correction** (-200 to 0 ms) - Normalize to pre-stimulus period
10. **Calculate ERP** (median) - Robust averaging

### Why This Order?
- **Filtering before ICA**: ICA works better on filtered data (removes non-neural noise)
- **Reference before bad channel detection**: Proper reference needed for accurate detection
- **Bad channels before ICA**: Don't want bad channels influencing ICA decomposition
- **High-pass for ICA (1 Hz)**: ICA is sensitive to slow drifts
- **Apply ICA to original filter settings**: Preserve low-frequency ERP components
- **Baseline correction after epoching**: Applied to each trial individually

## Results Interpretation

### ERP Components of Interest
- **SPN (Sustained Posterior Negativity)**: 300-1000 ms post-stimulus
  - Expected at posterior sites (PO7, PO8, O1, O2, Oz)
  - More negative for regular (symmetric) patterns
  - Reflects visual processing of symmetry

### Expected Findings
- Regular patterns should elicit:
  - More negative amplitude at posterior sites
  - Peak around 300-400 ms
  - Sustained negativity up to 1000 ms
  
## Next Steps (Milestone 4)

1. **Group-Level Analysis**:
   - Run pipeline on all 24 subjects
   - Create grand average ERPs
   - Implement automated quality control

2. **Time-Frequency Analysis** (Step 9):
   - Wavelet analysis (4-30 Hz)
   - Time-frequency representations
   - Induced vs evoked power

3. **Statistical Analysis** (Step 10):
   - Cluster-based permutation tests
   - Multiple comparison correction
   - Compare with authors' t-test results

4. **Validation**:
   - Compare our results with authors' published findings
   - Assess sensitivity of different preprocessing choices
   - Document any discrepancies

## Technical Notes

### Bad Channel Detection
The current implementation uses a simple statistical outlier detection based on variance. For production use, consider:
- **RANSAC** (from `autoreject` package): More robust iterative method
- **PREP pipeline** (from `pyprep` package): Standardized preprocessing
- **Visual inspection**: Always good to manually verify

### ICA Component Selection
The current implementation uses variance-based heuristics. For production use:
- **ICLabel** (from `mne-icalabel`): Deep learning-based automatic classification
- **Manual inspection**: Review component topographies and time courses
- **EOG correlation**: Identify eye movement artifacts

### Robust Averaging
We use median instead of mean for robustness. Other options:
- **Autoreject**: Automated trial rejection before averaging
- **Robust averaging**: Iteratively downweight outlier trials
- **Trimmed mean**: Average after removing extreme values

## References

1. Makin, A. D. J., Wilton, M. M., Pecchinenda, A., & Bertamini, M. (2012). Symmetry perception and affective responses: A combined EEG/EMG study. *Neuropsychologia*, 50(14), 3250-3261.

2. Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.

3. Jas, M., et al. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. *NeuroImage*, 159, 417-429.

## Contact

For questions about this implementation, please refer to the project documentation or supervisor.

---

**Last Updated**: December 2025  
**Status**: Milestone 3 Complete ✅  
**Next Milestone**: Group-level analysis and time-frequency decomposition
