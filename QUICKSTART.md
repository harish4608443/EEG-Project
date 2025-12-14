# Quick Start Guide - Milestone 3

## Installation

### Step 1: Install Required Package (MNE)

```bash
pip install mne
```

Or install all requirements at once:
```bash
cd d:\ds004347\code
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python check_environment.py
```

You should see all required packages marked with ✓

## Running the Pipeline

### Basic Execution

```bash
cd d:\ds004347\code
python preprocessing_pipeline.py
```

This will:
1. Load raw EEG data for sub-001
2. Process through both pipelines (Authors' and Ours)
3. Save epochs and evoked responses
4. Generate 3 comparison visualizations
5. Display summary statistics

### Expected Runtime
- **Single subject**: ~2-5 minutes (depending on computer)
- Most time spent on: ICA fitting and plotting

### Output Location
All results saved to: `d:\ds004347\derivatives\preprocessing_results\`

Files generated:
- `sub-001_authors_epo.fif` - Authors' epochs
- `sub-001_ours_epo.fif` - Our epochs  
- `sub-001_authors_ave.fif` - Authors' evoked
- `sub-001_ours_ave.fif` - Our evoked
- `sub-001_authors_pipeline.png` - Authors' visualization
- `sub-001_ours_pipeline.png` - Our visualization
- `sub-001_comparison.png` - Direct comparison

## Interpreting Results

### What to Look For

1. **ERP Waveforms** (Top left plot)
   - Blue line = Regular (symmetric) patterns
   - Red line = Random patterns
   - Regular should be more negative ~300-800ms

2. **Difference Wave** (Top right plot)
   - Shows Regular - Random
   - Negative deflection = SPN component
   - Peak around 300-400ms
   - Sustained to ~1000ms

3. **Topographic Map** (Bottom left)
   - Shows spatial distribution at 300ms
   - SPN strongest at posterior sites (back of head)
   - Red = positive, Blue = negative

4. **Comparison Figure**
   - Left: Authors' vs Our difference waves overlaid
   - Right: Summary statistics table
   - Should show similar patterns with some differences

### Expected Differences

| Aspect | Authors' | Ours | Why Different? |
|--------|----------|------|----------------|
| Amplitude | Slightly larger | Slightly smaller | Median less sensitive to outliers |
| Noise level | More variable | Smoother | Broader filtering, median averaging |
| Time course | Longer (-1 to 1s) | Shorter (-0.2 to 1s) | Different epoch window |

## Troubleshooting

### Error: "File not found"
- Check that data file exists: `d:\ds004347\sub-001\eeg\sub-001_task-jacobsen_eeg.bdf`
- File size should be ~111 MB

### Error: "No module named 'mne'"
- Install MNE: `pip install mne`

### Warning: "Bad channels detected"
- This is normal - automatic detection may find noisy channels
- They will be interpolated automatically

### Warning: "ICA did not converge"
- Try increasing `max_iter` parameter in ICA setup
- Or reduce number of components

### Plots not displaying
- Make sure you're not running in a headless environment
- Plots are also saved as PNG files automatically

## Common Issues

### Issue 1: Memory Error
If you get a memory error:
- Close other applications
- Reduce ICA components (change `n_components=20` to `n_components=15`)

### Issue 2: Slow Performance
If processing is very slow:
- Normal for first run (MNE compiles filters)
- Subsequent runs should be faster
- Consider running without visualization for batch processing

### Issue 3: Different Results
If results differ from expected:
- Check that you're using sub-001
- Verify data file is complete (111 MB)
- Review printed statistics for data quality

## Next Steps After Running

1. **Review Output Files**
   - Check that all 7 files were created
   - Open PNG files to view visualizations

2. **Examine Data Quality**
   - How many bad channels detected?
   - How many ICA components removed?
   - Are there enough epochs retained?

3. **Compare Pipelines**
   - Do both pipelines show similar SPN effect?
   - Is the difference wave consistent?
   - Which approach looks cleaner?

4. **Document Findings**
   - Note any unusual patterns
   - Save observations for discussion
   - Prepare questions about results

## Advanced Usage

### Modify Pipeline Parameters

Edit `preprocessing_pipeline.py` to change:

```python
# Line ~24: Change subject
SUBJECT = 'sub-002'  # Process different subject

# Line ~137: Change ICA components
ica_authors = ICA(n_components=25, ...)  # More components

# Line ~310: Change ROI channels
roi_channels = ['O1', 'Oz', 'O2']  # Different electrode selection

# Line ~179: Change epoch window
tmin=-0.5, tmax=1.5  # Longer epochs
```

### Batch Processing

To process multiple subjects, create a loop:

```python
for subject_num in range(1, 25):  # All 24 subjects
    subject = f'sub-{subject_num:03d}'
    # ... run pipeline ...
```

## Getting Help

### Check Documentation
- `README_milestone3.md` - Full technical documentation
- `MILESTONE3_SUMMARY.md` - Summary for supervisor
- `requirements.txt` - Package dependencies

### Common Questions

**Q: How long should processing take?**  
A: 2-5 minutes per subject on a modern computer

**Q: Can I run this on all subjects at once?**  
A: Yes, but modify the script to loop through subjects

**Q: What if I don't have MNE installed?**  
A: Install it with `pip install mne` (requires ~200 MB)

**Q: The visualizations look noisy, is that normal?**  
A: Single-subject ERPs are always noisier than group averages

**Q: Should results match the published paper exactly?**  
A: No - single subject results will vary; group averages should be similar

## File Overview

```
code/
├── preprocessing_pipeline.py    ← Main script - RUN THIS
├── check_environment.py         ← Verify installation
├── requirements.txt             ← Install dependencies
├── README_milestone3.md         ← Full documentation  
├── MILESTONE3_SUMMARY.md        ← Supervisor summary
└── QUICKSTART.md                ← This file
```

## Success Checklist

- [ ] MNE package installed
- [ ] Environment check passes (✓ for all required packages)
- [ ] Data file found (111 MB .bdf file)
- [ ] Script runs without errors
- [ ] 7 output files generated
- [ ] Visualizations show clear ERP patterns
- [ ] Both pipelines complete successfully
- [ ] Results documented for discussion

---

**Ready to run?**

```bash
python preprocessing_pipeline.py
```

**Having issues?** Run the environment check first:

```bash
python check_environment.py
```
