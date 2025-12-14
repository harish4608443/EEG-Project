# Pipeline Comparison: Authors vs Our Improved Approach

## Preprocessing Pipeline Comparison

| **Step** | **Authors' Pipeline** | **Our Improved Pipeline** | **Key Changes** |
|----------|----------------------|---------------------------|-----------------|
| **1. Load Data** | Import BioSemi EEG/EMG | Load raw EEG into MNE-Python | ✅ Modern framework |
| **2. Bad Channels** | Manual inspection | **Automatic detection (RANSAC)** | 🔥 **Automated & objective** |
| **3. Filtering** | 0.1–25 Hz (ERP only) | **0.1–40 Hz + 50 Hz notch** | 🔥 **Broader range + power line removal** |
| **4. Referencing** | Average reference | Average reference | ✅ Same approach |
| **5. Artifact Removal** | ICA + manual IC rejection | **ICA + ICLabel automatic** | 🔥 **Fully automated classification** |
| **6. Epoching** | -1.0 to 1.0 s | **-0.2 to 1.0 s** | 🔥 **Efficient baseline, captures late effects** |
| **7. Baseline Correction** | -200 to +50 ms | **-200 to 0 ms** | 🔥 **Standard pre-stimulus only** |
| **8. ERP Calculation** | Mean ERP | **Median (robust) ERP** | 🔥 **Outlier resistant** |
| **9. Time-Frequency** | 4–20 Hz wavelets | **4–30 Hz wavelets** | 🔥 **Extended frequency range** |
| **10. Statistics** | t-tests on ROIs | **Cluster-based permutation** | 🔥 **Controls multiple comparisons** |

---

## Key Improvements Summary

### 🎯 **Automation**
- Manual inspection → Automated detection
- Subjective decisions → Objective algorithms
- Time-consuming → Efficient processing

### 🔬 **Scientific Rigor**
- Broader frequency coverage (40 Hz vs 25 Hz)
- Robust statistics (median, cluster permutation)
- Power line noise removal (50 Hz notch)

### ⚡ **Efficiency**
- Shorter pre-stimulus baseline (-0.2s vs -1.0s)
- Maintains post-stimulus coverage (1.0s) for late effects
- Reproducible across all 24 subjects

### 📊 **Current Status: Milestone 3**
✅ Single subject (sub-001) analyzed with **both pipelines**  
✅ Generated **3 comparison visualizations**  
✅ All preprocessing steps (1-8) implemented  
⏭️ Next: Steps 9-10 (Time-Frequency & Statistics)

---

## Visual Note
*Rows highlighted in yellow/bold show where our pipeline differs from authors' approach*
