# EEG Symmetry Perception — Personal Notes

Course project for **Signal Processing and Analysis of Human Brain Potentials (EEG)**  
Universität Stuttgart | Supervisor: Jun.-Prof. Dr. Benedikt Ehinger  
Authors: Harish Renganathan, Rithika Byna, Aveen Vas | Due: March 31, 2026

---

## What this is

Personal code repo for the EEG coursework. Analyses EEG data (OpenNeuro ds004347, 24 subjects)
looking at symmetric vs. random visual patterns. Main result: random patterns drive stronger
occipital ERPs than symmetric ones (peak −1.03 µV at 420 ms, d = 1.73).

Analyses done:
- ERP at Oz + topomaps
- Morlet wavelet TFR
- Cluster permutation test
- LDA single-trial decoding (AUC = 0.62)
- Lateralisation (bilateral, as expected)
- dSPM source localisation (fsaverage template)
- Individual differences vs. preprocessing quality

---

## Repo structure

```
code/          analysis scripts (Python) + original MATLAB scripts
sub-001…024/   BIDS metadata only (no raw .bdf — too large)
```

Large/private stuff excluded via `.gitignore`: `derivatives/`, `.venv/`, `mne-data/`, `report/`

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r code/requirements.txt

python code/preprocessing_pipeline.py   # preprocess all 24 subjects
python code/final_analyses.py            # run analyses + save figures
python code/update_figures.py            # regenerate report figures only
```

Report (`report/eeg_report_final.tex`) is kept local — compile via Overleaf or:
`pdflatex → biber → pdflatex × 2`
