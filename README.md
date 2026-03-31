# EEG Symmetry Perception — Personal Notes

Course project for **Signal Processing and Analysis of Human Brain Potentials (EEG)**  
Universität Stuttgart | Supervisor: Jun.-Prof. Dr. Benedikt Ehinger  
Authors: Harish Renganathan, Rithika Byna, Aveen Vas 

---

## What this is

Personal code repo for the EEG coursework. Analyses EEG data (NEMAR ds004347, 24 subjects)
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

## Dataset setup (required before running anything)

The raw EEG data is **not included** in this repo — it must be downloaded separately and
placed in the correct folder structure before running any script.

**1. Download from OpenNeuro**

Go to https://nemar.org/dataexplorer/detail?dataset_id=ds004347 and download the dataset, or use the
OpenNeuro CLI:

```bash
pip install openneuro-py
openneuro download --dataset ds004347 --target d:\ds004347
```

**2. Expected folder structure**

After downloading, your directory should look like this:

```
d:\ds004347\
├── sub-001\
│   └── eeg\
│       ├── sub-001_task-jacobsen_eeg.bdf      ← raw EEG (needed)
│       ├── sub-001_task-jacobsen_events.tsv
│       └── ...
├── sub-002\
│   └── eeg\
│       └── sub-002_task-jacobsen_eeg.bdf
│   ...
└── sub-024\
```

Each subject folder needs the `.bdf` file inside `eeg/`. The scripts look for files at:
`sub-XXX/eeg/sub-XXX_task-jacobsen_eeg.bdf`

**3. Do NOT run the scripts without the data**

You will get a `FileNotFoundError` if the `.bdf` files are missing. The preprocessing
script (`preprocessing_pipeline.py`) must be run first — it generates the cleaned epochs
that `final_analyses.py` and `update_figures.py` depend on.

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


