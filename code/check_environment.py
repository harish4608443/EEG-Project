"""
Environment Check Script
Verifies that all required packages are installed and accessible
"""

import sys

print("="*80)
print("EEG PREPROCESSING PIPELINE - ENVIRONMENT CHECK")
print("="*80)
print()

# Check Python version
print(f"Python version: {sys.version}")
print()

# Required packages
required_packages = {
    'mne': 'Core EEG/MEG analysis',
    'numpy': 'Numerical computing',
    'matplotlib': 'Visualization',
    'scipy': 'Signal processing',
}

# Optional packages
optional_packages = {
    'autoreject': 'Automated artifact detection',
    'mne_icalabel': 'Automatic ICA classification',
    'pyprep': 'PREP pipeline',
}

print("REQUIRED PACKAGES")
print("-" * 80)

all_required_ok = True
for package, description in required_packages.items():
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package:20s} v{version:15s} - {description}")
    except ImportError:
        print(f"✗ {package:20s} {'NOT FOUND':15s} - {description}")
        all_required_ok = False

print()
print("OPTIONAL PACKAGES")
print("-" * 80)

for package, description in optional_packages.items():
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package:20s} v{version:15s} - {description}")
    except ImportError:
        print(f"○ {package:20s} {'NOT INSTALLED':15s} - {description}")

print()
print("="*80)

if all_required_ok:
    print("✓ ALL REQUIRED PACKAGES ARE INSTALLED")
    print()
    print("You can run the preprocessing pipeline:")
    print("  python preprocessing_pipeline.py")
else:
    print("✗ SOME REQUIRED PACKAGES ARE MISSING")
    print()
    print("Install missing packages with:")
    print("  pip install mne numpy matplotlib scipy")
    print()
    print("Or install all at once:")
    print("  pip install -r requirements.txt")

print("="*80)
print()

# Check data file
import os
data_file = r'd:\ds004347\sub-001\eeg\sub-001_task-jacobsen_eeg.bdf'
print("DATA FILE CHECK")
print("-" * 80)
if os.path.exists(data_file):
    size_mb = os.path.getsize(data_file) / (1024 * 1024)
    print(f"✓ Raw data file found: {data_file}")
    print(f"  File size: {size_mb:.2f} MB")
else:
    print(f"✗ Raw data file not found: {data_file}")
    print("  Make sure you have the dataset downloaded")

print("="*80)
