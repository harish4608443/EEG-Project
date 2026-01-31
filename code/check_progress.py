"""
Quick Progress Check - See how many subjects have been processed
"""

from pathlib import Path
import json

BASE_DIR = Path(r"d:\ds004347")
BATCH_DIR = BASE_DIR / "derivatives" / "batch_processing"
SUBJECTS = [f"sub-{i:03d}" for i in range(1, 25)]

print("="*80)
print("BATCH PROCESSING PROGRESS CHECK")
print("="*80)
print()

processed = []
pending = []

for subject in SUBJECTS:
    metadata_file = BATCH_DIR / subject / f"{subject}_metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        processed.append((subject, metadata['ours_n_epochs']))
    else:
        pending.append(subject)

print(f"Processed: {len(processed)}/{len(SUBJECTS)} subjects")
print()

if processed:
    print("✓ Completed subjects:")
    for subj, n_epochs in processed:
        print(f"  {subj}: {n_epochs} epochs")
    print()

if pending:
    print(f"⏳ Pending: {len(pending)} subjects")
    print(f"  {', '.join(pending[:10])}" + ("..." if len(pending) > 10 else ""))
    print()

print("To process all subjects, run:")
print("  python code\\batch_preprocessing.py")
print()
print("To run sanity checks on completed subjects:")
print("  python code\\sanity_checks.py")
print("="*80)
