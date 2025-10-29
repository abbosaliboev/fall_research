# Fall Research — Human Activity / Fall Detection

Short README for the repository. It explains where to get the dataset, how the repository is organised, and quick commands to prepare and run experiments.

## Dataset

Dataset link:
https://sites.google.com/up.edu.mx/har-up/

Download the dataset from the link above and place the extracted dataset under the repository `data/` directory (e.g. `data/fall_data/` and `data/nofall_data/`). Do NOT commit large raw datasets to Git — see the "Ignoring dataset files" section below.

## Project structure (important folders)

- `data/` - raw dataset (should be kept out of Git). Put downloaded data here.
- `scripts/` - training, evaluation and preprocessing scripts (e.g. `train_tcn.py`, `eval_tcn.py`, `extract_pose.py`).
- `models/` - model artifacts and dataset derivatives (should be ignored by Git).
- `experiments/` - saved checkpoints, logs and experiment outputs (should be ignored by Git).
- `*.csv` - dataset split/label files provided at repo root (small metadata files may be tracked).

## Setup

This project uses the Python packages listed in `requirements.txt`. Create a virtual environment and install requirements:

```powershell
# create venv (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Adjust the Python command and activation command to match your environment (PowerShell example shown above).

## Quick run examples

Train a model (example):

```powershell
python scripts\train_tcn.py --config configs/train_config.yaml
```

Evaluate a trained model (example):

```powershell
python scripts\eval_tcn.py --checkpoint experiments\myrun\best.pt --data-dir data
```

Inspect available scripts in the `scripts/` folder for specific options used in this repo.

## Ignoring dataset files (important)

You should add the dataset and other large files to `.gitignore` to avoid committing them. Example `.gitignore` entries:

```
# dataset
data/

# models and checkpoints
models/
*.pt
*.pth

# common
__pycache__/
*.pyc
venv/
.venv/
```

If the `data/` folder (or other large files) were already committed to the repo in previous commits, adding to `.gitignore` is not enough. To stop tracking those files but keep them locally, run:

```powershell
git rm -r --cached "data/"
git commit -m "Stop tracking dataset files in data/ and add to .gitignore"
git push
```

Repeat `git rm -r --cached` for other directories you want to untrack (for example `models/` or `experiments/`). The `--cached` flag removes files from the index but leaves them on disk.

## Notes and next steps

- I can add the `.gitignore` entries for you and run the `git rm --cached` commands if you want — tell me whether to just update `.gitignore` or also untrack the already committed `data/` files.
- For sharing models or large binary data between collaborators, consider using Git LFS or an external storage bucket.

## Contact

If you need more help (examples, config files, or CI), tell me what to add and I will update the repo.
