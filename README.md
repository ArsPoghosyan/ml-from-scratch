# ML From Scratch

Core machine learning algorithms implemented from scratch with Python and NumPy.

## Project Layout

```text
ml_from_scratch/
  base.py
  linear_model/
  tree/
  cluster/
  decomposition/
  utils/
tests/
examples/
```

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Development

Run tests with:

```powershell
pytest
```

Examples live in `examples/`. Package code lives in `ml_from_scratch/`.
