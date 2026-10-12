# Contributing

ActSteer is an early-stage research toolkit. Bug reports, documentation fixes,
and narrowly scoped pull requests are welcome.

## Development setup

```bash
git clone https://github.com/nssokada/actsteer.git
cd actsteer
conda env create -f environment.yml
conda activate actsteer
python -m pip install -e ".[models,data,dev]"
```

Run the lightweight suite with:

```bash
python -m unittest discover -s tests -v
```

Tests must not require downloading large model weights. Model-specific changes
should include a small contract fixture and document any checkpoint-level smoke
test performed separately.

## Design boundaries

- `actsteer.core` must not import `actsteer.emotion`.
- Model-family attribute paths belong in model adapters, not experiment code.
- Task-specific classifiers, statistical analyses, and plots stay downstream.
- New artifacts must be immutable and include scientifically relevant
  provenance.
- Unsupported capabilities should fail explicitly rather than fall back
  silently.

Before opening a pull request, describe the scientific behavior being changed,
the models or tokenizers affected, and how the change was verified.
