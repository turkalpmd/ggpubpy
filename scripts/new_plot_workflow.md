# Internal Guide: Adding a New Plot Type (One-Pass Checklist)

Follow this single document top-to-bottom to implement, document, test, and release a new plot in one go.

## 0) Prep
- Create a branch: `git switch -c feat/<plot-name>`
- Ensure venv is active and deps installed: `pip install -e .[dev,docs]`

## 1) Implement plot + minimal example
- Add new module under `ggpubpy/<plot_name>.py`. Keep API consistent (data-first, labels, title, subtitle; return `(fig, ax)` or `(fig, axes)`).
- Export via `ggpubpy/__init__.py` (explicit import at top; add to `__all__`).
- Create runnable example saving an image:
  - `examples/<plot_name>_example.py` → saves `examples/<plot_name>_example.png`.

Checklist:
- [ ] Implement `ggpubpy/<plot_name>.py`
- [ ] Export in `ggpubpy/__init__.py`
- [ ] Add `examples/<plot_name>_example.py` (saves PNG)

## 2) Tests (headless)
- Create `tests/test_<plot_name>.py`:
  - Use `matplotlib.use("Agg")`
  - Import the new function from `ggpubpy`
  - Smoke test: returns Axes; plot closes without error

Checklist:
- [ ] Add `tests/test_<plot_name>.py`
- [ ] `python -m pytest -q` passes locally

## 3) Docs
- Create `docs/<plot_name>.md` matching other pages (Features, Basic Usage with Iris, second example, Parameters, Tips) and embed images:
  - `![Main](../examples/<plot_name>_example.png)`
  - Optionally embed manipulated image if added to manipulation generator
- Add to `docs/index.rst` under Plot Types

Checklist:
- [ ] Add `docs/<plot_name>.md` with 1–2 Iris examples
- [ ] Include images from `examples/`
- [ ] Link in `docs/index.rst`

## 4) Manipulation examples integration (optional but recommended)
- Update `examples/plots_manipulation_examples.py`:
  - Add a generator function `save_<plot_name>()` and save `examples/plots_manip_<plot_name>.png`
  - Add call in `__main__`
- Run `python examples/plots_manipulation_examples.py`

Checklist:
- [ ] Add `save_<plot_name>()` to manipulation script
- [ ] Regenerate manipulation images

## 5) Version bump + CHANGELOG
- Decide next version: feature → bump minor, bugfix → bump patch
- Update versions:
  - `pyproject.toml` → `project.version = "X.Y.Z"`
  - `ggpubpy/__init__.py` → `__version__ = "X.Y.Z"`
- Update `CHANGELOG.md` (Added, Changed, Fixed)

Checklist:
- [ ] Bump versions in pyproject + __init__
- [ ] Update CHANGELOG

## 6) Full validation (one command set)
Run the following in order:
```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```
(Optionally) scripts:
```bash
python scripts/final_check.py
python scripts/pre_upload_check.py
```

Checklist:
- [ ] Tests pass
- [ ] Build succeeds
- [ ] Twine check passes

## 7) Commit, push, tag, release
```bash
git add -A
git commit -m "feat: add <plot_name> plot, docs, examples, tests"
git push origin HEAD
```
- Create release tag that matches version (triggers CI/CD via `.github/workflows/release.yml`):
```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Checklist:
- [ ] Pushed branch/main
- [ ] Tag pushed → GitHub Actions releases to PyPI

## 8) Post-release sanity
- Verify GitHub Action run success
- Check PyPI page updated
- If hotfix needed: bump patch (X.Y.Z+1), fix, re-run 6–7

---

## Quick TODO template (copy/paste)
- [ ] Implement `ggpubpy/<plot_name>.py` and export in `__init__.py`
- [ ] Example `examples/<plot_name>_example.py` (save PNG)
- [ ] Tests `tests/test_<plot_name>.py` (Agg, returns Axes)
- [ ] Docs `docs/<plot_name>.md` + link in `docs/index.rst`
- [ ] Manipulation generator update (optional)
- [ ] Bump version (pyproject + __init__)
- [ ] Update CHANGELOG
- [ ] Run tests → build → twine check
- [ ] Commit, push, tag `vX.Y.Z`
- [ ] Verify CI + PyPI
