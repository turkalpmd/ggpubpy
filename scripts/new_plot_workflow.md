# Internal Guide: Adding a New Plot Type

This guide is for maintainers. It documents the internal process for adding a new plot type to ggpubpy.

## 1) Implement plotting function and standalone example

- Implement in `ggpubpy/` (new module preferred). Maintain consistent API: data-first args, labels, `title`, `subtitle`, style, and `(fig, ax)` (or `(fig, axes)`) return.
- Create an example in `examples/` that saves an image (PNG) referencing the feature and label customization.

## 2) Tests in `tests/`

- Add unit/integration tests that:
  - Import and call the new function
  - Assert `matplotlib` objects are returned
  - Run with Agg backend

## 3) Docs page (public) and link

- Create `docs/<plot_name>.md` with intro, features, basic usage, parameters, and examples (use the saved image).
- Cross-reference `docs/plots_manipulation.md` for label/title tips.
- Add to `docs/index.rst` under Plot Types.

## 4) Update `examples/plots_manipulation_examples.py`

- Add a new function that shows manipulated labels/colors/titles for the new plot and saves a PNG.
- Call it from `__main__`.

## 5) Versioning and release

- Bump `ggpubpy.__version__` appropriately (feature: minor, bugfix: patch). Keep `pyproject.toml` in sync if needed.
- Update `CHANGELOG.md`.
- Build and validate:
  - `python -m build`
  - `twine check dist/*`
- Tag and push to trigger CI:
  - `git tag v<new_version>`
  - `git push origin v<new_version>`

## 6) Validation scripts

- Run `python scripts/final_check.py` (quality + unit + integration tests)
- Run `python scripts/pre_upload_check.py` (build + twine + optional upload prompts)

## 7) CI/docs/examples sanity

- Ensure docs are linked and examples run headless.
- Ensure images exist under `examples/`.

## 8) Git workflow

- Feature branch → PR → CI pass → merge to `main` → tag push.

Quick commands:

```bash
python scripts/final_check.py
python scripts/pre_upload_check.py
python -m build
python -m twine check dist/*
git tag v0.5.1
git push origin v0.5.1
```
