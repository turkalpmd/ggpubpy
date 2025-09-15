
# Changelog

## [0.4.3] - YYYY-MM-DD

- Bump version to 0.4.3.



All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - 2025-09-15

### Added
- Shift Plot: Support `x_label`, `y_label`, `title`, `subtitle`, `color`, `line_color`, `alpha`, and `figsize` parameters.

### Changed
- Docs: Shift Plot examples now return a `Figure` and demonstrate accessing axes via `fig.axes[0]`.
- README: Shift Plot example updated to assign only `fig`.
- Version: Bumped to 0.4.2 and aligned `__init__` with `pyproject.toml`.

## [0.4.1] - 2024-12-19

### Fixed
- **Type checking**: Fixed all mypy type checking errors across the codebase
- **Type annotations**: Added proper type hints to all functions and variables
- **Import issues**: Fixed missing matplotlib.pyplot import in pre_upload_check.py
- **Return types**: Corrected return type annotations in examples and test functions
- **Dictionary types**: Fixed type mismatches in datasets.py palette functions

### Enhanced
- **Code quality**: Improved type safety and code maintainability
- **Development workflow**: All type checking now passes cleanly
- **Documentation**: Updated Sphinx configuration with proper type annotations

## [0.4.0] - 2024-09-03

### Added
- **Alluvial plots**: New `plot_alluvial()` and `plot_alluvial_with_stats()` functions for creating flow diagrams
- **Comprehensive documentation**: Complete ReadTheDocs documentation with examples and images
- **API documentation**: Automatic API reference generation with Sphinx
- **Enhanced examples**: Multiple working examples for all plot types
- **Statistical integration**: Ready for future statistical enhancements in alluvial plots

### Enhanced
- **Documentation**: Complete rewrite with professional ReadTheDocs site
- **Examples**: Added real dataset examples with high-quality images
- **Package structure**: Improved organization and lazy loading
- **Type hints**: Enhanced type annotations throughout the codebase

### Fixed
- **Import issues**: Resolved all import and dependency conflicts
- **Documentation paths**: Fixed all image and cross-reference paths
- **Build process**: Streamlined Sphinx build configuration

### Changed
- **Development status**: Moved from Alpha to Beta status
- **Documentation URL**: Updated to use ReadTheDocs instead of GitHub wiki
- **Package metadata**: Enhanced with additional keywords and classifiers

## [0.3.0] - 2024-08-XX

### Added
- **Box plots**: `plot_boxplot_with_stats()` with statistical annotations
- **Violin plots**: `plot_violin_with_stats()` with distribution visualization
- **Shift plots**: `plot_shift()` for paired data analysis
- **Correlation matrices**: `plot_correlation_matrix()` for relationship visualization
- **Statistical tests**: Built-in ANOVA, t-tests, and correlation analysis
- **Dataset support**: Built-in Iris and Titanic datasets
- **Color palettes**: Flexible color customization options

### Features
- Publication-ready styling
- Automatic p-value annotations
- Multiple statistical test options
- Jittered data points
- Custom ordering and grouping
- Professional appearance suitable for scientific publications

## [0.2.0] - 2024-07-XX

### Added
- Basic plotting infrastructure
- Statistical test framework
- Helper functions for data validation

## [0.1.0] - 2024-06-XX

### Added
- Initial package structure
- Basic matplotlib integration
- Core plotting functions

---

## Unreleased

### Planned
- Additional statistical tests for alluvial plots
- More plot customization options
- Interactive plot support
- Additional dataset integrations
## [0.4.3] - 2025-09-15

### Added
- API harmonization across plotting functions:
  - Shift Plot: `x_label`, `y_label`, `title`, `subtitle`, `color`, `line_color`, `alpha`, `figsize`.
  - Violin/Boxplot: `title`, `subtitle`, `alpha` for jitter.
  - Correlation Matrix: `subtitle` support.
- CI/CD: GitHub Actions workflows for tests/docs, PyPI release, and TestPyPI release.
- Scripts: `scripts/bump_version.py` for safe version bumps.

### Changed
- Docs: Updated Shift/Box/Violin/Correlation pages to reflect new parameters and return types.
- README: Added CI/TestPyPI badges and detailed “Releasing” section.
- Examples/Scripts: Demonstrate new parameters (title/subtitle/alpha/etc.).

### Security
- Removed accidentally committed token artifact and added ignore rules.
- Release workflow now verifies tag-version consistency before publishing.
