[![PyPI version](https://img.shields.io/pypi/v/ggpubpy)](https://pypi.org/project/ggpubpy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ggpubpy)](https://pypi.org/project/ggpubpy)
[![Downloads](https://img.shields.io/pypi/dm/ggpubpy)](https://pypi.org/project/ggpubpy)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/ggpubpy)](https://pypi.org/project/ggpubpy)
[![License](https://img.shields.io/pypi/l/ggpubpy)](https://github.com/turkalpmd/ggpubpy/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/turkalpmd/ggpubpy?style=social)](https://github.com/turkalpmd/ggpubpy)
[![GitHub forks](https://img.shields.io/github/forks/turkalpmd/ggpubpy?style=social)](https://github.com/turkalpmd/ggpubpy)

# ggpubpy: 'matplotlib' Based Publication-Ready Plots

Matplotlib is an excellent and flexible package for elegant data visualization in Python. However, the default plotting routines often require extensive boilerplate and manual styling before figures are ready for publication. Customizing complex plots can be a barrier for researchers and analysts without advanced plotting expertise.

The **ggpubpy** library provides a suite of easy-to-use functions for creating and customizing Matplotlib-based, publication-ready plots—complete with built-in statistical tests and automatic p-value or significance star annotations. This project is directly inspired by R's [ggpubr](https://github.com/kassambara/ggpubr) package.

**📦 PyPI Package**: https://pypi.org/project/ggpubpy/  
**🐙 GitHub Repository**: https://github.com/turkalpmd/ggpubpy  


---

## Installation and loading

Install the latest stable release from PyPI (recommended):

```bash
pip install ggpubpy
```

**Why install from PyPI?**
- ✅ Stable, tested releases
- ✅ Automatic dependency management
- ✅ Easy updates with `pip install --upgrade ggpubpy`
- ✅ Compatible with virtual environments

Or install the development version directly from GitHub:

```bash
pip install git+https://github.com/turkalpmd/ggpubpy.git
```

Load the package:

```python
import ggpubpy
from ggpubpy import violinggplot, boxggplot
from ggpubpy.datasets import load_iris  # Built-in datasets
```

---

## Core Features

- **Violin + boxplot + jitter** in one call  
- **Automatic color palettes** with ColorBrewer-inspired defaults
- **Built-in datasets** (iris) for quick testing and examples
- **Flexible group comparisons** - works with 2-group, 3-group, or more
- **Built-in Kruskal–Wallis & Mann–Whitney U tests** (or ANOVA & t-tests for parametric option)  
- **Automatic p-value or "star" annotation** with dynamic bracket placement
- **Smart p-value formatting** - pairwise comparisons show significance stars (*, **, ns), global tests show formatted values (<0.001)  
- **Parametric and non-parametric statistical tests** with `parametric=True/False` option
- **Smart test selection** - t-test for 2 groups, ANOVA for 3+ groups (parametric mode)
- **Modular, data-driven API**: custom labels, ordering, figure sizing

---

## Quick Examples

### 🎻 Violin plots with boxplots & jitter + statistical tests

#### 3-Group Comparison (All Species)
```python
import ggpubpy
from ggpubpy.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Create the plot with default colors (automatic palette)
fig, ax = ggpubpy.violinggplot(
    df=iris, 
    x="species", 
    y="sepal_length",
    x_label="Species", 
    y_label="Sepal Length (cm)"
)
```

![Violin Plot Example](examples/violin_example.png)

#### 2-Group Comparison (Subset Analysis)
```python
# Filter for 2-group comparison
iris_2groups = iris[iris['species'].isin(['setosa', 'versicolor'])]

# Create 2-group comparison plot
fig, ax = ggpubpy.violinggplot(
    df=iris_2groups, 
    x="species", 
    y="sepal_length",
    x_label="Species", 
    y_label="Sepal Length (cm)"
)
```

![Violin Plot 2-Groups](examples/violin_2groups_example.png)

### 📊 Boxplots with jitter + statistical tests

#### 3-Group Box Plot with Default Colors
```python
# Create boxplot with default automatic colors
fig, ax = ggpubpy.boxggplot(
    df=iris, 
    x="species", 
    y="sepal_length",
    x_label="Species", 
    y_label="Sepal Length (cm)"
)
```

![Box Plot Example](examples/boxplot_example.png)

#### 2-Group Box Plot with Statistical Tests
```python
# 2-group comparison with Mann-Whitney U test (non-parametric default)
iris_2groups = iris[iris['species'].isin(['setosa', 'versicolor'])]

fig, ax = ggpubpy.boxggplot(
    df=iris_2groups, 
    x="species", 
    y="sepal_length",
    x_label="Species", 
    y_label="Sepal Length (cm)",
    parametric=False  # Non-parametric tests (default)
)
```

![Box Plot 2-Groups](examples/boxplot_2groups_example.png)

### 🎨 Advanced Features

```python
# Custom color palette
custom_palette = {
    "setosa": "#FF6B6B", 
    "versicolor": "#4ECDC4", 
    "virginica": "#45B7D1"
}

fig, ax = ggpubpy.violinggplot(
    df=iris, 
    x="species", 
    y="petal_length",
    x_label="Species", 
    y_label="Petal Length (cm)",
    palette=custom_palette
)

# Parametric tests (ANOVA + t-test instead of Kruskal-Wallis + Mann-Whitney)
fig, ax = ggpubpy.violinggplot(
    df=iris, 
    x="species", 
    y="sepal_length",
    x_label="Species", 
    y_label="Sepal Length (cm)",
    parametric=True
)

# Custom ordering
fig, ax = ggpubpy.violinggplot(
    df=iris, 
    x="species",
    y="petal_width",
    order=["virginica", "versicolor", "setosa"]  # Custom order
)
```

### 📊 Built-in Datasets

```python
# Load built-in datasets
iris = ggpubpy.datasets.load_iris()
print(f"Available datasets: {ggpubpy.datasets.list_datasets()}")

# Get recommended color palette for iris species
palette = ggpubpy.datasets.get_iris_palette()
print(palette)  # {'setosa': '#00AFBB', 'versicolor': '#E7B800', 'virginica': '#FC4E07'}
```

---

## 🤝 Contributing

**We welcome contributions!** This project is designed to be contribution-friendly.

### Ways to Contribute:
- 🐛 **Bug reports** and feature requests
- 📖 **Documentation** improvements  
- 🔧 **Code contributions** (new features, optimizations, tests)
- 🎨 **New plot types** and statistical tests
- 📊 **Additional datasets** and examples

### Getting Started:
```bash
# Clone and setup development environment
git clone https://github.com/turkalpmd/ggpubpy.git
cd ggpubpy
pip install -e .
pip install -r requirements-dev.txt

# Run tests to verify setup
python final_check.py
```

See [CONTRIBUTING.md](docs/contributing.rst) for detailed guidelines.

---

## 📚 Documentation & Support

- **Local Documentation**: [Build locally](#-view-documentation-locally) with Sphinx
- **API Reference**: Complete function documentation with examples
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussion

### 🚀 View Documentation Locally

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
sphinx-build -b html docs docs/_build

# Open in browser (Windows)
start docs/_build/index.html

# Open in browser (Mac/Linux)
open docs/_build/index.html  # Mac
xdg-open docs/_build/index.html  # Linux
```

---

## License

**ggpubpy** is released under the MIT License. See [LICENSE](https://github.com/turkalpmd/ggpubpy/blob/main/LICENSE) for details.

---

## 📈 Project Status

🎉 **PUBLISHED ON PyPI**: June 20, 2025  
📦 **Latest Version**: 0.1.0  
🌟 **Status**: Stable and ready for production use  
🤝 **Contributing**: Open for community contributions  

**Install now**: `pip install ggpubpy`
