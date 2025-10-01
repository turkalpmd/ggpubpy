# Bland–Altman Plot

Agreement plot to compare two measurement methods with mean bias and limits of agreement.

## Examples (Iris)

```python
from ggpubpy import plot_blandaltman, load_iris
import matplotlib.pyplot as plt

iris = load_iris()
fig, ax = plt.subplots(figsize=(6, 4))
plot_blandaltman(
    iris["sepal_length"].values,
    iris["petal_length"].values,
    agreement=1.96,
    confidence=0.95,
    annotate=True,
    ax=ax,
)
ax.set_title("Iris: Bland–Altman (Sepal Length vs Petal Length)")
plt.tight_layout()
```

```python
from ggpubpy import plot_blandaltman, load_iris
import matplotlib.pyplot as plt

iris = load_iris()
fig, ax = plt.subplots(figsize=(6, 4))
plot_blandaltman(
    iris["sepal_width"].values,
    iris["petal_width"].values,
    agreement=1.96,
    confidence=0.95,
    annotate=False,
    ax=ax,
)
ax.set_title("Iris: Bland–Altman (Sepal Width vs Petal Width)")
plt.tight_layout()
```

## Parameters
- **x, y**: arrays of measurements (same length)
- **agreement**: multiplier for limits of agreement (default: 1.96)
- **xaxis**: "mean", "x", or "y" for x-axis reference
- **confidence**: CI for mean bias and limits (float) or None
- **annotate**: add textual annotations (default: True)
- **ax**: Matplotlib Axes (optional)
- Additional kwargs are passed to `Axes.scatter`.
