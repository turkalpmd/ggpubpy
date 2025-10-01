# QQ Plot

Create a Quantile–Quantile (Q–Q) plot with optional confidence envelope and regression line.

## Examples (Iris)

```python
from ggpubpy import qqplot, load_iris
import matplotlib.pyplot as plt

iris = load_iris()
fig, ax = plt.subplots(figsize=(5, 5))
qqplot(iris["sepal_length"].values, dist="norm", confidence=0.95, ax=ax)
ax.set_title("Iris: QQ Plot (Sepal Length vs Normal)")
plt.tight_layout()
```

```python
from ggpubpy import qqplot, load_iris
import matplotlib.pyplot as plt

iris = load_iris()
fig, ax = plt.subplots(figsize=(5, 5))
qqplot(iris["petal_width"].values, dist="norm", confidence=0.95, ax=ax)
ax.set_title("Iris: QQ Plot (Petal Width vs Normal)")
plt.tight_layout()
```

## Parameters
- **x**: array-like sample data
- **dist**: scipy.stats distribution or name (default: "norm")
- **sparams**: shape/location/scale parameters tuple (optional)
- **confidence**: float in (0,1) for envelope, or False to disable
- **square**: keep equal aspect ratio (default: True)
- **ax**: Matplotlib Axes (optional)
- Additional kwargs are passed to `Axes.scatter`.

## Notes
- The 45° line indicates perfect agreement; deviations show distributional differences.
