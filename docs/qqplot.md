# QQ Plot

Create a Quantile–Quantile (Q–Q) plot with optional confidence envelope and regression line.

## Basic Usage

```python
from ggpubpy import qqplot
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.normal(size=200)
fig, ax = plt.subplots(figsize=(5, 5))
qqplot(x, dist="norm", confidence=0.95, ax=ax)
ax.set_title("QQ Plot (Normal)")
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
