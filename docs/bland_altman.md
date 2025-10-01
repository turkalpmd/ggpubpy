# Bland–Altman Plot

Agreement plot to compare two measurement methods with mean bias and limits of agreement.

## Basic Usage

```python
from ggpubpy import plot_blandaltman
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.normal(loc=10, scale=2, size=100)
y = x + np.random.normal(loc=0.2, scale=1.0, size=100)
fig, ax = plt.subplots(figsize=(6, 4))
plot_blandaltman(x, y, agreement=1.96, confidence=0.95, annotate=True, ax=ax)
ax.set_title("Bland–Altman")
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
