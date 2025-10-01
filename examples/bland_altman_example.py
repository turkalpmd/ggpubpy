import numpy as np
import matplotlib.pyplot as plt
from ggpubpy import plot_blandaltman

if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.normal(loc=10, scale=2, size=150)
    y = x + np.random.normal(loc=0.3, scale=1.1, size=150)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_blandaltman(x, y, agreement=1.96, confidence=0.95, annotate=True, ax=ax)
    ax.set_title("Bland–Altman")
    fig.savefig("examples/bland_altman_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
