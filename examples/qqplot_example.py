import numpy as np
import matplotlib.pyplot as plt
from ggpubpy import qqplot

if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.normal(size=300)
    fig, ax = plt.subplots(figsize=(5, 5))
    qqplot(x, dist="norm", confidence=0.95, ax=ax)
    ax.set_title("QQ Plot (Normal)")
    fig.savefig("examples/qqplot_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
