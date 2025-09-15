#!/usr/bin/env python3
"""
Aggregate runner to generate all example images.

Note: Individual examples are maintained in separate scripts:
- violinplot_examples.py
- boxplot_examples.py
- shiftplot_examples.py
- correlation_matrix_example.py
- alluvial_examples.py

This runner imports and executes their main functions for convenience.
"""

import os
import sys

# Add the parent directory to sys.path to import ggpubpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ggpubpy

    print("✓ Successfully imported ggpubpy")
except Exception as e:
    print(f"✗ Failed to import ggpubpy: {e}")
    sys.exit(1)

import matplotlib.pyplot as plt


def main() -> None:
    """Generate all example plots for README."""
    print("Running individual example scripts to generate all images...")
    # Violin
    try:
        import examples.violinplot_examples as vexp

        vexp.main()
    except Exception as e:
        print(f"✗ Violin examples failed: {e}")

    # Boxplot
    try:
        import examples.boxplot_examples as bexp

        bexp.main()
    except Exception as e:
        print(f"✗ Boxplot examples failed: {e}")

    # Shift plot
    try:
        import examples.shiftplot_examples as shfexp

        shfexp.main()
    except Exception as e:
        print(f"✗ Shift plot examples failed: {e}")

    # Correlation matrices
    try:
        import examples.correlation_matrix_example as cmexp

        cmexp.main()
    except Exception as e:
        print(f"✗ Correlation examples failed: {e}")

    print("\nSummary:")
    for filename in [
        "violin_example.png",
        "boxplot_example.png",
        "violin_2groups_example.png",
        "boxplot_2groups_example.png",
        "shift_plot_example.png",
        "shift_plot_with_diff_example.png",
        "correlation_matrix_example.png",
    ]:
        filepath = os.path.join(examples_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (MISSING)")


if __name__ == "__main__":
    main()
