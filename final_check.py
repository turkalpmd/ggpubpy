#!/usr/bin/env python3
"""
Final comprehensive test script to verify package is ready for publication.

This script runs:
- Code quality checks (black, isort, mypy)
- Unit tests (pytest)
- Integration/smoke tests
"""

import os
import subprocess
import sys
import traceback
from typing import List


def run_command(command: List[str], description: str) -> bool:
    """
    Run a shell command and report its success or failure.

    Parameters
    ----------
    command : list
        The command to execute as a list of strings.
    description : str
        A description of what the command is doing.

    Returns
    -------
    bool
        True if the command succeeded, False otherwise.
    """
    print(f"\n--- Running {description} ---")
    try:
        is_windows = sys.platform == "win32"
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            shell=is_windows,
        )
        print(process.stdout)
        if process.stderr:
            print("--- STDERR ---")
            print(process.stderr)
        print(f"[PASS] {description} successful.")
        return True
    except FileNotFoundError:
        print(f"[FAIL] Command '{command[0]}' not found.")
        print("Please ensure required packages are installed in your environment.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {description} failed with exit code {e.returncode}.")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"[FAIL] An unexpected error occurred: {e}")
        return False


def run_quality_and_unit_tests() -> bool:
    """
    Run all code quality checks and unit tests.
    """
    print("[INFO] Running Code Quality Checks and Unit Tests...")
    print("=" * 50)

    checks = [
        (["black", "."], "black (code formatter)"),
        (["isort", "."], "isort (import sorter)"),
        (["mypy", "."], "mypy (static type checker)"),
        (["pytest"], "pytest (unit tests)"),
    ]

    for command, description in checks:
        if not run_command(command, description):
            return False

    print("\n[SUCCESS] All quality checks and unit tests passed.")
    return True


def run_integration_tests() -> bool:
    """Run comprehensive integration tests for ggpubpy package."""
    print("\n[INFO] Running Integration and Smoke Tests...")
    print("=" * 50)

    try:
        import matplotlib
        import matplotlib.pyplot as plt

        import ggpubpy
        from ggpubpy import (
            plot_boxplot_with_stats,
            plot_shift,
            plot_violin_with_stats,
        )
        from ggpubpy.datasets import get_iris_palette, list_datasets, load_iris

        matplotlib.use("Agg")  # Configure backend before plotting

        print("[PASS] ggpubpy import successful")
        print("[PASS] Main functions import successful")

        iris = load_iris()
        print(
            f"[PASS] Iris dataset loaded: {len(iris)} rows, {len(iris.columns)} columns"
        )

        fig, ax = plot_violin_with_stats(iris, x="species", y="sepal_length")
        plt.close(fig)
        print("[PASS] Violin plot creation successful")

        fig, ax = plot_boxplot_with_stats(iris, x="species", y="sepal_length")
        plt.close(fig)
        print("[PASS] Box plot creation successful")

        x_data = iris[iris["species"] == "setosa"]["sepal_length"].values
        y_data = iris[iris["species"] == "versicolor"]["sepal_length"].values
        fig = plot_shift(x_data, y_data)
        plt.close(fig)
        print("[PASS] Shift plot creation successful")

        palette = get_iris_palette()
        datasets = list_datasets()
        print(
            f"[PASS] Dataset utilities successful: {len(palette)} colors, {len(datasets)} datasets"
        )

    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        traceback.print_exc()
        return False

    print("\n[SUCCESS] All integration tests passed.")
    return True


def main() -> None:
    """Main test runner."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    quality_passed = run_quality_and_unit_tests()
    integration_passed = False
    if quality_passed:
        integration_passed = run_integration_tests()

    print("\n" + "=" * 50)
    if quality_passed and integration_passed:
        print(
            "[SUCCESS] ALL CHECKS AND TESTS PASSED - Package is ready for publication!"
        )
        print()
        print("[PASS] Code Quality (black, isort, mypy): PASSED")
        print("[PASS] Unit Tests (pytest): PASSED")
        print("[PASS] Integration Tests (imports, plots, data): PASSED")
        print()
        print("[INFO] Ready to publish to PyPI!")
    else:
        print("[FAIL] CHECKS OR TESTS FAILED - Package needs fixes before publication.")
        sys.exit(1)


if __name__ == "__main__":
    main()
