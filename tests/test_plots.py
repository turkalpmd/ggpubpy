"""
Tests for ggpubpy plotting functions.
"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ggpubpy import (
    plot_boxplot_with_stats,
    plot_shift,
    plot_violin_with_stats,
    significance_stars,
)
from ggpubpy.plots import _perform_statistical_tests, _validate_inputs


class TestSignificanceStars:
    """Test the significance_stars function."""

    def test_significance_stars(self) -> None:
        """Test p-value to stars conversion."""
        assert significance_stars(1e-5) == "****"
        assert significance_stars(5e-4) == "***"
        assert significance_stars(0.005) == "**"
        assert significance_stars(0.03) == "*"
        assert significance_stars(0.1) == "ns"


class TestPlotViolinWithStats:
    """Test the plot_violin_with_stats function."""

    def test_basic_violin_plot(
        self, sample_data: pd.DataFrame, sample_palette: Dict[float, str]
    ) -> None:
        """Test basic violin plot creation."""
        fig, ax = plot_violin_with_stats(
            sample_data, x="dose", y="len", palette=sample_palette
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "dose"
        assert ax.get_ylabel() == "len"

        # Close the figure to prevent display
        plt.close(fig)

    def test_custom_labels(self, sample_data: pd.DataFrame) -> None:
        """Test custom axis labels."""
        fig, ax = plot_violin_with_stats(
            sample_data,
            x="dose",
            y="len",
            x_label="Dose (mg)",
            y_label="Length (units)",
        )

        assert ax.get_xlabel() == "Dose (mg)"
        assert ax.get_ylabel() == "Length (units)"
        plt.close(fig)

    def test_custom_order(self, sample_data: pd.DataFrame) -> None:
        """Test custom ordering of categories."""
        fig, ax = plot_violin_with_stats(
            sample_data, x="dose", y="len", order=[2.0, 1.0, 0.5]
        )

        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == ["2.0", "1.0", "0.5"]
        plt.close(fig)

    def test_no_jitter(self, sample_data: pd.DataFrame) -> None:
        """Test violin plot without jitter points."""
        fig, ax = plot_violin_with_stats(
            sample_data, x="dose", y="len", add_jitter=False
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_data: pd.DataFrame) -> None:
        """Test custom figure size."""
        figsize = (8, 10)
        fig, ax = plot_violin_with_stats(
            sample_data, x="dose", y="len", figsize=figsize
        )

        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]
        plt.close(fig)

    def test_small_dataset(self, small_data: pd.DataFrame) -> None:
        """Test with minimal dataset."""
        fig, ax = plot_violin_with_stats(small_data, x="group", y="value")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_group(self) -> None:
        """Test with single group (edge case)."""
        df = pd.DataFrame({"group": ["A"] * 5, "value": [1, 2, 3, 4, 5]})

        fig, ax = plot_violin_with_stats(df, x="group", y="value")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_parametric_violin_plot(
        self, sample_data: pd.DataFrame, sample_palette: Dict[float, str]
    ) -> None:
        """Test parametric violin plot."""
        fig, ax = plot_violin_with_stats(
            sample_data, x="dose", y="len", palette=sample_palette, parametric=True
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_invalid_parametric_parameter(self, sample_data: pd.DataFrame) -> None:
        """Test invalid parametric parameter."""
        with pytest.raises(AssertionError, match="parametric must be a boolean"):
            plot_violin_with_stats(sample_data, x="dose", y="len", parametric="invalid")

    def test_invalid_figsize(self, sample_data: pd.DataFrame) -> None:
        """Test invalid figsize parameter."""
        with pytest.raises(AssertionError, match="figsize must be a tuple"):
            plot_violin_with_stats(sample_data, x="dose", y="len", figsize="invalid")

    def test_negative_figsize_scale(self, sample_data: pd.DataFrame) -> None:
        """Test negative figsize_scale parameter."""
        with pytest.raises(AssertionError, match="figsize_scale must be positive"):
            plot_violin_with_stats(sample_data, x="dose", y="len", figsize_scale=-1)

    def test_negative_jitter_std(self, sample_data: pd.DataFrame) -> None:
        """Test negative jitter_std parameter."""
        with pytest.raises(AssertionError, match="jitter_std must be non-negative"):
            plot_violin_with_stats(sample_data, x="dose", y="len", jitter_std=-1)

    def test_zero_violin_width(self, sample_data: pd.DataFrame) -> None:
        """Test zero violin_width parameter."""
        with pytest.raises(AssertionError, match="violin_width must be positive"):
            plot_violin_with_stats(sample_data, x="dose", y="len", violin_width=0)

    def test_zero_box_width(self, sample_data: pd.DataFrame) -> None:
        """Test zero box_width parameter."""
        with pytest.raises(AssertionError, match="box_width must be positive"):
            plot_violin_with_stats(sample_data, x="dose", y="len", box_width=0)


class TestPlotBoxplotWithStats:
    """Test the plot_boxplot_with_stats function."""

    def test_basic_boxplot(
        self, sample_data: pd.DataFrame, sample_palette: Dict[float, str]
    ) -> None:
        """Test basic boxplot creation."""
        fig, ax = plot_boxplot_with_stats(
            sample_data, x="dose", y="len", palette=sample_palette
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "dose"
        assert ax.get_ylabel() == "len"
        plt.close(fig)

    def test_custom_labels_boxplot(self, sample_data: pd.DataFrame) -> None:
        """Test custom axis labels for boxplot."""
        fig, ax = plot_boxplot_with_stats(
            sample_data,
            x="dose",
            y="len",
            x_label="Dose (mg)",
            y_label="Length (units)",
        )

        assert ax.get_xlabel() == "Dose (mg)"
        assert ax.get_ylabel() == "Length (units)"
        plt.close(fig)

    def test_custom_order_boxplot(self, sample_data: pd.DataFrame) -> None:
        """Test custom ordering for boxplot."""
        fig, ax = plot_boxplot_with_stats(
            sample_data, x="dose", y="len", order=[2.0, 1.0, 0.5]
        )

        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert labels == ["2.0", "1.0", "0.5"]
        plt.close(fig)

    def test_custom_figsize_boxplot(self, sample_data: pd.DataFrame) -> None:
        """Test custom figure size for boxplot."""
        figsize = (8, 10)
        fig, ax = plot_boxplot_with_stats(
            sample_data, x="dose", y="len", figsize=figsize
        )

        assert fig.get_size_inches()[0] == figsize[0]
        assert fig.get_size_inches()[1] == figsize[1]
        plt.close(fig)

    def test_small_dataset_boxplot(self, small_data: pd.DataFrame) -> None:
        """Test boxplot with minimal dataset."""
        fig, ax = plot_boxplot_with_stats(small_data, x="group", y="value")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_parametric_boxplot(
        self, sample_data: pd.DataFrame, sample_palette: Dict[float, str]
    ) -> None:
        """Test parametric boxplot."""
        fig, ax = plot_boxplot_with_stats(
            sample_data, x="dose", y="len", palette=sample_palette, parametric=True
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_invalid_parametric_parameter_boxplot(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test invalid parametric parameter for boxplot."""
        with pytest.raises(AssertionError, match="parametric must be a boolean"):
            plot_boxplot_with_stats(
                sample_data, x="dose", y="len", parametric="invalid"
            )

    def test_invalid_figsize_boxplot(self, sample_data: pd.DataFrame) -> None:
        """Test invalid figsize parameter for boxplot."""
        with pytest.raises(AssertionError, match="figsize must be a tuple"):
            plot_boxplot_with_stats(sample_data, x="dose", y="len", figsize="invalid")

    def test_negative_jitter_std_boxplot(self, sample_data: pd.DataFrame) -> None:
        """Test negative jitter_std parameter for boxplot."""
        with pytest.raises(AssertionError, match="jitter_std must be non-negative"):
            plot_boxplot_with_stats(sample_data, x="dose", y="len", jitter_std=-1)

    def test_zero_box_width_boxplot(self, sample_data: pd.DataFrame) -> None:
        """Test zero box_width parameter for boxplot."""
        with pytest.raises(AssertionError, match="box_width must be positive"):
            plot_boxplot_with_stats(sample_data, x="dose", y="len", box_width=0)


class TestPlotShift:
    """Test the plot_shift function."""

    @pytest.fixture
    def shift_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sample data for shift plot."""
        np.random.seed(0)
        x = np.random.randn(30)
        y = np.random.randn(30) + 1
        return x, y

    def test_basic_shift_plot(self, shift_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic shift plot creation."""
        x, y = shift_data
        fig = plot_shift(x, y)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_paired_shift_plot(self, shift_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test paired shift plot."""
        x, y = shift_data
        fig = plot_shift(x, y, paired=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_quantile_diff(
        self, shift_data: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Test shift plot with quantile difference subplot."""
        x, y = shift_data
        fig = plot_shift(x, y, show_quantile_diff=True)
        assert len(fig.axes) > 1  # Should have more than one axes
        plt.close(fig)

    def test_show_quantiles(self, shift_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test shift plot with quantile connection lines."""
        x, y = shift_data
        fig = plot_shift(x, y, show_quantiles=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_parametric_test(self, shift_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Test shift plot with parametric test."""
        x, y = shift_data
        fig = plot_shift(x, y, parametric=True)
        ax = fig.axes[0]
        assert "t-test" in ax.get_title()
        plt.close(fig)


class TestDataValidation:
    """Test data validation and edge cases."""

    def test_missing_values(self) -> None:
        """Test handling of missing values."""
        df = pd.DataFrame(
            {"group": ["A", "A", "B", "B", "A"], "value": [1, 2, np.nan, 4, 5]}
        )

        fig, ax = plot_violin_with_stats(df, x="group", y="value")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig, ax = plot_boxplot_with_stats(df, x="group", y="value")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_groups(self) -> None:
        """Test handling of empty groups after filtering."""
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        # This should handle the groups gracefully
        fig, ax = plot_violin_with_stats(df, x="group", y="value")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestValidationFunctions:
    """Test the validation functions."""

    def test_validate_inputs_valid_data(self, sample_data: pd.DataFrame) -> None:
        """Test validation with valid data."""
        # Should not raise any exception
        _validate_inputs(sample_data, "dose", "len")

    def test_validate_inputs_empty_dataframe(self) -> None:
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(AssertionError, match="DataFrame cannot be empty"):
            _validate_inputs(df, "x", "y")

    def test_validate_inputs_missing_column(self, sample_data: pd.DataFrame) -> None:
        """Test validation with missing column."""
        with pytest.raises(AssertionError, match="Column 'missing' not found"):
            _validate_inputs(sample_data, "missing", "len")

    def test_validate_inputs_non_numeric_y(self) -> None:
        """Test validation with non-numeric y column."""
        df = pd.DataFrame({"x": ["A", "B"], "y": ["cat", "dog"]})
        with pytest.raises(AssertionError, match="must be numeric"):
            _validate_inputs(df, "x", "y")

    def test_validate_inputs_all_nan(self) -> None:
        """Test validation when all data is NaN."""
        df = pd.DataFrame({"x": ["A", "B"], "y": [np.nan, np.nan]})
        with pytest.raises(AssertionError, match="No valid data remaining"):
            _validate_inputs(df, "x", "y")

    def test_validate_inputs_invalid_order(self, sample_data: pd.DataFrame) -> None:
        """Test validation with invalid order."""
        with pytest.raises(AssertionError, match="All items in order must exist"):
            _validate_inputs(sample_data, "dose", "len", order=[99])


class TestStatisticalTests:
    """Test the statistical test functions."""

    def test_perform_statistical_tests_parametric(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test parametric statistical tests."""
        levels = sorted(sample_data["dose"].unique())
        groups: List[np.ndarray] = [
            sample_data[sample_data["dose"] == lvl]["len"].dropna().values
            for lvl in levels
        ]

        global_stat, global_p, pairwise = _perform_statistical_tests(
            groups, parametric=True
        )

        assert not np.isnan(global_stat)
        assert not np.isnan(global_p)
        assert len(pairwise) == 3  # 3 groups = 3 pairwise comparisons

    def test_perform_statistical_tests_nonparametric(
        self, sample_data: pd.DataFrame
    ) -> None:
        """Test non-parametric statistical tests."""
        levels = sorted(sample_data["dose"].unique())
        groups: List[np.ndarray] = [
            sample_data[sample_data["dose"] == lvl]["len"].dropna().values
            for lvl in levels
        ]

        global_stat, global_p, pairwise = _perform_statistical_tests(
            groups, parametric=False
        )

        assert not np.isnan(global_stat)
        assert not np.isnan(global_p)
        assert len(pairwise) == 3  # 3 groups = 3 pairwise comparisons

    def test_perform_statistical_tests_single_group(self) -> None:
        """Test statistical tests with single group."""
        groups = [np.array([1, 2, 3, 4, 5])]

        global_stat, global_p, pairwise = _perform_statistical_tests(groups)

        assert np.isnan(global_stat)
        assert np.isnan(global_p)
        assert len(pairwise) == 0

    def test_perform_statistical_tests_empty_groups(self) -> None:
        """Test statistical tests with empty groups."""
        groups: List[np.ndarray] = []

        with pytest.raises(AssertionError, match="At least one group is required"):
            _perform_statistical_tests(groups)


class TestDatasets:
    """Test the datasets module."""

    def test_load_iris(self) -> None:
        """Test loading iris dataset."""
        from ggpubpy.datasets import load_iris

        iris = load_iris()
        assert isinstance(iris, pd.DataFrame)
        assert iris.shape == (150, 5)
        assert list(iris.columns) == [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        ]
        assert set(iris["species"].unique()) == {"setosa", "versicolor", "virginica"}

    def test_get_iris_palette(self) -> None:
        """Test iris palette function."""
        from ggpubpy.datasets import get_iris_palette

        palette = get_iris_palette()
        assert isinstance(palette, dict)
        assert len(palette) == 3
        assert "setosa" in palette
        assert "versicolor" in palette
        assert "virginica" in palette

    def test_list_datasets(self) -> None:
        """Test list datasets function."""
        from ggpubpy.datasets import list_datasets

        datasets = list_datasets()
        assert isinstance(datasets, dict)
        assert "iris" in datasets
        assert "description" in datasets["iris"]


class TestDefaultPalette:
    """Test the default palette system."""

    def test_default_palette_import(self) -> None:
        """Test importing default palette."""
        from ggpubpy import DEFAULT_PALETTE

        assert isinstance(DEFAULT_PALETTE, list)
        assert len(DEFAULT_PALETTE) == 10
        assert all(color.startswith("#") for color in DEFAULT_PALETTE)

    def test_automatic_palette_generation(self, sample_data: pd.DataFrame) -> None:
        """Test automatic palette generation."""
        fig, ax = plot_violin_with_stats(sample_data, x="dose", y="len")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_mixed_palette_usage(self, sample_data: pd.DataFrame) -> None:
        """Test using partial custom palette with defaults."""
        partial_palette = {0.5: "#FF0000"}  # Only specify one color

        fig, ax = plot_violin_with_stats(
            sample_data, x="dose", y="len", palette=partial_palette
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
