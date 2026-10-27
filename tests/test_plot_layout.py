import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from atomity_forecasting.plotting import PLOTTERS, save_figure


def test_shared_legend_does_not_cover_panels_titles_or_axis_labels(tmp_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5))
    for ax, origin in zip(axes, (50, 70, 86)):
        ax.plot([0, 1, 2], [9.2, 9.8, 9.4], label="Actual validation CPU")
        ax.plot([0, 1, 2], [9.3] * 3, "--", label="Selected = baseline: persistence")
        ax.set(title=f"Training-only backtest, origin prefix {origin}",
               xlabel="Minutes since training start", ylabel="Aggregate CPU (%)")
        ax.legend()
    before = [[line.get_ydata().copy() for line in ax.lines] for ax in axes]
    save_figure(fig, tmp_path / "backtest.png")
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    assert len(fig.legends) == 1
    legend = fig.legends[0].get_window_extent(renderer)
    assert len(fig.legends[0].get_texts()) == 2
    panels = [ax.get_tightbbox(renderer) for ax in axes]
    for index, (ax, bounds) in enumerate(zip(axes, panels)):
        assert ax.get_legend() is None
        assert not legend.overlaps(bounds)
        for other in panels[index + 1:]:
            assert not bounds.overlaps(other)
        for line, expected in zip(ax.lines, before[index]):
            np.testing.assert_array_equal(line.get_ydata(), expected)


def test_multiline_legend_stays_inside_figure_and_outside_plot(tmp_path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for index, label in enumerate(("P50-P75", "P75-P90", "P90-P95", "P95-P99",
                                   "Mean VM average CPU", "Maximum VM average CPU")):
        ax.plot([0, 1], [index, index + 1], label=label)
    ax.set(title="Training cross-sectional quantiles", xlabel="Minutes", ylabel="CPU (%)")
    save_figure(fig, tmp_path / "fan.png")
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(renderer)
    assert fig.bbox.contains(legend.x0, legend.y0)
    assert fig.bbox.contains(legend.x1, legend.y1)
    assert not legend.overlaps(ax.get_tightbbox(renderer))


def test_plot_without_legend_keeps_no_empty_legend(tmp_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow([[.2, .3], [.1, .25]])
    save_figure(fig, tmp_path / "matrix.png")
    assert not fig.legends


def test_all_figures_have_one_canonical_data_driven_implementation():
    import inspect

    assert len(PLOTTERS) == len(set(PLOTTERS.values())) == 12
    for plotter in PLOTTERS.values():
        assert plotter.__module__ == "atomity_forecasting.plotting"
        source = inspect.getsource(plotter)
        assert "plt.subplots(" in source and "return fig" in source
        assert "read_csv" not in source and "imread" not in source and "open(" not in source


def test_reexport_does_not_duplicate_shared_legend(tmp_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([0, 1], [1, 2], label="Training data")
    save_figure(fig, tmp_path / "first.png")
    save_figure(fig, tmp_path / "second.png")
    assert len(fig.legends) == 1
    assert len(fig.legends[0].get_texts()) == 1
