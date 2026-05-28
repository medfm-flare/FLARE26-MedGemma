#!/usr/bin/env python3
"""Plot aggregate model performance from evaluation scores under output/.

The script writes an SVG using only the Python standard library so it can run
on login nodes or minimal cluster environments without a plotting stack.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Run:
    label: str
    scores_path: Path
    color: str


@dataclass(frozen=True)
class Panel:
    title: str
    primary: str
    metrics: tuple[tuple[str, str], ...]
    higher_is_better: bool = True


PANELS: tuple[Panel, ...] = (
    Panel(
        title="Classification",
        primary="Balanced Accuracy",
        metrics=(
            ("Accuracy", "classification_accuracy"),
            ("Balanced Accuracy", "balanced_accuracy"),
            ("F1 Score", "classification_f1_score"),
        ),
    ),
    Panel(
        title="Detection",
        primary="F1 Score",
        metrics=(
            ("Precision", "detection_precision_iou_0.5"),
            ("Recall", "detection_recall_iou_0.5"),
            ("F1 Score", "detection_f1_iou_0.5"),
        ),
    ),
    Panel(
        title="Regression",
        primary="Mean Absolute Error",
        metrics=(
            ("Mean Absolute Error", "regression_mean_absolute_error"),
            ("Root Mean Squared Error", "regression_root_mean_squared_error"),
        ),
        higher_is_better=False,
    ),
    Panel(
        title="Multi Label",
        primary="F1 Score",
        metrics=(
            ("Precision", "multi_label_precision"),
            ("Recall", "multi_label_recall"),
            ("F1 Score", "multi_label_f1_score"),
        ),
    ),
    Panel(
        title="Report Generation",
        primary="GREEN / CRIMSON Score",
        metrics=(
            ("GREEN Score", "green_score"),
            ("CRIMSON Score", "crimson_score"),
        ),
    ),
    Panel(
        title="Counting",
        primary="Mean Absolute Error",
        metrics=(
            ("Mean Absolute Error", "cell_counting_mean_absolute_error"),
            ("Root Mean Squared Error", "cell_counting_root_mean_squared_error"),
        ),
        higher_is_better=False,
    ),
)


DEFAULT_RUNS: tuple[tuple[str, str, str], ...] = (
    ("Base MG1", "flare-medgemma1-base-eval", "#ef7f9a"),
    ("FT MG1", "flare-medgemma1-eval", "#c7aa4b"),
    ("Base MG1.5", "flare-medgemma-base-eval", "#6fa8dc"),
    ("FT MG1.5", "flare-medgemma-eval", "#8fbc6a"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a six-panel performance comparison plot from evaluation scores.json files."
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"), help="Directory containing *-eval folders.")
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("output/model_performance_comparison.svg"),
        help="Where to write the SVG figure.",
    )
    parser.add_argument(
        "--no-auto-log",
        action="store_true",
        help="Disable automatic log scaling for error panels with extreme value ranges.",
    )
    return parser.parse_args()


def load_metrics(scores_path: Path) -> dict[str, float]:
    with scores_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    metrics = payload.get("mean") or payload.get("mean_metrics") or payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"No aggregate metrics found in {scores_path}")
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def format_value(value: float) -> str:
    if math.isnan(value):
        return "NA"
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 10000 or abs_value < 0.001:
        return f"{value:.2e}"
    if abs_value >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def format_delta(base: float, value: float, higher_is_better: bool) -> str:
    if not math.isfinite(base) or not math.isfinite(value) or base == 0:
        return ""
    delta = (value - base) / abs(base)
    if not higher_is_better:
        delta *= -1
    return f"{delta:+.0%}"


def should_use_log(values: list[float], panel: Panel, auto_log: bool) -> bool:
    positives = [value for value in values if math.isfinite(value) and value > 0]
    if not auto_log or panel.higher_is_better or len(positives) < 2:
        return False
    return max(positives) / min(positives) > 1000


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 13,
    weight: str = "400",
    fill: str = "#2b2b2b",
    anchor: str = "middle",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" font-family="Times New Roman, Times, serif"{transform}>'
        f"{html.escape(value)}</text>"
    )


def multiline_text(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: int = 12,
    line_height: float = 15,
    fill: str = "#2b2b2b",
    anchor: str = "middle",
    weight: str = "400",
) -> str:
    escaped_lines = [html.escape(line) for line in lines]
    tspans = []
    for index, line_text in enumerate(escaped_lines):
        dy = 0 if index == 0 else line_height
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{line_text}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" font-family="Times New Roman, Times, serif">'
        f"{''.join(tspans)}</text>"
    )


def wrap_label(label: str) -> list[str]:
    manual_wraps = {
        "Mean Absolute Error": ["Mean Absolute", "Error"],
        "Root Mean Squared Error": ["Root Mean", "Squared Error"],
        "Balanced Accuracy": ["Balanced", "Accuracy"],
        "GREEN Score": ["GREEN", "Score"],
        "CRIMSON Score": ["CRIMSON", "Score"],
    }
    return manual_wraps.get(label, [label])


def rect(x: float, y: float, width: float, height: float, fill: str, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" opacity="{opacity:.2f}" />'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#ffffff", width: float = 1.0) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.1f}" />'


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def linear_ticks(max_value: float) -> list[float]:
    if max_value <= 0 or not math.isfinite(max_value):
        return [0.0, 1.0]
    exponent = math.floor(math.log10(max_value))
    fraction = max_value / 10**exponent
    if fraction <= 1:
        step = 0.2 * 10**exponent
    elif fraction <= 2:
        step = 0.5 * 10**exponent
    elif fraction <= 5:
        step = 1.0 * 10**exponent
    else:
        step = 2.0 * 10**exponent
    top = math.ceil(max_value / step) * step
    ticks = []
    value = 0.0
    while value <= top + step / 2:
        ticks.append(value)
        value += step
    return ticks


def log_ticks(values: list[float]) -> list[float]:
    positives = [value for value in finite_values(values) if value > 0]
    if not positives:
        return [1.0]
    low = math.floor(math.log10(min(positives)))
    high = math.ceil(math.log10(max(positives)))
    if high - low <= 6:
        powers = list(range(low, high + 1))
    else:
        step = math.ceil((high - low) / 5)
        powers = list(range(low, high + 1, step))
        if powers[-1] != high:
            powers.append(high)
    return [10.0**power for power in powers]


def render_panel(
    panel: Panel,
    runs: list[Run],
    metrics_by_run: dict[str, dict[str, float]],
    *,
    panel_x: float,
    panel_y: float,
    panel_width: float,
    panel_height: float,
    auto_log: bool,
) -> str:
    keys = [key for _label, key in panel.metrics]
    metric_labels = [label for label, _key in panel.metrics]
    all_values = [metrics_by_run[run.label].get(key, float("nan")) for key in keys for run in runs]
    log_scale = should_use_log(all_values, panel, auto_log)
    plot_x = panel_x + 104
    plot_y = panel_y + 104
    plot_width = panel_width - 140
    plot_height = panel_height - 250
    baseline_y = plot_y + plot_height
    pieces = [
        rect(panel_x, panel_y, panel_width, panel_height, "#e9e9f2"),
        text(panel_x + panel_width / 2, panel_y + 38, panel.title, size=30, weight="700"),
        text(panel_x + panel_width / 2, panel_y + 72, f"(Primary: {panel.primary})", size=23, weight="700"),
    ]

    values = finite_values(all_values)
    if log_scale:
        positives = [value for value in values if value > 0]
        axis_min = 10.0 ** math.floor(math.log10(min(positives)))
        axis_max = 10.0 ** math.ceil(math.log10(max(positives)))

        def y_for(value: float) -> float:
            safe_value = max(value, axis_min)
            ratio = (math.log10(safe_value) - math.log10(axis_min)) / (math.log10(axis_max) - math.log10(axis_min))
            return baseline_y - ratio * plot_height

        ticks = log_ticks(values)
    else:
        axis_min = 0.0
        axis_max = max(values) * 1.18 if values else 1.0
        ticks = linear_ticks(axis_max)
        axis_max = ticks[-1] if ticks else axis_max

        def y_for(value: float) -> float:
            return baseline_y - ((value - axis_min) / (axis_max - axis_min)) * plot_height

    for tick in ticks:
        if tick < axis_min or tick > axis_max:
            continue
        y = y_for(tick)
        pieces.append(line(plot_x, y, plot_x + plot_width, y, stroke="#ffffff", width=1))
        pieces.append(text(plot_x - 16, y + 6, format_value(tick), size=16, anchor="end", fill="#333333"))
    pieces.append(line(plot_x, plot_y, plot_x, baseline_y, stroke="#555555", width=1.6))
    pieces.append(line(plot_x, baseline_y, plot_x + plot_width, baseline_y, stroke="#555555", width=1.6))
    y_label = "Value (log scale)" if log_scale else "Value"
    pieces.append(text(panel_x + 35, plot_y + plot_height / 2, y_label, size=21, rotate=-90))

    group_width = plot_width / len(keys)
    bar_area_width = group_width * 0.72
    bar_width = bar_area_width / len(runs)
    for metric_index, (metric_label, key) in enumerate(zip(metric_labels, keys)):
        group_center = plot_x + group_width * (metric_index + 0.5)
        x_start = group_center - bar_area_width / 2
        metric_values = [metrics_by_run[run.label].get(key, float("nan")) for run in runs]
        for run_index, run in enumerate(runs):
            value = metric_values[run_index]
            if not math.isfinite(value):
                continue
            bar_x = x_start + run_index * bar_width
            bar_top = y_for(value)
            bar_height = max(1.0, baseline_y - bar_top)
            pieces.append(rect(bar_x + 1, bar_top, bar_width - 2, bar_height, run.color, opacity=0.88))
            if bar_height > 84:
                pieces.append(
                    text(
                        bar_x + bar_width / 2,
                        bar_top + bar_height / 2,
                        format_value(value),
                        size=17,
                        weight="700",
                        fill="#ffffff",
                        rotate=-90,
                    )
                )
            elif bar_height > 28 and bar_width >= 42:
                pieces.append(
                    text(
                        bar_x + bar_width / 2,
                        max(plot_y + 18, bar_top - 12),
                        format_value(value),
                        size=15,
                        weight="700",
                        fill="#2b2b2b",
                    )
                )

        for base_index, ft_index in ((0, 1), (2, 3)):
            if ft_index >= len(metric_values):
                continue
            delta = format_delta(metric_values[base_index], metric_values[ft_index], panel.higher_is_better)
            if not delta:
                continue
            ft_value = metric_values[ft_index]
            if not math.isfinite(ft_value):
                continue
            ft_x = x_start + ft_index * bar_width + bar_width / 2
            pieces.append(text(ft_x, max(plot_y + 22, y_for(ft_value) - 30), delta, size=16, weight="700"))

        pieces.append(multiline_text(group_center, baseline_y + 54, wrap_label(metric_label), size=19, line_height=23))
    return "\n".join(pieces)


def render_svg(runs: list[Run], metrics_by_run: dict[str, dict[str, float]], auto_log: bool) -> str:
    width = 2400
    height = 3600
    columns = 2
    margin_x = 95
    margin_y = 165
    bottom_margin = 90
    gap_x = 90
    gap_y = 105
    rows = math.ceil(len(PANELS) / columns)
    panel_width = (width - 2 * margin_x - (columns - 1) * gap_x) / columns
    panel_height = (height - margin_y - bottom_margin - (rows - 1) * gap_y) / rows
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
        f'style="max-width:100%;height:auto;background:#ffffff">',
        rect(0, 0, width, height, "#ffffff"),
        text(width / 2, 58, "Model Performance Comparison: Base vs Finetuned", size=43, weight="700"),
    ]
    legend_width = 230 * len(runs)
    legend_x = width / 2 - legend_width / 2
    for index, run in enumerate(runs):
        item_x = legend_x + index * 230
        pieces.append(rect(item_x, 96, 42, 18, run.color, opacity=0.88))
        pieces.append(text(item_x + 58, 112, run.label, size=22, anchor="start"))
    for index, panel in enumerate(PANELS):
        row, col = divmod(index, columns)
        x = margin_x + col * (panel_width + gap_x)
        y = margin_y + row * (panel_height + gap_y)
        pieces.append(
            render_panel(
                panel,
                runs,
                metrics_by_run,
                panel_x=x,
                panel_y=y,
                panel_width=panel_width,
                panel_height=panel_height,
                auto_log=auto_log,
            )
        )
    pieces.append("</svg>")
    return "\n".join(pieces)


def main() -> None:
    args = parse_args()
    runs = [
        Run(label=label, scores_path=args.output_root / eval_dir / "scores.json", color=color)
        for label, eval_dir, color in DEFAULT_RUNS
    ]
    metrics_by_run = {run.label: load_metrics(run.scores_path) for run in runs}
    svg = render_svg(runs, metrics_by_run, auto_log=not args.no_auto_log)
    args.figure_path.parent.mkdir(parents=True, exist_ok=True)
    args.figure_path.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.figure_path}")


if __name__ == "__main__":
    main()
