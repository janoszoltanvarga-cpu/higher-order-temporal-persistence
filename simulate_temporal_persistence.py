#!/usr/bin/env python3
"""
Simulate and visualize the temporal persistence threshold model from the paper.

This version compares:
1. The reduced mean-field ODE from the manuscript.
2. An explicit finite-size temporal network / hypergraph simulator with:
   - a sparse pairwise graph,
   - a fixed set of candidate triangles,
   - Markov-persistent triangle activity,
   - synchronous SIS updates.

Outputs:
- phase_scan.csv
- trajectories.csv
- phase_scan.svg
- phase_scan_zoom.svg
- trajectory_below_threshold.svg
- trajectory_above_threshold.svg
- summary.txt

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class ModelParameters:
    mu: float = 1.0
    lam: float = 0.6
    beta_tri: float = 1.0
    triangles_per_node: int = 8
    a: float = 0.5


@dataclass(frozen=True)
class ThresholdInfo:
    eta_c: float
    p_c: Optional[float]
    regime: str


@dataclass(frozen=True)
class PhasePoint:
    p: float
    x0: float
    eta: float
    theory_final: float
    explicit_final_mean: float
    explicit_final_std: float


@dataclass(frozen=True)
class Series:
    label: str
    x: Sequence[float]
    y: Sequence[float]
    color: str
    dash: Optional[str] = None
    marker: bool = False
    marker_radius: float = 3.0


@dataclass(frozen=True)
class NetworkStructure:
    n_nodes: int
    edges: Sequence[tuple[int, int]]
    triangles: Sequence[tuple[int, int, int]]
    avg_pair_degree: float
    avg_triangle_incidence: float


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def linspace(start: float, stop: float, count: int) -> List[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def tail_average(values: Sequence[float], fraction: float = 0.2) -> float:
    if not values:
        return 0.0
    tail_count = max(1, int(len(values) * fraction))
    return mean(values[-tail_count:])


def effective_eta(params: ModelParameters, p: float) -> float:
    return params.beta_tri * params.triangles_per_node * params.a * (params.a + p * (1.0 - params.a))


def critical_eta(lam: float, mu: float) -> float:
    return 2.0 * mu - lam + 2.0 * math.sqrt(mu * (mu - lam))


def threshold_info(params: ModelParameters) -> ThresholdInfo:
    eta_c = critical_eta(params.lam, params.mu)
    eta_min = params.beta_tri * params.triangles_per_node * params.a * params.a
    eta_max = params.beta_tri * params.triangles_per_node * params.a

    if eta_c <= eta_min:
        return ThresholdInfo(eta_c=eta_c, p_c=0.0, regime="already_bistable")
    if eta_c >= eta_max:
        return ThresholdInfo(eta_c=eta_c, p_c=None, regime="unattainable")

    p_c = (eta_c / (params.beta_tri * params.triangles_per_node * params.a) - params.a) / (1.0 - params.a)
    return ThresholdInfo(eta_c=eta_c, p_c=p_c, regime="interior")


def mean_field_rhs(x: float, params: ModelParameters, p: float) -> float:
    eta = effective_eta(params, p)
    return -params.mu * x + (1.0 - x) * (params.lam * x + eta * x * x)


def rk4_step(x: float, dt: float, params: ModelParameters, p: float) -> float:
    k1 = mean_field_rhs(x, params, p)
    k2 = mean_field_rhs(clamp(x + 0.5 * dt * k1), params, p)
    k3 = mean_field_rhs(clamp(x + 0.5 * dt * k2), params, p)
    k4 = mean_field_rhs(clamp(x + dt * k3), params, p)
    return clamp(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def integrate_mean_field(
    params: ModelParameters,
    p: float,
    x0: float,
    steps: int,
    dt: float,
) -> tuple[List[float], List[float]]:
    times = [i * dt for i in range(steps + 1)]
    xs = [clamp(x0)]
    x = clamp(x0)
    for _ in range(steps):
        x = rk4_step(x, dt, params, p)
        xs.append(x)
    return times, xs


def initialize_infected(n_nodes: int, x0: float, rng: random.Random) -> List[bool]:
    infected = [False] * n_nodes
    n_seed = max(0, min(n_nodes, round(n_nodes * x0)))
    if n_seed > 0:
        for idx in rng.sample(range(n_nodes), n_seed):
            infected[idx] = True
    return infected


def sample_pair_graph(n_nodes: int, target_degree: float, rng: random.Random) -> tuple[List[tuple[int, int]], float]:
    if n_nodes < 2:
        return [], 0.0
    edge_prob = clamp(target_degree / max(1.0, n_nodes - 1.0), 0.0, 1.0)
    edges: List[tuple[int, int]] = []
    degree_sum = 0
    for u in range(n_nodes - 1):
        for v in range(u + 1, n_nodes):
            if rng.random() < edge_prob:
                edges.append((u, v))
                degree_sum += 2
    avg_degree = degree_sum / n_nodes if n_nodes else 0.0
    return edges, avg_degree


def sample_unique_triangles(n_nodes: int, triangle_count: int, rng: random.Random) -> List[tuple[int, int, int]]:
    if triangle_count <= 0:
        return []
    if n_nodes < 3:
        raise ValueError("Need at least 3 nodes to sample triangles.")

    triangles = set()
    max_attempts = max(1000, 50 * triangle_count)
    attempts = 0
    while len(triangles) < triangle_count and attempts < max_attempts:
        attempts += 1
        tri = tuple(sorted(rng.sample(range(n_nodes), 3)))
        triangles.add(tri)

    if len(triangles) < triangle_count:
        raise RuntimeError(
            f"Could only sample {len(triangles)} unique triangles out of requested {triangle_count}. "
            "Try fewer nodes or fewer triangles per node."
        )

    return sorted(triangles)


def build_random_structure(
    n_nodes: int,
    pair_degree: float,
    triangles_per_node: int,
    seed: int,
) -> NetworkStructure:
    rng = random.Random(seed)
    edges, avg_pair_degree = sample_pair_graph(n_nodes, pair_degree, rng)
    triangle_count = max(1, round(n_nodes * triangles_per_node / 3.0))
    triangles = sample_unique_triangles(n_nodes, triangle_count, rng)
    avg_triangle_incidence = 3.0 * len(triangles) / n_nodes if n_nodes else 0.0
    return NetworkStructure(
        n_nodes=n_nodes,
        edges=edges,
        triangles=triangles,
        avg_pair_degree=avg_pair_degree,
        avg_triangle_incidence=avg_triangle_incidence,
    )


def build_structure_ensemble(
    n_structures: int,
    n_nodes: int,
    pair_degree: float,
    triangles_per_node: int,
    seed: int,
) -> List[NetworkStructure]:
    structures = []
    for rep in range(n_structures):
        structures.append(
            build_random_structure(
                n_nodes=n_nodes,
                pair_degree=pair_degree,
                triangles_per_node=triangles_per_node,
                seed=seed + 1000 * rep,
            )
        )
    return structures


def initialize_triangle_activity(
    n_triangles: int,
    a: float,
    p: float,
    rng: random.Random,
) -> tuple[List[bool], List[bool]]:
    prev_active = [False] * n_triangles
    curr_active = [False] * n_triangles
    off_to_on = (1.0 - p) * a
    on_to_on = p + (1.0 - p) * a

    for idx in range(n_triangles):
        prev = rng.random() < a
        curr = rng.random() < (on_to_on if prev else off_to_on)
        prev_active[idx] = prev
        curr_active[idx] = curr
    return prev_active, curr_active


def simulate_explicit_temporal_hypergraph(
    params: ModelParameters,
    structure: NetworkStructure,
    p: float,
    x0: float,
    steps: int,
    dt: float,
    seed: int,
) -> tuple[List[float], List[float]]:
    rng = random.Random(seed)
    infected = initialize_infected(structure.n_nodes, x0, rng)
    prev_active, curr_active = initialize_triangle_activity(len(structure.triangles), params.a, p, rng)

    times = [i * dt for i in range(steps + 1)]
    xs = [sum(1 for state in infected if state) / structure.n_nodes]

    recover_prob = 1.0 - math.exp(-params.mu * dt)
    off_to_on = (1.0 - p) * params.a
    on_to_on = p + (1.0 - p) * params.a
    pair_beta = params.lam / structure.avg_pair_degree if structure.avg_pair_degree > 0.0 else 0.0
    n_nodes = structure.n_nodes
    triangles = structure.triangles
    edges = structure.edges

    for _ in range(steps):
        pair_counts = [0] * n_nodes
        tri_counts = [0] * n_nodes

        for u, v in edges:
            if infected[u]:
                pair_counts[v] += 1
            if infected[v]:
                pair_counts[u] += 1

        for tri_idx, (u, v, w) in enumerate(triangles):
            if not (prev_active[tri_idx] and curr_active[tri_idx]):
                continue
            iu = infected[u]
            iv = infected[v]
            iw = infected[w]
            if iv and iw:
                tri_counts[u] += 1
            if iu and iw:
                tri_counts[v] += 1
            if iu and iv:
                tri_counts[w] += 1

        next_infected = infected[:]
        for node in range(n_nodes):
            if infected[node]:
                next_infected[node] = rng.random() >= recover_prob
                continue
            total_rate = pair_beta * pair_counts[node] + params.beta_tri * tri_counts[node]
            infect_prob = 1.0 - math.exp(-total_rate * dt)
            next_infected[node] = rng.random() < infect_prob

        new_active = [False] * len(triangles)
        for tri_idx, is_active in enumerate(curr_active):
            new_active[tri_idx] = rng.random() < (on_to_on if is_active else off_to_on)

        infected = next_infected
        prev_active = curr_active
        curr_active = new_active
        xs.append(sum(1 for state in infected if state) / n_nodes)

    return times, xs


def average_trajectories(trajectories: Sequence[Sequence[float]]) -> List[float]:
    if not trajectories:
        return []
    length = len(trajectories[0])
    avg = [0.0] * length
    for series in trajectories:
        for idx, value in enumerate(series):
            avg[idx] += value
    return [value / len(trajectories) for value in avg]


def scan_phase_diagram(
    params: ModelParameters,
    structures: Sequence[NetworkStructure],
    p_values: Sequence[float],
    x0_values: Sequence[float],
    steps: int,
    dt: float,
    seed: int,
) -> List[PhasePoint]:
    results: List[PhasePoint] = []
    for p_idx, p in enumerate(p_values):
        eta = effective_eta(params, p)
        for x_idx, x0 in enumerate(x0_values):
            _, theory_xs = integrate_mean_field(params, p, x0, steps, dt)
            theory_final = tail_average(theory_xs)
            finals = []
            for rep, structure in enumerate(structures):
                run_seed = seed + 100000 * p_idx + 5000 * x_idx + rep
                _, explicit_xs = simulate_explicit_temporal_hypergraph(
                    params=params,
                    structure=structure,
                    p=p,
                    x0=x0,
                    steps=steps,
                    dt=dt,
                    seed=run_seed,
                )
                finals.append(tail_average(explicit_xs))
            results.append(
                PhasePoint(
                    p=p,
                    x0=x0,
                    eta=eta,
                    theory_final=theory_final,
                    explicit_final_mean=mean(finals),
                    explicit_final_std=std(finals),
                )
            )
    return results


def collect_average_trajectory(
    params: ModelParameters,
    structures: Sequence[NetworkStructure],
    p: float,
    x0: float,
    steps: int,
    dt: float,
    seed: int,
) -> tuple[List[float], List[float], List[float]]:
    times, theory_xs = integrate_mean_field(params, p, x0, steps, dt)
    explicit_runs = []
    for rep, structure in enumerate(structures):
        run_seed = seed + rep
        _, explicit_xs = simulate_explicit_temporal_hypergraph(
            params=params,
            structure=structure,
            p=p,
            x0=x0,
            steps=steps,
            dt=dt,
            seed=run_seed,
        )
        explicit_runs.append(explicit_xs)
    return times, theory_xs, average_trajectories(explicit_runs)


def write_phase_scan_csv(path: Path, points: Sequence[PhasePoint], threshold: ThresholdInfo) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "p",
                "x0",
                "eta",
                "eta_c",
                "p_c",
                "regime",
                "theory_final",
                "explicit_final_mean",
                "explicit_final_std",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    f"{point.p:.6f}",
                    f"{point.x0:.6f}",
                    f"{point.eta:.6f}",
                    f"{threshold.eta_c:.6f}",
                    "" if threshold.p_c is None else f"{threshold.p_c:.6f}",
                    threshold.regime,
                    f"{point.theory_final:.6f}",
                    f"{point.explicit_final_mean:.6f}",
                    f"{point.explicit_final_std:.6f}",
                ]
            )


def write_trajectory_csv(path: Path, rows: Sequence[tuple[str, float, float, float, float, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "p", "x0", "time", "theory_x", "explicit_x"])
        for row in rows:
            scenario, p, x0, time, theory_x, explicit_x = row
            writer.writerow(
                [
                    scenario,
                    f"{p:.6f}",
                    f"{x0:.6f}",
                    f"{time:.6f}",
                    f"{theory_x:.6f}",
                    f"{explicit_x:.6f}",
                ]
            )


def svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _polyline_points(xs: Sequence[float], ys: Sequence[float], x_map, y_map) -> str:
    return " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in zip(xs, ys))


def write_line_chart_svg(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series_list: Sequence[Series],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    vertical_lines: Optional[Sequence[tuple[float, str, str]]] = None,
) -> None:
    width, height = 960, 640
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 70, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_map(value: float) -> float:
        if x_max == x_min:
            return margin_left
        return margin_left + (value - x_min) / (x_max - x_min) * plot_w

    def y_map(value: float) -> float:
        if y_max == y_min:
            return margin_top + plot_h
        return margin_top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width / 2:.1f}" y="36" font-size="24" text-anchor="middle" font-family="Arial">{svg_escape(title)}</text>')

    for tick in range(6):
        x_value = x_min + (x_max - x_min) * tick / 5.0
        x_pixel = x_map(x_value)
        parts.append(
            f'<line x1="{x_pixel:.2f}" y1="{margin_top}" x2="{x_pixel:.2f}" y2="{margin_top + plot_h}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_pixel:.2f}" y="{height - 45}" font-size="12" text-anchor="middle" font-family="Arial">{x_value:.2f}</text>'
        )

    for tick in range(6):
        y_value = y_min + (y_max - y_min) * tick / 5.0
        y_pixel = y_map(y_value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y_pixel:.2f}" x2="{margin_left + plot_w}" y2="{y_pixel:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 14}" y="{y_pixel + 4:.2f}" font-size="12" text-anchor="end" font-family="Arial">{y_value:.2f}</text>'
        )

    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" '
        'stroke="#111827" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" '
        'stroke="#111827" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 12}" font-size="15" text-anchor="middle" font-family="Arial">{svg_escape(xlabel)}</text>'
    )
    parts.append(
        f'<text x="22" y="{height / 2:.1f}" font-size="15" text-anchor="middle" font-family="Arial" '
        f'transform="rotate(-90 22 {height / 2:.1f})">{svg_escape(ylabel)}</text>'
    )

    if vertical_lines:
        for x_value, color, label in vertical_lines:
            if x_value < x_min or x_value > x_max:
                continue
            x_pixel = x_map(x_value)
            parts.append(
                f'<line x1="{x_pixel:.2f}" y1="{margin_top}" x2="{x_pixel:.2f}" y2="{margin_top + plot_h}" '
                f'stroke="{color}" stroke-width="2" stroke-dasharray="8,6"/>'
            )
            parts.append(
                f'<text x="{x_pixel + 6:.2f}" y="{margin_top + 18}" font-size="12" text-anchor="start" '
                f'font-family="Arial" fill="{color}">{svg_escape(label)}</text>'
            )

    for series in series_list:
        points = _polyline_points(series.x, series.y, x_map, y_map)
        dash_attr = f' stroke-dasharray="{series.dash}"' if series.dash else ""
        parts.append(f'<polyline fill="none" stroke="{series.color}" stroke-width="2.5"{dash_attr} points="{points}"/>')
        if series.marker:
            for x_value, y_value in zip(series.x, series.y):
                parts.append(
                    f'<circle cx="{x_map(x_value):.2f}" cy="{y_map(y_value):.2f}" r="{series.marker_radius:.2f}" '
                    f'fill="{series.color}" stroke="white" stroke-width="1"/>'
                )

    legend_x = margin_left + plot_w - 265
    legend_y = margin_top + 18
    legend_height = 22 * len(series_list) + 16
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y - 14}" width="255" height="{legend_height}" '
        'fill="white" fill-opacity="0.88" stroke="#d1d5db"/>'
    )
    for idx, series in enumerate(series_list):
        y = legend_y + idx * 22
        dash_attr = f' stroke-dasharray="{series.dash}"' if series.dash else ""
        parts.append(
            f'<line x1="{legend_x + 10}" y1="{y}" x2="{legend_x + 38}" y2="{y}" '
            f'stroke="{series.color}" stroke-width="2.5"{dash_attr}/>'
        )
        if series.marker:
            parts.append(
                f'<circle cx="{legend_x + 24}" cy="{y}" r="{series.marker_radius:.2f}" '
                f'fill="{series.color}" stroke="white" stroke-width="1"/>'
            )
        parts.append(f'<text x="{legend_x + 48}" y="{y + 4}" font-size="12" font-family="Arial">{svg_escape(series.label)}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(
    path: Path,
    params: ModelParameters,
    threshold: ThresholdInfo,
    structures: Sequence[NetworkStructure],
    pair_degree_target: float,
    p_low: float,
    p_high: float,
    phase_points: Sequence[PhasePoint],
) -> None:
    avg_pair_degree = mean([structure.avg_pair_degree for structure in structures])
    avg_triangle_incidence = mean([structure.avg_triangle_incidence for structure in structures])
    triangle_count = mean([len(structure.triangles) for structure in structures])
    edge_count = mean([len(structure.edges) for structure in structures])

    lines = []
    lines.append("Temporal persistence threshold simulation summary")
    lines.append("=" * 52)
    lines.append("")
    lines.append("Simulator: explicit sparse pair graph + temporal triangle hypergraph")
    lines.append(f"mu = {params.mu}")
    lines.append(f"lambda = {params.lam}")
    lines.append(f"beta_tri = {params.beta_tri}")
    lines.append(f"triangles_per_node (target) = {params.triangles_per_node}")
    lines.append(f"a = {params.a}")
    lines.append(f"pair_degree_target = {pair_degree_target}")
    lines.append(f"avg_pair_degree_realized = {avg_pair_degree:.3f}")
    lines.append(f"avg_triangle_incidence_realized = {avg_triangle_incidence:.3f}")
    lines.append(f"avg_edge_count = {edge_count:.1f}")
    lines.append(f"avg_triangle_count = {triangle_count:.1f}")
    lines.append("")
    lines.append(f"eta_c = {threshold.eta_c:.6f}")
    lines.append(f"regime = {threshold.regime}")
    lines.append("p_c = unattainable in [0, 1]" if threshold.p_c is None else f"p_c = {threshold.p_c:.6f}")
    lines.append(f"trajectory p values = {p_low:.3f} (below/near threshold), {p_high:.3f} (above/near threshold)")
    lines.append("")
    lines.append("Files:")
    lines.append("- phase_scan.csv")
    lines.append("- trajectories.csv")
    lines.append("- phase_scan.svg")
    lines.append("- phase_scan_zoom.svg")
    lines.append("- trajectory_below_threshold.svg")
    lines.append("- trajectory_above_threshold.svg")
    lines.append("")
    lines.append("Phase-scan snapshots:")
    for point in phase_points[: min(6, len(phase_points))]:
        lines.append(
            f"- p={point.p:.3f}, x0={point.x0:.3f}, theory={point.theory_final:.3f}, "
            f"explicit_mean={point.explicit_final_mean:.3f}, explicit_std={point.explicit_final_std:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate and visualize the temporal persistence threshold model.")
    parser.add_argument("--outdir", default="simulation_output", help="Directory for generated CSV/SVG outputs.")
    parser.add_argument("--mu", type=float, default=1.0, help="Recovery rate.")
    parser.add_argument("--lam", type=float, default=0.6, help="Pairwise infection rate lambda.")
    parser.add_argument("--beta-tri", type=float, default=1.0, help="Triadic infection rate beta_Delta.")
    parser.add_argument("--triangles-per-node", type=int, default=8, help="Target mean number of candidate triangles incident to a node.")
    parser.add_argument("--pair-degree", type=float, default=12.0, help="Target mean degree of the explicit pair graph.")
    parser.add_argument("--a", type=float, default=0.5, help="Stationary activity probability of a candidate triangle.")
    parser.add_argument("--n-nodes", type=int, default=220, help="Number of nodes in the explicit temporal network simulation.")
    parser.add_argument("--steps", type=int, default=500, help="Number of time steps per trajectory.")
    parser.add_argument("--dt", type=float, default=0.04, help="Time step for both ODE integration and Monte Carlo updates.")
    parser.add_argument("--replicates", type=int, default=10, help="Number of independent explicit-network replicates.")
    parser.add_argument("--p-grid-count", type=int, default=15, help="Number of p values in the phase scan over [0, 1].")
    parser.add_argument("--x0-low", type=float, default=0.05, help="Low-seed initial infected fraction.")
    parser.add_argument("--x0-high", type=float, default=0.35, help="High-seed initial infected fraction.")
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    params = ModelParameters(
        mu=args.mu,
        lam=args.lam,
        beta_tri=args.beta_tri,
        triangles_per_node=args.triangles_per_node,
        a=args.a,
    )
    threshold = threshold_info(params)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    structures = build_structure_ensemble(
        n_structures=args.replicates,
        n_nodes=args.n_nodes,
        pair_degree=args.pair_degree,
        triangles_per_node=args.triangles_per_node,
        seed=args.seed + 700000,
    )

    p_values = linspace(0.0, 1.0, args.p_grid_count)
    x0_values = [args.x0_low, args.x0_high]
    phase_points = scan_phase_diagram(
        params=params,
        structures=structures,
        p_values=p_values,
        x0_values=x0_values,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
    )

    if threshold.p_c is None:
        p_low, p_high = 0.2, 0.8
    else:
        p_low = max(0.0, threshold.p_c - 0.15)
        p_high = min(1.0, threshold.p_c + 0.15)
        if abs(p_high - p_low) < 1e-9:
            p_low = max(0.0, p_low - 0.1)
            p_high = min(1.0, p_high + 0.1)

    trajectory_rows = []
    palette = {
        "low_seed": "#1f77b4",
        "high_seed": "#d62728",
        "threshold": "#111827",
    }

    for scenario_name, p_value, svg_name, title in [
        ("below_threshold", p_low, "trajectory_below_threshold.svg", f"Trajectories for p = {p_low:.3f}"),
        ("above_threshold", p_high, "trajectory_above_threshold.svg", f"Trajectories for p = {p_high:.3f}"),
    ]:
        series_list: List[Series] = []
        times: List[float] = []
        for x0, color, label in [
            (args.x0_low, palette["low_seed"], f"x0 = {args.x0_low:.2f}"),
            (args.x0_high, palette["high_seed"], f"x0 = {args.x0_high:.2f}"),
        ]:
            times, theory_xs, explicit_xs = collect_average_trajectory(
                params=params,
                structures=structures,
                p=p_value,
                x0=x0,
                steps=args.steps,
                dt=args.dt,
                seed=args.seed + int(100000 * p_value) + int(1000 * x0),
            )
            series_list.append(Series(label=f"Theory, {label}", x=times, y=theory_xs, color=color, dash="8,5"))
            series_list.append(Series(label=f"Explicit avg, {label}", x=times, y=explicit_xs, color=color))
            for time, theory_x, explicit_x in zip(times, theory_xs, explicit_xs):
                trajectory_rows.append((scenario_name, p_value, x0, time, theory_x, explicit_x))

        write_line_chart_svg(
            path=outdir / svg_name,
            title=title,
            xlabel="time",
            ylabel="infected fraction",
            series_list=series_list,
            x_min=0.0,
            x_max=times[-1] if times else args.steps * args.dt,
            y_min=0.0,
            y_max=1.0,
        )

    write_trajectory_csv(outdir / "trajectories.csv", trajectory_rows)
    write_phase_scan_csv(outdir / "phase_scan.csv", phase_points, threshold)

    phase_series: List[Series] = []
    for x0, color in [(args.x0_low, palette["low_seed"]), (args.x0_high, palette["high_seed"] )]:
        subset = [point for point in phase_points if abs(point.x0 - x0) < 1e-12]
        subset.sort(key=lambda item: item.p)
        phase_series.append(
            Series(
                label=f"Explicit avg, x0 = {x0:.2f}",
                x=[point.p for point in subset],
                y=[point.explicit_final_mean for point in subset],
                color=color,
                marker=True,
            )
        )
        phase_series.append(
            Series(
                label=f"Theory, x0 = {x0:.2f}",
                x=[point.p for point in subset],
                y=[point.theory_final for point in subset],
                color=color,
                dash="8,5",
            )
        )

    vertical_lines = []
    if threshold.p_c is not None:
        vertical_lines.append((threshold.p_c, palette["threshold"], f"theory p_c = {threshold.p_c:.3f}"))

    write_line_chart_svg(
        path=outdir / "phase_scan.svg",
        title="Final infected fraction versus temporal persistence",
        xlabel="p (temporal persistence)",
        ylabel="final infected fraction",
        series_list=phase_series,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        vertical_lines=vertical_lines,
    )

    zoom_margin = 0.22
    zoom_center = threshold.p_c if threshold.p_c is not None else 0.5
    zoom_min = max(0.0, zoom_center - zoom_margin)
    zoom_max = min(1.0, zoom_center + zoom_margin)
    write_line_chart_svg(
        path=outdir / "phase_scan_zoom.svg",
        title="Phase scan near the theoretical persistence threshold",
        xlabel="p (temporal persistence)",
        ylabel="final infected fraction",
        series_list=phase_series,
        x_min=zoom_min,
        x_max=zoom_max,
        y_min=0.0,
        y_max=1.0,
        vertical_lines=vertical_lines,
    )

    write_summary(
        outdir / "summary.txt",
        params=params,
        threshold=threshold,
        structures=structures,
        pair_degree_target=args.pair_degree,
        p_low=p_low,
        p_high=p_high,
        phase_points=phase_points,
    )

    print(f"Wrote outputs to: {outdir.resolve()}")
    print("Simulator type: explicit temporal network / hypergraph")
    print(f"Theoretical eta_c = {threshold.eta_c:.6f}")
    if threshold.p_c is None:
        print("Theoretical p_c is unattainable in [0, 1] for these parameters.")
    else:
        print(f"Theoretical p_c = {threshold.p_c:.6f}")


if __name__ == "__main__":
    main()
