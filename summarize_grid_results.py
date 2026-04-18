# summarize_grid_results.py
from __future__ import annotations

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_std(values):
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def pick_stablest(results):
    # definición práctica: menor std_reward entre runs con mean_reward > 0
    positive = [r for r in results if r["metrics"]["mean_reward"] > 0]
    pool = positive if positive else results
    return min(pool, key=lambda r: r["metrics"]["std_reward"])


def save_cost_plot(results, out_dir):
    by_res = defaultdict(list)
    for r in results:
        by_res[r["hp"]["img_size"]].append(r["metrics"]["elapsed_sec"] / 3600.0)

    labels = sorted(by_res.keys())
    means = [np.mean(by_res[k]) for k in labels]
    stds = [np.std(by_res[k]) for k in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color=["#42A5F5", "#EF5350"][:len(labels)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{k}x{k}" for k in labels])
    ax.set_ylabel("Training time (hours)")
    ax.set_title("Training cost by image resolution")
    plt.tight_layout()

    path = os.path.join(out_dir, "cost_by_resolution.png")
    plt.savefig(path, dpi=140)
    plt.close(fig)
    return path


def format_latex_table_cost(results):
    by_res = defaultdict(list)
    by_res_reward = defaultdict(list)

    for r in results:
        img = r["hp"]["img_size"]
        by_res[img].append(r["metrics"]["elapsed_sec"] / 3600.0)
        by_res_reward[img].append(r["metrics"]["mean_reward"])

    rows = []
    for img in sorted(by_res.keys()):
        mean_t, std_t = mean_std(by_res[img])
        mean_r, std_r = mean_std(by_res_reward[img])
        rows.append((img, mean_t, std_t, mean_r, std_r))

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Coste computacional medio por resolución de imagen.}",
        "\\label{tab:cost_by_resolution}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Resolución & Tiempo medio (h) & Reward medio \\\\",
        "\\midrule",
    ]
    for img, mt, st, mr, sr in rows:
        lines.append(f"${img}\\times{img}$ & {mt:.2f} $\\pm$ {st:.2f} & {mr:.1f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def format_latex_table_summary(best, worst, stable):
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Resumen de configuraciones destacadas del grid search.}",
        "\\label{tab:grid_best_worst_stable}",
        "\\begin{tabular}{p{2.0cm}p{4.0cm}}",
        "\\toprule",
        "Categoría & Configuración \\\\",
        "\\midrule",
        f"Mejor run & {best['run_id'].replace('_', '\\_')} "
        f"(reward={best['metrics']['mean_reward']:.1f}, std={best['metrics']['std_reward']:.1f}) \\\\",
        f"Peor run & {worst['run_id'].replace('_', '\\_')} "
        f"(reward={worst['metrics']['mean_reward']:.1f}, std={worst['metrics']['std_reward']:.1f}) \\\\",
        f"Más estable & {stable['run_id'].replace('_', '\\_')} "
        f"(reward={stable['metrics']['mean_reward']:.1f}, std={stable['metrics']['std_reward']:.1f}) \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize grid results for report")
    parser.add_argument("--summary", type=str, default="./grid_results/grid_summary.json")
    parser.add_argument("--out_dir", type=str, default="./grid_results/report_assets")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = load_summary(args.summary)

    best = max(results, key=lambda r: r["metrics"]["mean_reward"])
    worst = min(results, key=lambda r: r["metrics"]["mean_reward"])
    stable = pick_stablest(results)

    cost_plot = save_cost_plot(results, args.out_dir)
    latex_cost = format_latex_table_cost(results)
    latex_summary = format_latex_table_summary(best, worst, stable)

    tex_path = os.path.join(args.out_dir, "report_tables.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% --- Table: cost by resolution ---\n")
        f.write(latex_cost)
        f.write("\n\n% --- Table: best / worst / stable ---\n")
        f.write(latex_summary)
        f.write("\n")

    print(f"[saved] {cost_plot}")
    print(f"[saved] {tex_path}")
    print("\nBest run :", best["run_id"], best["metrics"])
    print("Worst run:", worst["run_id"], worst["metrics"])
    print("Stablest :", stable["run_id"], stable["metrics"])


if __name__ == "__main__":
    main()