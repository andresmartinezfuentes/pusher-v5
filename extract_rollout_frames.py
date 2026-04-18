# extract_rollout_frames.py
from __future__ import annotations

import os
import argparse

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Extract a contact sheet from rollout GIF")
    parser.add_argument("--gif", type=str, required=True, help="Path to rollout gif, e.g. ep00_rollout.gif")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    parser.add_argument("--n_frames", type=int, default=6, help="Number of frames to show")
    args = parser.parse_args()

    frames = imageio.mimread(args.gif)
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {args.gif}")

    n = min(args.n_frames, len(frames))
    idxs = [round(i * (len(frames) - 1) / max(n - 1, 1)) for i in range(n)]
    selected = [frames[i] for i in idxs]

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]

    for ax, idx, frame in zip(axes, idxs, selected):
        ax.imshow(frame)
        ax.set_title(f"t={idx}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    out_path = args.out
    if out_path is None:
        stem = os.path.splitext(os.path.basename(args.gif))[0]
        out_path = os.path.join(os.path.dirname(args.gif), f"{stem}_frames.png")

    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()