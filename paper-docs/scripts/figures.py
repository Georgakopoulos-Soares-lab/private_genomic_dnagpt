#!/usr/bin/env python3
"""Generate every manuscript figure from the evidence ledger.

No figure hard-codes a measurement. Every plotted value is read from
``paper-docs/evidence/*.yaml``, so a figure and a table cannot disagree.

    python paper-docs/scripts/figures.py                 # all figures
    python paper-docs/scripts/figures.py --only waterfall memory
    python paper-docs/scripts/figures.py --list
    python paper-docs/scripts/figures.py --png DIR       # also write PNGs, for visual review

Output: ``paper-docs/manuscript/figures/*.pdf`` (vector, committed).

Layout rules, learned by getting them wrong:
  * titles go through ``ax.set_title`` or ``transAxes`` -- never data coordinates;
  * text positions are computed from the data, never guessed;
  * anything that could collide gets its own reserved band.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from figstyle import (  # noqa: E402
    AMBER,
    BLACK,
    CHART_BLUE,
    CLIENT,
    CLIENT_PALE,
    DPI,
    FAILURE,
    GRAY,
    GREEN,
    LIGHT_GRAY,
    MEASURED_KW,
    NAVY,
    PALE_ORANGE,
    SERVER,
    SERVER_PALE,
    arrow,
    canvas,
    despine,
    labelled_box,
    save,
    save_png,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "paper-docs" / "evidence"
DEFAULT_OUT = ROOT / "paper-docs" / "manuscript" / "figures"

PNG_DIR: pathlib.Path | None = None


# --- Evidence access --------------------------------------------------------


def load(name: str) -> dict:
    with (EVIDENCE / f"{name}.yaml").open() as fh:
        return yaml.safe_load(fh)


class Ledger:
    """Flat id -> row view over measurements.yaml, so figures cite by id."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        for section in load("measurements").values():
            if isinstance(section, list):
                for row in section:
                    if isinstance(row, dict) and "id" in row:
                        self.rows[row["id"]] = row

    def value(self, rid: str):
        try:
            return self.rows[rid]["value"]
        except KeyError:
            raise SystemExit(
                f"figures.py: '{rid}' is not in evidence/measurements.yaml.\n"
                f"Add it to the ledger with a source before plotting it."
            )


def emit(fig, out: pathlib.Path, name: str) -> pathlib.Path:
    """Write the PDF, and a review PNG when --png was given."""
    if PNG_DIR is not None:
        save_png(fig, PNG_DIR, name, dpi=130)
    return save(fig, out, name)


def title(ax, text, subtitle=None, size=13.5, pad=None):
    """Title above the axes, in axes coordinates. Never in data coordinates."""
    ax.set_title(
        text,
        fontsize=size,
        fontweight="bold",
        color=BLACK,
        loc="left",
        pad=pad if pad is not None else (30 if subtitle else 14),
    )
    if subtitle:
        ax.text(
            0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            fontsize=9.6,
            color=GRAY,
            va="bottom",
            ha="left",
        )


# --- 1. Graphical abstract --------------------------------------------------


def fig_graphical_abstract(out: pathlib.Path) -> pathlib.Path:
    L = Ledger()
    fig, ax = canvas(figsize=(7.05, 4.45))

    ax.text(
        50,
        96,
        "A genomic fragment stays off the compute provider",
        fontsize=13,
        fontweight="bold",
        color=BLACK,
        ha="center",
        va="center",
    )

    labelled_box(
        ax,
        2,
        68,
        27,
        18,
        "Data owner",
        "600 bp genomic fragment\nCPU only; holds secret key",
        facecolor=CLIENT_PALE,
        edgecolor=CLIENT,
        title_size=9.6,
        sub_size=7.2,
    )
    labelled_box(
        ax,
        71,
        68,
        27,
        18,
        "Compute provider",
        "serves the model\non one GPU",
        facecolor=SERVER_PALE,
        edgecolor=SERVER,
        title_size=9.6,
        sub_size=7.2,
    )

    labelled_box(
        ax,
        4,
        43,
        25,
        15,
        "Local preprocessing",
        "tokenize + embedding lookup\nthen encrypt embedded vectors",
        facecolor=CLIENT_PALE,
        edgecolor=CLIENT,
        title_size=8.4,
        sub_size=6.5,
    )
    labelled_box(
        ax,
        34,
        39,
        32,
        23,
        "Client-assisted CKKS",
        "server: encrypted linear algebra\nclient: declared nonlinear boundaries",
        facecolor="white",
        edgecolor=BLACK,
        title_size=9.3,
        sub_size=7.2,
    )

    arrow(ax, 15.5, 67, 15.5, 59, color=CLIENT, lw=1.5)
    arrow(ax, 29.5, 50.5, 33.5, 50.5, color=CLIENT, lw=1.5)
    arrow(ax, 84.5, 67, 66.5, 57.5, color=SERVER, lw=1.5)

    ax.text(
        50,
        34.5,
        "The compute provider never receives a plaintext activation or the secret key.",
        fontsize=7.3,
        color=BLACK,
        ha="center",
        va="center",
        style="italic",
    )

    # Every headline value below comes from the complete-model run, the only clean
    # dedicated-node artifact. The earlier per-block timing and the client peak-memory
    # figure are not used here because neither has committed dedicated-node evidence.
    results = [
        (f"{L.value('prompt.gsr_total')} tokens", "GSR prompt"),
        (f"{L.value('full.wall_hours')} h", "one measured run"),
        (
            rf"${L.value('full.margin_rel_error'):.2e}$",
            "vs. float64 reference",
        ),
        (
            f"{L.value('full.client_share_of_eval')}%",
            "of encrypted evaluation",
        ),
    ]
    for i, (big, small) in enumerate(results):
        labelled_box(
            ax,
            2 + i * 24.4,
            4,
            22.4,
            16,
            big,
            small,
            facecolor=LIGHT_GRAY,
            edgecolor=GRAY,
            title_size=9.8,
            sub_size=6.6,
        )

    return emit(fig, out, "fig_graphical_abstract")


# --- 2. Protocol architecture -----------------------------------------------


def fig_architecture(out: pathlib.Path) -> pathlib.Path:
    """The evaluation order, and which party performs each step."""
    fig, ax = canvas(figsize=(7.0, 5.65))

    steps = [
        ("client", "Tokenize, embed, and encrypt"),
        ("server", "LayerNorm statistics"),
        ("client", "Inverse square root"),
        ("server", "Query / key / value projection"),
        ("server", "Causal attention scores"),
        ("client", "Softmax over the score tiles"),
        ("server", "Attention context, projection, residual"),
        ("server", "LayerNorm statistics, second site"),
        ("client", "Inverse square root, second site"),
        ("server", "MLP up-projection"),
        ("client", "GELU"),
        ("server", "MLP down-projection, residual"),
    ]

    top, bottom = 86.0, 18.0
    n = len(steps)
    pitch = (top - bottom) / n
    height = pitch * 0.72
    lx, rx, width = 4.0, 54.0, 42.0
    boundary = 50.0

    ax.text(
        4,
        96.0,
        "Data owner",
        fontsize=10.5,
        fontweight="bold",
        color=CLIENT,
        va="center",
    )
    ax.text(
        4,
        92.0,
        "holds the genomic fragment and secret key  ·  no GPU",
        fontsize=7.1,
        color=GRAY,
        va="center",
    )
    ax.text(
        54,
        96.0,
        "Compute provider",
        fontsize=10.5,
        fontweight="bold",
        color=SERVER,
        va="center",
    )
    ax.text(
        54,
        92.0,
        "holds the model  ·  encrypted activations only  ·  one A100",
        fontsize=7.1,
        color=GRAY,
        va="center",
    )

    ax.plot(
        [boundary, boundary],
        [4.0, 88.5],
        color=GRAY,
        linewidth=1.1,
        linestyle=(0, (6, 5)),
    )
    centres = []
    for i, (who, label) in enumerate(steps):
        y = top - (i + 1) * pitch + (pitch - height) / 2
        x = lx if who == "client" else rx
        labelled_box(
            ax,
            x,
            y,
            width,
            height,
            f"{i + 1}.  {label}",
            facecolor=CLIENT_PALE if who == "client" else SERVER_PALE,
            edgecolor=CLIENT if who == "client" else SERVER,
            title_size=7.4,
        )
        centres.append((who, x, y, y + height / 2))

    for i in range(n - 1):
        who_a, xa, ya, mid_a = centres[i]
        who_b, xb, yb, mid_b = centres[i + 1]
        if who_a == who_b:  # same column: drop straight down
            cx = xa + width / 2
            arrow(ax, cx, ya, cx, yb + height, color=GRAY, lw=1.3)
        else:  # cross the boundary
            x1 = xa + width if who_a == "client" else xa
            x2 = xb if who_b == "server" else xb + width
            arrow(
                ax,
                x1,
                mid_a - height * 0.15,
                x2,
                mid_b + height * 0.15,
                color=GRAY,
                lw=1.3,
            )

    # The retained implementation has two distinct block-end actions. A one-block validation
    # may decrypt the output for comparison; composition decrypts and freshly re-encrypts the
    # full hidden state before the next block. They are alternatives, not one generic readout.
    last_x = rx + width / 2
    last_y = centres[-1][3]
    box_h = 7.2
    labelled_box(
        ax,
        4,
        6.5,
        42,
        box_h,
        "Validation only: decrypt output",
        facecolor=CLIENT_PALE,
        edgecolor=CLIENT,
        title_size=7.3,
    )
    labelled_box(
        ax,
        54,
        6.5,
        42,
        box_h,
        "Composition: decrypt + re-encrypt hidden state",
        facecolor=CLIENT_PALE,
        edgecolor=CLIENT,
        title_size=7.1,
    )
    arrow(ax, last_x, last_y - height / 2, 25, 14.1, color=GRAY, lw=1.1)
    arrow(ax, last_x, last_y - height / 2, 75, 14.1, color=GRAY, lw=1.1)
    ax.text(
        boundary,
        1.2,
        "Only ciphertexts cross the in-process role boundary; fresh encryption resets level to 0.",
        fontsize=6.8,
        color=BLACK,
        ha="center",
        va="bottom",
        style="italic",
    )

    return emit(fig, out, "fig_architecture")


# --- 3. Packing -------------------------------------------------------------


def fig_packing(out: pathlib.Path) -> pathlib.Path:
    L = Ledger()
    copies = L.value("layout.copies")
    width_feat = L.value("layout.pack_width")
    lanes = L.value("layout.token_lanes")
    slots = L.value("param.slots")
    groups = L.value("layout.token_groups")
    tokens = L.value("prompt.gsr_total")
    bp = L.value("prompt.gsr_basepairs")
    kmer = L.value("prompt.kmer")
    specials = L.value("prompt.gsr_special_tokens")
    active = L.value("layout.embedding_width")

    fig, ax = canvas(figsize=(7.0, 4.05))

    ax.text(
        2,
        96,
        f"One ciphertext holds {slots:,} slots",
        fontsize=10.5,
        fontweight="bold",
        color=BLACK,
        va="center",
    )
    ax.text(
        2,
        90.5,
        f"{copies} copies × {width_feat:,} features × {lanes} token lanes = "
        f"{copies * width_feat * lanes:,} slots. The copies carry the {copies}× MLP expansion.",
        fontsize=7.2,
        color=GRAY,
        va="center",
    )

    x0, gap = 2.0, 1.6
    w = (96.0 - gap * (copies - 1)) / copies
    for c in range(copies):
        x = x0 + c * (w + gap)
        labelled_box(ax, x, 56, w, 26, "", facecolor=SERVER_PALE, edgecolor=SERVER)
        ax.text(
            x + w / 2,
            78.0,
            f"copy {c}",
            fontsize=8.4,
            fontweight="bold",
            color=BLACK,
            ha="center",
            va="center",
        )
        ax.text(
            x + w / 2,
            74.0,
            f"{width_feat:,} features",
            fontsize=6.7,
            color=GRAY,
            ha="center",
            va="center",
        )
        inner, pad = w - 3.0, 1.5
        active_w = inner * active / width_feat
        ax.add_patch(
            plt.Rectangle(
                (x + pad, 65.3),
                active_w,
                5.0,
                facecolor=SERVER,
                edgecolor=SERVER,
                linewidth=0.6,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (x + pad + active_w, 65.3),
                inner - active_w,
                5.0,
                facecolor=PALE_ORANGE,
                edgecolor=AMBER,
                linewidth=0.6,
            )
        )
        ax.text(
            x + pad + active_w / 2,
            67.8,
            f"{active} active",
            fontsize=5.9,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
        )
        ax.text(
            x + pad + active_w + (inner - active_w) / 2,
            67.8,
            f"{width_feat - active}\npad + stage",
            fontsize=5.2,
            color=BLACK,
            ha="center",
            va="center",
            linespacing=0.9,
        )
        for lane in range(lanes):
            lw_ = inner / lanes
            ax.add_patch(
                plt.Rectangle(
                    (x + pad + lane * lw_, 59.0),
                    lw_ * 0.82,
                    4.0,
                    facecolor=CLIENT_PALE,
                    edgecolor=CLIENT,
                    linewidth=0.8,
                )
            )
        ax.text(
            x + w / 2,
            57.0,
            f"{lanes} token lanes",
            fontsize=6.1,
            color=GRAY,
            ha="center",
            va="center",
        )

    ax.text(
        2,
        46,
        f"GSR: {bp} bp become {tokens} tokens in {groups} groups",
        fontsize=10.0,
        fontweight="bold",
        color=BLACK,
        va="center",
    )
    ax.text(
        2,
        40.5,
        f"{bp} bp ÷ {kmer}-mer = {bp // kmer} tokens, plus {specials} task specials = "
        f"{tokens}.   {tokens} tokens across {lanes} lanes rounds up to {groups} groups; "
        f"the final group is partly padded.",
        fontsize=7.0,
        color=GRAY,
        va="center",
    )

    gw = 96.0 / groups
    for g in range(groups):
        filled = min(lanes, tokens - g * lanes)
        full = filled == lanes
        labelled_box(
            ax,
            2 + g * gw,
            20,
            gw - 1.2,
            13,
            f"g{g}",
            f"{filled}/{lanes}",
            facecolor=SERVER_PALE if full else PALE_ORANGE,
            edgecolor=SERVER if full else AMBER,
            title_size=7.0,
            sub_size=6.0,
        )

    ax.text(
        2,
        13,
        f"The {groups} groups give {groups * (groups + 1) // 2} lower-triangular causal "
        f"score tiles, and just {(-tokens) % lanes} of the {groups * lanes} lanes is padding.",
        fontsize=6.9,
        color=GRAY,
        va="center",
    )

    return emit(fig, out, "fig_packing")


# --- 4. The optimization result ---------------------------------------------


def fig_waterfall(out: pathlib.Path) -> pathlib.Path:
    """Two measured endpoints, the changes between them, and the whole-model consequence."""
    opt = load("optimizations")
    baseline = opt["retained"][0]["block_time_s"]
    final = opt["net"]["to_encrypted_evaluation_s"]
    factor = opt["net"]["factor"]
    full = opt["net_full_pass"]

    changes = [
        "Thread affinity and NUMA-local allocation",
        "Encode once, clone per use",
        "Parallel batched encoding across 32 cores",
        "Encode at the level of use, flush between stages",
        "Release host-side key copies after upload",
    ]

    fig, (ax, axr) = plt.subplots(
        1,
        2,
        figsize=(13.6, 6.0),
        dpi=DPI,
        gridspec_kw={"width_ratios": [1.0, 1.12], "wspace": 0.06},
    )

    # Left: the two measured endpoints.
    for x, val in ((0.0, baseline), (1.0, final)):
        ax.bar(x, val, width=0.52, facecolor=SERVER, **MEASURED_KW, zorder=3)
        ax.text(
            x,
            val * 1.06,
            f"{val:,.0f} s\n{val / 60:,.0f} min",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=BLACK,
            zorder=4,
        )

    ax.set_yscale("log")
    ax.set_ylim(300, baseline * 4.2)
    ax.set_xlim(-0.62, 1.62)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(
        ["Re-encode at every\nmultiplication", "After the\ncampaign"],
        fontsize=11,
        color=BLACK,
    )
    ax.set_ylabel("Encrypted evaluation, seconds (log scale)", fontsize=11)
    despine(ax)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.annotate(
        "",
        xy=(0.86, final * 1.35),
        xytext=(0.16, baseline * 1.02),
        arrowprops=dict(
            arrowstyle="-|>",
            color=GREEN,
            linewidth=2.4,
            connectionstyle="arc3,rad=0.26",
        ),
        zorder=2,
    )
    ax.text(
        1.20,
        baseline * 1.45,
        f"{factor}×",
        fontsize=27,
        fontweight="bold",
        color=GREEN,
        ha="center",
        va="center",
    )
    ax.text(
        1.20,
        baseline * 0.92,
        "faster",
        fontsize=13.5,
        fontweight="bold",
        color=GREEN,
        ha="center",
        va="center",
    )
    title(ax, "One transformer block, 103 tokens")

    # Right: what changed, and what it means for a whole inference.
    axr.axis("off")
    axr.set_xlim(0, 100)
    axr.set_ylim(0, 100)

    axr.text(
        0,
        95,
        "What changed",
        fontsize=12.5,
        fontweight="bold",
        color=BLACK,
        va="center",
    )
    axr.text(
        0,
        89.5,
        "None of it alters a single encrypted operation.",
        fontsize=9.8,
        color=GRAY,
        va="center",
        style="italic",
    )
    for i, change in enumerate(changes):
        axr.text(0, 82 - i * 6.6, "—", fontsize=10.5, color=SERVER, va="center")
        axr.text(4.5, 82 - i * 6.6, change, fontsize=10.5, color=BLACK, va="center")

    axr.text(
        0,
        42,
        "Identical at both ends",
        fontsize=12.5,
        fontweight="bold",
        color=BLACK,
        va="center",
    )
    axr.text(
        0,
        30,
        "156 dense products  ·  177,734 ciphertext–plaintext and\n"
        "1,506 ciphertext–ciphertext multiplications  ·  8,173 rotations\n"
        "857 client boundary crossings  ·  multiplicative depth 13\n"
        "the same frozen plaintext reference, reproduced to $4.6\\times10^{-9}$",
        fontsize=10.2,
        color=GRAY,
        va="center",
        linespacing=1.7,
    )

    labelled_box(
        axr,
        0,
        2,
        100,
        13,
        f"A whole {int(full['from_hours'])}-hour inference becomes "
        f"{full['to_range_human'].split(',')[0]}",
        "projected across the encrypted-scope tasks, at fixed circuit",
        facecolor=LIGHT_GRAY,
        edgecolor=GRAY,
        title_size=12,
        sub_size=9.2,
    )

    return emit(fig, out, "fig_waterfall")


# --- 5. Systems behaviour of one complete inference (panels of fig_systems) -
#
# Review feedback asked for one composite figure rather than four standalone ones, so the
# resource trace, cost split, stage memory, and prompt-length schedule are panels (a)-(e) of
# fig_systems. Each panel keeps its own evidence source; no panel hard-codes a measurement.


def _panel_letter(ax, letter: str, x: float = -0.055, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=BLACK,
        ha="left",
        va="bottom",
    )


def _panel_title(ax, text: str, size: float = 8.4, pad: float = 6) -> None:
    ax.set_title(
        text, fontsize=size, fontweight="bold", color=BLACK, loc="left", pad=pad
    )


def _panel_trace(ax1, ax2) -> None:
    """(a) and (b): resource trace of the complete encrypted inference.

    Drawn from the complete-model run, which is the only clean dedicated-node artifact. The
    earlier per-block stage trace came from a shared partition and is therefore not plotted.
    """
    telemetry = load("telemetry")
    tel = telemetry["complete_model_trace"]
    trace = tel["trace"]
    total = tel["meta"]["total_wall_s"]
    full_samples = telemetry["complete_model_contention"]["meta"]["samples"]
    blocks = tel["meta"]["blocks"]

    t = np.array([r["t_s"] for r in trace], dtype=float)
    gpu_pct = np.array([r["gpu_pct"] for r in trace], dtype=float)
    rss = np.array([r["rss_mib"] for r in trace], dtype=float)
    gpu_mem = np.array([r["gpu_mib"] for r in trace], dtype=float)

    # No block-boundary guides: the twelve cycles are legible in the data itself, and drawn
    # boundaries would be nominal (equal division) rather than measured start times.

    ax1.plot(t, gpu_pct, color=NAVY, linewidth=1.7, zorder=3)
    ax1.set_ylabel("GPU use (%)", fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.tick_params(labelsize=7)
    despine(ax1)
    ax1.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax1.set_axisbelow(True)

    ax2.plot(t, rss, color=CHART_BLUE, linewidth=1.7, label="host memory", zorder=3)
    ax2.plot(
        t,
        gpu_mem,
        color=GREEN,
        linewidth=1.7,
        linestyle="--",
        label="GPU memory",
        zorder=3,
    )
    ax2.set_ylabel("Memory (MiB)", fontsize=8)
    ax2.set_xlabel("Seconds since process start", fontsize=8)
    ax2.set_ylim(0, max(rss) * 1.14)
    ax2.tick_params(labelsize=7)
    ax2.legend(
        frameon=False,
        fontsize=7.2,
        loc="upper left",
        ncol=2,
        bbox_to_anchor=(0.0, 1.04),
    )
    despine(ax2)
    ax2.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax2.set_axisbelow(True)

    ax1.set_xlim(0, total)
    _panel_title(
        ax1,
        f"Sampled GPU use over one complete inference "
        f"(thinned from {full_samples} samples)",
    )
    _panel_title(ax2, f"Memory over the same run, across {blocks} blocks")


def _panel_stage_memory(ax) -> None:
    """(d): stage-local host memory under the bounded plaintext cache."""
    fail = Ledger().value("mem.unbounded_failure")
    bounded = next(
        o
        for o in load("optimizations")["retained"]
        if o["id"] == "opt.bounded_host_memory"
    )
    values = bounded["effect_mib"]
    labels = [
        "Query / key / value\n(3 matrices)",
        "Attention\n(none)",
        "MLP\n(8 matrices)",
    ]

    bars = ax.bar(
        range(3), values, width=0.5, facecolor=SERVER, **MEASURED_KW, zorder=3
    )
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 1100,
            f"{v:,} MiB",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=BLACK,
        )

    ax.axhline(fail, color=FAILURE, linewidth=1.6, linestyle="--", zorder=2)
    ax.text(
        2.60,
        fail + 1400,
        f"Separate unbounded-cache attempt:\n{fail:,} MiB at operating-system kill",
        color=FAILURE,
        fontsize=6.6,
        ha="right",
        va="bottom",
        fontweight="bold",
    )

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("Peak process RSS (MiB)", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, fail * 1.22)
    ax.set_xlim(-0.62, 2.62)
    despine(ax)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)
    _panel_title(ax, "Host memory per stage, one block, bounded cache")


def _panel_cost_split(ax) -> None:
    """(c): how the encrypted evaluation interval divides between the two parties."""
    # The complete-model run is the only clean dedicated-node timing artifact, so the split is
    # drawn from it rather than from the single block.
    L = Ledger()
    server = L.value("full.server_linear_algebra")
    client = L.value("full.client_boundaries")
    evaluation = L.value("full.encrypted_evaluation")
    wall = L.value("full.wall")

    # Segments are labelled in place, so the figure needs no legend and nothing can collide
    # with the axis.
    for label, val, color, textcolor in (
        ("Compute provider", server, SERVER, "white"),
        ("Data owner", client, CLIENT, "white"),
    ):
        left = {"Compute provider": 0.0, "Data owner": server}[label]
        ax.barh(
            0,
            val,
            left=left,
            height=0.34,
            facecolor=color,
            edgecolor=BLACK,
            linewidth=1.1,
        )
        inside = (
            f"{val:,.0f} s  ·  {100 * val / evaluation:.1f}% of evaluation"
            if label == "Compute provider"
            else f"{val:,.0f} s"
        )
        ax.text(
            left + val / 2,
            0.02,
            inside,
            ha="center",
            va="center",
            fontsize=8.0 if label == "Compute provider" else 7.2,
            fontweight="bold",
            color=textcolor,
        )
        ax.text(
            left + val / 2,
            -0.31,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color=color,
            fontweight="bold",
        )

    ax.set_xlim(0, evaluation)
    ax.set_ylim(-0.52, 0.30)
    ax.set_yticks([])
    ax.set_xlabel("Seconds of encrypted evaluation", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    despine(ax, keep=("bottom",))
    _panel_title(
        ax,
        f"Encrypted evaluation: {evaluation:,.0f} s of {wall:,.0f} s wall clock",
    )


def _panel_scaling(ax) -> None:
    """(e): causal score-tile schedule across tasks, with the measured inference marked.

    This plots one schedule component, not total circuit size or time. Per-task time estimates
    were removed from the ledger:
    they were built from per-block timings measured on a shared partition, which the manuscript
    may not use. Only genomic signal recognition has a measured complete inference.
    """
    sc = load("scaling")
    tasks = list(reversed(sc["tasks"]))  # longest prompt at the top
    names = [t["name"] for t in tasks]
    tiles = [t["causal_score_tiles"] for t in tasks]
    toks = [t["tokens"] for t in tasks]
    groups = [t["token_groups"] for t in tasks]
    measured = [t.get("full_pass_tag") == "[V]" for t in tasks]

    y = np.arange(len(tasks))

    for i, (tv, m) in enumerate(zip(tiles, measured)):
        ax.barh(
            i,
            tv,
            height=0.56,
            facecolor=SERVER if m else "white",
            edgecolor=SERVER if m else GRAY,
            linewidth=1.6 if m else 1.2,
            hatch=None if m else "///",
            zorder=3,
        )
        ax.text(
            tv + max(tiles) * 0.018,
            i,
            f"{tv:,} score tiles",
            va="center",
            ha="left",
            fontsize=7.8,
            fontweight="bold" if m else "normal",
            color=BLACK if m else GRAY,
        )

    anchor = next(t for t in tasks if t.get("full_pass_tag") == "[V]")
    ai = names.index(anchor["name"])
    ax.text(
        max(tiles) * 0.035,
        ai,
        f"measured complete inference: {anchor['full_pass_human']}",
        va="center",
        ha="left",
        fontsize=7.3,
        color="white",
        fontweight="bold",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{n}\n{tk} tokens · {g} groups" for n, tk, g in zip(names, toks, groups)],
        fontsize=7.2,
    )
    ax.set_xlabel(
        "Causal attention score tiles per block  (lower-triangular group pairs)",
        fontsize=8,
    )
    ax.set_xlim(0, max(tiles) * 1.52)
    ax.set_ylim(-0.62, len(tasks) - 0.30)
    despine(ax, keep=("bottom",))
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.legend(
        handles=[
            Patch(
                facecolor=SERVER, edgecolor=SERVER, label="complete inference measured"
            ),
            Patch(
                facecolor="white",
                edgecolor=GRAY,
                hatch="///",
                label="score-tile schedule only, not timed",
            ),
        ],
        frameon=False,
        fontsize=7.0,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.94),
        ncol=1,
    )

    excl = sc["out_of_encrypted_scope"][0]
    ax.text(
        0.0,
        -0.30,
        f"Outside the encrypted scope: {excl['name']} needs {excl['tokens']:,} tokens and "
        f"{excl['token_groups']} groups, or {excl['causal_score_tiles']:,} score tiles — "
        f"a different regime, not a longer prompt.",
        transform=ax.transAxes,
        fontsize=6.6,
        color=GRAY,
        ha="left",
        va="top",
    )

    _panel_title(
        ax,
        "Prompt length sets the causal score-tile schedule (derived, not timed)",
    )


def fig_systems(out: pathlib.Path) -> pathlib.Path:
    """The systems behaviour of one complete encrypted inference, as five panels.

    (a) sampled GPU use and (b) memory over the run; (c) how the encrypted-evaluation
    interval divides between the two parties; (d) host memory per stage under the bounded
    plaintext cache, against the separate unbounded-cache attempt; (e) the causal score-tile
    schedule implied by each task's prompt length.
    """
    fig = plt.figure(figsize=(7.1, 7.35), dpi=DPI)
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[0.80, 0.80, 1.35, 1.55],
        hspace=0.62,
        wspace=0.38,
    )
    ax_gpu = fig.add_subplot(gs[0, :])
    ax_mem = fig.add_subplot(gs[1, :], sharex=ax_gpu)
    ax_cost = fig.add_subplot(gs[2, 0])
    ax_stage = fig.add_subplot(gs[2, 1])
    ax_scale = fig.add_subplot(gs[3, :])

    _panel_trace(ax_gpu, ax_mem)
    _panel_cost_split(ax_cost)
    _panel_stage_memory(ax_stage)
    _panel_scaling(ax_scale)

    # Half-width panels need a wider letter offset: the axes are narrower, so the same
    # fraction of axes width lands on top of the panel title.
    for ax, letter, dx in (
        (ax_gpu, "a", -0.055),
        (ax_mem, "b", -0.055),
        (ax_cost, "c", -0.13),
        (ax_stage, "d", -0.22),
        (ax_scale, "e", -0.055),
    ):
        _panel_letter(ax, letter, x=dx, y=1.10)

    return emit(fig, out, "fig_systems")


# --- 9. Plaintext baseline --------------------------------------------------


def fig_baseline(out: pathlib.Path) -> pathlib.Path:
    L = Ledger()
    rows = [
        ("Polyadenylation signal\n(accuracy)", "base.gsr_accuracy", True),
        ("Core promoter\n(MCC)", "base.gue_prom_core_mcc", True),
        ("300 bp promoter\n(MCC)", "base.gue_prom_300_mcc", True),
        ("Splice site\n(MCC)", "base.gue_splice_mcc", True),
        ("mRNA abundance\n($r^2$)", "base.mrna_r2", False),
    ]
    labels = [r[0] for r in rows]
    ours = [L.value(r[1]) for r in rows]
    refs = [L.rows[r[1]].get("reference") for r in rows]
    in_scope = [r[2] for r in rows]

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11.2, 5.9), dpi=DPI)
    ax.bar(
        x - 0.185,
        ours,
        width=0.35,
        facecolor=SERVER,
        label="this work",
        **MEASURED_KW,
        zorder=3,
    )
    ax.bar(
        x + 0.185,
        refs,
        width=0.35,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=1.2,
        label="published reference",
        zorder=3,
    )

    for xi, (o, r) in enumerate(zip(ours, refs)):
        ax.text(
            xi - 0.185,
            o + 0.016,
            f"{o:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        if r is not None:
            ax.text(
                xi + 0.185,
                r + 0.016,
                f"{r:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=GRAY,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.legend(
        frameon=False, fontsize=10, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0)
    )
    despine(ax)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)

    # Shade and label the task that is outside the encrypted scope.
    for xi, inside in enumerate(in_scope):
        if not inside:
            ax.axvspan(xi - 0.42, xi + 0.42, color=PALE_ORANGE, alpha=0.55, zorder=0)
            ax.text(
                xi,
                0.035,
                "outside the\nencrypted scope",
                fontsize=9,
                color=AMBER,
                ha="center",
                va="bottom",
                style="italic",
                fontweight="bold",
                linespacing=1.35,
            )

    title(
        ax,
        "The released model, reproduced locally",
        "Signal recognition and abundance regression use the released fine-tuned heads. "
        "The promoter and splice-site rows use a locally fine-tuned head, as none was "
        "released.",
    )
    return emit(fig, out, "fig_baseline")


# --- Driver -----------------------------------------------------------------

FIGURES = {
    "graphical_abstract": fig_graphical_abstract,
    "architecture": fig_architecture,
    "packing": fig_packing,
    # "waterfall" is deliberately not generated. It needs two supported timing endpoints, and
    # the pre-optimization baseline has no committed dedicated-node measurement -- only an
    # attested value (see open_provenance in optimizations.yaml). The generator is kept so the
    # figure can be restored if that measurement ever lands.
    # The resource trace, cost split, stage memory, and prompt-length schedule are panels of
    # one composite figure; they are no longer generated as standalone PDFs.
    "systems": fig_systems,
}


def main() -> int:
    global PNG_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES), metavar="NAME")
    ap.add_argument(
        "--png", type=pathlib.Path, help="also write review PNGs to this directory"
    )
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("\n".join(FIGURES))
        return 0

    PNG_DIR = args.png
    for name in args.only or list(FIGURES):
        print(f"  {FIGURES[name](args.output_dir).relative_to(ROOT)}")
    print(
        f"{len(args.only or FIGURES)} figure(s) written to "
        f"{args.output_dir.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
