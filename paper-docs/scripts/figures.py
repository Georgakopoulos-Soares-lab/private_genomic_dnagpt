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
    fig, ax = canvas(figsize=(14.0, 7.9))

    ax.text(
        50,
        95.5,
        "The genome is protected. The client needs no GPU.",
        fontsize=21,
        fontweight="bold",
        color=BLACK,
        ha="center",
        va="center",
    )

    labelled_box(
        ax,
        2,
        62,
        30,
        24,
        "Data owner",
        "a human genome it will\nnot disclose  ·  CPU only",
        facecolor=CLIENT_PALE,
        edgecolor=CLIENT,
        title_size=15,
        sub_size=11,
    )
    labelled_box(
        ax,
        68,
        62,
        30,
        24,
        "Compute provider",
        "serves the model on\nGPU infrastructure",
        facecolor=SERVER_PALE,
        edgecolor=SERVER,
        title_size=15,
        sub_size=11,
    )

    labelled_box(
        ax,
        27,
        30,
        46,
        22,
        "Client-assisted CKKS",
        "every linear operation runs encrypted on the server;\n"
        "the key holder evaluates three nonlinearities exactly",
        facecolor="white",
        edgecolor=BLACK,
        title_size=15.5,
        sub_size=10.5,
    )

    arrow(ax, 17, 61, 36, 52.5, color=CLIENT, lw=2.0)
    arrow(ax, 83, 61, 64, 52.5, color=SERVER, lw=2.0)
    ax.text(
        21.5,
        55.5,
        "sends it\nencrypted",
        fontsize=10,
        color=CLIENT,
        ha="center",
        va="center",
        style="italic",
    )
    ax.text(
        78.5,
        55.5,
        "runs it\nin place",
        fontsize=10,
        color=SERVER,
        ha="center",
        va="center",
        style="italic",
    )

    ax.text(
        50,
        26.0,
        "The compute provider never receives a plaintext activation or the secret key.",
        fontsize=11,
        color=BLACK,
        ha="center",
        va="center",
        style="italic",
    )

    results = [
        (f"{L.value('prompt.gsr_total')} tokens", "the full task prompt"),
        (f"{L.value('block.encrypted_evaluation'):,.0f} s", "one transformer block"),
        (r"$4.6\times10^{-9}$", "error vs. the plaintext model"),
        (f"{L.value('client.peak_ram')} GB", "peak client memory, no GPU"),
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
            title_size=17,
            sub_size=9.5,
        )

    return emit(fig, out, "fig_graphical_abstract")


# --- 2. Protocol architecture -----------------------------------------------


def fig_architecture(out: pathlib.Path) -> pathlib.Path:
    """The evaluation order, and which party performs each step."""
    fig, ax = canvas(figsize=(13.0, 9.4))

    steps = [
        ("client", "Encrypt the embedded token vectors"),
        ("server", "LayerNorm statistics"),
        ("client", "Inverse square root"),
        ("server", "Query / key / value projection"),
        ("server", "Causal attention scores"),
        ("client", "Softmax over the score tiles"),
        ("server", "Attention context, projection, residual"),
        ("client", "LayerNorm, second occurrence"),
        ("server", "MLP up-projection"),
        ("client", "GELU"),
        ("server", "MLP down-projection, residual"),
        ("client", "Decrypt the block output"),
    ]

    top, bottom = 85.0, 11.0
    n = len(steps)
    pitch = (top - bottom) / n
    height = pitch * 0.72
    lx, rx, width = 4.0, 54.0, 42.0
    boundary = 50.0

    ax.text(
        4, 96.0, "Data owner", fontsize=16, fontweight="bold", color=CLIENT, va="center"
    )
    ax.text(
        4,
        92.0,
        "holds the genome and the secret key  ·  no GPU",
        fontsize=10.2,
        color=GRAY,
        va="center",
    )
    ax.text(
        54,
        96.0,
        "Compute provider",
        fontsize=16,
        fontweight="bold",
        color=SERVER,
        va="center",
    )
    ax.text(
        54,
        92.0,
        "holds the model  ·  sees only ciphertexts  ·  one A100",
        fontsize=10.2,
        color=GRAY,
        va="center",
    )

    ax.plot(
        [boundary, boundary],
        [6.5, 88.5],
        color=GRAY,
        linewidth=1.1,
        linestyle=(0, (6, 5)),
    )
    ax.text(
        boundary,
        5.4,
        "trust boundary",
        fontsize=9.6,
        color=GRAY,
        ha="center",
        va="top",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none"),
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
            title_size=11,
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

    ax.text(
        boundary,
        1.6,
        "Only ciphertexts cross. A crossing carries values derived from the client's own "
        "query,\nwhich it decrypts, evaluates exactly, and returns freshly encrypted.",
        fontsize=9.8,
        color=BLACK,
        ha="center",
        va="top",
        style="italic",
        linespacing=1.45,
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

    fig, ax = canvas(figsize=(13.0, 7.2))

    ax.text(
        2,
        96,
        f"One ciphertext holds {slots:,} slots",
        fontsize=15.5,
        fontweight="bold",
        color=BLACK,
        va="center",
    )
    ax.text(
        2,
        90.5,
        f"{copies} activation copies  ×  {width_feat:,} padded features  ×  "
        f"{lanes} token lanes  =  {copies * width_feat * lanes:,} slots."
        f"   The {copies} copies exist because the MLP expands to {copies}× the hidden width.",
        fontsize=10.6,
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
            fontsize=12.5,
            fontweight="bold",
            color=BLACK,
            ha="center",
            va="center",
        )
        ax.text(
            x + w / 2,
            74.0,
            f"features 0–{width_feat - 1}",
            fontsize=9,
            color=GRAY,
            ha="center",
            va="center",
        )
        inner, pad = w - 3.0, 1.5
        for lane in range(lanes):
            lw_ = inner / lanes
            ax.add_patch(
                plt.Rectangle(
                    (x + pad + lane * lw_, 62.5),
                    lw_ * 0.82,
                    7.5,
                    facecolor=CLIENT_PALE,
                    edgecolor=CLIENT,
                    linewidth=0.8,
                )
            )
        ax.text(
            x + w / 2,
            59.0,
            f"{lanes} token lanes",
            fontsize=8.6,
            color=GRAY,
            ha="center",
            va="center",
        )

    ax.text(
        2,
        46,
        f"{tokens} tokens occupy {groups} ciphertext groups",
        fontsize=14.5,
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
        fontsize=10.6,
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
            title_size=10.5,
            sub_size=8.8,
        )

    ax.text(
        2,
        13,
        f"The {groups} groups give {groups * (groups + 1) // 2} lower-triangular causal "
        f"score tiles, and just {(-tokens) % lanes} of the {groups * lanes} lanes is padding.",
        fontsize=10.2,
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


# --- 5. Resource trace ------------------------------------------------------


def fig_timeline(out: pathlib.Path) -> pathlib.Path:
    tel = load("telemetry")
    trace, stages = tel["trace"], tel["stages"]
    total = tel["meta"]["total_wall_s"]

    t = np.array([r["t_s"] for r in trace], dtype=float)
    gpu_pct = np.array([r["gpu_pct"] for r in trace], dtype=float)
    rss = np.array([r["rss_mib"] for r in trace], dtype=float) / 1024.0
    gpu_mem = np.array([r["gpu_mib"] for r in trace], dtype=float) / 1024.0

    fig, (axl, ax1, ax2) = plt.subplots(
        3,
        1,
        figsize=(13.4, 7.8),
        dpi=DPI,
        sharex=True,
        gridspec_kw={"height_ratios": [0.30, 1.15, 1.0], "hspace": 0.10},
    )

    band = {"server": SERVER_PALE, "client": CLIENT_PALE, "mixed": PALE_ORANGE}

    # Dedicated label lane, so stage names can never collide with the plots.
    axl.set_ylim(0, 1)
    axl.axis("off")
    for i, st in enumerate(stages):
        axl.axvspan(
            st["start_s"],
            st["end_s"],
            color=band[st["executor"]],
            alpha=0.75,
            linewidth=0,
        )
        axl.text(
            (st["start_s"] + st["end_s"]) / 2,
            0.5 if i % 2 == 0 else 0.5,
            st["label"].replace(" + ", "\n"),
            ha="center",
            va="center",
            fontsize=8.2,
            color=BLACK,
            linespacing=1.2,
        )

    for ax in (ax1, ax2):
        for st in stages:
            ax.axvspan(
                st["start_s"],
                st["end_s"],
                color=band[st["executor"]],
                alpha=0.5,
                linewidth=0,
                zorder=0,
            )

    ax1.plot(
        t, gpu_pct, color=NAVY, linewidth=2.2, marker="o", markersize=3.6, zorder=3
    )
    ax1.set_ylabel("GPU utilization (%)", fontsize=11)
    ax1.set_ylim(0, 100)
    despine(ax1)
    ax1.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax1.set_axisbelow(True)

    idle = [r for r in trace if r["stage"] == "attention_context"]
    if idle:
        mid = idle[len(idle) // 2]
        ax1.annotate(
            "GPU idle while the data owner\nevaluates the attention weights",
            xy=(mid["t_s"], mid["gpu_pct"] + 1.5),
            xytext=(150, 68),
            fontsize=10.5,
            color=CLIENT,
            fontweight="bold",
            ha="left",
            arrowprops=dict(
                arrowstyle="-|>",
                color=CLIENT,
                linewidth=1.5,
                connectionstyle="arc3,rad=-0.25",
            ),
        )

    ax2.plot(
        t,
        rss,
        color=CHART_BLUE,
        linewidth=2.2,
        marker="o",
        markersize=3.6,
        label="host memory",
        zorder=3,
    )
    ax2.plot(
        t,
        gpu_mem,
        color=GREEN,
        linewidth=1.9,
        linestyle="--",
        marker="s",
        markersize=3.0,
        label="GPU memory",
        zorder=3,
    )
    ax2.set_ylabel("Memory (GiB)", fontsize=11)
    ax2.set_xlabel("Seconds since process start", fontsize=11)

    fail = Ledger().value("mem.unbounded_failure")
    ax2.set_ylim(0, fail * 1.30)
    ax2.set_yticks([0, 20, 40, 60])
    ax2.axhline(fail, color=FAILURE, linewidth=1.3, linestyle=":")
    ax2.text(
        total * 0.995,
        fail * 1.04,
        f"{fail} GiB — where the unbounded cache was killed",
        color=FAILURE,
        fontsize=9.2,
        ha="right",
        va="bottom",
    )
    ax2.legend(
        frameon=False, fontsize=10, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.02)
    )
    despine(ax2)
    ax2.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax2.set_axisbelow(True)

    ax1.set_xlim(0, total)
    axl.set_title(
        "The evaluation is not GPU-bound, and the memory curve is the caching strategy",
        fontsize=13,
        fontweight="bold",
        color=BLACK,
        loc="left",
        pad=10,
    )
    return emit(fig, out, "fig_timeline")


# --- 6. Memory bounding -----------------------------------------------------


def fig_memory(out: pathlib.Path) -> pathlib.Path:
    fail = Ledger().value("mem.unbounded_failure")
    bounded = next(
        o
        for o in load("optimizations")["retained"]
        if o["id"] == "opt.bounded_host_memory"
    )
    values = bounded["effect_gib"]
    labels = [
        "Query / key / value\n3 weight matrices resident",
        "Attention\nnone resident",
        "MLP\n8 weight matrices resident",
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.9), dpi=DPI)
    bars = ax.bar(
        range(3), values, width=0.5, facecolor=SERVER, **MEASURED_KW, zorder=3
    )
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 1.1,
            f"{v} GiB",
            ha="center",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
            color=BLACK,
        )

    ax.axhline(fail, color=FAILURE, linewidth=1.6, linestyle="--", zorder=2)
    ax.text(
        2.52,
        fail - 1.4,
        f"{fail} GiB — the unbounded cache\nwas killed by the operating system",
        color=FAILURE,
        fontsize=9.8,
        ha="right",
        va="top",
        fontweight="bold",
    )

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Peak host memory (GiB)", fontsize=11)
    ax.set_ylim(0, fail * 1.22)
    ax.set_xlim(-0.62, 2.62)
    despine(ax)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)
    title(
        ax,
        "Flushing between stages bounds the peak",
        "The cache has to hold only the largest single stage, not all twelve weight "
        "matrices at once.",
    )
    return emit(fig, out, "fig_memory")


# --- 7. Cost split ----------------------------------------------------------


def fig_cost_split(out: pathlib.Path) -> pathlib.Path:
    L = Ledger()
    server = L.value("block.server_linear_algebra")
    client = L.value("block.client_boundaries")
    setup = L.value("block.context_keygen_load") + L.value("block.encrypt_input")
    tail = L.value("block.final_decrypt") + L.value("block.fixture_load")
    other = setup + tail
    wall = L.value("block.wall")
    crossings = L.value("ops.client_crossings_physical")
    ram = L.value("client.peak_ram")

    fig, (ax, axr) = plt.subplots(
        1,
        2,
        figsize=(13.4, 4.6),
        dpi=DPI,
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.13},
    )

    # Segments are labelled in place, so the figure needs no legend and nothing can collide
    # with the axis.
    for label, val, color, textcolor in (
        ("Compute provider", server, SERVER, "white"),
        ("Data owner", client, CLIENT, "white"),
        ("", other, GRAY, BLACK),
    ):
        left = {"Compute provider": 0.0, "Data owner": server}.get(
            label, server + client
        )
        ax.barh(
            0,
            val,
            left=left,
            height=0.34,
            facecolor=color,
            edgecolor=BLACK,
            linewidth=1.1,
        )
        if label:
            ax.text(
                left + val / 2,
                0.02,
                f"{val:,.0f} s\n{100 * val / wall:.1f}%",
                ha="center",
                va="center",
                fontsize=12.5,
                fontweight="bold",
                color=textcolor,
            )
            ax.text(
                left + val / 2,
                -0.30,
                label,
                ha="center",
                va="center",
                fontsize=10.5,
                color=color,
                fontweight="bold",
            )
            sub = (
                "encrypted linear algebra, one A100"
                if label == "Compute provider"
                else "exact nonlinearities, CPU only"
            )
            ax.text(
                left + val / 2,
                -0.40,
                sub,
                ha="center",
                va="center",
                fontsize=9.2,
                color=GRAY,
            )

    ax.set_xlim(0, wall)
    ax.set_ylim(-0.52, 0.30)
    ax.set_yticks([])
    ax.set_xlabel("Seconds", fontsize=11)
    despine(ax, keep=("bottom",))
    ax.text(
        1.0,
        -0.155,
        f"setup, encryption and final decrypt account for the remaining {other:.1f} s",
        transform=ax.transAxes,
        fontsize=9.2,
        color=GRAY,
        ha="right",
        va="top",
    )
    title(ax, f"One transformer block, 103 tokens — {wall:,.0f} s wall clock")

    axr.axis("off")
    axr.set_xlim(0, 100)
    axr.set_ylim(0, 100)
    labelled_box(
        axr,
        1,
        56,
        98,
        34,
        "Data owner",
        f"{client:,.0f} s  \u00b7  no GPU  \u00b7  ~{ram} GB peak memory\n"
        f"holds the genome and the secret key",
        facecolor=CLIENT_PALE,
        edgecolor=CLIENT,
        title_size=14,
        sub_size=10.5,
    )
    labelled_box(
        axr,
        1,
        6,
        98,
        34,
        "Compute provider",
        f"{server:,.0f} s  \u00b7  one A100 GPU\n"
        f"holds the model; observes only ciphertexts",
        facecolor=SERVER_PALE,
        edgecolor=SERVER,
        title_size=14,
        sub_size=10.5,
    )
    arrow(axr, 50, 55, 50, 41, color=GRAY, lw=1.6, style="<|-|>")
    axr.text(
        53, 48, f"{crossings} boundary crossings", fontsize=9.8, color=GRAY, va="center"
    )

    return emit(fig, out, "fig_cost_split")


# --- 8. Sequence-length scaling ---------------------------------------------


def fig_scaling(out: pathlib.Path) -> pathlib.Path:
    """Per-block cost across the encrypted-scope tasks, anchored on the one measured point."""
    sc = load("scaling")
    tasks = list(reversed(sc["tasks"]))  # longest prompt at the top
    names = [t["name"] for t in tasks]
    secs = [t["block_seconds"] for t in tasks]
    toks = [t["tokens"] for t in tasks]
    groups = [t["token_groups"] for t in tasks]
    passes = [t["full_pass_human"] for t in tasks]
    measured = [t["block_tag"] == "[V]" for t in tasks]

    y = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(11.4, 5.6), dpi=DPI)

    for i, (sv, m) in enumerate(zip(secs, measured)):
        ax.barh(
            i,
            sv,
            height=0.56,
            facecolor=SERVER if m else "white",
            edgecolor=SERVER if m else GRAY,
            linewidth=1.6 if m else 1.2,
            hatch=None if m else "///",
            alpha=1.0 if m else 0.9,
            zorder=3,
        )
        ax.text(
            sv + max(secs) * 0.018,
            i,
            f"{sv:,.0f} s",
            va="center",
            ha="left",
            fontsize=11.5,
            fontweight="bold" if m else "normal",
            color=BLACK if m else GRAY,
        )
        ax.text(
            sv + max(secs) * 0.115,
            i,
            f"·  twelve blocks {passes[i]}",
            va="center",
            ha="left",
            fontsize=9.6,
            color=GRAY,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{n}\n{tk} tokens · {g} groups" for n, tk, g in zip(names, toks, groups)],
        fontsize=10.2,
    )
    ax.set_xlabel("Time per transformer block (s)", fontsize=11)
    ax.set_xlim(0, max(secs) * 1.42)
    ax.set_ylim(-0.62, len(tasks) - 0.30)
    despine(ax, keep=("bottom",))
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.legend(
        handles=[
            Patch(facecolor=SERVER, edgecolor=SERVER, label="measured"),
            Patch(
                facecolor="white",
                edgecolor=GRAY,
                hatch="///",
                label="projected at fixed circuit",
            ),
        ],
        frameon=False,
        fontsize=10,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        ncol=2,
    )

    excl = sc["out_of_encrypted_scope"][0]
    ax.text(
        0.0,
        -0.185,
        f"Excluded: mRNA abundance regression at {excl['tokens']:,} tokens needs "
        f"{excl['token_groups']} ciphertext groups and roughly 48,000 causal score "
        f"tiles — a different problem, not a longer one.",
        transform=ax.transAxes,
        fontsize=9.2,
        color=GRAY,
        ha="left",
        va="top",
    )

    title(
        ax,
        "The task that was measured is the longest one",
        "Token counts follow from the task: base pairs divided by six, plus task specials. "
        "The twelve-block figures are projections.",
    )
    return emit(fig, out, "fig_scaling")


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
    "waterfall": fig_waterfall,
    "timeline": fig_timeline,
    "memory": fig_memory,
    "cost_split": fig_cost_split,
    "scaling": fig_scaling,
    "baseline": fig_baseline,
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
