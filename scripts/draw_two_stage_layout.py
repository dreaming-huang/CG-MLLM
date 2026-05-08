"""Visualize a two-stage indoor scene layout using only bounding boxes.

Stage 1: a small number of large bounding boxes (coarse furniture layout).
Stage 2: the same large layout plus many small object bounding boxes.

Both figures share the same room, the same large boxes, and the same camera
view, so they can be placed side by side as a paper figure.

Run:
    python scripts/draw_two_stage_layout.py
Outputs:
    assets/stage1_large_layout.png
    assets/stage2_small_objects.png
    assets/two_stage_layout.png   (side-by-side composite)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """Axis-aligned bounding box defined by its min corner and size."""

    origin: tuple[float, float, float]
    size: tuple[float, float, float]
    color: str
    alpha: float = 0.18
    edge_width: float = 1.4
    edge_color: str | None = None  # defaults to a darker version of `color`

    def corners(self) -> np.ndarray:
        x, y, z = self.origin
        dx, dy, dz = self.size
        return np.array(
            [
                [x, y, z],
                [x + dx, y, z],
                [x + dx, y + dy, z],
                [x, y + dy, z],
                [x, y, z + dz],
                [x + dx, y, z + dz],
                [x + dx, y + dy, z + dz],
                [x, y + dy, z + dz],
            ]
        )

    def faces(self, include_bottom: bool = False) -> list[list[np.ndarray]]:
        c = self.corners()
        sides = [
            [c[4], c[5], c[6], c[7]],  # top
            [c[0], c[1], c[5], c[4]],  # front
            [c[2], c[3], c[7], c[6]],  # back
            [c[1], c[2], c[6], c[5]],  # right
            [c[0], c[3], c[7], c[4]],  # left
        ]
        if include_bottom:
            sides.append([c[0], c[1], c[2], c[3]])
        return sides

    def edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        c = self.corners()
        idx = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        return [(c[i], c[j]) for i, j in idx]


@dataclass
class Scene:
    room_size: tuple[float, float, float]
    large_boxes: list[Box] = field(default_factory=list)
    small_boxes: list[Box] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _darken(color: str, factor: float = 0.55) -> tuple[float, float, float, float]:
    r, g, b, a = to_rgba(color)
    return (r * factor, g * factor, b * factor, 1.0)


def draw_box(ax, box: Box, zorder: int = 5) -> None:
    edge_color = box.edge_color if box.edge_color is not None else _darken(box.color)
    face_rgba = to_rgba(box.color, alpha=box.alpha)

    faces = Poly3DCollection(
        box.faces(),
        facecolors=face_rgba,
        edgecolors=(0, 0, 0, 0),  # edges drawn separately for crispness
        linewidths=0,
        zorder=zorder,
    )
    ax.add_collection3d(faces)

    edges = Line3DCollection(
        box.edges(),
        colors=edge_color,
        linewidths=box.edge_width,
        zorder=zorder + 0.1,
    )
    ax.add_collection3d(edges)


def draw_room(ax, size: tuple[float, float, float]) -> None:
    """Draw the room as a thin wireframe, with a faint floor."""
    w, d, h = size

    floor = Poly3DCollection(
        [[(0, 0, 0), (w, 0, 0), (w, d, 0), (0, d, 0)]],
        facecolors=(0.96, 0.96, 0.97, 1.0),
        edgecolors=(0, 0, 0, 0),
        zorder=0,
    )
    ax.add_collection3d(floor)

    corners = np.array(
        [
            [0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0],
            [0, 0, h], [w, 0, h], [w, d, h], [0, d, h],
        ]
    )
    edge_idx = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    edges = [(corners[i], corners[j]) for i, j in edge_idx]
    room_edges = Line3DCollection(
        edges, colors=(0.35, 0.35, 0.4, 0.9), linewidths=1.1, zorder=1,
    )
    ax.add_collection3d(room_edges)


def setup_axes(ax, size: tuple[float, float, float]) -> None:
    w, d, h = size
    ax.set_xlim(-0.2, w + 0.2)
    ax.set_ylim(-0.2, d + 0.2)
    ax.set_zlim(0, h + 0.2)
    ax.set_box_aspect((w, d, h))

    ax.view_init(elev=20, azim=-58)
    # Disable matplotlib's automatic z-order computation so we can control it.
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    try:
        ax.computed_zorder = False
    except AttributeError:
        pass

    ax.set_axis_off()
    try:  # newer matplotlib supports per-pane styling
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------

# Color palette tuned for clean paper-style figures.
C_GREEN = "#4FA37A"   # wardrobe
C_PURPLE = "#9B7BC8"  # bed / nightstand
C_ORANGE = "#E8A24A"  # table
C_BLUE = "#5B86C5"    # chair / shelf

SMALL_PALETTE = [
    "#E25A4D",  # red
    "#F2B544",  # yellow
    "#5BB3D6",  # cyan
    "#8EBF63",  # light green
    "#C97BB8",  # pink
    "#7A7BD6",  # indigo
    "#E8865A",  # orange
    "#62B59B",  # teal
]


def build_scene() -> Scene:
    # Room: width(x) x depth(y) x height(z).  y grows toward the back.
    room = (6.0, 5.0, 3.0)

    large_boxes = [
        # Wardrobe — tall, against the back-left wall.
        Box(origin=(0.30, 3.60, 0.0), size=(1.10, 1.30, 2.45),
            color=C_GREEN, alpha=0.30, edge_width=1.6),
        # Bed — low, in the back-right, head against back wall.
        Box(origin=(3.60, 2.80, 0.0), size=(1.90, 2.10, 0.55),
            color=C_PURPLE, alpha=0.30, edge_width=1.6),
        # Nightstand — small, just to the left of the bed (clearly separated).
        Box(origin=(2.85, 3.55, 0.0), size=(0.55, 0.55, 0.75),
            color=C_PURPLE, alpha=0.34, edge_width=1.6),
        # Table — in the middle of the room, well separated from other items.
        Box(origin=(1.80, 1.30, 0.0), size=(1.80, 1.10, 0.78),
            color=C_ORANGE, alpha=0.34, edge_width=1.6),
        # Chair — front-left, in front of the wardrobe.
        Box(origin=(0.55, 0.55, 0.0), size=(0.75, 0.75, 1.05),
            color=C_BLUE, alpha=0.30, edge_width=1.6),
    ]

    small_boxes = build_small_boxes()

    return Scene(room_size=room, large_boxes=large_boxes, small_boxes=small_boxes)


def build_small_boxes() -> list[Box]:
    rng = random.Random(7)

    def rand_color() -> str:
        return rng.choice(SMALL_PALETTE)

    def small(o, s, c=None) -> Box:
        return Box(origin=o, size=s, color=c or rand_color(), alpha=0.32, edge_width=1.0)

    boxes: list[Box] = []

    # On the table (x=1.80..3.60, y=1.30..2.40, top z=0.78).
    tt = 0.78
    boxes += [
        small((2.00, 1.50, tt), (0.30, 0.30, 0.34)),   # cup / vase
        small((2.45, 1.45, tt), (0.55, 0.35, 0.05)),   # book / pad
        small((3.10, 1.55, tt), (0.20, 0.20, 0.42)),   # lamp
        small((2.80, 1.95, tt), (0.30, 0.25, 0.18)),   # small box
        small((3.20, 2.05, tt), (0.30, 0.22, 0.08)),   # tray
    ]

    # On the nightstand (x=2.85..3.40, y=3.55..4.10, top z=0.75).
    nt = 0.75
    boxes += [
        small((2.95, 3.65, nt), (0.20, 0.22, 0.34)),   # lamp
        small((3.18, 3.85, nt), (0.18, 0.16, 0.04)),   # book
    ]

    # On the bed (x=3.60..5.50, y=2.80..4.90, top z=0.55).
    bt = 0.55
    boxes += [
        small((3.75, 2.95, bt), (0.55, 0.35, 0.18)),   # pillow
        small((3.75, 3.40, bt), (0.55, 0.35, 0.18)),   # pillow
        small((4.55, 3.40, bt), (0.65, 0.35, 0.06)),   # folded blanket
    ]

    # On top of wardrobe (top z=2.45).
    wt = 2.45
    boxes += [
        small((0.45, 3.80, wt), (0.40, 0.35, 0.28)),
        small((0.95, 3.95, wt), (0.30, 0.30, 0.22)),
    ]

    # On the chair (top z=1.05).
    ct = 1.05
    boxes += [small((0.70, 0.70, ct), (0.32, 0.32, 0.30))]

    # Floor scatter — pots, bins, baskets.
    boxes += [
        small((1.30, 0.55, 0.0), (0.40, 0.30, 0.20)),   # basket
        small((4.50, 0.55, 0.0), (0.32, 0.32, 0.50)),   # plant pot
        small((1.85, 4.45, 0.0), (0.30, 0.30, 0.55)),   # plant pot
        small((5.45, 1.10, 0.0), (0.28, 0.28, 0.40)),   # bin
    ]

    return boxes


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def _proximity(box: Box, elev_deg: float, azim_deg: float) -> float:
    """Camera-space depth of a box center.  Larger = closer to camera."""
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    cam_dir = np.array([np.cos(elev) * np.cos(azim),
                        np.cos(elev) * np.sin(azim),
                        np.sin(elev)])
    cx = box.origin[0] + box.size[0] / 2
    cy = box.origin[1] + box.size[1] / 2
    cz = box.origin[2] + box.size[2] / 2
    return float(cam_dir @ np.array([cx, cy, cz]))


def render_stage(scene: Scene, include_small: bool, ax) -> None:
    setup_axes(ax, scene.room_size)
    draw_room(ax, scene.room_size)

    elev, azim = 20.0, -58.0

    boxes = list(scene.large_boxes)
    if include_small:
        boxes += scene.small_boxes

    # Sort back-to-front, then assign monotonically increasing zorder.
    boxes_with_z = sorted(boxes, key=lambda b: _proximity(b, elev, azim))
    for i, box in enumerate(boxes_with_z):
        draw_box(ax, box, zorder=10 + i * 2)


def save_single(scene: Scene, include_small: bool, out_path: str, title: str) -> None:
    fig = plt.figure(figsize=(6.0, 4.4), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    render_stage(scene, include_small, ax)
    ax.set_title(title, fontsize=13, pad=2)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def save_composite(scene: Scene, out_path: str) -> None:
    fig = plt.figure(figsize=(11.5, 4.4), dpi=220)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    render_stage(scene, include_small=False, ax=ax1)
    ax1.set_title("Stage 1: Coarse Layout (Large Objects)", fontsize=13, pad=2)

    render_stage(scene, include_small=True, ax=ax2)
    ax2.set_title("Stage 2: Fine Layout (+ Small Objects)", fontsize=13, pad=2)

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95, wspace=0.05)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "assets")
    os.makedirs(out_dir, exist_ok=True)

    scene = build_scene()

    save_single(
        scene,
        include_small=False,
        out_path=os.path.join(out_dir, "stage1_large_layout.png"),
        title="Stage 1: Coarse Layout (Large Objects)",
    )
    save_single(
        scene,
        include_small=True,
        out_path=os.path.join(out_dir, "stage2_small_objects.png"),
        title="Stage 2: Fine Layout (+ Small Objects)",
    )
    save_composite(scene, out_path=os.path.join(out_dir, "two_stage_layout.png"))

    print("Saved:")
    for name in ("stage1_large_layout.png", "stage2_small_objects.png", "two_stage_layout.png"):
        print(" -", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
