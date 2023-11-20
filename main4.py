#!/usr/bin/env python3
from __future__ import annotations

from typing import Literal, Optional

from mpmath import mp  # type: ignore[import-untyped]

from example_drawers import ExampleFixedColorDrawer
from triangle_side import TriangleSide
from util import (STEP_BASE, STEP_LEFT, STEP_RIGHT, AngleID, Subtimers,
                  TriangleSideType, create_canvas, get_angles)

mp.prec = 100
radius = 1

random_angle_count: Literal[0, 1, 2] = 0
right_angle: Optional[AngleID] = None

# angles s.t. height = base
theta_A = mp.pi - 2*mp.atan(2)
theta_B = mp.atan(2)


def main(timers: Subtimers) -> None:
    # theta A, theta B, theta C
    angles = get_angles(theta_A, theta_B, random_angle_count, right_angle)
    with timers.pushed('initializing'):
        a = TriangleSide(TriangleSideType.A, *angles, right_angle, radius)
        b = TriangleSide(TriangleSideType.B, *angles, right_angle, radius)
        c = TriangleSide(TriangleSideType.C, *angles, right_angle, radius)
    with timers.pushed('calculating'):
        with timers.pushed('calculating side a'):
            a.step_all(1, STEP_BASE)
            a.step_all(1, STEP_RIGHT)
            print(f"{len(a.tree)} triangles calculated")
        with timers.pushed('calculating side b'):
            print(f"{len(b.tree)} triangles calculated")
        with timers.pushed('calculating side c'):
            c.step_all(1, STEP_BASE)
            c.step_all(1, STEP_LEFT)
            print(f"{len(c.tree)} triangles calculated")
    with timers.pushed('finalizing'):
        tree_a = a.finalized_tree()
        tree_b = b.finalized_tree()
        tree_c = c.finalized_tree()
    with timers.pushed('drawing'):
        with timers.pushed('setting up figure and drawers'):
            fig, ax = create_canvas(radius)
            draw_a = ExampleFixedColorDrawer(fig, ax, tree_a, draw_zero=True)
            draw_b = ExampleFixedColorDrawer(fig, ax, tree_b)
            draw_c = ExampleFixedColorDrawer(fig, ax, tree_c)
        with timers.pushed('drawing side a'):
            draw_a.draw_tree()
        with timers.pushed('drawing side b'):
            draw_b.draw_tree()
        with timers.pushed('drawing side c'):
            draw_c.draw_tree()
    with timers.pushed('saving'):
        fig.savefig("output/output_fixed.png", pad_inches=0)


if __name__ == '__main__':
    with Subtimers() as timers, timers.pushed("main"):
        main(timers)
