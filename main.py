#!/usr/bin/env python3
from __future__ import annotations

from typing import Literal, Optional

from mpmath import mp  # type: ignore[import-untyped]

from example_data import PFP, PFP2
from example_drawers import (ColorDataDrawerWithSymbolAlt,
                             ColorDataDrawerWithSymbolMain)
from triangle_side import TriangleSide
from util import (STEP_BASE, STEP_HORIZONTAL, STEP_NONBASE, AngleID, Subtimers,
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
        a = TriangleSide(TriangleSideType.A, *angles,
                         right_angle, radius, data=(PFP, PFP2))
        b = TriangleSide(TriangleSideType.B, *angles,
                         right_angle, radius, data=(PFP, PFP2))
        c = TriangleSide(TriangleSideType.C, *angles,
                         right_angle, radius, data=(PFP, PFP2))
    with timers.pushed('calculating'):
        with timers.pushed('calculating side a'):
            a.step_all(2, STEP_BASE)
            a.step_all(12, STEP_NONBASE)
            a.step_all(12, STEP_HORIZONTAL)
            triangle_count_a = len(a.tree)
            print(f"{triangle_count_a} triangles calculated")
        with timers.pushed('calculating side b'):
            b.step_all(2, STEP_BASE)
            b.step_all(12, STEP_NONBASE)
            b.step_all(12, STEP_HORIZONTAL)
            triangle_count_b = len(b.tree)
            print(f"{triangle_count_b} triangles calculated")
        with timers.pushed('calculating side c'):
            c.step_all(2, STEP_BASE)
            c.step_all(12, STEP_NONBASE)
            c.step_all(12, STEP_HORIZONTAL)
            triangle_count_c = len(c.tree)
            print(f"{triangle_count_c} triangles calculated")
    with timers.pushed('finalizing'):
        tree_a = a.finalized_tree()
        tree_b = b.finalized_tree()
        tree_c = c.finalized_tree()
    with timers.pushed('drawing'):
        with timers.pushed('drawing main profile picture'):
            with timers.pushed('setting up figure and drawers'):
                fig_1, ax_1 = create_canvas(radius)
                draw_a_1 = ColorDataDrawerWithSymbolMain(fig_1, ax_1, tree_a,
                                                         draw_zero=True)
                draw_b_1 = ColorDataDrawerWithSymbolMain(fig_1, ax_1, tree_b)
                draw_c_1 = ColorDataDrawerWithSymbolMain(fig_1, ax_1, tree_c)
            with timers.pushed('drawing side a'):
                draw_a_1.draw_tree()
            with timers.pushed('drawing side b'):
                draw_b_1.draw_tree()
            with timers.pushed('drawing side c'):
                draw_c_1.draw_tree()
        with timers.pushed('drawing alt profile picture'):
            with timers.pushed('setting up figure and drawers'):
                fig_2, ax_2 = create_canvas(radius)
                draw_a_2 = ColorDataDrawerWithSymbolAlt(fig_2, ax_2, tree_a,
                                                        draw_zero=True)
                draw_b_2 = ColorDataDrawerWithSymbolAlt(fig_2, ax_2, tree_b)
                draw_c_2 = ColorDataDrawerWithSymbolAlt(fig_2, ax_2, tree_c)
            with timers.pushed('drawing side a'):
                draw_a_2.draw_tree()
            with timers.pushed('drawing side b'):
                draw_b_2.draw_tree()
            with timers.pushed('drawing side c'):
                draw_c_2.draw_tree()
    with timers.pushed('saving'):
        with timers.pushed('saving main profile picture'):
            fig_1.savefig("output/output.png", pad_inches=0)
        with timers.pushed('saving alt profile picture'):
            fig_2.savefig("output/output_alt.png", pad_inches=0)


if __name__ == '__main__':
    with Subtimers() as timers, timers.pushed("main"):
        main(timers)
