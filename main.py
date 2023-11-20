#!/usr/bin/env python3
from __future__ import annotations

from typing import Literal, Optional, TYPE_CHECKING, Any
from data import Data
from itertools import chain

from mpmath import mp  # type: ignore[import-untyped]
import mpmath

from example_data import PFP, PFP2
from drawer import MPLDrawer
from example_drawers import (ColorDataDrawerWithSymbolAlt,
                             ColorDataDrawerWithSymbolMain, KyzaDrawer, ExampleRandomHPLuvColorDrawer, ExampleFixedColorDrawer)
from triangle_side import TriangleSide
from util import (STEP_BASE, STEP_HORIZONTAL, STEP_NONBASE, STEP_LEFT, STEP_RIGHT, AngleID, Subtimers,
                  TriangleSideType)
if TYPE_CHECKING:
    from util import RealNumber
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

mp.prec = 100
radius = 1

def get_angles(
    angle1: RealNumber, angle2: RealNumber,
    random_angle_count: Literal[0, 1, 2] = 0,
    right_angle: Optional[AngleID] = None,
) -> tuple[RealNumber, RealNumber, RealNumber]:

    match right_angle, random_angle_count:
        case None, 0:
            theta_A = angle1
            theta_B = angle2
            theta_C = mp.pi - theta_A - theta_B
        case None, 1:
            theta_A = angle1
            theta_B = mpmath.rand() * (mp.pi-angle1)
            theta_C = mp.pi - theta_A - theta_B
        case None, 2:
            theta_A = mpmath.rand() * mp.pi
            theta_B = mpmath.rand() * (mp.pi-theta_A)
            theta_C = mp.pi - theta_A - theta_B
        case AngleID.A, 0:
            theta_A = mp.pi/2
            theta_B = angle2
            theta_C = mp.pi/2 - theta_B
        case AngleID.A, 1:
            theta_A = mp.pi/2
            theta_B = mpmath.rand() * mp.pi/2
            theta_C = mp.pi/2 - theta_B
        case AngleID.B, 0:
            theta_A = angle1
            theta_B = mp.pi/2
            theta_C = mp.pi/2 - theta_A
        case AngleID.B, 1:
            theta_A = mpmath.rand() * mp.pi/2
            theta_B = mp.pi/2
            theta_C = mp.pi/2 - theta_A
        case AngleID.C, 0:
            theta_A = angle1
            theta_B = mp.pi/2 - theta_A
            theta_C = mp.pi/2
        case AngleID.C, 1:
            theta_A = mpmath.rand() * mp.pi/2
            theta_B = mp.pi/2 - theta_A
            theta_C = mp.pi/2
        case AngleID.A | AngleID.B | AngleID.C, 2:
            raise ValueError(
                "Cannot randomize two angles and "
                "have a right angle at the same time."
            )
        case _:
            # doing this at runtime because static type checkers
            # don't type narrow tuples apparently
            # i'd have to nest match statements if i wanted to
            # do it 'properly'
            # i am sure this will never change either way so whatever
            assert False, "reached supposedly unreachable code"

    return theta_A, theta_B, theta_C


def create_canvas(radius: RealNumber) -> tuple[Figure, Axes]:
    fig = plt.figure(figsize=(204.8, 204.8))
    ax = Axes(fig, (0, 0, 1, 1))
    ax.set_xlim(left=-radius, right=radius)
    ax.set_ylim(bottom=-radius, top=radius)
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.margins(0, 0)
    return fig, ax


def process_picture(
    timers: Subtimers,
    name: str,
    *,
    side_data: tuple[type[Data], ...] = (),
    side_a_data: tuple[type[Data], ...] = (),
    side_b_data: tuple[type[Data], ...] = (),
    side_c_data: tuple[type[Data], ...] = (),
    steps: tuple[tuple[int, int], ...] = (),
    a_steps: tuple[tuple[int, int], ...] = (),
    b_steps: tuple[tuple[int, int], ...] = (),
    c_steps: tuple[tuple[int, int], ...] = (),
    drawer: type[MPLDrawer[Any]],
    angles: tuple[RealNumber, RealNumber, RealNumber], 
    right_angle: Optional[AngleID] = None,
) -> None:
    # theta A, theta B, theta C
    with timers.pushed('initializing'):
        a = TriangleSide(TriangleSideType.A, *angles, right_angle=right_angle,
                         data=(*side_data, *side_a_data))
        b = TriangleSide(TriangleSideType.B, *angles, right_angle=right_angle,
                         data=(*side_data, *side_b_data))
        c = TriangleSide(TriangleSideType.C, *angles, right_angle=right_angle,
                         data=(*side_data, *side_c_data))
    with timers.pushed('calculating'):
        with timers.pushed('calculating side a'):
            for step in chain(steps, a_steps):
                a.step_all(*step)
            triangle_count_a = len(a.tree)
            print(f"{triangle_count_a} triangles calculated")
        with timers.pushed('calculating side b'):
            for step in chain(steps, b_steps):
                b.step_all(*step)
            triangle_count_b = len(b.tree)
            print(f"{triangle_count_b} triangles calculated")
        with timers.pushed('calculating side c'):
            for step in chain(steps, c_steps):
                c.step_all(*step)
            triangle_count_c = len(c.tree)
            print(f"{triangle_count_c} triangles calculated")
    with timers.pushed('finalizing'):
        tree_a = a.finalized_tree()
        tree_b = b.finalized_tree()
        tree_c = c.finalized_tree()
    with timers.pushed('drawing'):
        with timers.pushed('setting up figure and drawers'):
            fig, ax = create_canvas(radius)
            draw_a = drawer(fig, ax, tree_a, draw_zero=True)
            draw_b = drawer(fig, ax, tree_b)
            draw_c = drawer(fig, ax, tree_c)
        with timers.pushed('drawing side a'):
            draw_a.draw_tree()
        with timers.pushed('drawing side b'):
            draw_b.draw_tree()
        with timers.pushed('drawing side c'):
            draw_c.draw_tree()
    with timers.pushed('saving'):
        fig.savefig(f"output/output_{name}.png", pad_inches=0)

SETUPS = {
    'main': {
        'drawer': ColorDataDrawerWithSymbolMain,
        'steps': ((2, STEP_BASE),
                  (12, STEP_NONBASE),
                  (12, STEP_HORIZONTAL)),
        'angles': (mp.pi - 2*mp.atan(2), mp.atan(2), mp.atan(2)),
        'side_data': (PFP, PFP2),
    },
    'alt': {
        'drawer': ColorDataDrawerWithSymbolAlt,
        'steps': ((2, STEP_BASE),
                  (12, STEP_NONBASE),
                  (12, STEP_HORIZONTAL)),
        'angles': (mp.pi - 2*mp.atan(2), mp.atan(2), mp.atan(2)),
        'side_data': (PFP, PFP2),
    },
    'kyza': {
        'drawer': KyzaDrawer,
        'steps': ((2, STEP_BASE),
                  (6, STEP_NONBASE),
                  (12, STEP_HORIZONTAL)),
        'angles': (mp.pi - 2*mp.atan(2), mp.atan(2), mp.atan(2)),
    },
    'random': {
        'drawer': ExampleRandomHPLuvColorDrawer,
        'steps': ((2, STEP_BASE),
                  (6, STEP_NONBASE),
                  (12, STEP_HORIZONTAL)),
        'angles': (mp.pi - 2*mp.atan(2), mp.atan(2), mp.atan(2)),
    },
    'fixed': {
        'drawer': ExampleFixedColorDrawer,
        'a_steps': ((1, STEP_BASE), (1, STEP_RIGHT)),
        'c_steps': ((1, STEP_BASE), (1, STEP_LEFT)),
        'angles': (mp.pi - 2*mp.atan(2), mp.atan(2), mp.atan(2)),
    },
}

def main(setup: str) -> None:
    assert setup in SETUPS
    with Subtimers() as timers, timers.pushed(setup):
        process_picture(timers, setup, **SETUPS[setup])
    

if __name__ == '__main__':
    possible_setups = ', '.join(SETUPS)
    while (setup := input(f"Choose ({possible_setups}): ")) not in SETUPS:
        pass
    main(setup)
