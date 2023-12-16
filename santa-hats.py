#!/usr/bin/env python3
from __future__ import annotations

import io
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, Optional

import cairosvg
import mpmath  # type: ignore[import-untyped]
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpmath import mp  # type: ignore[import-untyped, unused-ignore]
from PIL import Image

from data import Data
from drawer import MPLDrawer
from example_data import PFP, PFP2
from example_drawers import (ColorDataDrawerWithSymbolAlt,
                             ColorDataDrawerWithSymbolMain,
                             ExampleFixedColorDrawer,
                             ExampleRandomHPLuvColorDrawer, KyzaDrawer)
from shapes import Point
from triangle_side import TriangleSide
from util import (STEP_BASE, STEP_HORIZONTAL, STEP_LEFT, STEP_NONBASE,
                  STEP_RIGHT, AngleID, Subtimers, TriangleSideType)

if TYPE_CHECKING:
    from util import RealNumber


Image.MAX_IMAGE_PIXELS = None


def attach_hats_to_tree(tree, hats):
    for x in tree.walk_bfs():
        A, B, C = x.shapes.triangle.points
        a, b, c = B.dist(C), A.dist(C), A.dist(B)
        incenter = (a*A + b*B + c*C)/(a + b + c)
        inradius = 0.5 * mp.sqrt((b+c-a) * (c+a-b) * (a+b-c) / (a+b+c))

        hp = (a + b + c)/2
        area = mp.sqrt(hp * (hp - a) * (hp - b) * (hp - c))

        area_ratio = area / 4

        hat_image_area = area_ratio * 20480**2
        hat_image_dimension = int(mp.sqrt(hat_image_area) * 0.67)
        if hat_image_dimension <= 0:
            continue
        hat_png = io.BytesIO()
        cairosvg.svg2png(url='santa-hat.svg',
                         output_width=hat_image_dimension,
                         write_to=hat_png)
        hat_image = Image.open(hat_png)

        anchor = incenter

        # image space is a left-handed coordinate system
        anchor = Point(anchor.x, -anchor.y)

        anchor -= Point(0, inradius*0.8)

        # align (-1, -1) with (0, 0)
        anchor += Point(1, 1)

        anchor *= 10240

        anchor -= Point(0.42*hat_image_dimension, 0.5*hat_image_dimension)

        anchor_coordinates = int(anchor.x), int(anchor.y)

        hats.paste(hat_image, anchor_coordinates, hat_image)


mp.prec = 100
radius = 1


def process_image(
    timers: Subtimers,
    name: str,
    *,
    steps: tuple[tuple[int, int], ...] = (),
    a_steps: tuple[tuple[int, int], ...] = (),
    b_steps: tuple[tuple[int, int], ...] = (),
    c_steps: tuple[tuple[int, int], ...] = (),
    angles: tuple[RealNumber, RealNumber, RealNumber],
    right_angle: Optional[AngleID] = None,
) -> None:
    with timers.pushed('initializing'):
        a = TriangleSide(TriangleSideType.A, *angles, right_angle=right_angle)
        b = TriangleSide(TriangleSideType.B, *angles, right_angle=right_angle)
        c = TriangleSide(TriangleSideType.C, *angles, right_angle=right_angle)
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
    with timers.pushed('attaching santa hats'):
        with timers.pushed('opening original image'):
            hats = Image.open(f'output/output_{setup}.png')
        with timers.pushed('attaching santa hats to side a'):
            attach_hats_to_tree(tree_a, hats)
        with timers.pushed('attaching santa hats to side b'):
            attach_hats_to_tree(tree_b, hats)
        with timers.pushed('attaching santa hats to side c'):
            attach_hats_to_tree(tree_c, hats)
    with timers.pushed('saving'):
        hats.save(f'output/output_{setup}_hats.png')


SETUPS = {
    'main': {
        'steps': ((2, STEP_BASE),
                  (12, STEP_NONBASE),
                  (12, STEP_HORIZONTAL)),
        'angles': (mp.pi - 2*mp.atan(2), mp.atan(2), mp.atan(2)),
    },
    'alt': {
        'steps': ((2, STEP_BASE),
                  (12, STEP_NONBASE),
                  (12, STEP_HORIZONTAL)),
        'angles': (mp.pi - 2*mp.atan(2), mp.atan(2), mp.atan(2)),
    },
}


def main(setup: str) -> None:
    assert setup in SETUPS
    with Subtimers() as timers, timers.pushed(setup):
        process_image(timers, setup, **SETUPS[setup])


if __name__ == '__main__':
    possible_setups = ', '.join(SETUPS)
    while (setup := input(f"Choose ({possible_setups}): ")) not in SETUPS:
        pass
    main(setup)
