from __future__ import annotations

import functools
import itertools
from random import uniform
from typing import (TYPE_CHECKING, Any, ClassVar, Generic, Literal, Mapping,
                    Never, Optional, TypeVar, cast, overload)

import mpmath as mp  # type: ignore[import-untyped]
import numpy as np
import shapely  # type: ignore[import-untyped]
import shapely.geometry  # type: ignore[import-untyped]
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from basic_data import ColorData
from basic_drawers import ColorDataDrawer, ColorDrawer
from colors import Color, ColorHPLuv, ColorHSL
from drawer import MPLDrawer
from identifier import ContextualizedIdentifier, Identifier
from shapes import Point, Triangle
from tree import CBaseNode, CNormalNode, CRealNode, CZeroNode, TriangleSideTree
from util import (MPLColor, apply_unpacked, is_mpl_color_transparent,
                  offset_polygon)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from util import LookupKeyType, RealNumber


class ExampleRandomHPLuvColorDrawer(ColorDrawer):

    def choose_triangle_color(*args: Any, **kwargs: Any) -> MPLColor:
        return ColorHPLuv(uniform(0, 360), 100, 85).to_hex()

    def choose_negative_triangle_color(*args: Any, **kwargs: Any) -> MPLColor:
        return ColorHPLuv(uniform(0, 360), 100, 95).to_hex()

    def choose_vertical_arc_color(*args: Any, **kwargs: Any) -> MPLColor:
        return ColorHPLuv(uniform(0, 360), 100, 95).to_hex()

    def choose_horizontal_arc_color(*args: Any, **kwargs: Any) -> MPLColor:
        return ColorHPLuv(uniform(0, 360), 100, 95).to_hex()

    def choose_base_left_arc_color(
        *args: Any, **kwargs: Any,
    ) -> MPLColor:
        return ColorHPLuv(uniform(0, 360), 100, 95).to_hex()

    def choose_base_right_arc_color(
        *args: Any, **kwargs: Any,
    ) -> MPLColor:
        return ColorHPLuv(uniform(0, 360), 100, 95).to_hex()

    def choose_segment_color(*args: Any, **kwargs: Any) -> MPLColor:
        return ColorHPLuv(uniform(0, 360), 100, 95).to_hex()


class ExampleFixedColorDrawer(ColorDrawer):

    def choose_triangle_color(*args: Any, **kwargs: Any) -> MPLColor:
        return 'r'

    def choose_negative_triangle_color(*args: Any, **kwargs: Any) -> MPLColor:
        return 'g'

    def choose_vertical_arc_color(*args: Any, **kwargs: Any) -> MPLColor:
        return 'b'

    def choose_horizontal_arc_color(*args: Any, **kwargs: Any) -> MPLColor:
        return 'c'

    def choose_base_left_arc_color(
        *args: Any, **kwargs: Any
    ) -> MPLColor:
        return 'y'

    def choose_base_right_arc_color(
        *args: Any, **kwargs: Any
    ) -> MPLColor:
        return 'm'

    def choose_segment_color(*args: Any, **kwargs: Any) -> MPLColor:
        return 'k'


ColorType = TypeVar('ColorType', bound=Color, covariant=True)


class OutlineDrawer(MPLDrawer[Never]):

    def __init__(
        self, fig: Figure, ax: Axes, /,
        tree: TriangleSideTree[ContextualizedIdentifier, Never],
        *,
        draw_zero: bool = False,
        radius: RealNumber = 1,
        outline_scale: float = 0.002
    ):
        self.fig = fig
        self.ax = ax
        self.tree = tree
        self.draw_zero = draw_zero
        self.radius = radius
        self.outline_scale = outline_scale

    def choose_outline_color(self, /, node: CRealNode[Never]) -> MPLColor:
        return 'black'

    # guaranteed to be the same orientation as the coords that get
    # passed in (this is assumed to be counter-clockwise and the
    # triangle gets offset with that assumption)
    def get_inner_triangle(self, triangle: Triangle, /) -> Triangle:
        DATA = self.ax.transData
        FIGURE = self.fig.dpi_scale_trans
        # matplotlib does not like it when you try to transform
        # non-floats (well actually it's numpy that does not like it,
        # but i have no idea how to tell matplotlib to tell numpy to
        # keep the dtype as object, and if that even makes sense).
        p = [p.transform(DATA, FIGURE) for p in triangle.points]
        d = Point.dist(*min(itertools.combinations(p, 2),
                            key=functools.partial(apply_unpacked, Point.dist)))
        small_p = offset_polygon(p, -self.outline_scale*d)
        small_p_data = [p.transform(FIGURE, DATA) for p in small_p]
        return Triangle(*small_p_data)

    def get_outline_path(self, /, triangle: Triangle) -> Path:
        float_triangle = Triangle(
            *[Point(float(x), float(y))
              for x, y in triangle.coords.T]
        )
        inner_triangle = self.get_inner_triangle(float_triangle)
        path = Path(float_triangle.draw_coords.T, closed=True)
        inner_path = Path(inner_triangle.draw_coords.T[::-1], closed=True)
        return Path.make_compound_path(path, inner_path)

    def draw_outline(
        self, /, node: CRealNode[Never],
    ) -> None:
        outline_path = self.get_outline_path(node.shapes.triangle)
        patch = PathPatch(
            outline_path, edgecolor='none',
            facecolor=self.choose_outline_color(node)
        )
        self.ax.add_patch(patch)


class KyzaDrawer(MPLDrawer[Never], Generic[ColorType]):

    subdivision_dict: Mapping[
        LookupKeyType,
        tuple[Optional[int], Optional[int], int,
              Optional[int], Optional[int], int]
    ] = {
        (None, None, Identifier()): (None, 1, -2, None, 1, -2),
        (None, None, None): (None, 1, 0, None, 1, 0),
    }

    outline_scale: RealNumber

    def subdivision_lookup(
        self, /, contextualized_identifier: ContextualizedIdentifier,
    ) -> Optional[LookupKeyType]:
        triangle_side = contextualized_identifier.triangle_side
        side = contextualized_identifier.side
        identifier = contextualized_identifier.identifier
        possibilities = [(triangle_side, side, identifier),
                         (triangle_side, None, identifier),
                         (None, side, identifier),
                         (None, None, identifier),
                         (triangle_side, side, None),
                         (triangle_side, None, None),
                         (None, side, None),
                         (None, None, None)]
        for possibility in possibilities:
            if possibility in self.subdivision_dict:
                return possibility
        return None

    @overload
    def __init__(
        self: KyzaDrawer[ColorHSL], fig: Figure, ax: Axes, /,
        tree: TriangleSideTree[ContextualizedIdentifier, Never],
        *,
        draw_zero: bool = ...,
        start_color: None = ...,
        end_color: None = ...,
        radius: RealNumber = ...,
        power: RealNumber = ...,
        max_cuts: int = ...,
        subdivision_dict: Optional[Mapping[
            LookupKeyType,
            tuple[Optional[int], Optional[int], int,
                  Optional[int], Optional[int], int]
        ]] = ...,
        outline_scale: RealNumber = ...,
    ) -> None:
        ...

    @overload
    def __init__(
        self: KyzaDrawer[ColorType], fig: Figure, ax: Axes, /,
        tree: TriangleSideTree[ContextualizedIdentifier, Never],
        *,
        draw_zero: bool = ...,
        start_color: ColorType,
        end_color: ColorType,
        radius: RealNumber = ...,
        power: RealNumber = ...,
        max_cuts: int = ...,
        subdivision_dict: Optional[Mapping[
            LookupKeyType,
            tuple[Optional[int], Optional[int], int,
                  Optional[int], Optional[int], int]
        ]] = ...,
        outline_scale: RealNumber = ...,
    ) -> None:
        ...

    def __init__(
        self, fig: Figure, ax: Axes, /,
        tree: TriangleSideTree[ContextualizedIdentifier, Never],
        *,
        draw_zero: bool = False,
        start_color: Optional[ColorType] = None,
        end_color: Optional[ColorType] = None,
        radius: RealNumber = 1,
        power: RealNumber = 1,
        max_cuts: int = 8,
        subdivision_dict: Optional[Mapping[
            LookupKeyType,
            tuple[Optional[int], Optional[int], int,
                  Optional[int], Optional[int], int]
        ]] = None,
        outline_scale: RealNumber = 0.002,
    ) -> None:
        if start_color is None and end_color is None:
            start_color = cast(ColorType, ColorHSL(275, 50, 50))
            end_color = cast(ColorType, ColorHSL(300, 100, 98))
        assert start_color is not None
        assert end_color is not None
        self.fig = fig
        self.ax = ax
        self.tree = tree
        self.draw_zero = draw_zero
        self.start_color = start_color
        self.end_color = end_color
        self.radius = radius
        self.max_cuts = max_cuts
        self.exponent = power/2
        self.area_circle = radius**2 * np.pi
        if subdivision_dict is None:
            self.subdivision_dict = self.__class__.subdivision_dict
        else:
            self.subdivision_dict = subdivision_dict
        self._outline_drawer = OutlineDrawer(fig, ax, tree,
                                             draw_zero=draw_zero,
                                             radius=radius,
                                             outline_scale=outline_scale)

    def compute_color(self, /, x: NDArray[Any], y: NDArray[Any]) -> str:
        x_avg = sum(x) / len(x)
        y_avg = sum(y) / len(y)
        t = (x_avg**2 + y_avg**2)**self.exponent
        return self.start_color.__class__.mix(
            self.start_color, self.radius**self.exponent - t,
            self.end_color, t
        ).to_hex()

    def _subdivide_triangle(
        self, /,
        triangle: Triangle,
        *,
        cuts: Optional[int] = None,
        sierpinski_cutoff: Optional[int] = None,
        additional_cuts: int = 0,
    ) -> tuple[list[NDArray[Any]], list[NDArray[Any]]]:

        if cuts is None:
            area_triangle = 1/2 * abs(
                triangle.a.x*(triangle.b.y - triangle.c.y) +
                triangle.b.x*(triangle.c.y - triangle.a.y) +
                triangle.c.x*(triangle.a.y - triangle.b.y)
            )
            portion = np.sqrt(area_triangle/self.area_circle)
            cuts = round(portion * self.max_cuts)
        cuts = min(cast(int, cuts) + additional_cuts, self.max_cuts)

        # normal subdivision subdivides all triangles
        # sierpinski subdivition does not keep subdividing the
        # upside down triangle at the center

        # never do sierpinski subdivision.
        if sierpinski_cutoff is None:
            sierpinski_cutoff = cuts

        def subdivide(
            t: NDArray[Any]
        ) -> tuple[
            NDArray[Any], NDArray[Any],
            NDArray[Any], NDArray[Any]
        ]:
            pA = t[:, 0]
            pB = t[:, 1]
            pC = t[:, 2]
            mAB = (pA + pB)/2
            mAC = (pA + pC)/2
            mBC = (pB + pC)/2
            triangle_1 = np.row_stack((pA, mAC, mAB)).T
            triangle_2 = np.row_stack((pB, mBC, mAB)).T
            triangle_3 = np.row_stack((pC, mBC, mAC)).T
            triangle_4 = np.row_stack((mAB, mBC, mAC)).T
            return triangle_1, triangle_2, triangle_3, triangle_4

        current_positive_triangles: list[NDArray[Any]] = [triangle.coords]
        current_negative_triangles: list[NDArray[Any]] = []
        positive_sierpinski_triangles: list[NDArray[Any]] = []
        negative_sierpinski_triangles: list[NDArray[Any]] = []

        for iteration in range(cuts):
            pos = len(current_positive_triangles)
            neg = len(current_negative_triangles)
            for i in range(pos):
                t = current_positive_triangles.pop(0)
                t1, t2, t3, t4 = subdivide(t)
                current_positive_triangles.extend((t1, t2, t3))
                if iteration < sierpinski_cutoff:
                    current_negative_triangles.append(t4)
                else:
                    negative_sierpinski_triangles.append(t4)
            for i in range(neg):
                t = current_negative_triangles.pop(0)
                t1, t2, t3, t4 = subdivide(t)
                current_negative_triangles.extend((t1, t2, t3))
                if iteration < sierpinski_cutoff:
                    current_positive_triangles.append(t4)
                else:
                    positive_sierpinski_triangles.append(t4)

        return (positive_sierpinski_triangles + current_positive_triangles,
                negative_sierpinski_triangles + current_negative_triangles)

    def draw_triangle(self, /, node: CRealNode[Never]) -> None:
        subdivision = self.subdivision_lookup(node.identifier)
        if subdivision:
            cuts, sierpinski_cutoff, additional_cuts, *_ = (
                self.subdivision_dict[subdivision]
            )
        else:
            cuts, sierpinski_cutoff, additional_cuts = (None, None, 0)

        triangles = itertools.chain(*self._subdivide_triangle(
            node.shapes.triangle,
            cuts=cuts,
            sierpinski_cutoff=sierpinski_cutoff,
            additional_cuts=additional_cuts
        ))
        for t in triangles:
            self.ax.fill(*t, facecolor=self.compute_color(*t))

        self._outline_drawer.draw_outline(node)

    def draw_negative_triangle(self, /, node: CNormalNode[Never]) -> None:

        subdivision = self.subdivision_lookup(node.identifier)
        if subdivision:
            *_, cuts, sierpinski_cutoff, additional_cuts = (
                self.subdivision_dict[subdivision]
            )
        else:
            cuts, sierpinski_cutoff, additional_cuts = (None, None, 0)

        triangles = itertools.chain(*self._subdivide_triangle(
            node.shapes.negative_triangle,
            cuts=cuts,
            sierpinski_cutoff=sierpinski_cutoff,
            additional_cuts=additional_cuts
        ))

        for t in triangles:
            self.ax.fill(*t, facecolor=self.compute_color(*t))

    def draw_vertical_arc(self, /, node: CNormalNode[Never]) -> None:
        self.ax.fill(*node.shapes.vertical_arc.draw_coords,
                     facecolor=self.compute_color(
                         *node.shapes.vertical_arc.coords
                     ))

    def draw_horizontal_arc(self, /, node: CNormalNode[Never]) -> None:
        self.ax.fill(*node.shapes.horizontal_arc.draw_coords,
                     facecolor=self.compute_color(
                         *node.shapes.horizontal_arc.coords
                     ))

    def draw_base_left_arc(
        self, /, node: CBaseNode[Never],
    ) -> None:
        self.ax.fill(*node.shapes.left_arc.draw_coords,
                     facecolor=self.compute_color(
                         *node.shapes.left_arc.coords
                     ))

    def draw_base_right_arc(
        self, /, node: CBaseNode[Never],
    ) -> None:
        self.ax.fill(*node.shapes.right_arc.draw_coords,
                     facecolor=self.compute_color(
                         *node.shapes.right_arc.coords
                     ))

    def draw_segment(
        self, /, node: CBaseNode[Never] | CZeroNode[Never],
    ) -> None:
        self.ax.fill(*node.shapes.segment.draw_coords,
                     facecolor=self.compute_color(*node.shapes.segment.coords))


class ColorDataDrawerFilterOutNonTriangles(ColorDataDrawer):

    def filter_negative_triangle(*args: Any, **kwargs: Any) -> bool:
        return False

    def filter_vertical_arc(*args: Any, **kwargs: Any) -> bool:
        return False

    def filter_horizontal_arc(*args: Any, **kwargs: Any) -> bool:
        return False

    def filter_base_left_arc(*args: Any, **kwargs: Any) -> bool:
        return False

    def filter_base_right_arc(*args: Any, **kwargs: Any) -> bool:
        return False

    def filter_segment(*args: Any, **kwargs: Any) -> bool:
        return False


class ColorDataDrawerWithSymbol(ColorDataDrawer):

    symbol_edgecolor: ClassVar[MPLColor | Literal['data']]
    symbol_facecolor: ClassVar[MPLColor | Literal['data']]
    negative_symbol_edgecolor: ClassVar[MPLColor | Literal['data']]
    negative_symbol_facecolor: ClassVar[MPLColor | Literal['data']]
    triangle_edgecolor: ClassVar[MPLColor | Literal['data']]
    triangle_facecolor: ClassVar[MPLColor | Literal['data']]
    negative_triangle_edgecolor: ClassVar[MPLColor | Literal['data']]
    negative_triangle_facecolor: ClassVar[MPLColor | Literal['data']]

    linewidth_scale: ClassVar[RealNumber] = 0.015
    symbol_part_distance_scale: ClassVar[RealNumber] = 0.2
    symbol_outline_gap_scale: ClassVar[RealNumber] = 0.2
    inradius_scale: ClassVar[RealNumber] = 0.8
    quad_segs: ClassVar[int] = 64

    _triangle_paths: list[Path]
    _outline_paths: list[Path]
    _triangle_edgecolors: list[MPLColor]
    _triangle_facecolors: list[MPLColor]

    _symbol_paths: list[Path]
    _symbol_edgecolors: list[MPLColor]
    _symbol_facecolors: list[MPLColor]
    _symbol_linewidths: list[float]

    def __init__(
        self, /,
        fig: Figure, ax: Axes,
        tree: TriangleSideTree[ContextualizedIdentifier, ColorData[Any]],
        *, draw_zero: bool = False, radius: RealNumber = 1,
    ):
        self.fig = fig
        self.ax = ax
        self.tree = tree
        self.draw_zero = draw_zero
        self.radius = radius

        self._triangle_paths = []
        self._outline_paths = []
        self._triangle_edgecolors = []
        self._triangle_facecolors = []

        self._symbol_paths = []
        self._symbol_edgecolors = []
        self._symbol_facecolors = []
        self._symbol_linewidths = []

        # no need to actually remove anything from the tree
        # the drawer you're passing it to does not do anything
        # with the data because it assumes it's empty, so it's enough
        # to just cast it as empty
        no_data_tree = cast(TriangleSideTree[ContextualizedIdentifier, Never],
                            tree)

        self._outline_drawer = OutlineDrawer(
            fig, ax, no_data_tree, draw_zero=draw_zero, radius=radius,
            outline_scale=self.linewidth_scale
        )

    def _get_symbol_parts(
        self, /,
        triangle: Triangle, distance: float,
        inradius: float, incenter: Point,
        a: RealNumber, b: RealNumber, c: RealNumber,
    ) -> Any:
        circle = shapely.geometry.Point(incenter.x, incenter.y).buffer(
            self.inradius_scale*inradius, quad_segs=self.quad_segs
        )

        rectangle = shapely.geometry.Polygon((
            (incenter.x-distance, -float(self.radius)),
            (incenter.x-distance, float(self.radius)),
            (incenter.x+distance, float(self.radius)),
            (incenter.x+distance, -float(self.radius)),
            (incenter.x-distance, -float(self.radius)),
        ))
        return circle.difference(rectangle)

    def _prepare_symbol_parts(
        self, /,
        symbol_parts: Any, linewidth_points: float, delta: float,
        *, edgecolor: MPLColor, facecolor: MPLColor,
    ) -> None:
        already_done = 0
        if len(symbol_parts.geoms) != 2:
            print(len(symbol_parts.geoms), "symbol part(s) found, expected 2.")
        for geom in symbol_parts.geoms:
            try:
                smaller_geom = shapely.geometry.Polygon(
                    geom.exterior.offset_curve(
                        -delta,
                        quad_segs=self.quad_segs,
                        join_style=shapely.BufferJoinStyle.mitre
                    )
                )
                symbol_part = geom.difference(smaller_geom)
                path = Path.make_compound_path(*[
                    Path(np.asarray(shape.coords), closed=True)
                    for shape in (
                        symbol_part.exterior, *symbol_part.interiors
                    )
                ])
            except (
                shapely.GEOSException, ValueError,
                TypeError, AttributeError
            ):
                for i in range(already_done):
                    del self._symbol_paths[-1]
                    del self._symbol_edgecolors[-1]
                    del self._symbol_facecolors[-1]
                    del self._symbol_linewidths[-1]
                return
            self._symbol_paths.append(path)
            self._symbol_edgecolors.append(edgecolor)
            self._symbol_facecolors.append(facecolor)
            self._symbol_linewidths.append(linewidth_points)
            already_done += 1

    def prepare_symbol(
        self, /,
        triangle: Triangle, data: ColorData[Any],
        *, negative: bool,
    ) -> None:
        DATA = self.ax.transData
        FIGURE = self.fig.dpi_scale_trans
        edgecolor = (self.symbol_edgecolor
                     if not negative else
                     self.negative_symbol_edgecolor)
        facecolor = (self.symbol_facecolor
                     if not negative else
                     self.negative_symbol_facecolor)
        edgecolor = (data.color.to_hex() if edgecolor == 'data' else edgecolor)
        facecolor = (data.color.to_hex() if facecolor == 'data' else facecolor)
        if (
            is_mpl_color_transparent(edgecolor) and
            is_mpl_color_transparent(facecolor)
        ):
            return
        a, b, c = (triangle.b.dist(triangle.c),
                   triangle.a.dist(triangle.c),
                   triangle.a.dist(triangle.b))
        incenter = (a*triangle.a + b*triangle.b + c*triangle.c)/(a + b + c)
        inradius = float(
            0.5 * mp.sqrt((b+c-a) * (c+a-b) * (a+b-c) / (a+b+c))
        )
        symbol_part_distance = (inradius *
                                self.inradius_scale *
                                self.symbol_part_distance_scale)
        symbol_outline_gap = (inradius *
                              self.inradius_scale *
                              self.symbol_outline_gap_scale)

        float_triangle = Triangle(
            *[Point(float(x), float(y))
              for x, y in triangle.coords.T]
        )
        p = [p.transform(DATA, FIGURE) for p in float_triangle.points]
        d = Point.dist(*min(itertools.combinations(p, 2),
                            key=functools.partial(apply_unpacked, Point.dist)))
        linewidth_inches = self.linewidth_scale * d * 72
        symbol_parts = self._get_symbol_parts(triangle, symbol_part_distance,
                                              inradius, incenter, a, b, c)
        if not hasattr(symbol_parts, 'geoms'):
            return
        self._prepare_symbol_parts(symbol_parts, linewidth_inches,
                                   symbol_outline_gap, edgecolor=edgecolor,
                                   facecolor=facecolor)

    def prepare_triangle(
        self, /,
        triangle: Triangle, data: ColorData[Any],
        *, negative: bool,
    ) -> None:
        edgecolor = (self.triangle_edgecolor
                     if not negative else
                     self.negative_triangle_edgecolor)
        facecolor = (self.triangle_facecolor
                     if not negative else
                     self.negative_triangle_facecolor)
        edgecolor = (data.color.to_hex() if edgecolor == 'data' else edgecolor)
        facecolor = (data.color.to_hex() if facecolor == 'data' else facecolor)
        if (
            is_mpl_color_transparent(edgecolor) and
            is_mpl_color_transparent(facecolor)
        ):
            return
        path = Path(triangle.draw_coords.T, closed=True)
        try:
            outline_path = self._outline_drawer.get_outline_path(triangle)
        except ZeroDivisionError:
            return
        self._triangle_paths.append(path)
        self._outline_paths.append(outline_path)
        self._triangle_edgecolors.append(edgecolor)
        self._triangle_facecolors.append(facecolor)

    def commit_draw(self, /) -> None:

        outline_patches = PatchCollection(
            [PathPatch(p) for p in self._outline_paths],
            facecolors=self._triangle_edgecolors,
            linewidths=0
        )

        triangle_patches = PatchCollection(
            [PathPatch(p) for p in self._triangle_paths],
            facecolors=self._triangle_facecolors,
            linewidths=0,
        )

        symbol_patches = PatchCollection(
            [PathPatch(p) for p in self._symbol_paths],
            facecolors=self._symbol_facecolors,
            edgecolors=self._symbol_edgecolors,
            linewidths=self._symbol_linewidths,
            joinstyle='miter',
        )

        self.ax.add_collection(triangle_patches)
        self.ax.add_collection(outline_patches)
        self.ax.add_collection(symbol_patches)

    def draw_triangle(self, /, node: CRealNode[ColorData[Any]]) -> None:
        index = self.triangle_data_index % len(node.data)
        self.prepare_triangle(node.shapes.triangle,
                              node.data[index],
                              negative=False)
        self.prepare_symbol(node.shapes.triangle,
                            node.data[index],
                            negative=False)

    def draw_negative_triangle(
        self, /, node: CNormalNode[ColorData[Any]],
    ) -> None:
        index = self.negative_triangle_data_index % len(node.data)
        self.prepare_triangle(node.shapes.negative_triangle,
                              node.data[index], negative=True)
        self.prepare_symbol(node.shapes.negative_triangle,
                            node.data[index], negative=True)

    def draw_tree(self, /) -> None:
        super().draw_tree()
        self.commit_draw()


class ColorDataDrawerWithSymbolMain(ColorDataDrawerWithSymbol):

    symbol_edgecolor = 'w'
    symbol_facecolor = 'none'
    negative_symbol_edgecolor = 'k'
    negative_symbol_facecolor = 'none'
    triangle_edgecolor = 'k'
    triangle_facecolor = 'data'
    negative_triangle_facecolor = 'data'
    negative_triangle_edgecolor = 'none'

    triangle_data_index = 0
    negative_triangle_data_index = 1
    horizontal_arc_data_index = 1
    vertical_arc_data_index = 1
    segment_data_index = 1
    base_left_arc_data_index = 1
    base_right_arc_data_index = 1


class ColorDataDrawerWithSymbolAlt(ColorDataDrawerWithSymbol):

    symbol_edgecolor = 'k'
    symbol_facecolor = 'none'
    negative_symbol_edgecolor = 'w'
    negative_symbol_facecolor = 'none'
    triangle_edgecolor = 'none'
    triangle_facecolor = 'data'
    negative_triangle_facecolor = 'data'
    negative_triangle_edgecolor = 'k'

    triangle_data_index = 1
    negative_triangle_data_index = 0
    horizontal_arc_data_index = 0
    vertical_arc_data_index = 0
    segment_data_index = 0
    base_left_arc_data_index = 0
    base_right_arc_data_index = 0
