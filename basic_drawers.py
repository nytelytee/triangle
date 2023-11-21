from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Never

from basic_data import ColorData
from drawer import MPLDrawer
from tree import CBaseNode, CNormalNode, CRealNode, CZeroNode

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from util import MPLColor, RealNumber


class ColorDrawer(MPLDrawer[Never]):

    fig: Figure
    ax: Axes
    draw_zero: bool
    radius: RealNumber

    def choose_triangle_color(self, /, node: CRealNode[Never]) -> MPLColor:
        raise NotImplementedError

    def choose_negative_triangle_color(
        self, /, node: CNormalNode[Never],
    ) -> MPLColor:
        raise NotImplementedError

    def choose_horizontal_arc_color(
        self, /, node: CNormalNode[Never]
    ) -> MPLColor:
        raise NotImplementedError

    def choose_vertical_arc_color(
        self, /, node: CNormalNode[Never]
    ) -> MPLColor:
        raise NotImplementedError

    def choose_base_left_arc_color(
        self, /, node: CBaseNode[Never],
    ) -> MPLColor:
        raise NotImplementedError

    def choose_base_right_arc_color(
        self, /, node: CBaseNode[Never],
    ) -> MPLColor:
        raise NotImplementedError

    def choose_segment_color(
        self, /, node: CZeroNode[Never] | CBaseNode[Never],
    ) -> MPLColor:
        raise NotImplementedError

    def draw_triangle(self, /, node: CRealNode[Never]) -> None:
        self.ax.fill(*node.shapes.triangle.draw_coords,
                     facecolor=self.choose_triangle_color(node))

    def draw_negative_triangle(self, /, node: CNormalNode[Never]) -> None:
        self.ax.fill(*node.shapes.negative_triangle.draw_coords,
                     facecolor=self.choose_negative_triangle_color(node))

    def draw_horizontal_arc(self, /, node: CNormalNode[Never]) -> None:
        self.ax.fill(*node.shapes.horizontal_arc.draw_coords,
                     facecolor=self.choose_horizontal_arc_color(node))

    def draw_vertical_arc(self, /, node: CNormalNode[Never]) -> None:
        self.ax.fill(*node.shapes.vertical_arc.draw_coords,
                     facecolor=self.choose_vertical_arc_color(node))

    def draw_base_left_arc(self, /, node: CBaseNode[Never]) -> None:
        self.ax.fill(*node.shapes.left_arc.draw_coords,
                     facecolor=self.choose_base_left_arc_color(node))

    def draw_base_right_arc(self, /, node: CBaseNode[Never]) -> None:
        self.ax.fill(*node.shapes.right_arc.draw_coords,
                     facecolor=self.choose_base_right_arc_color(node))

    def draw_segment(
        self, /, node: CBaseNode[Never] | CZeroNode[Never],
    ) -> None:
        self.ax.fill(*node.shapes.segment.draw_coords,
                     facecolor=self.choose_segment_color(node))


class ColorDataDrawer(MPLDrawer[ColorData[Any]]):

    fig: Figure
    ax: Axes
    draw_zero: bool
    radius: RealNumber

    triangle_data_index: ClassVar[int] = 0
    negative_triangle_data_index: ClassVar[int] = 1
    vertical_arc_data_index: ClassVar[int] = 1
    horizontal_arc_data_index: ClassVar[int] = 1
    base_left_arc_data_index: ClassVar[int] = 1
    base_right_arc_data_index: ClassVar[int] = 1
    segment_data_index: ClassVar[int] = 1

    def draw_triangle(self, /, node: CRealNode[ColorData[Any]]) -> None:
        index = self.triangle_data_index % len(node.data)
        self.ax.fill(*node.shapes.triangle.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_negative_triangle(
        self, /, node: CNormalNode[ColorData[Any]],
    ) -> None:
        index = self.negative_triangle_data_index % len(node.data)
        self.ax.fill(*node.shapes.negative_triangle.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_horizontal_arc(
        self, /, node: CNormalNode[ColorData[Any]],
    ) -> None:
        index = self.horizontal_arc_data_index % len(node.data)
        self.ax.fill(*node.shapes.horizontal_arc.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_vertical_arc(
        self, /, node: CNormalNode[ColorData[Any]],
    ) -> None:
        index = self.vertical_arc_data_index % len(node.data)
        self.ax.fill(*node.shapes.vertical_arc.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_base_left_arc(
        self, /, node: CBaseNode[ColorData[Any]],
    ) -> None:
        index = self.base_left_arc_data_index % len(node.data)
        self.ax.fill(*node.shapes.left_arc.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_base_right_arc(
        self, /, node: CBaseNode[ColorData[Any]],
    ) -> None:
        index = self.base_right_arc_data_index % len(node.data)
        self.ax.fill(*node.shapes.right_arc.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_segment(
        self, /,
        node: CBaseNode[ColorData[Any]] | CZeroNode[ColorData[Any]],
    ) -> None:
        assert node.shapes.segment is not None
        index = self.segment_data_index % len(node.data)
        self.ax.fill(*node.shapes.segment.draw_coords,
                     facecolor=node.data[index].color.to_hex())


class AlternatingColorDataDrawer(MPLDrawer[ColorData[Any]]):

    fig: Figure
    ax: Axes
    draw_zero: bool
    radius: RealNumber

    triangle_offset: ClassVar[int] = 0
    negative_triangle_offset: ClassVar[int] = 1
    vertical_arc_offset: ClassVar[int] = 1
    horizontal_arc_offset: ClassVar[int] = 1
    base_left_arc_offset: ClassVar[int] = 1
    base_right_arc_offset: ClassVar[int] = 1
    segment_offset: ClassVar[int] = 1

    def draw_triangle(
        self, /, node: CRealNode[ColorData[Any]],
    ) -> None:
        index = sum(node.identifier.parts) + self.triangle_offset
        index %= len(node.data)
        self.ax.fill(*node.shapes.triangle.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_negative_triangle(
        self, /, node: CNormalNode[ColorData[Any]],
    ) -> None:
        index = sum(node.identifier.parts) + self.negative_triangle_offset
        index %= len(node.data)
        self.ax.fill(*node.shapes.negative_triangle.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_horizontal_arc(
        self, /, node: CNormalNode[ColorData[Any]],
    ) -> None:
        index = sum(node.identifier.parts) + self.horizontal_arc_offset
        index %= len(node.data)
        self.ax.fill(*node.shapes.horizontal_arc.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_vertical_arc(
        self, /, node: CNormalNode[ColorData[Any]],
    ) -> None:
        index = sum(node.identifier.parts) + self.vertical_arc_offset
        index %= len(node.data)
        self.ax.fill(*node.shapes.vertical_arc.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_base_left_arc(
        self, /, node: CBaseNode[ColorData[Any]],
    ) -> None:
        index = sum(node.identifier.parts) + self.base_left_arc_offset
        index %= len(node.data)
        self.ax.fill(*node.shapes.left_arc.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_base_right_arc(
        self, /, node: CBaseNode[ColorData[Any]],
    ) -> None:
        index = sum(node.identifier.parts) + self.base_right_arc_offset
        index %= len(node.data)
        self.ax.fill(*node.shapes.right_arc.draw_coords,
                     facecolor=node.data[index].color.to_hex())

    def draw_segment(
        self, /,
        node: CBaseNode[ColorData[Any]] | CZeroNode[ColorData[Any]],
    ) -> None:
        index = sum(node.identifier.parts) + self.segment_offset
        index %= len(node.data)
        self.ax.fill(*node.shapes.segment.draw_coords,
                     facecolor=node.data[index].color.to_hex())
