from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, final

from identifier import ContextualizedIdentifier
from tree import (BaseNode, CBaseNode, CNormalNode, CRealNode, CZeroNode,
                  NormalNode, TriangleSideTree, ZeroNode)
from util import SideType

if TYPE_CHECKING:
    from data import Data
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


DataType = TypeVar('DataType', bound='Data', contravariant=True)


class Drawer(Protocol, Generic[DataType]):

    def triangle(self, /, node: CRealNode[DataType]) -> None:
        return

    def negative_triangle(self, /, node: CNormalNode[DataType]) -> None:
        return

    def vertical_arc(self, /, node: CNormalNode[DataType]) -> None:
        return

    def horizontal_arc(self, /, node: CNormalNode[DataType]) -> None:
        return

    def base_left_arc(self, /, node: CBaseNode[DataType]) -> None:
        return

    def base_right_arc(self, /, node: CBaseNode[DataType]) -> None:
        return

    def segment(
        self, /, node: CZeroNode[DataType] | CBaseNode[DataType],
    ) -> None:
        return

    def node(self, /, node: CRealNode[DataType]) -> None:
        self.triangle(node)
        match node:
            case NormalNode():
                self.negative_triangle(node)
                self.horizontal_arc(node)
                self.vertical_arc(node)
            case BaseNode():
                self.base_left_arc(node)
                self.base_right_arc(node)
                self.segment(node)
            case ZeroNode():
                self.segment(node)


class FilteringDrawer(Drawer[DataType]):

    def filter_triangle(self, /, node: CRealNode[DataType]) -> bool:
        return True

    def filter_negative_triangle(
        self, /, node: CNormalNode[DataType],
    ) -> bool:
        return True

    def filter_vertical_arc(self, /, node: CNormalNode[DataType]) -> bool:
        return True

    def filter_horizontal_arc(
        self, /, node: CNormalNode[DataType],
    ) -> bool:
        return True

    def filter_base_left_arc(self, /, node: CBaseNode[DataType]) -> bool:
        return True

    def filter_base_right_arc(self, /, node: CBaseNode[DataType]) -> bool:
        return True

    def filter_segment(
        self, /, node: CZeroNode[DataType] | CBaseNode[DataType],
    ) -> bool:
        return True

    def draw_triangle(self, /, node: CRealNode[DataType]) -> None:
        return

    def draw_negative_triangle(
        self, /, node: CNormalNode[DataType],
    ) -> None:
        return

    def draw_vertical_arc(self, /, node: CNormalNode[DataType]) -> None:
        return

    def draw_horizontal_arc(
        self, /, node: CNormalNode[DataType],
    ) -> None:
        return

    def draw_base_left_arc(self, /, node: CBaseNode[DataType]) -> None:
        return

    def draw_base_right_arc(self, /, node: CBaseNode[DataType]) -> None:
        return

    def draw_segment(
        self, /, node: CZeroNode[DataType] | CBaseNode[DataType],
    ) -> None:
        return

    @final
    def triangle(self, /, node: CRealNode[DataType]) -> None:
        if self.filter_triangle(node):
            self.draw_triangle(node)

    @final
    def negative_triangle(self, /, node: CNormalNode[DataType]) -> None:
        if self.filter_negative_triangle(node):
            self.draw_negative_triangle(node)

    @final
    def horizontal_arc(self, /, node: CNormalNode[DataType]) -> None:
        if self.filter_horizontal_arc(node):
            self.draw_horizontal_arc(node)

    @final
    def vertical_arc(self, /, node: CNormalNode[DataType]) -> None:
        if self.filter_vertical_arc(node):
            self.draw_vertical_arc(node)

    @final
    def base_left_arc(self, /, node: CBaseNode[DataType]) -> None:
        if self.filter_base_left_arc(node):
            self.draw_base_left_arc(node)

    @final
    def base_right_arc(self, /, node: CBaseNode[DataType]) -> None:
        if self.filter_base_right_arc(node):
            self.draw_base_right_arc(node)

    @final
    def segment(
        self, /, node: CZeroNode[DataType] | CBaseNode[DataType],
    ) -> None:
        if self.filter_segment(node):
            self.draw_segment(node)


class DefaultTreeFilters(FilteringDrawer[DataType]):
    tree: TriangleSideTree[ContextualizedIdentifier, DataType]
    draw_zero: bool

    def __init__(
        self, /, tree: TriangleSideTree[ContextualizedIdentifier, DataType],
        *, draw_zero: bool = False
    ) -> None:
        self.tree = tree
        self.draw_zero = draw_zero

    def filter_triangle(self, /, node: CRealNode[DataType]) -> bool:
        return node.identifier.side != SideType.ZERO or self.draw_zero

    def filter_negative_triangle(
        self, /, node: CNormalNode[DataType],
    ) -> bool:
        return True

    def filter_vertical_arc(self, /, node: CNormalNode[DataType]) -> bool:
        return node.vertical is None

    def filter_horizontal_arc(
        self, /, node: CNormalNode[DataType],
    ) -> bool:
        return node.horizontal is None

    def filter_base_left_arc(self, /, node: CBaseNode[DataType]) -> bool:
        return node.left.horizontal is None

    def filter_base_right_arc(self, /, node: CBaseNode[DataType]) -> bool:
        return node.right.horizontal is None

    def filter_segment(
        self, /, node: CZeroNode[DataType] | CBaseNode[DataType],
    ) -> bool:
        return node.base is None

    def draw_tree(self) -> None:
        for obj in self.tree.walk_dfs():
            self.node(obj)

class MPLDrawer(DefaultTreeFilters[DataType]):
    def __init__(
        self, /,
        fig: Figure,
        ax: Axes,
        tree: TriangleSideTree[ContextualizedIdentifier, DataType],
        *, draw_zero: bool = False,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.tree = tree
        self.draw_zero = draw_zero
