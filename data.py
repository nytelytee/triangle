from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Self

from util import SideType, TriangleSideType

if TYPE_CHECKING:
    from identifier import Identifier
    from shapes import (BaseShapeCollection, NormalShapeCollection,
                        ZeroShapeCollection)


class Data(ABC):

    @classmethod
    @abstractmethod
    def for_zero_triangle(
        cls, /,
        triangle_side_type: TriangleSideType, shapes: ZeroShapeCollection,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def for_base_triangle(
        cls, /,
        triangle_side_type: TriangleSideType, identifier: Identifier,
        shapes: BaseShapeCollection, above_identifier: Identifier,
        above_shapes: BaseShapeCollection | ZeroShapeCollection,
        above_data: Self,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def for_horizontal_triangle(
        cls, /,
        triangle_side_type: TriangleSideType,
        side: Literal[SideType.LEFT, SideType.RIGHT],
        identifier: Identifier, shapes: NormalShapeCollection,
        touching_horizontal_identifier: Identifier,
        touching_horizontal_shapes: (NormalShapeCollection |
                                     BaseShapeCollection),
        touching_horizontal_data: Self,
        touching_vertical_identifier: Identifier,
        touching_vertical_shapes: (NormalShapeCollection |
                                   BaseShapeCollection |
                                   ZeroShapeCollection),
        touching_vertical_data: Self,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def for_vertical_triangle(
        cls, /,
        triangle_side_type: TriangleSideType,
        side: Literal[SideType.LEFT, SideType.RIGHT],
        identifier: Identifier, shapes: NormalShapeCollection,
        touching_horizontal_identifier: Identifier,
        touching_horizontal_shapes: (NormalShapeCollection |
                                     BaseShapeCollection),
        touching_horizontal_data: Self,
        touching_vertical_identifier: Identifier,
        touching_vertical_shapes: NormalShapeCollection,
        touching_vertical_data: Self,
    ) -> Self:
        raise NotImplementedError
