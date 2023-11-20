from __future__ import annotations

from typing import (TYPE_CHECKING, ClassVar, Generic, Literal, Mapping,
                    Optional, Self, TypeVar, get_args)

from data import Data
from identifier import ContextualizedIdentifier, Identifier
from util import SideType, TriangleSideType

if TYPE_CHECKING:
    from colors import Color
    from shapes import (BaseShapeCollection, NormalShapeCollection,
                        ZeroShapeCollection)
    from util import LookupKeyType

ColorType = TypeVar('ColorType', bound='Color')


class ColorData(Data, Generic[ColorType]):
    predefined_colors: ClassVar[  # type:ignore [misc]
        Mapping[LookupKeyType, ColorType],  # type: ignore[misc, unused-ignore]
    ]
    circle_color: ClassVar[ColorType]  # type: ignore[misc]
    base_above_weight: ClassVar[float] = 1
    base_circle_weight: ClassVar[float] = 1
    horizontal_touching_vertical_weight: ClassVar[float] = 1
    horizontal_touching_horizontal_weight: ClassVar[float] = 1
    horizontal_circle_weight: ClassVar[float] = 1
    vertical_touching_vertical_weight: ClassVar[float] = 1
    vertical_touching_horizontal_weight: ClassVar[float] = 1
    vertical_circle_weight: ClassVar[float] = 1

    def __init__(self, /, color: ColorType):
        self.color = color

    @classmethod
    def predefined_lookup(
        cls, /,
        contextualized_identifier: ContextualizedIdentifier,
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
            if possibility in cls.predefined_colors:
                return possibility
        return None

    @classmethod
    def for_zero_triangle(
        cls, /,
        triangle_side_type: TriangleSideType,
        shape_collection: ZeroShapeCollection,
    ) -> Self:
        contextualized_identifier = ContextualizedIdentifier(
            triangle_side_type, SideType.ZERO, Identifier(),
        )
        predefined_found = cls.predefined_lookup(contextualized_identifier)
        assert predefined_found is not None
        return cls(cls.predefined_colors[predefined_found])

    @classmethod
    def for_base_triangle(
        cls, /,
        triangle_side_type: TriangleSideType,
        identifier: Identifier, shapes: BaseShapeCollection,
        above_identifier: Identifier,
        above_shapes: BaseShapeCollection | ZeroShapeCollection,
        above_data: Self,
    ) -> Self:
        contextualized_identifier = ContextualizedIdentifier(
            triangle_side_type, SideType.BASE, identifier,
        )
        predefined_found = cls.predefined_lookup(contextualized_identifier)
        if predefined_found is not None:
            return cls(cls.predefined_colors[predefined_found])
        color_class = get_args(
            cls.__orig_bases__[0],  # type: ignore[attr-defined]
        )[0]
        return cls(color_class.mix(above_data.color, cls.base_above_weight,
                                   cls.circle_color, cls.base_circle_weight))

    @classmethod
    def for_horizontal_triangle(
        cls, /,
        triangle_side_type: TriangleSideType,
        side: Literal[SideType.LEFT, SideType.RIGHT],
        identifier: Identifier,
        shapes: NormalShapeCollection,
        touching_horizontal_identifier: Identifier,
        touching_horizontal_shapes: (BaseShapeCollection |
                                     NormalShapeCollection),
        touching_horizontal_data: Self,
        touching_vertical_identifier: Identifier,
        touching_vertical_shapes: (ZeroShapeCollection |
                                   BaseShapeCollection |
                                   NormalShapeCollection),
        touching_vertical_data: Self,
    ) -> Self:
        contextualized_identifier = ContextualizedIdentifier(
            triangle_side_type, side, identifier,
        )
        predefined_found = cls.predefined_lookup(contextualized_identifier)
        if predefined_found is not None:
            return cls(cls.predefined_colors[predefined_found])
        color_class = get_args(
            cls.__orig_bases__[0],  # type: ignore[attr-defined]
        )[0]
        return cls(color_class.mix(touching_vertical_data.color,
                                   cls.horizontal_touching_vertical_weight,
                                   touching_horizontal_data.color,
                                   cls.horizontal_touching_horizontal_weight,
                                   cls.circle_color,
                                   cls.horizontal_circle_weight))

    @classmethod
    def for_vertical_triangle(
        cls, /,
        triangle_side_type: TriangleSideType, side: SideType,
        identifier: Identifier,
        shapes: NormalShapeCollection,
        touching_horizontal_identifier: Identifier,
        touching_horizontal_shapes: (BaseShapeCollection |
                                     NormalShapeCollection),
        touching_horizontal_data: Self,
        touching_vertical_identifier: Identifier,
        touching_vertical_shapes: NormalShapeCollection,
        touching_vertical_data: Self,
    ) -> Self:
        contextualized_identifier = ContextualizedIdentifier(
            triangle_side_type, side, identifier,
        )
        predefined_found = cls.predefined_lookup(contextualized_identifier)
        if predefined_found is not None:
            return cls(cls.predefined_colors[predefined_found])
        color_class = get_args(
            cls.__orig_bases__[0],  # type: ignore[attr-defined]
        )[0]
        return cls(color_class.mix(touching_vertical_data.color,
                                   cls.vertical_touching_vertical_weight,
                                   touching_horizontal_data.color,
                                   cls.vertical_touching_horizontal_weight,
                                   cls.circle_color,
                                   cls.vertical_circle_weight))
