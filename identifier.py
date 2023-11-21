from dataclasses import dataclass
from itertools import chain, cycle, repeat
from typing import Self

from util import IdentifierType, SideType, TriangleSideType


@dataclass(frozen=True)
class Identifier:

    parts: tuple[int, ...] = ()

    def pop_back(self, /) -> Self:
        return self.__class__(self.parts[:-1])

    def push_back(self, value: int, /) -> Self:
        return self.__class__((*self.parts, value))

    @property
    def last_value(self) -> int:
        if not self.parts:
            return 0
        return self.parts[-1]

    @property
    def last_type(self) -> IdentifierType:
        if len(self.parts) == 0:
            return IdentifierType.ZERO
        if len(self.parts) == 1:
            return IdentifierType.BASE
        if len(self.parts) % 2 == 0:
            return IdentifierType.HORIZONTAL
        if len(self.parts) % 2 == 1:
            return IdentifierType.VERTICAL
        assert False

    def change_last(self, value: int, /) -> Self:
        # if it's the zero triangle, pop_back returns the zero triangle
        # and last_value returns 0, so you get a proper base triangle
        # when calling this on the zero_triangle (if value >= 0)
        new = self.pop_back().push_back(self.last_value + value)
        if new.last_value < 0:
            raise ValueError('nope.')
        return new if new.last_value != 0 else new.pop_back()

    def increment(self, /) -> Self:
        return self.change_last(1)

    def decrement(self, /) -> Self:
        return self.change_last(-1)

    def __str__(self, /) -> str:
        if not self.parts:
            return '0'
        labels = chain(('',), cycle('hv'))
        return ';'.join(f"{value}{label}" for
                        value, label in
                        zip(self.parts, labels))

    def new_base_id(self, /) -> Self:
        # self is the triangle from which we will
        # calculate the new base triangle
        if self.last_type in (IdentifierType.ZERO, IdentifierType.BASE):
            return self.increment()
        raise ValueError("Cannot calculate a new base triangle from "
                         "a horizontal or vertical triangle. Calculate "
                         "it from a base triangle or the zero triangle.")

    def new_horizontal_id(self, /) -> Self:
        # self is the triangle from which we will
        # calculate the new horizontal triangle
        if self.last_type == IdentifierType.ZERO:
            raise ValueError("Cannot calculate a new horizontal "
                             "triangle from the zero triangle. "
                             "Calculate the next base triangle instead.")
        if self.last_type == IdentifierType.BASE:
            return self.push_back(1)
        if self.last_type == IdentifierType.HORIZONTAL:
            return self.increment()
        if self.last_type == IdentifierType.VERTICAL:
            return self.push_back(1)
        assert False

    def new_vertical_id(self, /) -> Self:
        # self is the triangle from which we will
        # calculate the new vertical triangle
        if self.last_type == IdentifierType.ZERO:
            raise ValueError("Cannot calculate a new vertical "
                             "triangle from the zero triangle. "
                             "Calculate the next base triangle instead.")
        if self.last_type == IdentifierType.BASE:
            raise ValueError("Cannot calculate a new vertical "
                             "triangle from a base triangle. "
                             "Calculate the next base triangle instead.")
        if self.last_type == IdentifierType.HORIZONTAL:
            return self.push_back(1)
        if self.last_type == IdentifierType.VERTICAL:
            return self.increment()
        assert False

    def get_attrgetter_parts(self, /) -> tuple[str, str]:
        base_count = self.parts[0] if self.parts else 0
        part1 = '.'.join(chain(('zero',), repeat('base', base_count)))
        part2 = '.'.join('.'.join(repeat(direction, count))
                         for direction, count in
                         zip(cycle(['horizontal', 'vertical']), self.parts[1:])
                         )
        return part1, part2


@dataclass(frozen=True)
class ContextualizedIdentifier:
    triangle_side: TriangleSideType
    side: SideType
    identifier: Identifier

    @property
    def parts(self) -> tuple[int, ...]:
        return self.identifier.parts

    def get_attrgetter_string(self, /) -> str:
        part1, part2 = self.identifier.get_attrgetter_parts()
        if self.side == SideType.LEFT:
            return part1 + '.left.' + part2
        elif self.side == SideType.RIGHT:
            return part1 + '.right.' + part2
        return part1

    def __str__(self, /) -> str:
        return f"{self.triangle_side.name}@{self.identifier}|{self.side.name}"
