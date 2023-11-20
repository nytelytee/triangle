from __future__ import annotations

from dataclasses import dataclass
from math import prod
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Optional, Self, TypeVar

import numpy as np
from matplotlib.transforms import IdentityTransform, Transform
from mpmath import mp  # type: ignore[import-untyped]
from numpy.typing import NDArray

from util import SideType

if TYPE_CHECKING:
    from data import Data
    from util import RealNumber

DataType = TypeVar('DataType', bound='Data')


@dataclass(frozen=True)
class Point:

    x: RealNumber
    y: RealNumber

    @classmethod
    def fromarray(cls, /, array: NDArray[Any]) -> Self:
        assert np.shape(array) == (2,)
        return cls(array[0], array[1])

    def __str__(self, /) -> str:
        return f"{float(self.x):.5f}, {float(self.y):.5f}"

    @property
    def angle(self) -> RealNumber:
        return mp.arg(self.x + 1j*self.y)  # % (2*mp.pi)

    def __add__(self, other: Point, /) -> Self:
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self, /) -> Self:
        return self.__class__(self.x - other.x, self.y - other.y)

    def __mul__(self, other: RealNumber, /) -> Self:
        return self.__class__(self.x * other, self.y * other)

    def __rmul__(self, other: RealNumber, /) -> Self:
        return self.__class__(other * self.x, other * self.y)

    def __truediv__(self, other: RealNumber, /) -> Self:
        return self.__class__(self.x / other, self.y / other)

    def rotate(
        self, angle: RealNumber, /,
        *, around: Optional[Self] = None
    ) -> Self:
        around_ = around or ORIGIN

        s, c = mp.sin(angle), mp.cos(angle)
        rotation_matrix = np.array(((c, s), (-s, c)), dtype=object).T

        x = self.x
        y = self.y

        x -= around_.x
        y -= around_.y

        coords = np.dot(rotation_matrix, ((x,), (y,)))
        x = coords[0][0]
        y = coords[1][0]

        x += around_.x
        y += around_.y

        return self.__class__(x, y)

    def dist(self, other: Optional[Self] = None, /) -> RealNumber:
        otherx = other.x if other else 0
        othery = other.y if other else 0
        return mp.sqrt((self.x-otherx)**2 + (self.y-othery)**2)

    def transform(
        self, /,
        transform_from: Optional[Transform] = None,
        transform_to: Optional[Transform] = None,
    ) -> Self:
        tfrom = transform_from or IdentityTransform()
        tto = (transform_to or IdentityTransform()).inverted()
        x, y = tto.transform(tfrom.transform((self.x, self.y)))
        return self.__class__(x, y)

    def coord(self, /) -> tuple[RealNumber, RealNumber]:
        return float(self.x), float(self.y)

    def normalize(self, /) -> Self:
        return self / self.dist()

    def rotate90(self, /) -> Self:
        return self.__class__(-self.y, self.x)

    def rotate180(self, /) -> Self:
        return self.__class__(-self.x, -self.y)

    def rotate270(self, /) -> Self:
        return self.__class__(self.y, -self.x)

    def dot(self, *others: Point) -> RealNumber:
        return (self.x*prod(map(attrgetter('x'), others)) +
                self.y*prod(map(attrgetter('y'), others)))


ORIGIN = Point(0, 0)
NAN_POINT = Point(float('nan'), float('nan'))


@dataclass(frozen=True)
class Triangle:

    a: Point
    b: Point
    c: Point

    def __str__(self, /) -> str:
        return (f"{self.__class__.__name__}("
                f"A({self.a}), "
                f"B({self.b}), "
                f"C({self.c}))")

    @classmethod
    def from_coords(
        cls, coords: (NDArray[Any] |
                      tuple[tuple[RealNumber, RealNumber, RealNumber],
                            tuple[RealNumber, RealNumber, RealNumber]]),
        /,
    ) -> Self:
        if isinstance(coords, np.ndarray):
            assert np.shape(coords) == (2, 3)
        return cls(Point(coords[0][0], coords[1][0]),
                   Point(coords[0][1], coords[1][1]),
                   Point(coords[0][2], coords[1][2]))

    @property
    def coords(self) -> NDArray[Any]:
        return np.array(((self.a.x, self.b.x, self.c.x),
                         (self.a.y, self.b.y, self.c.y)),
                        dtype=object)

    @property
    def draw_coords(self) -> NDArray[Any]:
        return np.array(((self.a.x, self.b.x, self.c.x, self.a.x),
                         (self.a.y, self.b.y, self.c.y, self.a.y)),
                        dtype=object)

    def rotate_coords(
        self, angle: RealNumber, /,
        *, around: Point = ORIGIN,
    ) -> NDArray[Any]:

        coords = self.coords
        s, c = mp.sin(angle), mp.cos(angle)
        rotation_matrix = np.array(((c, s), (-s, c)), dtype=object).T

        coords[0] -= around.x
        coords[1] -= around.y

        coords = np.dot(rotation_matrix, coords)

        coords[0] += around.x
        coords[1] += around.y

        return coords

    def rotate(self, angle: RealNumber, /, *, around: Point = ORIGIN) -> Self:
        coords = self.rotate_coords(angle, around=around)
        return self.__class__.from_coords(coords)

    @property
    def points(self) -> tuple[Point, Point, Point]:
        return (self.a, self.b, self.c)


class PositiveTriangle(Triangle):

    @property
    def top(self) -> Point:
        return self.a

    @property
    def left(self) -> Point:
        return self.b

    @property
    def right(self) -> Point:
        return self.c

    def __str__(self, /) -> str:
        return (
            f"{self.__class__.__name__}("
            f"T({self.top}), "
            f"L({self.left}), "
            f"R({self.right}))"
        )


class NegativeTriangle(Triangle):

    @property
    def bottom(self) -> Point:
        return self.a

    @property
    def right(self) -> Point:
        return self.b

    @property
    def left(self) -> Point:
        return self.c

    def __str__(self, /) -> str:
        return (
            f"{self.__class__.__name__}("
            f"B({self.bottom}), "
            f"R({self.right}), "
            f"L({self.left}))"
        )


@dataclass(frozen=True)
class Arc:
    point: Point
    angle_start: RealNumber
    angle_end: RealNumber
    radius: RealNumber = 1

    def rotate(self, angle: RealNumber, /) -> Self:
        angle_start = self.angle_start
        angle_end = self.angle_end
        new_point = self.point.rotate(angle)
        angle_start += angle
        angle_end += angle

        if angle_start > angle_end:
            angle_end += 2*mp.pi
        return self.__class__(new_point, angle_start, angle_end, self.radius)

    @property
    def coords(self) -> NDArray[Any]:
        angles = np.linspace(float(self.angle_start), float(self.angle_end))
        r = self.radius
        x = np.insert(r*np.cos(angles), 0, self.point.x)
        y = np.insert(r*np.sin(angles), 0, self.point.y)
        return np.row_stack((x, y))

    @property
    def draw_coords(self) -> NDArray[Any]:
        angles = np.linspace(float(self.angle_start), float(self.angle_end))
        r = self.radius
        x = np.concatenate(((self.point.x,),
                            r*np.cos(angles),
                            (self.point.x,)))
        y = np.concatenate(((self.point.y,),
                            r*np.sin(angles),
                            (self.point.y,)))
        return np.row_stack((x, y))

    @property
    def triangle_coords(self) -> NDArray[Any]:
        p, s, e, r = self.point, self.angle_start, self.angle_end, self.radius
        return np.array(((p.x, r*mp.cos(s), r*mp.cos(e)),
                         (p.y, r*mp.sin(s), r*mp.sin(e))),
                        dtype=object)

    @property
    def triangle_draw_coords(self) -> NDArray[Any]:
        p, s, e, r = self.point, self.angle_start, self.angle_end, self.radius
        return np.array(((p.x, r*mp.cos(s), r*mp.cos(e), p.x),
                         (p.y, r*mp.sin(s), r*mp.sin(e), p.x)),
                        dtype=object)


@dataclass(frozen=True)
class Segment:
    angle_start: RealNumber
    angle_end: RealNumber
    radius: RealNumber = 1

    def rotate(self, angle: RealNumber, /) -> Self:
        angle_start = self.angle_start
        angle_end = self.angle_end
        angle_start += angle
        angle_end += angle
        if angle_start > angle_end:
            angle_end += 2*mp.pi

        return self.__class__(angle_start, angle_end, self.radius)

    @property
    def coords(self) -> NDArray[Any]:
        r = self.radius
        angles = np.linspace(float(self.angle_start), float(self.angle_end))
        return np.row_stack((r*np.cos(angles), r*np.sin(angles)))

    @property
    def draw_coords(self) -> NDArray[Any]:
        r = self.radius
        angles = np.linspace(float(self.angle_start), float(self.angle_end))
        x = np.append(r*np.cos(angles), r*np.cos(float(self.angle_start)))
        y = np.append(r*np.sin(angles), r*np.sin(float(self.angle_start)))
        return np.row_stack((x, y))

    @property
    def triangle_coords(self) -> NDArray[Any]:
        s, e, r = self.angle_start, self.angle_end, self.radius
        c = (s + e)/2
        return np.array(((r*mp.cos(s), r*mp.cos(c), r*mp.cos(e)),
                         (r*mp.sin(s), r*mp.sin(c), r*mp.sin(e))),
                        dtype=object)

    @property
    def triangle_draw_coords(self) -> NDArray[Any]:
        s, e, r = self.angle_start, self.angle_end, self.radius
        c = (s + e)/2
        return np.array(((r*mp.cos(s), r*mp.cos(c), r*mp.cos(e), r*mp.cos(s)),
                         (r*mp.sin(s), r*mp.sin(c), r*mp.sin(e), r*mp.sin(s))),
                        dtype=object)


@dataclass
class ZeroShapeCollection:

    triangle: PositiveTriangle
    segment: Segment

    @classmethod
    def create(cls, /, triangle: PositiveTriangle) -> Self:
        segment = Segment(triangle.left.angle,
                          triangle.right.angle,
                          triangle.top.dist())
        return cls(triangle, segment)

    def rotate(self, angle: RealNumber, /) -> Self:
        return self.__class__(self.triangle.rotate(angle),
                              self.segment.rotate(angle))


@dataclass(frozen=True)
class BaseShapeCollection:

    triangle: PositiveTriangle
    left_arc: Arc
    right_arc: Arc
    segment: Segment

    @classmethod
    def create(
        cls, /, triangle: PositiveTriangle, above: PositiveTriangle,
    ) -> Self:
        r = triangle.left.dist()
        la = triangle.left.angle
        ra = triangle.right.angle
        ala = above.left.angle
        ara = above.right.angle
        left_arc = Arc(triangle.top, ala, la, r)
        right_arc = Arc(triangle.top, ra, ara, r)
        segment = Segment(la, ra, r)
        return cls(triangle, left_arc, right_arc, segment)

    def rotate(self, angle: RealNumber, /) -> Self:
        return self.__class__(self.triangle.rotate(angle),
                              self.left_arc.rotate(angle),
                              self.right_arc.rotate(angle),
                              self.segment.rotate(angle))


@dataclass(frozen=True)
class NormalShapeCollection:

    triangle: PositiveTriangle
    negative_triangle: NegativeTriangle
    horizontal_arc: Arc
    vertical_arc: Arc

    @classmethod
    def create(
        cls, /,
        side: SideType,
        triangle: PositiveTriangle,
        touching_horizontal: PositiveTriangle,
        touching_vertical: PositiveTriangle,
    ) -> Self:
        radius = (triangle.left
                  if side == SideType.LEFT else
                  triangle.right).dist()
        horizontal_angle_start = (touching_vertical.left
                                  if side == SideType.LEFT else
                                  triangle.right).angle
        horizontal_angle_end = (triangle.left
                                if side == SideType.LEFT else
                                touching_vertical.right).angle
        vertical_angle_start = (triangle.left
                                if side == SideType.LEFT else
                                touching_horizontal.right).angle
        vertical_angle_end = (touching_horizontal.left
                              if side == SideType.LEFT else
                              triangle.right).angle
        horizontal_arc = Arc(
            triangle.top,
            horizontal_angle_start,
            horizontal_angle_end,
            radius,
        )
        vertical_arc = Arc(
            triangle.right if side == SideType.LEFT else triangle.left,
            vertical_angle_start,
            vertical_angle_end,
            radius,
        )
        rotation_side_point = (triangle.right
                               if side == SideType.LEFT else
                               triangle.left)
        rotation_center = 0.5*(triangle.top + rotation_side_point)
        bottom = 2*rotation_center - triangle.top
        right = 2*rotation_center - triangle.left
        left = 2*rotation_center - triangle.right
        negative_triangle = NegativeTriangle(bottom, right, left)

        return cls(triangle, negative_triangle, horizontal_arc, vertical_arc)

    def rotate(self, angle: RealNumber, /) -> Self:

        return self.__class__(self.triangle.rotate(angle),
                              self.negative_triangle.rotate(angle),
                              self.horizontal_arc.rotate(angle),
                              self.vertical_arc.rotate(angle))
