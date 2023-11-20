from __future__ import annotations

from typing import (TYPE_CHECKING, Generic, Never, Optional, Type, TypeVar,
                    overload)

from mpmath import mp  # type: ignore[import-untyped]

from data import Data
from identifier import ContextualizedIdentifier, Identifier
from shapes import (BaseShapeCollection, NormalShapeCollection, Point,
                    PositiveTriangle, ZeroShapeCollection)
from tree import (BaseNode, CBaseNode, CNormalNode, CSideNode, CZeroNode,
                  IBaseNode, INormalNode, IRealNode, ISideNode, IZeroNode,
                  NormalNode, SideNode, TriangleSideTree, ZeroNode)
from util import (STEP_ALL, STEP_BASE, STEP_LEFT, STEP_LEFT_HORIZONTAL,
                  STEP_LEFT_VERTICAL, STEP_RIGHT, STEP_RIGHT_HORIZONTAL,
                  STEP_RIGHT_VERTICAL, AngleID, SideType, TriangleSideType,
                  stick_args)

LEFT_ANGLES = {
    TriangleSideType.A: AngleID.B,
    TriangleSideType.B: AngleID.C,
    TriangleSideType.C: AngleID.A,
}
RIGHT_ANGLES = {
    TriangleSideType.A: AngleID.C,
    TriangleSideType.B: AngleID.A,
    TriangleSideType.C: AngleID.B,
}

if TYPE_CHECKING:
    from util import RealNumber

DataType = TypeVar('DataType', bound=Data, covariant=True)


class TriangleSide(Generic[DataType]):

    type: TriangleSideType
    angle_left: RealNumber
    angle_right: RealNumber
    right_angle_left: bool
    right_angle_right: bool
    rotation_angle: RealNumber
    radius: RealNumber = mp.mpf(1)
    data_classes: tuple[Type[DataType], ...]
    side_to_height_ratio: RealNumber
    tree: TriangleSideTree[Identifier, DataType]

    @overload
    def __init__(
        self: TriangleSide[Never],
        _type: TriangleSideType, theta_A: RealNumber,
        theta_B: RealNumber, theta_C: RealNumber,
        /,
        right_angle: Optional[AngleID], radius: RealNumber = ...,
        *, data: tuple[()] = ...
    ) -> None:
        ...

    @overload
    def __init__(
        self: TriangleSide[DataType],
        _type: TriangleSideType, theta_A: RealNumber,
        theta_B: RealNumber, theta_C: RealNumber,
        /,
        right_angle: Optional[AngleID], radius: RealNumber = ...,
        *, data: (Type[DataType] |
                  tuple[Type[DataType], *tuple[Type[DataType], ...]]),
    ) -> None:
        ...

    def __init__(
        self,
        _type: TriangleSideType, theta_A: RealNumber,
        theta_B: RealNumber, theta_C: RealNumber,
        /,
        right_angle: Optional[AngleID], radius: RealNumber = 1,
        *,
        data: (Type[DataType] | tuple[()] |
               tuple[Type[DataType], *tuple[Type[DataType], ...]]) = (),
    ) -> None:

        self.type = _type
        angles = {AngleID.A: theta_A, AngleID.B: theta_B, AngleID.C: theta_C}
        rotation_angles = {TriangleSideType.A: theta_B - theta_C,
                           TriangleSideType.B: -theta_A - theta_C,
                           TriangleSideType.C: theta_A + theta_B}

        self.angle_left = angles[LEFT_ANGLES[self.type]]
        self.angle_right = angles[RIGHT_ANGLES[self.type]]
        self.right_angle_left = (
            right_angle == LEFT_ANGLES[self.type]
            if right_angle else
            False
        )
        self.right_angle_right = (
            right_angle == RIGHT_ANGLES[self.type]
            if right_angle else
            False
        )
        self.rotation_angle = rotation_angles[self.type]

        self.radius = radius
        self.data_classes = (()
                             if data == () else
                             (stick_args(data),)
                             if not isinstance(data, tuple) else
                             tuple(stick_args(d) for d in data))

        A = Point(0, radius).rotate(self.rotation_angle)
        B = A.rotate(2*theta_C)
        C = A.rotate(-2*theta_B)

        point_order = {TriangleSideType.A: (A, B, C),
                       TriangleSideType.B: (B, C, A),
                       TriangleSideType.C: (C, A, B)}

        zero_triangle = PositiveTriangle(*point_order[_type])

        self.side_to_height_ratio = (
            zero_triangle.right.x - zero_triangle.left.x
        ) / (
            zero_triangle.top.y - zero_triangle.left.y
        )

        shape_collection = ZeroShapeCollection.create(zero_triangle)
        data_sequence = tuple(
            data_class.for_zero_triangle(self.type, shape_collection)
            for data_class in self.data_classes
        )

        self.tree = TriangleSideTree(ZeroNode(Identifier(),
                                              shape_collection,
                                              data_sequence))

        self._base_triangle_top_y_cache = [zero_triangle.top.y]

    def get_nth_base_top_y(self, n: int, /) -> RealNumber:
        if len(self._base_triangle_top_y_cache) > n:
            return self._base_triangle_top_y_cache[n]
        k = self.side_to_height_ratio/2
        c1 = k**2 + 1
        c2 = -2*k**2 * self.get_nth_base_top_y(n-1)
        c3 = k**2*self.get_nth_base_top_y(n-1)**2 - self.radius**2
        self._base_triangle_top_y_cache.append(
            (-c2 - mp.sqrt(c2**2 - 4*c1*c3))/(2*c1)
        )
        return self._base_triangle_top_y_cache[n]

    def get_nth_base_top_x(self, n: int, /) -> RealNumber:
        Lnp1 = self.get_nth_base_top_y(n+1)
        Ln = self.get_nth_base_top_y(n)
        # calculate with the smaller of the two angles to make sure
        # you're not calculating with a right angle, a trick i don't
        # think you can do when calculating non-base triangles,
        # hence two different calculation functions for them
        tan_angle = (
            mp.tan(self.angle_left) if
            self.angle_left < self.angle_right else
            mp.tan(-self.angle_right)
        )
        c_angle = -1 if self.angle_left < self.angle_right else 1
        square_root = c_angle*mp.sqrt(self.radius**2 - Lnp1**2)

        return (Ln - Lnp1)/tan_angle + square_root

    def next_normal_calculation_right_angle(
        self, /,
        touching_horizontal: IRealNode[DataType],
        touching_vertical: IRealNode[DataType],
        side: SideType,
    ) -> PositiveTriangle:
        c_side = -1 if side == SideType.LEFT else 1
        c1 = -self.side_to_height_ratio
        c2 = (
            self.side_to_height_ratio *
            touching_vertical.shapes.triangle.right.y
        ) + c_side*touching_horizontal.shapes.triangle.top.x
        c3 = c1**2 + 1
        c4 = 2*c1*c2
        c5 = c2**2 - self.radius**2
        # C is the point that lies on the circle,
        # and P is the point opposite it
        Py = (-c4 - mp.sqrt(c4**2 - 4*c3*c5))/(2*c3)
        Px = touching_horizontal.shapes.triangle.top.x
        Cy = Py
        Cx = c_side*mp.sqrt(self.radius**2 - Py**2)
        Ty = touching_vertical.shapes.triangle.right.y
        Tx = Cx
        Rx, Ry = (Px, Py) if side == SideType.LEFT else (Cx, Cy)
        Lx, Ly = (Cx, Cy) if side == SideType.LEFT else (Px, Py)
        new_triangle = PositiveTriangle.from_coords(((Tx, Lx, Rx),
                                                     (Ty, Ly, Ry)))
        return new_triangle

    def next_normal_calculation(
        self, /,
        touching_horizontal: IRealNode[DataType],
        touching_vertical: IRealNode[DataType],
        side: SideType,
    ) -> PositiveTriangle:
        angle = (self.angle_left
                 if side == SideType.LEFT else
                 -self.angle_right)
        tan_angle = mp.tan(angle)
        c_side = -1 if side == SideType.LEFT else 1
        c_angle = (-1 if mp.fabs(angle) > 0.5*mp.pi else 1)
        c1 = tan_angle
        c2 = (
            -tan_angle *
            touching_horizontal.shapes.triangle.top.x
        ) + touching_horizontal.shapes.triangle.top.y
        c3 = c_side - self.side_to_height_ratio*tan_angle
        c4 = self.side_to_height_ratio * (
            touching_vertical.shapes.triangle.right.y + (
                tan_angle *
                touching_horizontal.shapes.triangle.top.x
            ) - touching_horizontal.shapes.triangle.top.y
        )
        c5 = c1**2 + c3**2
        c6 = 2*c1*c2 + 2*c3*c4
        c7 = c2**2 + c4**2 - self.radius**2
        solution_branch = c_side*c_angle
        # C is the point that lies on the circle,
        # and P is the point opposite it
        Px = (-c6 + solution_branch*mp.sqrt(c6**2 - 4*c5*c7))/(2*c5)
        Py = tan_angle*(
            Px - touching_horizontal.shapes.triangle.top.x
        ) + touching_horizontal.shapes.triangle.top.y
        Cy = Py
        Cx = c_side*mp.sqrt(self.radius**2 - Py**2)
        Ty = touching_vertical.shapes.triangle.right.y
        Tx = (Ty - Cy)/tan_angle + Cx

        Rx, Ry = (Px, Py) if side == SideType.LEFT else (Cx, Cy)
        Lx, Ly = (Cx, Cy) if side == SideType.LEFT else (Px, Py)

        new_triangle = PositiveTriangle.from_coords(((Tx, Lx, Rx),
                                                     (Ty, Ly, Ry)))
        return new_triangle

    def next_base(
        self, node: IBaseNode[DataType] | IZeroNode[DataType], /,
    ) -> IBaseNode[DataType]:
        if node.base is not None:
            return node.base
        new_identifier = node.identifier.new_base_id()
        Ty = self.get_nth_base_top_y(new_identifier.parts[0])
        Tx = self.get_nth_base_top_x(new_identifier.parts[0])
        Ry = Ly = self.get_nth_base_top_y(new_identifier.last_value + 1)
        Rx = mp.sqrt(self.radius**2 - Ry**2)
        Lx = -Rx

        new_triangle = PositiveTriangle.from_coords(((Tx, Lx, Rx),
                                                     (Ty, Ly, Ry)))
        new_shape_collection = BaseShapeCollection.create(
            new_triangle, node.shapes.triangle
        )
        new_data = tuple(
            data_class.for_base_triangle(self.type,
                                         new_identifier,
                                         new_shape_collection,
                                         node.identifier,
                                         node.shapes,
                                         node.data[i])
            for i, data_class in enumerate(self.data_classes)
        )

        node.base = BaseNode(
            new_identifier, new_shape_collection, new_data, node
        )

        return node.base

    def next_horizontal(
        self,
        node: ISideNode[DataType] | INormalNode[DataType],
        /,
    ) -> INormalNode[DataType]:
        if node.horizontal is not None:
            return node.horizontal

        real_node: IBaseNode[DataType] | INormalNode[DataType]  # thanks, mypy
        real_node = node.base if isinstance(node, SideNode) else node

        new_identifier = real_node.identifier.new_horizontal_id()

        assert node.touching_vertical is not None
        assert node.touching_horizontal is not None

        if (
            (node.side.type == SideType.LEFT and self.right_angle_left) or
            (node.side.type == SideType.RIGHT and self.right_angle_right)
        ):
            new_triangle = self.next_normal_calculation_right_angle(
                real_node, node.touching_vertical, node.side.type
            )
        else:
            new_triangle = self.next_normal_calculation(real_node,
                                                        node.touching_vertical,
                                                        node.side.type)

        new_shape_collection = NormalShapeCollection.create(
            node.side.type, new_triangle, real_node.shapes.triangle,
            node.touching_vertical.shapes.triangle,
        )
        new_data = tuple(data_class.for_horizontal_triangle(
            self.type,
            node.side.type,
            new_identifier,
            new_shape_collection,
            node.touching_horizontal.identifier,
            node.touching_horizontal.shapes,
            node.touching_horizontal.data[i],
            node.touching_vertical.identifier,
            node.touching_vertical.shapes,
            node.touching_vertical.data[i],
        ) for i, data_class in enumerate(self.data_classes))

        node.horizontal = NormalNode(
            new_identifier, new_shape_collection, new_data,
            node.side, real_node, real_node, node.touching_vertical,
        )
        return node.horizontal

    def next_vertical(
        self, node: INormalNode[DataType], /,
    ) -> INormalNode[DataType]:
        if node.vertical is not None:
            return node.vertical

        new_identifier = node.identifier.new_vertical_id()
        assert node.touching_horizontal is not None

        if (
            (node.side.type == SideType.LEFT and self.right_angle_left) or
            (node.side.type == SideType.RIGHT and self.right_angle_right)
        ):
            new_triangle = self.next_normal_calculation_right_angle(
                node.touching_horizontal, node, node.side.type,
            )
        else:
            new_triangle = self.next_normal_calculation(
                node.touching_horizontal, node, node.side.type,
            )

        new_shape_collection = NormalShapeCollection.create(
            node.side.type, new_triangle,
            node.touching_horizontal.shapes.triangle,
            node.shapes.triangle,
        )
        new_data = tuple(data_class.for_vertical_triangle(
            self.type,
            node.side.type,
            new_identifier,
            new_shape_collection,
            node.touching_horizontal.identifier,
            node.touching_horizontal.shapes,
            node.touching_horizontal.data[i],
            node.identifier,
            node.shapes,
            node.data[i],
        ) for i, data_class in enumerate(self.data_classes))

        node.vertical = NormalNode(
            new_identifier, new_shape_collection,
            new_data, node.side, node, node.touching_horizontal, node,
        )
        return node.vertical

    def _lookup_edge_children(
        self,
        node: Optional[IRealNode[DataType]],
        /,
        result: list[IRealNode[DataType]],
    ) -> None:

        match node:
            case ZeroNode():
                if node.base is None:
                    result.append(node)
                self._lookup_edge_children(node.base, result)
            case BaseNode():
                if None in (
                    node.base, node.left.horizontal, node.right.horizontal
                ):
                    result.append(node)
                self._lookup_edge_children(node.base, result)
                self._lookup_edge_children(node.left.horizontal, result)
                self._lookup_edge_children(node.right.horizontal, result)
            case NormalNode():
                if node.horizontal is None or node.vertical is None:
                    result.append(node)
                self._lookup_edge_children(node.horizontal, result)
                self._lookup_edge_children(node.vertical, result)

    def _step_zero(
        self, node: IZeroNode[DataType], /,
        count: int = 1, flag: int = STEP_ALL,
        *, ignore_existing: bool = False,
    ) -> list[IBaseNode[DataType] | INormalNode[DataType]]:

        if (
            count == 0 or
            not (flag & STEP_BASE) or
            (ignore_existing and node.base is not None)
        ):
            return []
        returns: list[IBaseNode[DataType] | INormalNode[DataType]] = []
        returns.append(new := self.next_base(node))
        returns.extend(self._step_base(new, count=count-1, flag=flag))
        return returns

    def _step_base(
        self, node: IBaseNode[DataType], /,
        count: int = 1, flag: int = STEP_ALL,
        *, ignore_existing: bool = False,
    ) -> list[IBaseNode[DataType] | INormalNode[DataType]]:
        if count == 0 or not flag:
            return []
        returns: list[IBaseNode[DataType] | INormalNode[DataType]] = []
        new: IBaseNode[DataType] | INormalNode[DataType]
        if (
            flag & STEP_BASE and
            not (ignore_existing and node.base is not None)
        ):
            returns.append(new := self.next_base(node))
            returns.extend(self._step_base(new, count=count-1, flag=flag))

        if (
            flag & STEP_LEFT_HORIZONTAL and
            not (ignore_existing and node.left.horizontal is not None)
        ):
            returns.append(new := self.next_horizontal(node.left))
            returns.extend(self._step_normal(new, count=count-1, flag=flag))
        if (
            flag & STEP_RIGHT_HORIZONTAL and
            not (ignore_existing and node.right.horizontal is not None)
        ):
            returns.append(new := self.next_horizontal(node.right))
            returns.extend(self._step_normal(new, count=count-1, flag=flag))
        return returns

    def _step_normal(
        self, node: INormalNode[DataType], /,
        count: int = 1, flag: int = STEP_ALL,
        *, ignore_existing: bool = False,
    ) -> list[INormalNode[DataType] | IBaseNode[DataType]]:
        side_flag = (STEP_LEFT
                     if node.side.type == SideType.LEFT else
                     STEP_RIGHT)
        if count == 0 or not (flag & side_flag):
            return []
        horizontal_flag = (STEP_LEFT_HORIZONTAL
                           if node.side.type == SideType.LEFT else
                           STEP_RIGHT_HORIZONTAL)
        vertical_flag = (STEP_LEFT_VERTICAL
                         if node.side.type == SideType.LEFT else
                         STEP_RIGHT_VERTICAL)

        returns: list[INormalNode[DataType] | IBaseNode[DataType]] = []

        if (
            flag & horizontal_flag and
            not (ignore_existing and node.horizontal is not None)
        ):
            returns.append(new := self.next_horizontal(node))
            returns.extend(self._step_normal(new, count=count-1, flag=flag))

        if (
            flag & vertical_flag and
            not (ignore_existing and node.vertical is not None)
        ):
            returns.append(new := self.next_vertical(node))
            returns.extend(self._step_normal(new, count=count-1, flag=flag))
        return returns

    def step(
        self,
        node: IRealNode[DataType],
        /,
        count: int = 1, flag: int = STEP_ALL,
        *, ignore_existing: bool = False,
    ) -> list[IBaseNode[DataType] | INormalNode[DataType]]:
        match node:
            case ZeroNode():
                return self._step_zero(node, count=count, flag=flag,
                                       ignore_existing=ignore_existing)
            case BaseNode():
                return self._step_base(node, count=count, flag=flag,
                                       ignore_existing=ignore_existing)
            case NormalNode():
                return self._step_normal(node, count=count, flag=flag,
                                         ignore_existing=ignore_existing)

    def step_all(
        self, /,
        count: int = 1, flag: int = STEP_ALL,
    ) -> list[IBaseNode[DataType] | INormalNode[DataType]]:
        nodes_to_step: list[IRealNode[DataType]] = []
        self._lookup_edge_children(self.tree.zero, nodes_to_step)
        new_nodes: list[IBaseNode[DataType] | INormalNode[DataType]] = []
        for node in nodes_to_step:
            new_nodes.extend(self.step(node, count=count,
                                       flag=flag, ignore_existing=True))
        return new_nodes

    def finalized_tree(
        self, /, rotate: RealNumber = 0,
    ) -> TriangleSideTree[ContextualizedIdentifier, DataType]:

        final_tree = TriangleSideTree(ZeroNode(
            ContextualizedIdentifier(self.type, SideType.ZERO,
                                     self.tree.zero.identifier),
            self.tree.zero.shapes.rotate(rotate - self.rotation_angle),
            self.tree.zero.data,
        ))

        def finalize_base(
            parent: CZeroNode[DataType] | CBaseNode[DataType],
            child_to_process: IBaseNode[DataType],
        ) -> None:
            new_base = BaseNode(
                ContextualizedIdentifier(self.type, SideType.BASE,
                                         child_to_process.identifier),
                child_to_process.shapes.rotate(rotate - self.rotation_angle),
                child_to_process.data,
                parent,
            )
            parent.base = new_base
            if child_to_process.left.horizontal is not None:
                finalize_horizontal(
                    new_base.left, child_to_process.left.horizontal
                )
            if child_to_process.right.horizontal is not None:
                finalize_horizontal(
                    new_base.right, child_to_process.right.horizontal
                )
            if child_to_process.base is not None:
                finalize_base(parent.base, child_to_process.base)

        def finalize_horizontal(
            parent: CNormalNode[DataType] | CSideNode[DataType],
            child_to_process: INormalNode[DataType],
        ) -> None:
            new_horizontal = NormalNode(
                ContextualizedIdentifier(self.type, child_to_process.side.type,
                                         child_to_process.identifier),
                child_to_process.shapes.rotate(rotate - self.rotation_angle),
                child_to_process.data,
                parent.side,
                parent.base if isinstance(parent, SideNode) else parent,
                parent.base if isinstance(parent, SideNode) else parent,
                parent.touching_vertical
            )
            parent.horizontal = new_horizontal

            if child_to_process.horizontal is not None:
                finalize_horizontal(
                    parent.horizontal, child_to_process.horizontal,
                )
            if child_to_process.vertical is not None:
                finalize_vertical(
                    parent.horizontal, child_to_process.vertical,
                )

        def finalize_vertical(
            parent: CNormalNode[DataType],
            child_to_process: INormalNode[DataType],
        ) -> None:
            new_vertical = NormalNode(
                ContextualizedIdentifier(self.type, child_to_process.side.type,
                                         child_to_process.identifier),
                child_to_process.shapes.rotate(rotate - self.rotation_angle),
                child_to_process.data, parent.side,
                parent, parent.touching_horizontal, parent,
            )
            parent.vertical = new_vertical

            if child_to_process.horizontal is not None:
                finalize_horizontal(
                    parent.vertical, child_to_process.horizontal,
                )
            if child_to_process.vertical is not None:
                finalize_vertical(
                    parent.vertical, child_to_process.vertical,
                )

        if self.tree.zero.base is not None:
            finalize_base(final_tree.zero, self.tree.zero.base)

        return final_tree
