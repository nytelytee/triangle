from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import (Generator, Generic, Literal, Optional, TypeAlias, TypeVar,
                    final)

from data import Data
from identifier import ContextualizedIdentifier, Identifier
from shapes import (BaseShapeCollection, NormalShapeCollection,
                    ZeroShapeCollection)
from util import SideType

IdentifierType = TypeVar('IdentifierType',
                         Identifier,
                         ContextualizedIdentifier)
DataType = TypeVar('DataType', bound=Data, covariant=True)


@final
@dataclass
class NormalNode(Generic[IdentifierType, DataType]):

    identifier: IdentifierType
    shapes: NormalShapeCollection
    data: tuple[DataType, ...]
    side: SideNode[IdentifierType, DataType]
    parent: RealNode[IdentifierType, DataType]
    touching_horizontal: (BaseNode[IdentifierType, DataType] |
                          NormalNode[IdentifierType, DataType])
    touching_vertical: RealNode[IdentifierType, DataType]

    horizontal: Optional[NormalNode[IdentifierType, DataType]] = None
    vertical: Optional[NormalNode[IdentifierType, DataType]] = None

    def walk_dfs_pre(
        self, /,
    ) -> Generator[NormalNode[IdentifierType, DataType], None, None]:
        yield self
        yield from self.horizontal.walk_dfs_pre() if self.horizontal else ()
        yield from self.vertical.walk_dfs_pre() if self.vertical else ()

    def walk_dfs_post(
        self, /,
    ) -> Generator[NormalNode[IdentifierType, DataType], None, None]:
        yield from self.horizontal.walk_dfs_post() if self.horizontal else ()
        yield from self.vertical.walk_dfs_post() if self.vertical else ()
        yield self

    walk_dfs = walk_dfs_pre

    def walk_bfs(
        self, /,
    ) -> Generator[NormalNode[IdentifierType, DataType], None, None]:
        queue: deque[NormalNode[IdentifierType, DataType]] = deque((self,))
        while queue:
            yield (element := queue.popleft())
            queue.extend(c for c in element.children if c is not None)

    @property
    def children(self) -> tuple[
        Optional[NormalNode[IdentifierType, DataType]],
        Optional[NormalNode[IdentifierType, DataType]]
    ]:
        return (self.horizontal, self.vertical)


# Not actually a real node (does not have any triangles attached to it)
@final
@dataclass
class SideNode(Generic[IdentifierType, DataType]):

    type: Literal[SideType.LEFT, SideType.RIGHT]
    base: BaseNode[IdentifierType, DataType]
    horizontal: Optional[NormalNode[IdentifierType, DataType]] = None

    @property
    def touching_vertical(
        self
    ) -> (
        BaseNode[IdentifierType, DataType] | ZeroNode[IdentifierType, DataType]
    ):
        return self.base.parent

    @property
    def touching_horizontal(self) -> BaseNode[IdentifierType, DataType]:
        return self.base

    @property
    def side(self) -> SideNode[IdentifierType, DataType]:
        return self

    def walk_dfs_pre(
        self, /,
    ) -> Generator[NormalNode[IdentifierType, DataType], None, None]:
        yield from self.horizontal.walk_dfs_pre() if self.horizontal else ()

    def walk_dfs_post(
        self, /,
    ) -> Generator[NormalNode[IdentifierType, DataType], None, None]:
        yield from self.horizontal.walk_dfs_post() if self.horizontal else ()

    walk_dfs = walk_dfs_pre

    def walk_bfs(
        self, /,
    ) -> Generator[NormalNode[IdentifierType, DataType], None, None]:
        yield from self.horizontal.walk_bfs() if self.horizontal else ()


@final
@dataclass
class BaseNode(Generic[IdentifierType, DataType]):

    identifier: IdentifierType
    shapes: BaseShapeCollection
    data: tuple[DataType, ...]
    base: Optional[BaseNode[IdentifierType, DataType]]
    parent: (BaseNode[IdentifierType, DataType] |
             ZeroNode[IdentifierType, DataType])
    left: SideNode[IdentifierType, DataType]
    right: SideNode[IdentifierType, DataType]

    def __init__(
        self,
        identifier: IdentifierType,
        shapes: BaseShapeCollection,
        data: tuple[DataType, ...],
        parent: (BaseNode[IdentifierType, DataType] |
                 ZeroNode[IdentifierType, DataType]),
    ):
        self.identifier = identifier
        self.shapes = shapes
        self.data = data
        self.parent = parent
        self.base = None
        self.left = SideNode(SideType.LEFT, self)
        self.right = SideNode(SideType.RIGHT, self)

    def walk_dfs_pre(
        self, /
    ) -> Generator[
        BaseNode[IdentifierType, DataType] |
        NormalNode[IdentifierType, DataType],
        None, None
    ]:
        yield self
        yield from (self.left.horizontal.walk_dfs_pre()
                    if self.left.horizontal else
                    ())
        yield from self.base.walk_dfs_pre() if self.base else ()
        yield from (self.right.horizontal.walk_dfs_pre()
                    if self.right.horizontal else
                    ())

    def walk_dfs_post(
        self, /
    ) -> Generator[
        BaseNode[IdentifierType, DataType] |
        NormalNode[IdentifierType, DataType],
        None, None
    ]:
        yield from (self.left.horizontal.walk_dfs_post()
                    if self.left.horizontal else
                    ())
        yield from self.base.walk_dfs_post() if self.base else ()
        yield from (self.right.horizontal.walk_dfs_post()
                    if self.right.horizontal else
                    ())
        yield self

    walk_dfs = walk_dfs_pre

    def walk_bfs(
        self, /,
    ) -> Generator[
        BaseNode[IdentifierType, DataType] |
        NormalNode[IdentifierType, DataType],
        None, None
    ]:
        queue: deque[BaseNode[IdentifierType, DataType] |
                     NormalNode[IdentifierType, DataType]] = deque((self,))
        while queue:
            yield (element := queue.popleft())
            queue.extend(c for c in element.children if c is not None)

    @property
    def children(self) -> tuple[
        Optional[NormalNode[IdentifierType, DataType]],
        Optional[BaseNode[IdentifierType, DataType]],
        Optional[NormalNode[IdentifierType, DataType]]
    ]:
        return (self.left.horizontal, self.base, self.right.horizontal)


@final
@dataclass
class ZeroNode(Generic[IdentifierType, DataType]):

    identifier: IdentifierType
    shapes: ZeroShapeCollection
    data: tuple[DataType, ...]
    parent: ZeroNode[IdentifierType, DataType]
    base: Optional[BaseNode[IdentifierType, DataType]] = None

    def __init__(
        self,
        identifier: IdentifierType,
        shapes: ZeroShapeCollection,
        data: tuple[DataType, ...],
    ) -> None:
        self.identifier = identifier
        self.shapes = shapes
        self.data = data
        self.parent = self
        self.base = None

    def walk_dfs_pre(
        self, /,
    ) -> Generator[RealNode[IdentifierType, DataType], None, None]:
        yield self
        yield from self.base.walk_dfs_pre() if self.base else ()

    def walk_dfs_post(
        self, /,
    ) -> Generator[RealNode[IdentifierType, DataType], None, None]:
        yield from self.base.walk_dfs_post() if self.base else ()
        yield self

    walk_dfs = walk_dfs_pre

    def walk_bfs(
        self, /,
    ) -> Generator[RealNode[IdentifierType, DataType], None, None]:
        queue: deque[RealNode[IdentifierType, DataType]] = deque((self,))
        while queue:
            yield (element := queue.popleft())
            queue.extend(c for c in element.children if c is not None)

    @property
    def children(self) -> tuple[Optional[BaseNode[IdentifierType, DataType]]]:
        return (self.base,)


RealNode: TypeAlias = (ZeroNode[IdentifierType, DataType] |
                       BaseNode[IdentifierType, DataType] |
                       NormalNode[IdentifierType, DataType])

INormalNode: TypeAlias = NormalNode[Identifier, DataType]
IBaseNode: TypeAlias = BaseNode[Identifier, DataType]
IZeroNode: TypeAlias = ZeroNode[Identifier, DataType]
IRealNode: TypeAlias = RealNode[Identifier, DataType]
ISideNode: TypeAlias = SideNode[Identifier, DataType]
CNormalNode: TypeAlias = NormalNode[ContextualizedIdentifier, DataType]
CBaseNode: TypeAlias = BaseNode[ContextualizedIdentifier, DataType]
CZeroNode: TypeAlias = ZeroNode[ContextualizedIdentifier, DataType]
CRealNode: TypeAlias = RealNode[ContextualizedIdentifier, DataType]
CSideNode: TypeAlias = SideNode[ContextualizedIdentifier, DataType]


@dataclass
class TriangleSideTree(Generic[IdentifierType, DataType]):

    zero: ZeroNode[IdentifierType, DataType]

    def walk_dfs_pre(
        self, /,
    ) -> Generator[RealNode[IdentifierType, DataType], None, None]:
        yield from self.zero.walk_dfs_pre()

    def walk_dfs_post(
        self, /,
    ) -> Generator[RealNode[IdentifierType, DataType], None, None]:
        yield from self.zero.walk_dfs_post()

    walk_dfs = walk_dfs_pre

    def walk_bfs(
        self, /,
    ) -> Generator[RealNode[IdentifierType, DataType], None, None]:
        yield from self.zero.walk_dfs()

    def __len__(self) -> int:
        c = count()
        deque(zip(self.walk_dfs(), c), maxlen=0)
        return next(c)
