from __future__ import annotations

import builtins
import sys
from enum import Enum, auto
from functools import cache
from time import perf_counter_ns
from types import TracebackType
from typing import (TYPE_CHECKING, Callable, Generator, Literal, Optional,
                    Protocol, Self, Sequence, TextIO, TypeAlias, TypeVar,
                    TypeVarTuple, cast, get_args, get_origin, overload)

import numpy as np

if TYPE_CHECKING:
    from mpmath import mpf  # type: ignore[import-untyped]

    from identifier import Identifier
    from shapes import Point
    RealNumber: TypeAlias = float | mpf


class IdentifierType(Enum):
    ZERO = auto()
    BASE = auto()
    HORIZONTAL = auto()
    VERTICAL = auto()


class TriangleSideType(Enum):
    A = auto()
    B = auto()
    C = auto()


class SideType(Enum):
    ZERO = auto()
    BASE = auto()
    LEFT = auto()
    RIGHT = auto()


class AngleID(Enum):
    A = auto()
    B = auto()
    C = auto()


MPLColor: TypeAlias = (tuple[float, float, float, float] |
                       tuple[float, float, float] |
                       str |
                       tuple[tuple[float, float, float, float], float] |
                       tuple[tuple[float, float, float], float] |
                       tuple[str, float] |
                       None)

if TYPE_CHECKING:
    LookupKeyType: TypeAlias = tuple[Optional[TriangleSideType],
                                     Optional[SideType],
                                     Optional[Identifier]]


def is_mpl_color_transparent(color: MPLColor) -> bool:
    if color is None:
        return False
    if isinstance(color, str):
        return (color == 'none' or (color.startswith('#') and
                                    len(color) == 9 and
                                    color.endswith('00')))
    assert isinstance(color, tuple)
    if len(color) == 2:
        return color[1] == 0 or color[0] == 'none'
    if len(color) == 4:
        return color[3] == 0
    assert False


def is_mpl_color_opaque(color: MPLColor) -> bool:
    if color is None:
        return True
    if isinstance(color, str):
        return (color == 'none' or (color.startswith('#') and
                                    len(color) == 9 and
                                    color.lower().endswith('ff')))
    assert isinstance(color, tuple)
    if len(color) == 2:
        return color[1] == 1 and color[0] != 'none'
    if len(color) == 4:
        return color[3] == 1
    return True


def is_mpl_color_semitransparent(color: MPLColor) -> bool:
    return (
        not is_mpl_color_opaque(color) and not is_mpl_color_transparent(color)
    )


STEP_BASE = 1 << 0
STEP_LEFT_HORIZONTAL = 1 << 1
STEP_LEFT_VERTICAL = 1 << 2
STEP_RIGHT_HORIZONTAL = 1 << 3
STEP_RIGHT_VERTICAL = 1 << 4

STEP_LEFT = STEP_LEFT_HORIZONTAL | STEP_LEFT_VERTICAL
STEP_RIGHT = STEP_RIGHT_HORIZONTAL | STEP_RIGHT_VERTICAL
STEP_HORIZONTAL = STEP_LEFT_HORIZONTAL | STEP_RIGHT_HORIZONTAL
STEP_VERTICAL = STEP_LEFT_VERTICAL | STEP_RIGHT_VERTICAL

STEP_BASE_HORIZONTAL = STEP_BASE | STEP_HORIZONTAL
STEP_BASE_VERTICAL = STEP_BASE | STEP_VERTICAL
STEP_NONBASE = STEP_LEFT | STEP_RIGHT

STEP_ALL = STEP_BASE | STEP_LEFT | STEP_RIGHT


TypeType = TypeVar('TypeType', bound=type)


def stick_args(x: TypeType, /) -> TypeType:

    if not get_args(x) or not get_origin(x):
        return x

    o = get_origin(x)
    a = get_args(x)

    class _Newx(o[a]):  # type: ignore[valid-type, misc]
        pass

    try:
        _Newx.__module__ = x.__module__
    except AttributeError:
        pass
    try:
        _Newx.__name__ = x.__name__
    except AttributeError:
        pass
    try:
        _Newx.__qualname__ = x.__qualname__
    except AttributeError:
        pass
    try:
        _Newx.__annotations__ = x.__annotations__
    except AttributeError:
        pass
    try:
        _Newx.__doc__ = x.__doc__
    except AttributeError:
        pass

    return cast(TypeType, _Newx)


class PrintFunction(Protocol):
    def __call__(
        self, *values: object,
        sep: Optional[str] = ...,
        end: Optional[str] = ...,
        file: Optional[TextIO] = ...,
        flush: bool = ...
    ) -> None:
        ...


def format_time(time_ns: int) -> str:
    time = time_ns / 10**9
    time, time_s = divmod(time, 60)
    time_h, time_m = divmod(time, 60)
    return (f"{'{:02d}h'.format(int(time_h)) if time_h else ''}"
            f"{'{:02d}m'.format(int(time_m)) if time_m or time_h else ''}"
            f"{'{:06.3f}s'.format(time_s) if time_s else '00.000s'}")


# this assumes stdout
# hint: sed -E 's/.*\x1b\[2K\r//'
# if you want to output it to a file
class Subtimers:

    subtimers: list[tuple[str, int]]
    subtimer_types: list[Literal['manual', 'context-manager']]
    state: Literal['newline', 'inline-noinfoline', 'inline-infoline']
    last_printed_line: str
    padding_function: Callable[[int, Subtimers], str]
    subtimer_name_format: Callable[[str, Subtimers], str]
    subtimer_name_format_inline: Callable[[str, Subtimers], str]
    time_format: Callable[[int, Subtimers], str]
    time_format_inline: Callable[[int, Subtimers], str]
    prefix: str
    suffix: str
    waiting_suffix: str
    inline_prefix: str
    inline_suffix: str
    inline_waiting_suffix: str
    noinfoline_separator: str
    infoline_is_inline: bool
    noinfoline_is_inline: bool
    newline: bool
    _print: PrintFunction
    _print_overridden: bool

    class SubtimerError(Exception):
        pass

    class SubtimerContextManager:

        def __init__(self, name: str, subtimers: Subtimers, /) -> None:
            self.name = name
            self.subtimers = subtimers

        def __enter__(self, /) -> None:
            self.subtimers.subtimer_types.append('context-manager')
            self.subtimers._push(self.name)

        @overload
        def __exit__(
            self, exc_type: None, exc_val: None, exc_tb: None, /,
        ) -> None:
            ...

        @overload
        def __exit__(
            self,
            exc_type: type[BaseException],
            exc_val: BaseException,
            exc_tb: TracebackType,
            /,
        ) -> None:
            ...

        def __exit__(
            self,
            exc_type: Optional[type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
            /,
        ) -> None:
            if exc_type is not None:
                return
            if self.subtimers.subtimer_types[-1] != 'context-manager':
                raise self.subtimers.SubtimerError(
                    "Unpopped manual subtimers in context."
                )
            del self.subtimers.subtimer_types[-1]
            self.subtimers._pop()

    def __init__(
        self, /,
        padding: str | Callable[[int, Subtimers], str] = '│  ',
        subtimer_name_format: Callable[[str, Subtimers], str] = (
            lambda subtimer_name, subtimers: '┌► ' + subtimer_name
        ),
        subtimer_name_format_inline: Callable[[str, Subtimers], str] = (
            lambda subtimer_name, subtimers: subtimer_name
        ),
        time_format: Callable[[int, Subtimers], str] = (
            lambda time_ns, subtimers: format_time(time_ns)
        ),
        time_format_inline: Callable[[int, Subtimers], str] = (
            lambda time_ns, subtimers: format_time(time_ns)
        ),
        inline_prefix: str = ' ══► ',
        inline_suffix: str = ' ──► ',
        inline_waiting_suffix: str = ' --> ',
        noinfoline_separator: str = ' ──► ',
        prefix: str = '',
        suffix: str = '└► ',
        waiting_suffix: str = '└> ',
        infoline_is_inline: bool = True,
        noinfoline_is_inline: bool = True
    ) -> None:
        self.subtimers = []
        self.subtimer_types = []
        self.state = 'newline'
        self.last_printed_line = ''
        self.padding_function = (
            padding
            if callable(padding) else
            cast(
                Callable[[int, Subtimers], str],
                cache(lambda level, subtimers: level*padding)
            )
        )
        self.subtimer_name_format = subtimer_name_format
        self.subtimer_name_format_inline = subtimer_name_format_inline
        self.time_format = time_format
        self.time_format_inline = time_format_inline
        self.inline_prefix = inline_prefix
        self.inline_suffix = inline_suffix
        self.inline_waiting_suffix = inline_waiting_suffix
        self.noinfoline_separator = noinfoline_separator
        self.prefix = prefix
        self.suffix = suffix
        self.waiting_suffix = waiting_suffix
        self.infoline_is_inline = noinfoline_is_inline and infoline_is_inline
        self.noinfoline_is_inline = noinfoline_is_inline
        self.newline = False
        # i have no idea how to fix these type ignores
        self._print = builtins.print  # type: ignore[assignment]
        self._print_overridden = False

    def __enter__(self, /) -> Self:
        if self._print_overridden:
            raise self.SubtimerError(
                "Can't override print multiple times (nested?)."
            )
        self._print_overridden = True
        # i have no idea how to fix these type ignores
        self._print = builtins.print  # type: ignore[assignment]
        builtins.print = self.print  # type: ignore[assignment]
        return self

    @overload
    def __exit__(
        self, exc_type: None, exc_val: None, exc_tb: None, /,
    ) -> None:
        ...

    @overload
    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException,
        exc_tb: TracebackType,
        /,
    ) -> None:
        ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
        /,
    ) -> None:
        if not self._print_overridden:
            raise self.SubtimerError(
                "Print not overridden (manually called __exit__?)."
            )
        # i have no idea how to fix these type ignores
        builtins.print = self._print  # type: ignore[assignment]

    def _push(self, subtimer_name: str, /) -> None:
        if len(self.subtimers) == 0 and self.newline:
            self.newline = False
            self._print()
        padding = self.padding_function(len(self.subtimers), self)
        prev_padding = self.padding_function(len(self.subtimers) - 1, self)
        prefix = self.prefix.replace(
            '{padding}', padding
        ).replace(
            '{prev_padding}', prev_padding
        )
        waiting_suffix = self.waiting_suffix.replace(
            '{next_padding}', padding
        ).replace(
            '{padding}', prev_padding
        )
        clear = '\33[2K\r'
        match self.state:
            case 'newline':
                pass
            case 'inline-infoline':
                name = self.subtimers[-1][0]
                self._print(
                    f"{clear}{prev_padding}"
                    f"{self.subtimer_name_format(name, self)}{prefix}\n"
                    f"{padding}{self.last_printed_line}"
                )
            case 'inline-noinfoline':
                name = self.subtimers[-1][0]
                self._print(
                    f"{clear}{prev_padding}"
                    f"{self.subtimer_name_format(name, self)}{prefix}"
                )
        if self.noinfoline_is_inline:
            self._print(
                f"{clear}{padding}"
                f"{self.subtimer_name_format_inline(subtimer_name, self)}"
                f"{self.inline_waiting_suffix}", end='', flush=True
            )
            self.state = 'inline-noinfoline'
        else:
            self._print(
                f"{clear}{padding}"
                f"{self.subtimer_name_format(subtimer_name, self)}{prefix}"
                f"\n{padding}{waiting_suffix}", end='', flush=True
            )
            self.state = 'newline'
        self.subtimers.append((subtimer_name, perf_counter_ns()))

    def _pop(self, /) -> None:
        prev_padding = self.padding_function(len(self.subtimers), self)
        name, timer = self.subtimers.pop()
        padding = self.padding_function(len(self.subtimers), self)
        waiting_padding = self.padding_function(len(self.subtimers) - 1, self)
        clear = '\33[2K\r'
        waiting_suffix = self.waiting_suffix.replace(
            '{next_padding}', padding
        ).replace(
            '{padding}', waiting_padding
        )
        match self.state:
            case 'newline':
                suffix = self.suffix.replace(
                    '{padding}', padding
                ).replace(
                    '{next_padding}', prev_padding
                )
                self._print(
                    f"{clear}{padding}{suffix}"
                    f"{self.time_format(perf_counter_ns() - timer, self)}"
                    f"\n{waiting_padding}{waiting_suffix}",
                    end='', flush=True
                )
            case 'inline-infoline':
                self._print(
                    f"{clear}{padding}"
                    f"{self.subtimer_name_format_inline(name, self)}"
                    f"{self.inline_prefix}"
                    f"{self.last_printed_line}"
                    f"{self.inline_suffix}"
                    f"{self.time_format_inline(perf_counter_ns()-timer, self)}"
                    f"\n{waiting_padding}{waiting_suffix}",
                    end='', flush=True
                )
            case 'inline-noinfoline':
                self._print(
                    f"{clear}{padding}"
                    f"{self.subtimer_name_format_inline(name, self)}"
                    f"{self.noinfoline_separator}"
                    f"{self.time_format_inline(perf_counter_ns()-timer, self)}"
                    f"\n{waiting_padding}{waiting_suffix}",
                    end='', flush=True
                )
        if not len(self.subtimers):
            self._print(clear, end='')
        self.state = 'newline'
        self.newline = len(self.subtimers) > 0

    def push(self, subtimer_name: str, /) -> None:
        self.subtimer_types.append('manual')
        self._push(subtimer_name)

    def pushed(
        self, subtimer_name: str, /,
    ) -> Subtimers.SubtimerContextManager:
        return self.SubtimerContextManager(subtimer_name, self)

    def pop(self, /) -> None:
        if self.subtimer_types[-1] != 'manual':
            raise self.SubtimerError(
                "Cannot pop context manager subtimer manaully."
            )
        del self.subtimer_types[-1]
        self._pop()

    def print(
        self, /,
        *values: object,
        sep: Optional[str] = ' ',
        end: Optional[str] = '\n',
        file: Optional[TextIO] = None,
        flush: bool = False,
    ) -> None:
        sep = ' ' if sep is None else sep
        end = '\n' if end is None else end
        # we are not printing to stdout, just call the overloaded print
        if file is not None and file is not sys.stdout:
            return self._print(
                *values, sep=sep, end=end, file=file, flush=False
            )
        # we are not inside a subtimer, just call the overloaded print
        if len(self.subtimers) == 0:
            return self._print(
                *values, sep=sep, end=end, file=file, flush=flush
            )
            strings = (sep.join(str(o) for o in values) + end).split('\n')
            self.newline = strings[-1] != ''
        padding = self.padding_function(len(self.subtimers), self)
        prev_padding = self.padding_function(len(self.subtimers) - 1, self)
        clear = '\33[2K\r'
        if (self.state == 'inline-noinfoline' and not self.infoline_is_inline):
            prefix = self.prefix.replace(
                '{padding}', padding
            ).replace(
                '{prev_padding}', prev_padding
            )
            name = self.subtimers[-1][0]
            self._print(
                f"{clear}{prev_padding}"
                f"{self.subtimer_name_format(name, self)}{prefix}"
            )
            self.state = 'newline'

        strings = (sep.join(str(o) for o in values) + end).split('\n')
        for i, content in enumerate(strings, 0):
            is_last = (i == len(strings)-1)
            if is_last and content == '':
                self.newline = True
                break
            elif is_last and content != '':
                self.newline = False
            match self.state:
                case 'newline':
                    waiting_suffix = self.waiting_suffix.replace(
                        '{next_padding}', padding
                    ).replace(
                        '{padding}', prev_padding
                    ) if len(self.subtimers) > 0 else ''
                    if self.newline:
                        self._print(
                            f"{clear}{padding}{content}"
                            f"\n{prev_padding}{waiting_suffix}",
                            end='', flush=True
                        )
                        self.last_printed_line = content
                    else:
                        self._print(
                            f"{clear}{padding}"
                            f"{self.last_printed_line}{content}"
                            f"{self.inline_waiting_suffix}",
                            end='', flush=True
                        )
                        self.newline = True
                        self.last_printed_line = self.last_printed_line+content
                    self.state = 'newline'
                case 'inline-infoline':
                    name = self.subtimers[-1][0]
                    prefix = self.prefix.replace(
                        '{padding}', padding
                    ).replace(
                        '{prev_padding}', prev_padding
                    )
                    waiting_suffix = self.waiting_suffix.replace(
                        '{next_padding}', padding
                    ).replace(
                        '{padding}', prev_padding
                    )
                    if self.newline:
                        self._print(
                            f"{clear}{prev_padding}"
                            f"{self.subtimer_name_format(name, self)}"
                            f"{prefix}\n{padding}{self.last_printed_line}\n"
                            f"{padding}{content}"
                            f"\n{prev_padding}{waiting_suffix}",
                            end='', flush=True
                        )
                        self.state = 'newline'
                        self.last_printed_line = content
                    else:
                        self._print(
                            f"{clear}{prev_padding}"
                            f"{self.subtimer_name_format_inline(name, self)}"
                            f"{self.inline_prefix}{self.last_printed_line}"
                            f"{content}{self.inline_waiting_suffix}",
                            end='', flush=True
                        )
                        self.state = 'inline-infoline'
                        self.newline = True
                        self.last_printed_line = self.last_printed_line+content
                case 'inline-noinfoline':
                    name = self.subtimers[-1][0]
                    self._print(
                        f"{clear}{prev_padding}"
                        f"{self.subtimer_name_format_inline(name, self)}"
                        f"{self.inline_prefix}{content}"
                        f"{self.inline_waiting_suffix}",
                        end='',
                        flush=True
                    )
                    self.newline = True
                    self.state = 'inline-infoline'
                    self.last_printed_line = content


T = TypeVar('T')


def sliding_window(
    elements: Sequence[T],
    window_size: int
) -> Generator[list[T], None, None]:

    elems = iter(elements)
    try:
        to_return = [next(elems) for _ in range(window_size)]
    except StopIteration:
        # do not yield anything if there aren't enough elements
        # to complete the window
        return
    yield to_return
    while True:
        to_return.pop(0)
        try:
            to_return.append(next(elems))
        except StopIteration:
            return
        yield to_return


A = TypeVarTuple('A')
R = TypeVar('R')


def apply_unpacked(f: Callable[[*A], R], args: tuple[*A]) -> R:
    return f(*args)


def offset_polygon(points: Sequence[Point], dist: float, /) -> list[Point]:
    # this should work for all triangles, which is the only thing i
    # actually care about here, but it should work for polygons
    # generally, as long as their offset does not result in multiple
    # polygons (i.e. if shapely returns a MultiPolygon when offsetting
    # this polygon, then this will return an incorrect result, sorry).

    # when processing for example, a triangle ABC each point's offset
    # point is calculated based on the lines connecting them to the
    # previous and next point. the previous point of A should be C
    # and the next point of C should be A
    points = [points[-1], *points, points[0]]

    new_points = []
    for prev, curr, next in sliding_window(points, 3):
        n1 = (prev - curr).rotate90().normalize()
        n2 = (curr - next).rotate90().normalize()
        bisector = (n1 + n2).normalize()
        length = dist / np.sqrt(0.5 + 0.5*n1.dot(n2))
        new_points.append(curr + length*bisector)
    return new_points
