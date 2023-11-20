from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from itertools import chain, repeat
from typing import Callable, Self, Sequence, cast

import hsluv  # type: ignore[import-untyped]


class Color(ABC):

    @classmethod
    @abstractmethod
    def mix(cls, /, *args: Self | float) -> Self:
        raise NotImplementedError

    @classmethod
    def lerp(cls, t: float, /, color_start: Self, color_end: Self) -> Self:
        return cls.mix(color_start, 1-t, color_end, t)

    @classmethod
    @abstractmethod
    def from_rgb(cls, r: float, g: float, b: float, /) -> Self:
        raise NotImplementedError

    @abstractmethod
    def to_rgb(self, /) -> tuple[float, float, float]:
        raise NotImplementedError

    def to_hex(self, /) -> str:
        r, g, b = [int(x) for x in self.to_rgb()]
        rs, gs, bs = hex(r)[2:], hex(g)[2:], hex(b)[2:]
        rs = ('00' if rs[0] == 'x' else  # "-0x..."[2:] = "x..."
              'ff' if len(rs) > 2 else
              f'0{rs}' if len(rs) == 1 else
              rs)
        gs = ('00' if gs[0] == 'x' else  # "-0x..."[2:] = "x..."
              'ff' if len(gs) > 2 else
              f'0{gs}' if len(gs) == 1 else
              gs)
        bs = ('00' if bs[0] == 'x' else  # "-0x..."[2:] = "x..."
              'ff' if len(bs) > 2 else
              f'0{bs}' if len(bs) == 1 else
              bs)
        return f'#{rs}{gs}{bs}'

    @classmethod
    @abstractmethod
    def from_hex(cls, hex_color: str, /) -> Self:
        raise NotImplementedError


# i am not implementing any colors that include alpha
# (although it would not be hard), but i will just keep
# this function here anyway
def hex_to_rgba(
    hex_color: str, /,
    *, strict: bool = False,
) -> tuple[float, float, float, float]:
    rgb = hex_color.lstrip("#")
    if len(rgb) == 8:  # true color + alpha
        r = int(rgb[0:2], 16)
        g = int(rgb[2:4], 16)
        b = int(rgb[4:6], 16)
        a = int(rgb[6:8], 16)
        return r, g, b, a
    elif len(rgb) == 6:  # true color
        r = int(rgb[0:2], 16)
        g = int(rgb[2:4], 16)
        b = int(rgb[4:6], 16)
        return r, g, b, 255
    elif len(rgb) == 4 and not strict:  # 4096 colors + alpha
        r = int(2*rgb[0], 16)
        g = int(2*rgb[1], 16)
        b = int(2*rgb[2], 16)
        a = int(2*rgb[3], 16)
        return r, g, b, a
    elif len(rgb) == 3 and not strict:  # 4096 colors
        r = int(2*rgb[0], 16)
        g = int(2*rgb[1], 16)
        b = int(2*rgb[2], 16)
        return r, g, b, 255
    elif len(rgb) == 2 and not strict:  # 256 grays
        r = int(rgb, 16)
        g = int(rgb, 16)
        b = int(rgb, 16)
        return r, g, b, 255
    elif len(rgb) == 1 and not strict:  # 16 grays
        r = int(2*rgb, 16)
        g = int(2*rgb, 16)
        b = int(2*rgb, 16)
        return r, g, b, 255
    elif rgb == '' and not strict:  # transparent black
        return 0, 0, 0, 0

    raise ValueError("Invalid hex color (strict mode is on)."
                     if strict else
                     "Invalid hex color")


def hex_to_rgb(
    hex_color: str, /,
    *, strict: bool = False,
) -> tuple[float, float, float]:
    return hex_to_rgba(hex_color, strict=strict)[:3]


# shared code between hsv and hsl conversions
def _rgb_to_hue_cmax_cmin_delta(
    r: float, g: float, b: float, /
) -> tuple[float, float, float, float]:
    rp, gp, bp = r/255, g/255, b/255
    cmax, cmin = max(rp, gp, bp), min(rp, gp, bp)
    delta = cmax - cmin
    h = 60 * (
        0 if delta == 0 else
        ((gp - bp)/delta) % 6 if cmax == rp else
        ((bp - rp)/delta) + 2 if cmax == gp else
        ((rp - gp)/delta) + 4 if cmax == bp else
        float('NaN')
    )
    assert h == h  # not NaN
    return h, cmax, cmin, delta


# shared code between hsv and hsl conversions
def _cxmh_to_rgb(
    c: float, x: float, m: float, h: float, /,
) -> tuple[float, float, float]:
    rp, gp, bp = (
        (c, x, 0) if 0 <= h < 60 else
        (x, c, 0) if 60 <= h < 120 else
        (0, c, x) if 120 <= h < 180 else
        (0, x, c) if 180 <= h < 240 else
        (x, 0, c) if 240 <= h < 300 else
        (c, 0, x) if 300 <= h < 360 else
        (float('NaN'), float('NaN'), float('NaN'))
    )
    assert rp == rp and gp == gp and bp == bp  # not NaN
    return 255*(rp + m), 255*(gp + m), 255*(bp + m)


def rgb_to_hsl(r: float, g: float, b: float, /) -> tuple[float, float, float]:
    h, cmax, cmin, delta = _rgb_to_hue_cmax_cmin_delta(r, g, b)
    l = 0.5*(cmax + cmin)
    s = 0 if delta == 0 else delta/(1 - abs(2*l - 1))
    return h, 100*s, 100*l


def rgb_to_hsv(r: float, g: float, b: float, /) -> tuple[float, float, float]:
    h, cmax, cmin, delta = _rgb_to_hue_cmax_cmin_delta(r, g, b)
    s = 0 if cmax == 0 else delta/cmax
    v = cmax
    return h, 100*s, 100*v


def hsl_to_rgb(h: float, s: float, l: float, /) -> tuple[float, float, float]:
    h, s, l = h % 360, s / 100, l / 100
    c = (1 - abs(2*l - 1))*s
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = l - 0.5*c
    return _cxmh_to_rgb(c, x, m, h)


def hsv_to_rgb(h: float, s: float, v: float, /) -> tuple[float, float, float]:
    h, s, v = h % 360, s / 100, v / 100
    c = v*s
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = v - c
    return _cxmh_to_rgb(c, x, m, h)


@dataclass(frozen=True)
class ColorRGB(Color):
    r: float
    g: float
    b: float

    @classmethod
    def mix(cls, /, *args: Self | float) -> Self:
        colors = [v for v in args if isinstance(v, cls)]
        if not colors:
            raise ValueError("At least 1 color argument is required.")
        weights = cast(list[float],
                       [v for v in args if not isinstance(v, cls)])
        weights = list(chain(weights, repeat(1, len(colors)-len(weights))))
        r = sum(w*c.r for c, w in zip(colors, weights))/sum(w for w in weights)
        g = sum(w*c.g for c, w in zip(colors, weights))/sum(w for w in weights)
        b = sum(w*c.b for c, w in zip(colors, weights))/sum(w for w in weights)

        return cls(r, g, b)

    @classmethod
    def from_rgb(cls, r: float, g: float, b: float, /) -> Self:
        return cls(r, g, b)

    def to_rgb(self, /) -> tuple[float, float, float]:
        return self.r, self.g, self.b

    @classmethod
    def from_hex(cls, hex_color: str, /, *, strict: bool = False) -> Self:
        return cls.from_rgb(*hex_to_rgb(hex_color, strict=strict))

    def __add__(self, other: Self, /) -> Self:
        return self.__class__(self.r + other.r,
                              self.g + other.g,
                              self.b + other.b)

    def __sub__(self, other: Self, /) -> Self:
        return self.__class__(self.r - other.r,
                              self.g - other.g,
                              self.b - other.b)

    def __mul__(self, v: float, /) -> Self:
        return self.__class__(self.r*v, self.g*v, self.b*v)

    def __rmul__(self, v: float, /) -> Self:
        return self.__class__(self.r*v, self.g*v, self.b*v)

    def __truediv__(self, v: float, /) -> Self:
        return self.__class__(self.r/v, self.g/v, self.b/v)

    def clamp(self, /) -> Self:
        r = 0 if self.r < 0 else 255 if self.r > 255 else self.r
        g = 0 if self.g < 0 else 255 if self.g > 255 else self.g
        b = 0 if self.b < 0 else 255 if self.b > 255 else self.b
        return self.__class__(r, g, b)


@dataclass(frozen=True)
class _ColorHSLuv_or_ColorHPLuv(Color, metaclass=ABCMeta):
    _h: float
    _sp: float
    _l: float

    @staticmethod
    @abstractmethod
    def _from_rgb_function(
        floats: Sequence[float], /,
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _to_rgb_function(
        floats: Sequence[float], /,
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _to_hex_function(
        floats: Sequence[float], /,
    ) -> str:
        raise NotImplementedError

    @classmethod
    def from_rgb(cls, r: float, g: float, b: float, /) -> Self:
        return cls(*cls._from_rgb_function((r/255, g/255, b/255)))

    def to_rgb(self, /) -> tuple[float, float, float]:
        r, g, b = self._to_rgb_function((self._h, self._sp, self._l))
        return 255*r, 255*g, 255*b

    @classmethod
    def mix(cls, /, *args: Self | float) -> Self:
        colors = [v for v in args if isinstance(v, cls)]
        if not colors:
            raise ValueError("At least 1 color argument is required.")
        weights = cast(
            list[float], [v for v in args if not isinstance(v, cls)]
        )
        weights = list(chain(weights, repeat(1, len(colors)-len(weights))))
        h = (sum(w*c._h for c, w in zip(colors, weights)) /
             sum(w for w in weights))
        sp = (sum(w*c._sp for c, w in zip(colors, weights)) /
              sum(w for w in weights))
        l = (sum(w*c._l for c, w in zip(colors, weights)) /
             sum(w for w in weights))

        return cls(h, sp, l)

    def to_hex(self, /) -> str:
        return self._to_hex_function((self._h, self._sp, self._l))

    @classmethod
    def from_hex(cls, /, hex_color: str, *, strict: bool = False) -> Self:
        r, g, b = hex_to_rgb(hex_color, strict=strict)
        return cls(*cls._from_rgb_function((r/255, g/255, b/255)))

    def clamp(self, /, *, mod_hue: bool = True) -> Self:
        h = self._h % 360 if mod_hue else self._h
        sp = 0 if self._sp < 0 else 100 if self._sp > 100 else self._sp
        l = 0 if self._l < 0 else 100 if self._l > 100 else self._l
        return self.__class__(h, sp, l)


class ColorHSLuv(_ColorHSLuv_or_ColorHPLuv):
    _from_rgb_function = cast(Callable[[Sequence[float]],
                                       tuple[float, float, float]],
                              staticmethod(hsluv.rgb_to_hsluv))

    _to_rgb_function = cast(Callable[[Sequence[float]],
                                     tuple[float, float, float]],
                            staticmethod(hsluv.hsluv_to_rgb))

    _to_hex_function = cast(Callable[[Sequence[float]], str],
                            staticmethod(hsluv.hsluv_to_hex))

    @property
    def h(self) -> float: return self._h
    @property
    def s(self) -> float: return self._sp
    @property
    def l(self) -> float: return self._l


class ColorHPLuv(_ColorHSLuv_or_ColorHPLuv):
    _from_rgb_function = cast(Callable[[Sequence[float]],
                                       tuple[float, float, float]],
                              staticmethod(hsluv.rgb_to_hpluv))

    _to_rgb_function = cast(Callable[[Sequence[float]],
                                     tuple[float, float, float]],
                            staticmethod(hsluv.hpluv_to_rgb))

    _to_hex_function = cast(Callable[[Sequence[float]], str],
                            staticmethod(hsluv.hpluv_to_hex))

    @property
    def h(self) -> float: return self._h
    @property
    def p(self) -> float: return self._sp
    @property
    def l(self) -> float: return self._l


@dataclass(frozen=True)
class _ColorHSL_or_ColorHSV(Color, metaclass=ABCMeta):
    _h: float
    _s: float
    _lv: float

    @staticmethod
    @abstractmethod
    def _from_rgb_function(
        _: float, __: float, ___: float, /,
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _to_rgb_function(
        _: float, __: float, ___: float, /,
    ) -> tuple[float, float, float]:
        raise NotImplementedError

    @classmethod
    def from_rgb(cls, /, r: float, g: float, b: float) -> Self:
        return cls(*cls._from_rgb_function(r, g, b))

    def to_rgb(self, /) -> tuple[float, float, float]:
        return self._to_rgb_function(self._h, self._s, self._lv)

    @classmethod
    def mix(cls, /, *args: Self | float) -> Self:
        colors = [v for v in args if isinstance(v, cls)]
        if not colors:
            raise ValueError("At least 1 color argument is required.")
        weights = cast(
            list[float], [v for v in args if not isinstance(v, cls)]
        )
        weights = list(chain(weights, repeat(1, len(colors)-len(weights))))
        h = (sum(w*c._h for c, w in zip(colors, weights)) /
             sum(w for w in weights))
        s = (sum(w*c._s for c, w in zip(colors, weights)) /
             sum(w for w in weights))
        lv = (sum(w*c._lv for c, w in zip(colors, weights)) /
              sum(w for w in weights))
        return cls(h, s, lv)

    @classmethod
    def from_hex(cls, /, hex_color: str, *, strict: bool = False) -> Self:
        return cls.from_rgb(*hex_to_rgb(hex_color, strict=strict))

    def clamp(self, /, *, mod_hue: bool = True) -> Self:
        h = self._h % 360 if mod_hue else self._h
        s = 0 if self._s < 0 else 100 if self._s > 100 else self._s
        lv = 0 if self._lv < 0 else 100 if self._lv > 100 else self._lv
        return self.__class__(h, s, lv)


class ColorHSL(_ColorHSL_or_ColorHSV):
    _from_rgb_function = staticmethod(rgb_to_hsl)
    _to_rgb_function = staticmethod(hsl_to_rgb)

    @property
    def h(self) -> float: return self._h
    @property
    def s(self) -> float: return self._s
    @property
    def l(self) -> float: return self._lv


class ColorHSV(_ColorHSL_or_ColorHSV):
    _from_rgb_function = staticmethod(rgb_to_hsv)
    _to_rgb_function = staticmethod(hsv_to_rgb)

    @property
    def h(self) -> float: return self._h
    @property
    def s(self) -> float: return self._s
    @property
    def v(self) -> float: return self._lv
