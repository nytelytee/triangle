# pyright: strict
from __future__ import annotations

from basic_data import ColorData
from colors import ColorRGB
from identifier import Identifier
from util import SideType


class ExampleColorDataWeights(ColorData[ColorRGB]):
    base_above_weight = 4
    base_circle_weight = 1
    horizontal_touching_vertical_weight = 1
    horizontal_touching_horizontal_weight = 4
    horizontal_circle_weight = 1
    vertical_touching_vertical_weight = 4
    vertical_touching_horizontal_weight = 1
    vertical_circle_weight = 1


purple = ColorRGB.from_hex('#270627')
green = ColorRGB.from_hex('#e8f9e8')
white = ColorRGB.from_hex('#ffffff')


class PFP(ExampleColorDataWeights):
    circle_color = white
    predefined_colors = {
        (None, SideType.ZERO, Identifier()): purple,
    }


class PFP2(ExampleColorDataWeights):
    circle_color = white
    predefined_colors = {
        (None, SideType.ZERO, None): green,
        (None, SideType.BASE, None): green,
        (None, None, Identifier((1, 1))): green,
    }


class Black(ExampleColorDataWeights):
    circle_color = ColorRGB.from_hex('#FFFFFF')
    predefined_colors = {
        (None, SideType.ZERO, Identifier()): ColorRGB.from_hex('#000000'),
    }


class Red(ExampleColorDataWeights):
    circle_color = ColorRGB.from_hex('#FFFFFF')
    predefined_colors = {
        (None, SideType.ZERO, Identifier()): ColorRGB.from_hex('#FF0000'),
    }
