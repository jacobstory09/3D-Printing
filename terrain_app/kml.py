"""Parse KML boundaries into WGS84 polygons."""

from __future__ import annotations

import re
from typing import List
from xml.etree import ElementTree as ET

from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union


def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_coord_triplets(text: str) -> List[tuple[float, float]]:
    pts: List[tuple[float, float]] = []
    tokens = [t for t in re.split(r"[\s]+", text.strip()) if t]
    for tok in tokens:
        nums = [float(x) for x in tok.split(",") if x != ""]
        if len(nums) >= 2:
            pts.append((nums[0], nums[1]))
    return pts


def _polygons_from_element(elem: ET.Element) -> List[Polygon]:
    polys: List[Polygon] = []
    for el in elem.iter():
        if _strip_namespace(el.tag) != "Polygon":
            continue
        outer = None
        for child in el:
            if _strip_namespace(child.tag) != "outerBoundaryIs":
                continue
            for ring in child.iter():
                if _strip_namespace(ring.tag) != "coordinates":
                    continue
                if ring.text:
                    outer = _parse_coord_triplets(ring.text)
                    break
        if outer and len(outer) >= 3:
            if outer[0] != outer[-1]:
                outer = outer + [outer[0]]
            polys.append(Polygon(outer))
    return polys


def parse_kml_bytes(data: bytes) -> Polygon:
    root = ET.fromstring(data)
    polys: List[Polygon] = []
    for el in root.iter():
        if _strip_namespace(el.tag) == "Placemark":
            polys.extend(_polygons_from_element(el))
    if not polys:
        polys = _polygons_from_element(root)
    if not polys:
        raise ValueError("No Polygon found in KML")
    geom = unary_union(polys)
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    if not isinstance(geom, Polygon) or geom.is_empty:
        raise ValueError("Could not build a single project polygon")
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def polygon_geojson(poly: Polygon) -> dict:
    return mapping(poly)
