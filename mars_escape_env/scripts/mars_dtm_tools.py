# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class PdsDtmInfo:
    """Metadata needed to read a HiRISE PDS3 DTM raster."""

    path: Path
    product_id: str
    rationale: str
    record_bytes: int
    image_record: int
    lines: int
    samples: int
    map_scale_m_per_pixel: float
    maximum_latitude_deg: float
    minimum_latitude_deg: float
    westernmost_longitude_deg: float
    easternmost_longitude_deg: float
    valid_minimum_m: float
    valid_maximum_m: float

    @property
    def image_offset_bytes(self) -> int:
        """Byte offset where the image raster starts."""
        return (self.image_record - 1) * self.record_bytes


def read_pds_label(path: Path) -> str:
    """Read the fixed-length PDS3 label record from a HiRISE DTM `.IMG` file."""
    with path.open("rb") as infile:
        head = infile.read(65536)
    text = head.decode("ascii", errors="ignore")
    end_match = re.search(r"(?m)^END\s*$", text)
    if end_match is None:
        raise ValueError(f"Could not find END marker in PDS label: {path}")
    return text[: end_match.end()]


def _value_for(label: str, key: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", label)
    if match is None:
        raise KeyError(f"Missing PDS label key: {key}")
    value = match.group(1).strip()
    if "/*" in value:
        value = value.split("/*", 1)[0].strip()
    if "<" in value:
        value = value.split("<", 1)[0].strip()
    return value.strip().strip('"')


def _value_for_optional(label: str, key: str, default: str = "") -> str:
    try:
        return _value_for(label, key)
    except KeyError:
        return default


def _as_int(label: str, key: str) -> int:
    return int(_value_for(label, key))


def _as_float(label: str, key: str) -> float:
    return float(_value_for(label, key))


def read_pds_dtm_info(path: str | Path) -> PdsDtmInfo:
    """Read raster metadata from a HiRISE DTM `.IMG` file."""
    img_path = Path(path)
    label = read_pds_label(img_path)
    return PdsDtmInfo(
        path=img_path,
        product_id=_value_for(label, "PRODUCT_ID"),
        rationale=_value_for_optional(label, "RATIONALE_DESC"),
        record_bytes=_as_int(label, "RECORD_BYTES"),
        image_record=_as_int(label, "^IMAGE"),
        lines=_as_int(label, "LINES"),
        samples=_as_int(label, "LINE_SAMPLES"),
        map_scale_m_per_pixel=_as_float(label, "MAP_SCALE"),
        maximum_latitude_deg=_as_float(label, "MAXIMUM_LATITUDE"),
        minimum_latitude_deg=_as_float(label, "MINIMUM_LATITUDE"),
        westernmost_longitude_deg=_as_float(label, "WESTERNMOST_LONGITUDE"),
        easternmost_longitude_deg=_as_float(label, "EASTERNMOST_LONGITUDE"),
        valid_minimum_m=_as_float(label, "VALID_MINIMUM"),
        valid_maximum_m=_as_float(label, "VALID_MAXIMUM"),
    )


def open_dtm_memmap(info: PdsDtmInfo) -> np.memmap:
    """Open the DTM image as a read-only NumPy memmap."""
    return np.memmap(
        info.path,
        dtype="<f4",
        mode="r",
        offset=info.image_offset_bytes,
        shape=(info.lines, info.samples),
    )


def valid_height_mask(height: np.ndarray, info: PdsDtmInfo) -> np.ndarray:
    """Return a mask for finite DTM elevations inside the label-declared valid range [m]."""
    return (
        np.isfinite(height)
        & (height >= info.valid_minimum_m - 1.0)
        & (height <= info.valid_maximum_m + 1.0)
    )


def replace_invalid_with_median(height: np.ndarray, info: PdsDtmInfo) -> np.ndarray:
    """Replace invalid elevation samples with the valid median elevation [m]."""
    result = np.asarray(height, dtype=np.float32).copy()
    mask = valid_height_mask(result, info)
    if not np.any(mask):
        raise ValueError("No valid height samples found in crop.")
    median = float(np.median(result[mask]))
    result[~mask] = median
    return result


def pixel_to_lon_lat(info: PdsDtmInfo, sample: float, line: float) -> tuple[float, float]:
    """Convert zero-based sample/line coordinates to east longitude and latitude [deg]."""
    lon_span = info.easternmost_longitude_deg - info.westernmost_longitude_deg
    lat_span = info.maximum_latitude_deg - info.minimum_latitude_deg
    lon = info.westernmost_longitude_deg + lon_span * sample / float(info.samples - 1)
    lat = info.maximum_latitude_deg - lat_span * line / float(info.lines - 1)
    return lon, lat


def lon_lat_to_pixel(info: PdsDtmInfo, lon: float, lat: float) -> tuple[float, float]:
    """Convert east longitude and latitude [deg] to zero-based sample/line coordinates."""
    lon_span = info.easternmost_longitude_deg - info.westernmost_longitude_deg
    lat_span = info.maximum_latitude_deg - info.minimum_latitude_deg
    sample = (lon - info.westernmost_longitude_deg) / lon_span * float(info.samples - 1)
    line = (info.maximum_latitude_deg - lat) / lat_span * float(info.lines - 1)
    return sample, line


def clamp_crop_offsets(xoff: int, yoff: int, crop_px: int, info: PdsDtmInfo) -> tuple[int, int]:
    """Clamp crop offsets so a square crop remains inside the raster bounds."""
    xoff = max(0, min(xoff, info.samples - crop_px))
    yoff = max(0, min(yoff, info.lines - crop_px))
    return xoff, yoff


def crop_offsets_around_pixel(sample: float, line: float, crop_px: int, info: PdsDtmInfo) -> tuple[int, int]:
    """Compute clamped square crop offsets around a center sample/line."""
    xoff = int(round(sample - crop_px * 0.5))
    yoff = int(round(line - crop_px * 0.5))
    return clamp_crop_offsets(xoff, yoff, crop_px, info)


def candidate_crops(info: PdsDtmInfo, crop_px: int, region: str = "") -> dict[str, dict[str, object]]:
    """Return useful first-pass crop candidates for a HiRISE DTM."""
    candidates: dict[str, dict[str, object]] = {
        "image_center": {
            "description": "Geometric center of the DTM raster.",
            "center_sample": (info.samples - 1) * 0.5,
            "center_line": (info.lines - 1) * 0.5,
            "color": [255, 210, 60],
        },
    }
    if region == "gusev_spirit" or info.product_id == "DTEEC_001513_1655_001777_1650_U01":
        candidates.update({
        "center_columbia_hills": {
            "description": "HiRISE observation center near the Spirit rover / Columbia Hills target.",
            "center_lon_deg": 175.501,
            "center_lat_deg": -14.589,
            "color": [255, 80, 60],
        },
        "spirit_landing_site": {
            "description": "Estimated original Spirit landing site from MER map coordinates.",
            "center_lon_deg": 175.4729,
            "center_lat_deg": -14.5692,
            "color": [80, 180, 255],
        },
        })
    result: dict[str, dict[str, object]] = {}
    for name, spec in candidates.items():
        if "center_sample" in spec and "center_line" in spec:
            sample = float(spec["center_sample"])
            line = float(spec["center_line"])
            lon, lat = pixel_to_lon_lat(info, sample, line)
        else:
            lon = float(spec["center_lon_deg"])
            lat = float(spec["center_lat_deg"])
            sample, line = lon_lat_to_pixel(info, lon, lat)
        xoff, yoff = crop_offsets_around_pixel(sample, line, crop_px, info)
        west, north = pixel_to_lon_lat(info, xoff, yoff)
        east, south = pixel_to_lon_lat(info, xoff + crop_px - 1, yoff + crop_px - 1)
        result[name] = {
            "description": spec["description"],
            "xoff": xoff,
            "yoff": yoff,
            "crop_px": crop_px,
            "center_sample": sample,
            "center_line": line,
            "center_lon_deg": lon,
            "center_lat_deg": lat,
            "west_lon_deg": west,
            "east_lon_deg": east,
            "north_lat_deg": north,
            "south_lat_deg": south,
            "color": spec["color"],
        }
    return result


def normalize_height_to_rgb(height: np.ndarray, info: PdsDtmInfo) -> Image.Image:
    """Convert an elevation array [m] to a Mars-tinted preview image."""
    data = np.asarray(height, dtype=np.float32)
    mask = valid_height_mask(data, info)
    if not np.any(mask):
        mask = np.isfinite(data)
    if np.any(mask):
        lo, hi = np.percentile(data[mask], [1.0, 99.0])
        fill = float(np.median(data[mask]))
    else:
        lo, hi, fill = 0.0, 1.0, 0.0
    if math.isclose(float(hi), float(lo)):
        hi = lo + 1.0
    clean = np.where(mask, data, fill)
    norm = np.clip((clean - lo) / (hi - lo), 0.0, 1.0)
    rgb = np.empty((*norm.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (64.0 + norm * 180.0).astype(np.uint8)
    rgb[..., 1] = (48.0 + norm * 138.0).astype(np.uint8)
    rgb[..., 2] = (36.0 + norm * 88.0).astype(np.uint8)
    rgb[~mask] = np.array([0, 0, 0], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_full_preview(
    info: PdsDtmInfo,
    out_path: str | Path,
    candidates: dict[str, dict[str, object]],
    max_side_px: int = 1400,
) -> None:
    """Save a downsampled full-DTM preview with candidate crop boxes."""
    dtm = open_dtm_memmap(info)
    stride = max(1, int(math.ceil(max(info.samples, info.lines) / max_side_px)))
    sampled = np.asarray(dtm[::stride, ::stride], dtype=np.float32)
    image = normalize_height_to_rgb(sampled, info)
    draw = ImageDraw.Draw(image)
    for name, crop in candidates.items():
        color = tuple(int(channel) for channel in crop["color"])
        x0 = int(round(float(crop["xoff"]) / stride))
        y0 = int(round(float(crop["yoff"]) / stride))
        x1 = int(round((float(crop["xoff"]) + float(crop["crop_px"])) / stride))
        y1 = int(round((float(crop["yoff"]) + float(crop["crop_px"])) / stride))
        for inset in range(3):
            draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=color)
        draw.text((x0 + 5, y0 + 5), name, fill=color)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def save_crop_preview(height: np.ndarray, info: PdsDtmInfo, out_path: str | Path) -> None:
    """Save a preview image for a cropped height field [m]."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    normalize_height_to_rgb(height, info).save(out_path)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    """Write a JSON file with stable formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def info_as_json_dict(info: PdsDtmInfo) -> dict[str, object]:
    """Convert :class:`PdsDtmInfo` to JSON-safe values."""
    payload = asdict(info)
    payload["path"] = str(info.path)
    payload["image_offset_bytes"] = info.image_offset_bytes
    return payload
