"""Shared Overture Maps data functions for marimo notebooks."""

import warnings

import numpy as np
import pyarrow as pa
from geoarrow.rust.core import get_type_id, to_wkb
from geoarrow.rust.io import GeoParquetDataset
from lonboard import PathLayer, PolygonLayer, ScatterplotLayer
from obstore.store import S3Store

warnings.filterwarnings("ignore", message="No CRS exists on data")

OVERTURE_S3 = "s3://overturemaps-us-west-2/release/2026-01-21.0/"


def get_store():
    """Create an anonymous S3Store for Overture Maps."""
    return S3Store.from_url(OVERTURE_S3, region="us-west-2", skip_signature=True)


def load_geoarrow(store, path, bbox):
    """Load GeoParquet data for a path and bbox, return raw GeoArrow data.

    Preserves extension metadata needed by geoarrow-rust-core (get_type_id, etc.).
    Convert to PyArrow table with: pa.RecordBatchReader.from_stream(data).read_all()
    """
    objects = store.list_with_delimiter(path)["objects"]
    dataset = GeoParquetDataset.open(objects, store=store)
    return dataset.read(bbox=bbox)


def load_data(store, path, bbox):
    """Load GeoParquet data for a path and bbox, return a PyArrow table."""
    return pa.RecordBatchReader.from_stream(load_geoarrow(store, path, bbox)).read_all()


def filter_by_class(table, class_value):
    """Filter a PyArrow table by class column value."""
    classes = table.column("class").to_pylist()
    mask = [c == class_value for c in classes]
    return table.filter(mask)


def filter_to_lines(table):
    """Filter a table to only line geometries (LineString, MultiLineString + Z/M variants)."""
    type_ids = []
    for chunk in get_type_id(table.column("geometry")):
        type_ids.extend(chunk.to_pylist())
    idx = [i for i, t in enumerate(type_ids) if t in LINE_IDS]
    return table.take(idx)


def _get_voltage(tags):
    """Extract voltage from an Overture source_tags map entry."""
    if tags is None:
        return 0
    for key, val in tags:
        if key == "voltage":
            try:
                # Handle ranges like "115000;230000" — take max
                return max(int(v) for v in str(val).replace(";", ",").split(",") if v.strip().isdigit())
            except (ValueError, TypeError):
                return 0
    return 0


def load_power_lines(store, bbox, min_voltage=115000):
    """Load major power lines from Overture infrastructure.

    Filters to class=power_line, line geometries only, voltage >= min_voltage.
    Keeps native GeoArrow geometry so lonboard can consume directly.
    """
    data = load_geoarrow(store, "theme=base/type=infrastructure", bbox)

    # Extract type IDs while we have GeoArrow metadata
    type_ids = []
    for chunk in get_type_id(data.column("geometry")):
        type_ids.extend(chunk.to_pylist())

    # Filter to line geometries on the arro3 table (preserves GeoArrow geometry)
    line_idx = [i for i, t in enumerate(type_ids) if t in LINE_IDS]
    data = data.take(line_idx)

    # Need PyArrow for class/voltage filtering, but keep arro3 table for geometry
    pa_table = pa.RecordBatchReader.from_stream(data).read_all()

    # Filter to power_line class
    classes = pa_table.column("class").to_pylist()
    keep = [i for i, c in enumerate(classes) if c == "power_line"]
    data = data.take(keep)
    pa_table = pa_table.take(keep)

    # Filter by voltage from source_tags
    source_tags = pa_table.column("source_tags").to_pylist()
    voltages = [_get_voltage(tags) for tags in source_tags]
    keep = [i for i, v in enumerate(voltages) if v >= min_voltage]
    return data.take(keep)


# Geometry type ID sets (base + Z/M/ZM variants)
POLY_IDS = {3, 6, 13, 16, 23, 26, 33, 36}
POINT_IDS = {1, 4, 11, 14, 21, 24, 31, 34}
LINE_IDS = {2, 5, 12, 15, 22, 25, 32, 35}


def build_layers(data, cfg):
    """Build lonboard layers from a GeoArrow table + a LAYER_OPTIONS config dict.

    Splits mixed geometries by type, applies cmap or flat color, merges
    per-geometry-type kwargs from cfg["polygon"], cfg["point"], cfg["line"].

    Accepts raw arro3 GeoArrow data (preferred) or PyArrow tables.
    get_type_id must run on raw data BEFORE PyArrow conversion (metadata is lost).
    """
    # Extract type IDs from raw GeoArrow data before any conversion
    type_ids = []
    for chunk in get_type_id(data.column("geometry")):
        type_ids.extend(chunk.to_pylist())

    if isinstance(data, pa.RecordBatch):
        table = pa.Table.from_batches([data])
    elif not isinstance(data, pa.Table):
        table = pa.RecordBatchReader.from_stream(data).read_all()
    else:
        table = data

    # Per-row colors from cmap, or flat fallback
    if "cmap" in cfg:
        cmap = cfg["cmap"]
        classes = table.column(cmap["column"]).to_pylist()
        all_colors = [cmap["colors"].get(c, cmap["default"]) + [cmap["alpha"]] for c in classes]
    else:
        all_colors = None
    flat = cfg.get("fill_color", [70, 130, 180, 160])

    layers = []
    for type_set, key, LayerCls, color_prop in [
        (POLY_IDS, "polygon", PolygonLayer, "get_fill_color"),
        (POINT_IDS, "point", ScatterplotLayer, "get_fill_color"),
        (LINE_IDS, "line", PathLayer, "get_color"),
    ]:
        idx = [i for i, t in enumerate(type_ids) if t in type_set]
        if not idx:
            continue
        subset = table.take(idx)
        colors = np.array([all_colors[i] for i in idx], dtype=np.uint8) if all_colors else flat
        opts = cfg.get(key, {})
        layers.append(LayerCls(subset, **{color_prop: colors}, auto_highlight=True, **opts))
    return layers
