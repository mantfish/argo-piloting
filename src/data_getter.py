from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator, Literal

import yaml

import copernicusmarine
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial / temporal primitives
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    min_depth: float = 0.0
    max_depth: float = 0.0

    def __post_init__(self):
        if self.min_lat > self.max_lat:
            raise ValueError(f"min_lat ({self.min_lat}) must be ≤ max_lat ({self.max_lat})")
        if self.min_lon > self.max_lon:
            raise ValueError(f"min_lon ({self.min_lon}) must be ≤ max_lon ({self.max_lon})")
        if self.min_depth > self.max_depth:
            raise ValueError(f"min_depth ({self.min_depth}) must be ≤ max_depth ({self.max_depth})")

    def split(self, lat_tiles: int = 1, lon_tiles: int = 1) -> list[BoundingBox]:
        """Divide into a (lat_tiles × lon_tiles) grid of sub-boxes."""
        lat_edges = _linspace(self.min_lat, self.max_lat, lat_tiles)
        lon_edges = _linspace(self.min_lon, self.max_lon, lon_tiles)
        return [
            BoundingBox(
                min_lat=lat_edges[r],   max_lat=lat_edges[r + 1],
                min_lon=lon_edges[c],   max_lon=lon_edges[c + 1],
                min_depth=self.min_depth, max_depth=self.max_depth,
            )
            for r in range(lat_tiles)
            for c in range(lon_tiles)
        ]


@dataclass
class SubsetRequest:
    dataset_id: str
    variables: list[str]
    bbox: BoundingBox
    start_datetime: str          # ISO-8601, e.g. "2024-01-01T00:00:00"
    end_datetime: str
    output_dir: Path = field(default_factory=Path.cwd)
    output_filename: str | None = None


# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------

TemporalUnit = Literal["yearly", "monthly", "weekly", "daily"]


@dataclass
class ChunkStrategy:
    """Describe how to split a :class:`SubsetRequest` into smaller pieces.

    Parameters
    ----------
    temporal:
        Split the time window into chunks of this size.
        Pass a :data:`TemporalUnit` string or a plain :class:`int` for days.
    lat_tiles, lon_tiles:
        Split the bounding box into this many tiles along each axis.
        Defaults to 1 (no spatial splitting).
    """
    temporal: TemporalUnit | int | None = None
    lat_tiles: int = 1
    lon_tiles: int = 1

    @property
    def splits_spatially(self) -> bool:
        return self.lat_tiles > 1 or self.lon_tiles > 1

    @property
    def splits_temporally(self) -> bool:
        return self.temporal is not None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CopernicusDataGetter:
    """Thin, reusable wrapper around the Copernicus Marine toolbox."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._login(username, password)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subset(self, request: SubsetRequest) -> Path:
        """Download a single spatial/temporal subset and return the output path."""
        request.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Subsetting '%s' | %s – %s",
            request.dataset_id,
            request.start_datetime,
            request.end_datetime,
        )

        response = copernicusmarine.subset(
            dataset_id=request.dataset_id,
            variables=request.variables,
            minimum_longitude=request.bbox.min_lon,
            maximum_longitude=request.bbox.max_lon,
            minimum_latitude=request.bbox.min_lat,
            maximum_latitude=request.bbox.max_lat,
            start_datetime=request.start_datetime,
            end_datetime=request.end_datetime,
            minimum_depth=request.bbox.min_depth,
            maximum_depth=request.bbox.max_depth,
            output_directory=str(request.output_dir),
            output_filename=request.output_filename,
        )

        self._update_manifest(response.file_path, request)
        return response.file_path

    def subset_and_open(self, request: SubsetRequest) -> xr.Dataset:
        """Download a subset and immediately open it as an xarray Dataset."""
        path = self.subset(request)
        logger.info("Opening %s", path)
        return xr.open_dataset(path)

    def subset_chunked(
        self,
        request: SubsetRequest,
        strategy: ChunkStrategy,
        *,
        merge: bool = True,
    ) -> list[Path] | xr.Dataset:
        """Download a request in chunks, then optionally merge into one Dataset.

        Parameters
        ----------
        request:
            The full request to be split.
        strategy:
            How to split it — temporally, spatially, or both.
        merge:
            If ``True`` (default), merge all downloaded files with
            :func:`xarray.open_mfdataset` and return a single Dataset.
            If ``False``, return the list of paths instead.
        """
        chunks = list(self._explode(request, strategy))
        total = len(chunks)
        logger.info("Downloading %d chunk(s) for dataset '%s'.", total, request.dataset_id)

        paths: list[Path] = []
        for i, chunk in enumerate(chunks, 1):
            logger.info("Chunk %d / %d | %s – %s", i, total, chunk.start_datetime, chunk.end_datetime)
            paths.append(self.subset(chunk))

        if not merge:
            return paths

        logger.info("Merging %d files with xarray.open_mfdataset …", len(paths))
        return xr.open_mfdataset(paths, combine="by_coords")

    # ------------------------------------------------------------------
    # Chunking internals
    # ------------------------------------------------------------------

    def _explode(
        self,
        request: SubsetRequest,
        strategy: ChunkStrategy,
    ) -> Generator[SubsetRequest, None, None]:
        """Yield one SubsetRequest per (time window × spatial tile) combination."""
        time_windows = (
            list(_split_time(request.start_datetime, request.end_datetime, strategy.temporal))
            if strategy.splits_temporally
            else [(request.start_datetime, request.end_datetime)]
        )

        spatial_tiles = (
            request.bbox.split(strategy.lat_tiles, strategy.lon_tiles)
            if strategy.splits_spatially
            else [request.bbox]
        )

        for t_idx, (t_start, t_end) in enumerate(time_windows):
            for s_idx, bbox in enumerate(spatial_tiles):
                filename = _chunk_filename(request, t_idx, s_idx, strategy)
                yield SubsetRequest(
                    dataset_id=request.dataset_id,
                    variables=request.variables,
                    bbox=bbox,
                    start_datetime=t_start,
                    end_datetime=t_end,
                    output_dir=request.output_dir,
                    output_filename=filename,
                )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    MANIFEST_FILENAME = "manifest.yaml"

    def _update_manifest(self, file_path: Path, request: SubsetRequest) -> None:
        """Append an entry for *file_path* to the manifest in its output directory."""
        manifest_path = request.output_dir / self.MANIFEST_FILENAME
        entries: list[dict] = []
        if manifest_path.exists():
            with manifest_path.open() as f:
                entries = yaml.safe_load(f) or []

        entry = {
            "file": file_path.name,
            "dataset_id": request.dataset_id,
            "variables": request.variables,
            "lat": [request.bbox.min_lat, request.bbox.max_lat],
            "lon": [request.bbox.min_lon, request.bbox.max_lon],
            "depth": [request.bbox.min_depth, request.bbox.max_depth],
            "time": [request.start_datetime, request.end_datetime],
        }

        # Replace existing entry for the same file, or append.
        for i, e in enumerate(entries):
            if e.get("file") == entry["file"]:
                entries[i] = entry
                break
        else:
            entries.append(entry)

        with manifest_path.open("w") as f:
            yaml.dump(entries, f, sort_keys=False, allow_unicode=True)

        logger.info("Manifest updated: %s", manifest_path)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _login(self, username: str | None, password: str | None) -> None:
        kwargs = {}
        username = username or os.environ.get("COPERNICUS_USERNAME")
        password = password or os.environ.get("COPERNICUS_PASSWORD")
        if username:
            kwargs["username"] = username
        if password:
            kwargs["password"] = password
        success = copernicusmarine.login(**kwargs, check_credentials_valid=True)
        if not success:
            raise RuntimeError(
                "Copernicus Marine login failed — check your credentials. "
                "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in a .env file."
            )
        logger.info("Copernicus Marine login successful.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _linspace(lo: float, hi: float, n: int) -> list[float]:
    """Return n+1 evenly-spaced edges between lo and hi."""
    step = (hi - lo) / n
    return [lo + i * step for i in range(n + 1)]


def _split_time(
    start: str,
    end: str,
    unit: TemporalUnit | int,
) -> Generator[tuple[str, str], None, None]:
    """Yield (start, end) ISO-8601 string pairs covering [start, end]."""
    fmt = "%Y-%m-%dT%H:%M:%S"
    t0 = datetime.fromisoformat(start)
    t_end = datetime.fromisoformat(end)

    current = t0
    while current < t_end:
        next_t = _advance(current, unit)
        # Clamp the last window so it never exceeds the requested end.
        window_end = min(next_t - timedelta(seconds=1), t_end)
        yield current.strftime(fmt), window_end.strftime(fmt)
        current = next_t


def _advance(dt: datetime, unit: TemporalUnit | int) -> datetime:
    if isinstance(unit, int):
        return dt + timedelta(days=unit)
    if unit == "daily":
        return dt + timedelta(days=1)
    if unit == "weekly":
        return dt + timedelta(weeks=1)
    if unit == "monthly":
        # Roll forward one calendar month.
        month = dt.month % 12 + 1
        year = dt.year + (1 if dt.month == 12 else 0)
        return dt.replace(year=year, month=month, day=1)
    if unit == "yearly":
        return dt.replace(year=dt.year + 1, month=1, day=1)
    raise ValueError(f"Unknown temporal unit: {unit!r}")


def _chunk_filename(
    request: SubsetRequest,
    t_idx: int,
    s_idx: int,
    strategy: ChunkStrategy,
) -> str:
    stem = Path(request.output_filename).stem if request.output_filename else request.dataset_id
    parts = [stem]
    if strategy.splits_temporally:
        parts.append(f"t{t_idx:03d}")
    if strategy.splits_spatially:
        parts.append(f"s{s_idx:03d}")
    return "_".join(parts) + ".nc"


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    getter = CopernicusDataGetter()

    bbox = BoundingBox(min_lat=54, max_lat=56, min_lon=12.0, max_lon=16.0,
                       min_depth=0.5016462206840515, max_depth=90.0)

    request = SubsetRequest(
        dataset_id="cmems_mod_bal_phy_anfc_PT1H-i",
        variables=["uo", "vo"],
        bbox=bbox,
        start_datetime="2025-01-01T00:00:00",
        end_datetime="2025-12-31T23:59:59",
        output_dir=Path("data/raw"),
        output_filename="baltic2025.nc",
    )

    # Do both at once — 4 years × 4 tiles = 48 downloads, merged at the end
    ds = getter.subset_chunked(
        request,
        ChunkStrategy(temporal="monthly", lat_tiles=4, lon_tiles=4),
        merge=False,
    )
