from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import copernicusmarine
import xarray as xr

logger = logging.getLogger(__name__)


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


@dataclass
class SubsetRequest:
    dataset_id: str
    variables: list[str]
    bbox: BoundingBox
    start_datetime: str          # ISO-8601, e.g. "2024-01-01T00:00:00"
    end_datetime: str
    output_dir: Path = field(default_factory=Path.cwd)
    output_filename: str | None = None


class CopernicusDataGetter:
    """Thin, reusable wrapper around the Copernicus Marine toolbox."""

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        """Log in once at construction time.

        Credentials are optional — if omitted, the toolbox falls back to
        the ~/.copernicusmarine/credentials file or environment variables.
        """
        self._login(username, password)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subset(self, request: SubsetRequest) -> Path:
        """Download a spatial/temporal subset and return the output path.

        Parameters
        ----------
        request:
            A :class:`SubsetRequest` describing what to fetch.

        Returns
        -------
        Path
            Path to the downloaded NetCDF file.
        """
        request.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Subsetting dataset '%s' | %s–%s",
            request.dataset_id,
            request.start_datetime,
            request.end_datetime,
        )

        output_path = copernicusmarine.subset(
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

        return Path(output_path)

    def subset_and_open(self, request: SubsetRequest) -> xr.Dataset:
        """Download a subset and immediately open it as an xarray Dataset."""
        path = self.subset(request)
        logger.info("Opening %s", path)
        return xr.open_dataset(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _login(self, username: str | None, password: str | None) -> None:
        kwargs = {}
        if username:
            kwargs["username"] = username
        if password:
            kwargs["password"] = password

        try:
            copernicusmarine.login(**kwargs)
            logger.info("Copernicus Marine login successful.")
        except Exception as exc:
            raise RuntimeError("Copernicus Marine login failed.") from exc

if __name__ == "__main__":
