"""Yaw-oriented 3D boxes in an explicitly named coordinate frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class YawBox:
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    yaw_rad: float
    frame: str = "top"

    def __post_init__(self) -> None:
        if not self.frame:
            raise ValueError("box frame must be non-empty")
        values = self.center_m + self.size_m + (self.yaw_rad,)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("box values must be finite")
        if any(value <= 0 for value in self.size_m):
            raise ValueError("box sizes must be positive")

    @property
    def volume_m3(self) -> float:
        return float(math.prod(self.size_m))

    @property
    def center_range_m(self) -> float:
        return float(np.linalg.norm(np.asarray(self.center_m, dtype=np.float64)))

    def local_coordinates(self, points: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        if not np.isfinite(array).all():
            raise ValueError("point coordinates must be finite")
        translated = array - np.asarray(self.center_m, dtype=np.float64)
        cosine = math.cos(self.yaw_rad)
        sine = math.sin(self.yaw_rad)
        # Row-vector equivalent of a -yaw rotation.
        local_x = cosine * translated[:, 0] + sine * translated[:, 1]
        local_y = -sine * translated[:, 0] + cosine * translated[:, 1]
        return np.column_stack((local_x, local_y, translated[:, 2]))

    def contains(
        self, points: ArrayLike, *, tolerance_m: float = 1e-6
    ) -> NDArray[np.bool_]:
        if not math.isfinite(tolerance_m) or tolerance_m < 0:
            raise ValueError("boundary tolerance must be finite and nonnegative")
        local = self.local_coordinates(points)
        half = np.asarray(self.size_m, dtype=np.float64) / 2.0 + tolerance_m
        return np.asarray(np.all(np.abs(local) <= half, axis=1), dtype=np.bool_)

    def count_points(self, points: ArrayLike, *, tolerance_m: float = 1e-6) -> int:
        return int(np.count_nonzero(self.contains(points, tolerance_m=tolerance_m)))
