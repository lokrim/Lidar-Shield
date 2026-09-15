"""Rigid SE(3) transforms with explicit source and target frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from lidar_shield.geometry.poses import Pose


@dataclass(frozen=True)
class RigidTransform:
    """Transform coordinates from ``source_frame`` into ``target_frame``."""

    source_frame: str
    target_frame: str
    pose: Pose

    def __post_init__(self) -> None:
        if not self.source_frame or not self.target_frame:
            raise ValueError("frame names must be non-empty")

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(points, dtype=np.float64)
        if array.shape == (3,):
            pass
        elif array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("points must have shape (3,) or (n, 3)")
        if not np.isfinite(array).all():
            raise ValueError("point coordinates must be finite")
        rotated = Rotation.from_quat(self.pose.quaternion_xyzw).apply(array)
        return np.asarray(
            rotated + np.asarray(self.pose.translation_m), dtype=np.float64
        )

    def inverse(self) -> RigidTransform:
        rotation = Rotation.from_quat(self.pose.quaternion_xyzw)
        inverse_rotation = rotation.inv()
        inverse_translation = inverse_rotation.apply(
            -np.asarray(self.pose.translation_m)
        )
        inverse_quaternion = inverse_rotation.as_quat()
        return RigidTransform(
            self.target_frame,
            self.source_frame,
            Pose(
                (
                    float(inverse_translation[0]),
                    float(inverse_translation[1]),
                    float(inverse_translation[2]),
                ),
                (
                    float(inverse_quaternion[0]),
                    float(inverse_quaternion[1]),
                    float(inverse_quaternion[2]),
                    float(inverse_quaternion[3]),
                ),
            ),
        )

    def then(self, following: RigidTransform) -> RigidTransform:
        """Compose this transform followed by ``following``."""

        if self.target_frame != following.source_frame:
            raise ValueError(
                f"cannot compose {self.target_frame!r} with {following.source_frame!r}"
            )
        first_rotation = Rotation.from_quat(self.pose.quaternion_xyzw)
        next_rotation = Rotation.from_quat(following.pose.quaternion_xyzw)
        rotation = next_rotation * first_rotation
        translation = following.apply(np.asarray(self.pose.translation_m))
        quaternion = rotation.as_quat()
        return RigidTransform(
            self.source_frame,
            following.target_frame,
            Pose(
                (float(translation[0]), float(translation[1]), float(translation[2])),
                (
                    float(quaternion[0]),
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                ),
            ),
        )
