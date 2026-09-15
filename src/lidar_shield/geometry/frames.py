"""Rigid SE(3) transforms with explicit source and target frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from lidar_shield.contracts.reasons import ReasonCode
from lidar_shield.contracts.schemas import Extrinsics
from lidar_shield.geometry.poses import Pose


@dataclass(frozen=True)
class TransformResolution:
    transform: RigidTransform | None
    quality: str
    reason_code: ReasonCode


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

    @classmethod
    def from_matrix(
        cls, source_frame: str, target_frame: str, matrix_4x4: ArrayLike
    ) -> RigidTransform:
        """Construct a checked rigid transform from a homogeneous matrix."""

        matrix = np.asarray(matrix_4x4, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("transform matrix must be finite with shape (4, 4)")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
            raise ValueError("transform matrix must have homogeneous final row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            raise ValueError("transform rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
            raise ValueError("transform rotation must be proper")
        quaternion = Rotation.from_matrix(rotation).as_quat()
        return cls(
            source_frame,
            target_frame,
            Pose(
                (float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])),
                (
                    float(quaternion[0]),
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                ),
            ),
        )

    def as_matrix(self) -> NDArray[np.float64]:
        """Return the homogeneous matrix implementing this transform."""

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(self.pose.quaternion_xyzw).as_matrix()
        matrix[:3, 3] = self.pose.translation_m
        return matrix


def resolve_extrinsics(extrinsics: Extrinsics) -> TransformResolution:
    """Resolve only provenance-verified matrices; unresolved means no transform."""

    if extrinsics.status == "unresolved":
        return TransformResolution(None, "unresolved", ReasonCode.UNRESOLVED_TRANSFORM)
    assert extrinsics.matrix_4x4 is not None
    return TransformResolution(
        RigidTransform.from_matrix(
            extrinsics.from_frame, extrinsics.to_frame, extrinsics.matrix_4x4
        ),
        "verified",
        ReasonCode.MATCHED,
    )
