"""Thin temporal baselines for the M3 integration skeleton."""

from lidar_shield.temporal.ewma import EWMAConfig, EWMAUpdate, update_ewma

__all__ = ["EWMAConfig", "EWMAUpdate", "update_ewma"]
