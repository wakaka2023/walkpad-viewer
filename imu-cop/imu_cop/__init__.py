"""Standalone 50 Hz foot-IMU CoP model and right-foot load experiment."""
from .predict import RightFootCoP
from .settings import CONTEXT, RATE

__all__ = ['CONTEXT', 'RATE', 'RightFootCoP']
