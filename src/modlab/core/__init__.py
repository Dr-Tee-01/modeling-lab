"""Shared numerics used by every pillar.

ode         fixed-step RK4 for systems of ODEs
montecarlo  seeded random generators, random-walk paths, drawdowns
pca         PCA by SVD with scores in standard deviations
fitting     least squares wrappers that report what they did
"""
from . import fitting, montecarlo, ode, pca

__all__ = ["fitting", "montecarlo", "ode", "pca"]
