"""ShapeLab: geometry processing from first principles.

Modules
-------
io          read/write OFF meshes, outlines, landmark sets, skinning weights
primitives  procedural test meshes (icosphere, open cylinder)
fourier     Fourier descriptors of closed 2D curves
procrustes  generalised Procrustes analysis and PCA shape models
curvature   fundamental forms and curvatures of parametric patches and meshes
spectral    cotangent Laplacian, mass matrices, eigenbasis, GPS segmentation
skinning    skinning weights, rotations/quaternions, LBS, rotation blending, DQS
"""

from . import curvature, fourier, io, primitives, procrustes, skinning, spectral

__all__ = ["curvature", "fourier", "io", "primitives", "procrustes", "skinning", "spectral"]
__version__ = "0.1.0"
