"""Built-in graphene tight-binding model and its Brillouin-zone tools.

Re-exports the model API so that ``from ..graphene import GrapheneModel`` (and
the package-level ``floquet_toolkit.builtin_models`` imports) keep resolving
after ``graphene.py`` was moved into this subpackage. The adaptive
band-structure integration grid lives alongside it in
:mod:`graphene_bz_grid`.
"""

from .bz_geometry import bz_corners, conduction_band, dirac_points, reciprocal_vectors
from .graphene import GrapheneModel, GrapheneParameters, driven_graphene_model
from .graphene_bz_grid import build_graphene_bz_grid, graphene_bz_quadrature
from .grid_presets import (
    dirac_refined_grid_kwargs,
    dirac_refined_pocket_levels,
    local_dirac_polar_cut_grid,
)

__all__ = [
    "GrapheneModel",
    "GrapheneParameters",
    "driven_graphene_model",
    "build_graphene_bz_grid",
    "dirac_refined_grid_kwargs",
    "dirac_refined_pocket_levels",
    "local_dirac_polar_cut_grid",
    "graphene_bz_quadrature",
    "bz_corners",
    "conduction_band",
    "dirac_points",
    "reciprocal_vectors",
]
