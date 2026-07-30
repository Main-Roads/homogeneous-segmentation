from ._seg_mcv import (
	segment_ids_to_minimize_coefficient_of_variation as segment_ids_to_minimize_coefficient_of_variation,
)
from ._seg_shs import (
	segment_ids_to_maximize_spatial_heterogeneity as segment_ids_to_maximize_spatial_heterogeneity,
)

__all__ = [
	"segment_ids_to_maximize_spatial_heterogeneity",
	"segment_ids_to_minimize_coefficient_of_variation",
]
