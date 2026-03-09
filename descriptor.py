from dataclasses import dataclass
from typing import Optional, List


@dataclass(slots=True)
class MCCCell:
    # Discrete indices in the cuboid grid
    i: int
    j: int
    k: int

    # Spatial center of this cell in image coordinates
    x: float
    y: float

    # Center of the directional bin represented by k
    angle_center: float

    # Final accumulated value stored in this cell
    contribution: float

    valid: bool = True

    # Optional breakdown for debugging
    spatial_contribution: Optional[float] = None
    directional_contribution: Optional[float] = None

@dataclass(slots=True)
class MCCCylinder:
    center_index: int
    center_x: float
    center_y: float
    center_theta: float

    radius: float
    angle_height: float
    ns: int
    nd: int
    delta_s: float
    delta_d: float
    sigma_s: float
    sigma_d: float

    cells: List[MCCCell]