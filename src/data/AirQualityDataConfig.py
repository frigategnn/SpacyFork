from dataclasses import dataclass
from typing import Tuple


@dataclass
class AirQualityDataConfig:
    # data properties
    model: str = "air_quality"
    time_length: int = 100

    # grid properties
    nx: int = 20
    ny: int = 20
    node_dist: int = 1
    node_extent_low: int = 1
    node_extent_high: int = 1
    random_node: bool = False
    num_variates: int = 1
    disjoint_nodes: bool = False
    map_type: str = "custom"
    grid_noise: str = "none"
    grid_noise_scale: float = 0.0

    # graph properties
    functional_relationships: str = "custom"
    inst_graph_type: str = "custom"
    lag_graph_type: str = "custom"
    lag: int = 1
    num_nodes: int = 400
    base_noise_type: str = "custom"
    hist_dep_noise_type: str = "custom"
    noise_scale: float = 0.0
    hist_dep_noise: bool = False
    disable_inst: bool = False

    # optional
    seed: int = 0

    # Geographical bounds
    lat_bounds: Tuple[float, float] = (0.0, 90.0)
    long_bounds: Tuple[float, float] = (0.0, 180.0)

    def __post_init__(self):
        self.num_nodes = self.nx * self.ny
