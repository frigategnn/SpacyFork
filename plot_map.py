import enum
from pathlib import Path
import torch
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import contextily as ctx
import networkx as nx
from matplotlib.lines import Line2D


def get_variate_map(alpha):
    return alpha.argmax(0).numpy()


def find_centers(center, alpha):
    centers = center[alpha.to(bool)].squeeze()
    return centers.cpu().numpy()


def get_graph(graph, *, disease_idx_start, instantaneous=False, timelagged_idx=0):
    def get_edge_list(graph: torch.Tensor):
        row, col = torch.where(graph.to(bool))
        edge_list = [(r, c) for r, c in zip(row.numpy(), col.numpy())]
        return edge_list

    def remove_self_loops(edges):
        return list(filter(lambda pair: pair[0] != pair[1], edges))

    def remove_effect_to_cause_edges(edges):
        def valid_pair(pair):
            src, dst = pair
            if src < disease_idx_start:
                return True
            if dst >= disease_idx_start:
                # this is essentially equivalent to src >= disease_idx_start and dst >= disease_idx_start
                return True
            # this case is when src >= disease_idx_start and dst < disease_idx_start (that is an edge from a disease to a pollutant = invalid)
            return False

        return list(filter(lambda pair: valid_pair(pair), edges))

    if instantaneous:
        return remove_effect_to_cause_edges(remove_self_loops(get_edge_list(graph[0])))
    else:
        return remove_effect_to_cause_edges(
            remove_self_loops(get_edge_list(graph[timelagged_idx - 1]))
        )  # offset for the first instantaneous instance


def main():
    logs_base = Path(
        "/mnt/database/mridul/air_quality_and_health.project/SPACY/logs/numnodes_8_size_30_numvariates_4/seed_0_12_05_2026_02_23_54"
    )
    data = {}
    variate_names = ["PM10", "PM2_5_DRY", "COPD", "TB"]
    for p in logs_base.iterdir():
        if p.suffix == ".pt":
            print(f"Loading {p.stem}.pt")
            data[p.stem] = torch.load(p)
    lat_bounds = [28.4326, 28.8734]
    long_bounds = [76.8616, 77.3474]
    node_locations = find_centers(data["center"], data["alpha"])
    #
    lats = node_locations[:, 0]
    lats = lat_bounds[0] + lats * (lat_bounds[1] - lat_bounds[0])
    lats = lats.tolist()
    #
    longs = node_locations[:, 1]
    longs = long_bounds[0] + longs * (long_bounds[1] - long_bounds[0])
    longs = longs.tolist()
    #
    nodes_to_variates = get_variate_map(data["alpha"])
    graph = get_graph(data["G_pred"], disease_idx_start=2 * 2)
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(long_bounds + longs, lat_bounds + lats),
        crs="EPSG:4326",
    )
    gdf = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, markersize=5)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    pos = {i: (x, y) for i, (x, y) in enumerate(zip(gdf.geometry.x, gdf.geometry.y))}
    G = nx.DiGraph()
    G.add_nodes_from(range(8))
    G.add_edges_from(graph)
    variate_to_colors = {0: "red", 1: "blue", 2: "green", 3: "C1"}
    color_and_names = "\n".join(
        [f"{name}: {variate_to_colors[i]}" for i, name in enumerate(variate_names)]
    )
    print(color_and_names)
    node_colors = [variate_to_colors[nodes_to_variates[i]] for i in range(8)]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=600,
        node_color=node_colors,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="->",
        arrowsize=20,
        width=2,
        node_size=600,
    )
    custom_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=f"{variate_to_colors[i]}",
            label=f"{name}",
            markerfacecolor=f"{variate_to_colors[i]}",
            markersize=10,
        )
        for i, name in enumerate(variate_names)
    ]
    ax.legend(handles=custom_handles)
    plt.axis("off")
    fig.savefig("map.png")


if __name__ == "__main__":
    main()
