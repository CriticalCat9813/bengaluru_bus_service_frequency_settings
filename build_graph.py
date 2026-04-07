from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import osmium
import pandas as pd
from pyrosm import OSM

PBF_PATH = r"C:\game_heory\bengaluru_india.osm.pbf"
RELPATH = "."
FILENAME = "bengaluru_india"
CENTER = (12.9716, 77.5946)

LINES_DIST_TRGT = 400  
LINES_DIST_CTFF = 1600 
MIN_ROUTE_STOPS = 6
N_SYNTHETIC_TRIPS = 100_000
SNAP_TOL_M = 75

HEXCOLORS = [
    "#0072B2",
    "#F0E442",
    "#D55E00",
    "#56B4E9",
    "#E69F00",
    "#009E73",
    "#CC79A7",
]

BLR_HWY_SPEEDS = {
    "motorway": 60,
    "trunk": 60,
    "primary": 40,
    "secondary": 35,
    "tertiary": 30,
    "residential": 20,
    "unclassified": 25,
    "service": 15,
}

POI_FILTER = {
    "highway": ["bus_stop", "platform"],
    "public_transport": ["platform", "stop_position", "station"],
    "amenity": ["bus_station"],
}

def ensure_dirs() -> None:
    os.makedirs(f"{RELPATH}/results/preprocess", exist_ok=True)


@dataclass
class RouteRelation:
    relation_id: int
    name: str | None
    ref: str | None
    operator: str | None
    network: str | None
    members: list[tuple[str, int, str]]


class BusRouteHandler(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.routes: list[RouteRelation] = []

    def relation(self, r) -> None:
        tags = {t.k: t.v for t in r.tags}
        if tags.get("type") == "route" and tags.get("route") == "bus":
            members: list[tuple[str, int, str]] = []
            for m in r.members:
                members.append((m.type, int(m.ref), m.role or ""))
            self.routes.append(
                RouteRelation(
                    relation_id=int(r.id),
                    name=tags.get("name"),
                    ref=tags.get("ref"),
                    operator=tags.get("operator"),
                    network=tags.get("network"),
                    members=members,
                )
            )


def normalize_stops_columns(stops_raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    stops = stops_raw.copy()

    if "id" in stops.columns and "osmid" not in stops.columns:
        stops = stops.rename(columns={"id": "osmid"})
    if "osmid" not in stops.columns:
        raise ValueError("Could not find an OSM id column in the stops GeoDataFrame.")

    if "osm_type" not in stops.columns:
        stops["osm_type"] = "node"

    if "name" not in stops.columns:
        stops["name"] = None
    if "ref" not in stops.columns:
        stops["ref"] = None
    if "local_ref" not in stops.columns:
        stops["local_ref"] = None
    if "operator" not in stops.columns:
        stops["operator"] = None
    if "network" not in stops.columns:
        stops["network"] = None

    stops = stops[stops.geometry.geom_type == "Point"].copy()
    stops["osmid"] = stops["osmid"].astype("int64")
    return stops


def snap_points_to_graph(
    G: nx.MultiDiGraph,
    stops: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if stops.empty:
        stops["node"] = []
        stops["distance_to_node_m"] = []
        stops["snapped_ok"] = []
        return stops

    G_proj = ox.project_graph(G)
    projected_crs = G_proj.graph["crs"]
    stops_proj = stops.to_crs(projected_crs)

    X = stops_proj.geometry.x.to_numpy()
    Y = stops_proj.geometry.y.to_numpy()
    nn, dist = ox.distance.nearest_nodes(G_proj, X, Y, return_dist=True)

    stops = stops.copy()
    stops["node"] = [int(x) for x in nn]
    stops["distance_to_node_m"] = dist
    stops["snapped_ok"] = stops["distance_to_node_m"] <= SNAP_TOL_M
    return stops


def dedupe_keep_order(values: Iterable[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return tuple(out)


def build_graph(osm: OSM) -> tuple[nx.MultiDiGraph, nx.Graph]:
    print("... building road graph from local PBF")
    nodes, edges = osm.get_network(network_type="driving", nodes=True)
    G = osm.to_graph(nodes, edges, graph_type="networkx", osmnx_compatible=True)
    G = ox.routing.add_edge_speeds(G, hwy_speeds=BLR_HWY_SPEEDS, fallback=20)
    G = ox.routing.add_edge_travel_times(G)

    ox.save_graphml(G, f"{RELPATH}/results/preprocess/g_{FILENAME}.graphml")
    G_und = nx.Graph(G)
    return G, G_und


def build_stops(osm: OSM, G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    print("... extracting bus stops / platforms")
    stops_raw = osm.get_pois(
        custom_filter=POI_FILTER,
        extra_attributes=["name", "ref", "local_ref", "operator", "network"],
    )
    stops = normalize_stops_columns(stops_raw)
    stops = snap_points_to_graph(G, stops)
    stops.to_file(f"{RELPATH}/results/preprocess/stops_df_{FILENAME}.gpkg", driver="GPKG")
    return stops


def build_lines(
    pbf_path: str,
    G_und: nx.Graph,
    stops_df: gpd.GeoDataFrame,
) -> pd.DataFrame:
    print("... extracting bus route relations from local PBF")

    stop_node_lookup = {
        int(row.osmid): int(row.node)
        for _, row in stops_df.iterrows()
        if str(row.get("osm_type", "node")).lower() in {"node", "n"}
    }

    handler = BusRouteHandler()
    handler.apply_file(pbf_path, locations=False)

    lines: list[list] = []
    stop_sequences_rows: list[dict] = []

    for route_idx, route_rel in enumerate(handler.routes):
        stop_nodes: list[int] = []
        stop_osmids: list[int] = []

        for obj_type, ref, role in route_rel.members:
            role = (role or "").lower()
            if obj_type != "n":
                continue
            if role.startswith("stop") or role.startswith("platform") or ref in stop_node_lookup:
                if ref in stop_node_lookup:
                    node = stop_node_lookup[ref]
                    if not stop_nodes or stop_nodes[-1] != node:
                        stop_nodes.append(node)
                        stop_osmids.append(ref)

        stop_nodes = list(dedupe_keep_order(stop_nodes))
        stop_osmids = list(dedupe_keep_order(stop_osmids))
        if len(stop_nodes) < MIN_ROUTE_STOPS:
            continue

        path_nodes: list[int] = []
        total_length_m = 0.0
        valid_route = True

        for s, t in zip(stop_nodes[:-1], stop_nodes[1:]):
            if s == t:
                continue
            try:
                seg_len, seg_path = nx.bidirectional_dijkstra(G_und, s, t, weight="length")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                valid_route = False
                break
            total_length_m += float(seg_len)
            if path_nodes:
                path_nodes.extend(seg_path[1:])
            else:
                path_nodes.extend(seg_path)

        if not valid_route or not path_nodes:
            continue

        coords = tuple((float(G_und.nodes[node]["y"]), float(G_und.nodes[node]["x"])) for node in path_nodes)
        hexcolor = HEXCOLORS[len(lines) % len(HEXCOLORS)]
        name = route_rel.name or route_rel.ref or f"route_{route_rel.relation_id}"

        line = [
            route_rel.relation_id,
            {
                "name": route_rel.name,
                "ref": route_rel.ref,
                "operator": route_rel.operator,
                "network": route_rel.network,
            },
            tuple(stop_nodes),
            total_length_m / 1000.0,
            tuple(path_nodes),
            coords,
            hexcolor,
            name,
        ]
        lines.append(line)

        for seq, (stop_osmid, stop_node) in enumerate(zip(stop_osmids, stop_nodes), start=1):
            stop_row = stops_df.loc[stops_df["osmid"] == stop_osmid].head(1)
            if stop_row.empty:
                stop_name = None
                stop_lon = float(G_und.nodes[stop_node]["x"])
                stop_lat = float(G_und.nodes[stop_node]["y"])
            else:
                stop_name = stop_row.iloc[0].get("name")
                stop_lon = float(stop_row.iloc[0].geometry.x)
                stop_lat = float(stop_row.iloc[0].geometry.y)
            stop_sequences_rows.append(
                {
                    "route_id": route_rel.relation_id,
                    "direction_id": 0,
                    "stop_sequence": seq,
                    "stop_id": stop_osmid,
                    "stop_name": stop_name,
                    "stop_lat": stop_lat,
                    "stop_lon": stop_lon,
                    "matched_node_id": stop_node,
                    "source": "osm_relation_member_order",
                }
            )

    lines_df = pd.DataFrame(
        lines,
        columns=["id", "tags", "stops", "length", "route", "coords", "hexcolor", "name"],
    )

    if lines_df.empty:
        raise RuntimeError(
            "No usable bus routes were extracted from the PBF. "
            "This usually means the local extract has sparse/irregular route relations."
        )

    print(f"...... built {len(lines_df)} candidate bus routes")
    lines_df["dist"] = lines_df.apply(
        lambda row: nx.multi_source_dijkstra_path_length(
            G_und,
            row.stops,
            weight="length",
            cutoff=LINES_DIST_CTFF,
        ),
        axis=1,
    )
    lines_df = lines_df.drop_duplicates(subset=["length"]).copy()
    lines_df.set_index("id", inplace=True)
    lines_df.to_pickle(f"{RELPATH}/results/preprocess/lines_df_{FILENAME}.pkl")

    stop_sequences_df = pd.DataFrame(stop_sequences_rows)
    stop_sequences_df.to_csv(
        f"{RELPATH}/results/preprocess/stop_sequences_{FILENAME}.csv",
        index=False,
    )

    return lines_df


def build_synthetic_trips(
    G_und: nx.Graph,
    lines_df: pd.DataFrame,
    n_trips: int = N_SYNTHETIC_TRIPS,) -> pd.DataFrame:
    print("... building synthetic trips_df (because there is no Chicago ride-hailing file here)")

    stop_counts = Counter()
    for _, line in lines_df.iterrows():
        stop_counts.update(line.stops)

    candidate_nodes = np.array(list(stop_counts.keys()), dtype=np.int64)
    if candidate_nodes.size < 2:
        raise RuntimeError("Not enough stop nodes were extracted to build synthetic demand.")

    weights = np.array([stop_counts[n] for n in candidate_nodes], dtype=float)
    weights /= weights.sum()

    rng = np.random.default_rng(20260405)
    origins = rng.choice(candidate_nodes, size=n_trips, replace=True, p=weights)
    destinations = rng.choice(candidate_nodes, size=n_trips, replace=True, p=weights)

    same_mask = origins == destinations
    while same_mask.any():
        destinations[same_mask] = rng.choice(candidate_nodes, size=same_mask.sum(), replace=True, p=weights)
        same_mask = origins == destinations

    trips_df = pd.DataFrame({"o_node": origins, "d_node": destinations})
    trips_df["o_x"] = trips_df["o_node"].map(lambda n: float(G_und.nodes[int(n)]["x"]))
    trips_df["o_y"] = trips_df["o_node"].map(lambda n: float(G_und.nodes[int(n)]["y"]))
    trips_df["d_x"] = trips_df["d_node"].map(lambda n: float(G_und.nodes[int(n)]["x"]))
    trips_df["d_y"] = trips_df["d_node"].map(lambda n: float(G_und.nodes[int(n)]["y"]))

    def hav_m(row):
        lon1, lat1, lon2, lat2 = map(math.radians, [row.o_x, row.o_y, row.d_x, row.d_y])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * 6_371_000 * math.asin(math.sqrt(a))

    trips_df["euclid_m"] = trips_df.apply(hav_m, axis=1)
    trips_df = trips_df[trips_df["euclid_m"] >= 300].copy()
    trips_df.drop(columns=["euclid_m"], inplace=True)
    trips_df.reset_index(drop=True, inplace=True)
    trips_df.to_csv(f"{RELPATH}/results/preprocess/trips_df_{FILENAME}.csv", index=False)
    return trips_df


def main() -> None:
    ensure_dirs()

    osm = OSM(PBF_PATH)
    G, G_und = build_graph(osm)
    stops_df = build_stops(osm, G)
    lines_df = build_lines(PBF_PATH, G_und, stops_df)
    trips_df = build_synthetic_trips(G_und, lines_df)

    print("\nDone.")
    print(f"Graph: {RELPATH}/results/preprocess/g_{FILENAME}.graphml")
    print(f"Stops: {RELPATH}/results/preprocess/stops_df_{FILENAME}.gpkg")
    print(f"Lines: {RELPATH}/results/preprocess/lines_df_{FILENAME}.pkl")
    print(f"Trips: {RELPATH}/results/preprocess/trips_df_{FILENAME}.csv")
    print(f"Stop sequences: {RELPATH}/results/preprocess/stop_sequences_{FILENAME}.csv")
    print(f"#stops={len(stops_df)}, #lines={len(lines_df)}, #synthetic_trips={len(trips_df)}")


if __name__ == "__main__":
    main()
