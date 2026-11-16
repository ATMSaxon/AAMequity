"""Full-extent street-network travel times between tracts and vertiport candidates.

For each metro the OSM driving network is built for the whole study area
(convex hull of the tract layer plus a buffer) from Geofabrik state extracts in
data/raw/osm, stored as flat edge arrays, and routed with a sparse-graph
Dijkstra from every tract centroid.

Outputs (data/processed):
    network_graph_<metro>.npz   node coordinates and edge attributes
    network_od_<metro>.npz      for tract->tract and tract->candidate pairs, the
                                free-flow minutes spent in each congestion class
                                (area type x freeway/surface) and the network km
                                along the fastest path, plus the centroid-to-node
                                connector distance

Run:  python -m src.network_od <metro> [n_workers]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .candidates import _metric_crs_for, _to_metric, build_vertiport_candidates
from .config import PROC, RAW
from .data_layer import load_lodes_wac, load_tracts_joined

HULL_BUFFER_M = 5000.0
MAX_CANDIDATES = 50
CHUNK = 16
FREEWAY_CLASSES = ("motorway", "motorway_link", "trunk", "trunk_link")
# Activity density (residents + jobs per km2) separating the four area types
AREA_TYPE_BREAKS = (1000.0, 5000.0, 20000.0)
N_CLASS = 2 * (len(AREA_TYPE_BREAKS) + 1)


def graph_path(metro: str) -> Path:
    return PROC / f"network_graph_{metro}.npz"


def od_path(metro: str) -> Path:
    return PROC / f"network_od_{metro}.npz"


METRO_PBF = {
    "la": ["socal"], "sfbay": ["norcal"], "dfw": ["texas"],
    "nyc": ["new-york", "new-jersey"],
}
# Roads open to general motor traffic (the OSMnx "drive" network definition)
EXCLUDED_HIGHWAY = {
    "abandoned", "bridleway", "bus_guideway", "construction", "corridor", "cycleway",
    "elevator", "escalator", "footway", "no", "path", "pedestrian", "planned", "platform",
    "proposed", "raceway", "razed", "service", "steps", "track", "busway", "rest_area",
}
MPH_TO_KPH = 1.609344


def _drivable(tags) -> bool:
    if tags.get("highway") in EXCLUDED_HIGHWAY or tags.get("area") == "yes":
        return False
    if tags.get("motor_vehicle") == "no" or tags.get("motorcar") == "no":
        return False
    return tags.get("access") not in ("private", "no")


def _maxspeed_kph(value) -> float:
    """Posted speed in km/h; NaN when the tag is absent or not numeric."""
    if not value:
        return np.nan
    part = value.split(";")[0].strip().lower()
    num = "".join(ch for ch in part if ch.isdigit() or ch == ".")
    if not num:
        return np.nan
    kph = float(num) * (MPH_TO_KPH if "mph" in part else 1.0)
    return kph if 5.0 <= kph <= 140.0 else np.nan


def _read_ways(metro: str, bbox: tuple[float, float, float, float]) -> dict:
    """Drivable ways with at least one node inside the lon/lat bounding box."""
    import osmium

    west, south, east, north = bbox
    seen = set()
    refs, lons, lats, offsets = [], [], [], [0]
    highway, speed, direction = [], [], []
    for region in METRO_PBF[metro]:
        fp = (osmium.FileProcessor(str(RAW / "osm" / f"{region}-latest.osm.pbf"))
              .with_locations()
              .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
              .with_filter(osmium.filter.KeyFilter("highway")))
        for w in fp:
            if w.id in seen or not _drivable(w.tags):
                continue
            nodes = [(n.ref, n.lon, n.lat) for n in w.nodes if n.location.valid()]
            if len(nodes) < 2 or not any(west <= x <= east and south <= y <= north for _, x, y in nodes):
                continue
            seen.add(w.id)
            oneway = w.tags.get("oneway", "")
            if oneway in ("-1", "reverse"):
                d = -1
            elif (oneway in ("yes", "true", "1") or w.tags.get("junction") in ("roundabout", "circular")
                  or (w.tags.get("highway") == "motorway" and oneway != "no")):
                d = 1
            else:
                d = 0
            for r, x, y in nodes:
                refs.append(r); lons.append(x); lats.append(y)
            offsets.append(len(refs))
            highway.append(w.tags.get("highway"))
            speed.append(_maxspeed_kph(w.tags.get("maxspeed")))
            direction.append(d)
    return {"ref": np.array(refs, dtype=np.int64), "lon": np.array(lons), "lat": np.array(lats),
            "offset": np.array(offsets, dtype=np.int64), "highway": np.array(highway),
            "speed": np.array(speed), "direction": np.array(direction, dtype=np.int8)}


def build_graph(metro: str) -> Path:
    """Build the drive network for the full metro extent from OSM extracts.

    Ways are split at intersections, posted speeds are used where tagged and
    otherwise imputed with the mean posted speed of the same road class, and the
    largest strongly connected component is kept.
    """
    from scipy.sparse.csgraph import connected_components
    from shapely import contains_xy

    out = graph_path(metro)
    if out.exists():
        return out
    t0 = time.time()
    epsg_m = _metric_crs_for(metro)
    tracts_m = _to_metric(load_tracts_joined(metro), epsg_m)
    hull_m = tracts_m.geometry.union_all().convex_hull.buffer(HULL_BUFFER_M)
    hull_ll = gpd.GeoSeries([hull_m], crs=f"EPSG:{epsg_m}").to_crs("EPSG:4326").iloc[0]
    w = _read_ways(metro, hull_ll.bounds)

    # Project nodes once and keep ways that touch the study polygon
    node_ref, first, inv = np.unique(w["ref"], return_index=True, return_inverse=True)
    pts = gpd.GeoSeries(gpd.points_from_xy(w["lon"][first], w["lat"][first]),
                        crs="EPSG:4326").to_crs(f"EPSG:{epsg_m}")
    nx_, ny_ = pts.x.values, pts.y.values
    inside = contains_xy(hull_m, nx_, ny_)
    n_way = len(w["highway"])
    way_of = np.repeat(np.arange(n_way), np.diff(w["offset"]))
    keep_way = np.zeros(n_way, dtype=bool)
    np.logical_or.at(keep_way, way_of, inside[inv])

    # Speed per way: posted, else class mean, else overall mean
    speed = w["speed"].copy()
    overall = np.nanmean(speed[keep_way])
    for cls in np.unique(w["highway"][keep_way]):
        m = keep_way & (w["highway"] == cls)
        cls_mean = np.nanmean(speed[m]) if np.isfinite(speed[m]).any() else overall
        speed[m & ~np.isfinite(speed)] = cls_mean

    # Break ways at intersections: nodes used more than once, and way end points
    pos_keep = keep_way[way_of]
    use_count = np.bincount(inv[pos_keep], minlength=len(node_ref))
    is_break = use_count[inv] > 1
    is_break[w["offset"][:-1]] = True
    is_break[w["offset"][1:] - 1] = True

    x, y = nx_[inv], ny_[inv]
    seg = np.zeros(len(inv))
    seg[1:] = np.hypot(np.diff(x), np.diff(y))
    seg[w["offset"][:-1]] = 0.0                      # no segment into a way's first node
    cum = np.cumsum(seg)

    bpos = np.where(is_break & pos_keep)[0]
    same_way = way_of[bpos[1:]] == way_of[bpos[:-1]]
    a, b = bpos[:-1][same_way], bpos[1:][same_way]
    length = cum[b] - cum[a]
    wid = way_of[a]
    ok = length > 0
    a, b, length, wid = a[ok], b[ok], length[ok], wid[ok]
    na, nb, d = inv[a], inv[b], w["direction"][wid]

    fwd, rev = d >= 0, d <= 0
    u = np.concatenate([na[fwd], nb[rev]])
    v = np.concatenate([nb[fwd], na[rev]])
    e_len = np.concatenate([length[fwd], length[rev]])
    e_way = np.concatenate([wid[fwd], wid[rev]])

    # Largest strongly connected component, re-indexed
    n_all = len(node_ref)
    _, label = connected_components(csr_matrix((np.ones(len(u)), (u, v)), shape=(n_all, n_all)),
                                    directed=True, connection="strong")
    main = np.bincount(label).argmax()
    node_keep = label == main
    new_id = np.full(n_all, -1, dtype=np.int64)
    new_id[node_keep] = np.arange(node_keep.sum())
    e_keep = node_keep[u] & node_keep[v]
    u, v, e_len, e_way = new_id[u[e_keep]], new_id[v[e_keep]], e_len[e_keep], e_way[e_keep]

    # Area type of each edge: density class of the tract nearest to its start node
    centroid, area_type = tract_area_types(metro)
    _, nearest = cKDTree(centroid).query(np.column_stack([nx_[node_keep], ny_[node_keep]]))
    edge_area = area_type[nearest][u]

    np.savez_compressed(out, node_id=node_ref[node_keep], x=nx_[node_keep], y=ny_[node_keep],
                        u=u.astype(np.int32), v=v.astype(np.int32), length_m=e_len,
                        speed_kph=speed[e_way], highway=w["highway"][e_way].astype(str),
                        area_type=edge_area.astype(np.int8))
    posted = float(np.isfinite(w["speed"][e_way]).mean())
    print(f"  {metro}: graph nodes={int(node_keep.sum())} edges={len(u)} "
          f"posted-speed share={posted:.2f} in {time.time() - t0:.0f}s", flush=True)
    return out


def tract_area_types(metro: str) -> tuple[np.ndarray, np.ndarray]:
    """Tract centroids (metric CRS) and area type from activity density.

    Activity density is residents plus jobs per km2 of tract area, classed with
    AREA_TYPE_BREAKS (0 = low density ... 3 = urban core).
    """
    tracts = load_tracts_joined(metro)
    tracts_m = _to_metric(tracts, _metric_crs_for(metro))
    jobs = (load_lodes_wac(metro).set_index("dest_tract")["jobs_total"]
            .reindex(tracts["tract_geoid"].values).fillna(0.0).values)
    density = (tracts["pop"].fillna(0).values + jobs) / np.maximum(tracts_m.geometry.area.values / 1e6, 1e-6)
    centroid = np.array([(p.x, p.y) for p in tracts_m.geometry.centroid])
    return centroid, np.digitize(density, AREA_TYPE_BREAKS)


def edge_minutes(g) -> np.ndarray:
    """Free-flow traversal time per edge in minutes."""
    return g["length_m"] / 1000.0 / g["speed_kph"] * 60.0


def edge_class(g) -> np.ndarray:
    """Congestion class of each edge: 2 * area type + (1 if freeway else 0)."""
    return 2 * g["area_type"].astype(np.int64) + np.isin(g["highway"], FREEWAY_CLASSES)


def _fastest_edges(g, weight: np.ndarray):
    """Indices of the fastest edge for every ordered node pair."""
    order = np.lexsort((weight, g["v"], g["u"]))
    u, v = g["u"][order], g["v"][order]
    keep = np.ones(len(u), dtype=bool)
    keep[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
    return order[keep]


_W = {}


def _init_worker(path: str, kappa):
    g = np.load(path, allow_pickle=False)
    n = len(g["node_id"])
    minutes, cls = edge_minutes(g), edge_class(g)
    weight = minutes * (np.ones(N_CLASS) if kappa is None else np.asarray(kappa))[cls]
    e = _fastest_edges(g, weight)
    u, v = g["u"][e].astype(np.int64), g["v"][e].astype(np.int64)
    _W["n"] = n
    _W["time"] = csr_matrix((weight[e], (u, v)), shape=(n, n))
    _W["edge_key"] = u * n + v            # already sorted by (u, v)
    # Per-edge quantities summed along each path: free-flow minutes by class, then km
    step = np.zeros((len(e), N_CLASS + 1))
    step[np.arange(len(e)), cls[e]] = minutes[e]
    step[:, N_CLASS] = g["length_m"][e] / 1000.0
    _W["edge_step"] = step


def _route_chunk(args):
    """Along each fastest path: free-flow minutes by congestion class, and km."""
    src_nodes, dst_nodes = args
    t, pred = dijkstra(_W["time"], directed=True, indices=src_nodes, return_predecessors=True)
    n, key, edge_step = _W["n"], _W["edge_key"], _W["edge_step"]
    node_ids = np.arange(n, dtype=np.int64)
    out = np.full((len(src_nodes), len(dst_nodes), N_CLASS + 1), np.inf, dtype=np.float32)
    for r in range(len(src_nodes)):
        parent = pred[r].astype(np.int64)
        has = parent >= 0
        cum = np.zeros((n, N_CLASS + 1))
        cum[has] = edge_step[np.searchsorted(key, parent[has] * n + node_ids[has])]
        # Pointer jumping: each round doubles the length of the summed path prefix
        while True:
            idx = np.where(parent >= 0)[0]
            if idx.size == 0:
                break
            par = parent[idx]
            cum[idx] += cum[par]
            parent[idx] = parent[par]
        reach = np.isfinite(t[r][dst_nodes])
        out[r, reach] = cum[dst_nodes[reach]]
    return out


def build_od(metro: str, n_workers: int = 16, kappa=None) -> Path:
    """Route every tract to all tracts and candidates.

    With `kappa` (one congestion multiplier per class) paths minimise congested
    time and the existing matrix is replaced; otherwise paths minimise free-flow
    time and an existing matrix is reused.
    """
    out = od_path(metro)
    if out.exists() and kappa is None:
        return out
    g = np.load(build_graph(metro), allow_pickle=False)
    epsg_m = _metric_crs_for(metro)

    tracts = load_tracts_joined(metro)
    tracts_m = _to_metric(tracts, epsg_m)
    t_xy = np.array([(p.x, p.y) for p in tracts_m.geometry.centroid])
    cand = build_vertiport_candidates(metro=metro, max_candidates=MAX_CANDIDATES)
    c_xy = np.array([(p.x, p.y) for p in _to_metric(cand, epsg_m).geometry])

    tree = cKDTree(np.column_stack([g["x"], g["y"]]))
    t_conn, t_node = tree.query(t_xy)
    c_conn, c_node = tree.query(c_xy)
    dst_nodes = np.concatenate([t_node, c_node])
    n_t = len(t_node)

    chunks = [(t_node[i:i + CHUNK], dst_nodes) for i in range(0, n_t, CHUNK)]
    print(f"  {metro}: routing {n_t} origins to {len(dst_nodes)} destinations "
          f"on {len(g['node_id'])} nodes, {len(chunks)} chunks, {n_workers} workers", flush=True)
    t0 = time.time()
    parts = []
    with ProcessPoolExecutor(n_workers, initializer=_init_worker,
                             initargs=(str(graph_path(metro)), kappa)) as ex:
        for i, part in enumerate(ex.map(_route_chunk, chunks)):
            parts.append(part)
            if (i + 1) % 20 == 0 or i + 1 == len(chunks):
                print(f"    [{i + 1}/{len(chunks)}] {time.time() - t0:.0f}s", flush=True)
    res = np.concatenate(parts, axis=0)

    np.savez_compressed(
        out,
        tract_geoid=np.asarray(tracts["tract_geoid"].tolist(), dtype=str),
        candidate_id=np.asarray(cand["candidate_id"].tolist(), dtype=str),
        tract_area_type=tract_area_types(metro)[1].astype(np.int8),
        tt_class_min=res[:, :n_t, :N_CLASS], tt_km=res[:, :n_t, N_CLASS],
        tc_class_min=res[:, n_t:, :N_CLASS], tc_km=res[:, n_t:, N_CLASS],
        tract_conn_m=t_conn.astype(np.float32), cand_conn_m=c_conn.astype(np.float32),
    )
    unreachable = float(np.isinf(res[:, :n_t, N_CLASS]).mean())
    print(f"  {metro}: saved {out.name}; unreachable tract pairs {unreachable:.4%}; "
          f"median connector {np.median(t_conn):.0f} m", flush=True)
    return out


if __name__ == "__main__":
    metro_ = sys.argv[1]
    workers_ = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    build_od(metro_, workers_)
