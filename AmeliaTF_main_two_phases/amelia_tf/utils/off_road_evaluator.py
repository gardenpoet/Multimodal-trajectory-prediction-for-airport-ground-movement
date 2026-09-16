import os
import json
import torch
import numpy as np
import pickle
import shapely.geometry as sg
import geopandas as gpd
from typing import List, Tuple, Dict, Any, Optional
from geographiclib.geodesic import Geodesic

from amelia_tf.utils import global_masks as G
from amelia_scenes.utils.transform_utils import (
    xy_to_ll,
    inv_transform_batch,
    direct_wrapper,
    direct_wrapper_batch,
)


class OffRoadEvaluator:
    """
    Off-road detection evaluator for predicted trajectories.
    Uses Amelia's semantic_graph.pkl as the reference network.

    This class loads the airport road network and provides:
        1. Road network reference for off-road detection
        2. Original graph (NetworkX) for BFS exploration
        3. Spatial index for fast nearest-road queries
        4. Node KD-tree for fast nearest-node queries
    """

    # GMM-derived turn thresholds (degrees), computed once by
    # SceneProcessor._compute_global_thresholds from each airport's
    # prediction-segment cumulative-turn distribution.
    AIRPORT_TURN_THRESHOLDS = {
        'kmsy': {'turn_left_threshold': 31.52, 'turn_right_threshold': -31.16},
    }
    DEFAULT_TURN_LEFT_THRESHOLD = 25.0
    DEFAULT_TURN_RIGHT_THRESHOLD = -25.0

    def __init__(self, asset_dir: str, airport_code: str):
        """
        Args:
            asset_dir:    path to the assets directory (contains limits.json).
            airport_code: ICAO airport code (e.g. 'kmsy').
        """
        self.asset_dir = asset_dir
        self.airport_code = airport_code
        self.geodesic = Geodesic.WGS84

        self.limits_path = os.path.join(asset_dir, airport_code, 'limits.json')
        self.pkl_path = os.path.join(asset_dir, airport_code, 'semantic_graph.pkl')

        # Turn thresholds for heading-anchor BFS
        thr = self.AIRPORT_TURN_THRESHOLDS.get(airport_code.lower())
        if thr is not None:
            self.turn_left_threshold = thr['turn_left_threshold']
            self.turn_right_threshold = thr['turn_right_threshold']
        else:
            print(f"Warning: no recorded turn thresholds for airport "
                  f"'{airport_code}', using default "
                  f"+-{self.DEFAULT_TURN_LEFT_THRESHOLD:.1f}deg")
            self.turn_left_threshold = self.DEFAULT_TURN_LEFT_THRESHOLD
            self.turn_right_threshold = self.DEFAULT_TURN_RIGHT_THRESHOLD

        # Load reference point first (needed for coordinate conversion)
        self._load_reference_point()

        # Load road network
        self._load_reference_network()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_reference_network(self):
        """Load Amelia's semantic_graph.pkl as the reference network."""
        if not os.path.exists(self.pkl_path):
            print(f"Warning: PKL file not found at {self.pkl_path}")
            self.reference_gdf = None
            self.graph_nx = None
            return

        print(f"Loading reference network from: {self.pkl_path}")
        with open(self.pkl_path, 'rb') as f:
            amelia_data = pickle.load(f)

        # Store the raw NetworkX graph for BFS exploration
        self.graph_nx = amelia_data['graph_networkx']

        # Build GeoDataFrame for off-road detection
        edges = []
        geometries = []

        for u, v, data in self.graph_nx.edges(data=True):
            u_data = self.graph_nx.nodes[u]
            v_data = self.graph_nx.nodes[v]

            if 'x' in u_data and 'y' in u_data:
                line = sg.LineString([
                    (u_data['x'], u_data['y']),
                    (v_data['x'], v_data['y'])
                ])
                geometries.append(line)

                node_type_u = u_data.get('node_type', 0)
                node_type_v = v_data.get('node_type', 0)
                edges.append({
                    'u': u,
                    'v': v,
                    'node_type_u': node_type_u,
                    'node_type_v': node_type_v,
                    'length': data.get('length', 0),
                })

        self.reference_gdf = gpd.GeoDataFrame(edges, geometry=geometries, crs="EPSG:4326")
        self.reference_gdf_proj = self.reference_gdf.to_crs("EPSG:3857")
        self.spatial_index = self.reference_gdf_proj.geometry.sindex

        # Road widths by node type (FAA standards)
        DEFAULT_TAXIWAY_WIDTH = 23.0
        DEFAULT_RUNWAY_WIDTH = 45.0
        DEFAULT_HOLD_LINE_WIDTH = 5.0
        DEFAULT_EXIT_WIDTH = 15.0

        def get_width_by_node_type(row):
            # Check if the edge connects to a runway first (highest priority)
            if row['node_type_u'] == 1 or row['node_type_v'] == 1:
                return DEFAULT_RUNWAY_WIDTH
            # Then check for taxiway
            if row['node_type_u'] == 2 or row['node_type_v'] == 2:
                return DEFAULT_TAXIWAY_WIDTH
            # Then check for exit
            if row['node_type_u'] == 4 or row['node_type_v'] == 4:
                return DEFAULT_EXIT_WIDTH
            # Only pure hold lines get the narrow width
            if row['node_type_u'] == 3 and row['node_type_v'] == 3:
                return DEFAULT_HOLD_LINE_WIDTH
            # Fallback: taxiway width
            return DEFAULT_TAXIWAY_WIDTH

        self.reference_gdf['width_m'] = self.reference_gdf.apply(
            get_width_by_node_type, axis=1)

        print(f"Loaded {len(self.reference_gdf)} reference segments from PKL")
        for node_type, width in [
            (1, DEFAULT_RUNWAY_WIDTH), (2, DEFAULT_TAXIWAY_WIDTH),
            (3, DEFAULT_HOLD_LINE_WIDTH), (4, DEFAULT_EXIT_WIDTH)
        ]:
            count = len(self.reference_gdf[self.reference_gdf['width_m'] == width])
            if count > 0:
                print(f"  node_type {node_type}: {count} edges, width={width:.1f}m")

        # Build KD-tree over ALL nodes in the original graph
        self._build_node_kdtree()

        # Print graph statistics
        print(f"Original graph: {self.graph_nx.number_of_nodes()} nodes, "
              f"{self.graph_nx.number_of_edges()} edges")

    def _build_node_kdtree(self):
        """Build a KD-tree over all nodes in the original graph."""
        from scipy.spatial import cKDTree

        if self.graph_nx is None:
            self._node_kdtree = None
            self._kdtree_node_ids = []
            self._kdtree_node_xys = np.empty((0, 2))
            return

        node_ids = []
        node_xys = []

        for node in self.graph_nx.nodes():
            xy = self._get_node_xy_from_graph(node)
            if xy is not None:
                node_ids.append(node)
                node_xys.append(xy)

        self._kdtree_node_ids = node_ids
        self._kdtree_node_xys = np.array(node_xys) if node_xys else np.empty((0, 2))

        if len(node_xys) > 0:
            self._node_kdtree = cKDTree(self._kdtree_node_xys)
        else:
            self._node_kdtree = None

        print(f"Node KD-tree: {len(node_ids)} nodes indexed.")

    def _get_node_xy_from_graph(self, node) -> Optional[np.ndarray]:
        """
        Return a node's position in the local km-XY coordinate system.

        Converts WGS84 lon/lat from the graph to the same local km-XY frame
        used by the model (sequences[..., G.XY]).
        """
        data = self.graph_nx.nodes.get(node, {})
        if 'x' not in data or 'y' not in data:
            return None
        if self.ref is None:
            return None

        node_lon, node_lat = data['x'], data['y']
        ref_lat, ref_lon, range_scale = self.ref

        R_EARTH_M = 6371000.0
        dlat = np.radians(node_lat - ref_lat)
        dlon = np.radians(node_lon - ref_lon)
        mean_lat_rad = np.radians((node_lat + ref_lat) / 2.0)

        dx_m = dlon * R_EARTH_M * np.cos(mean_lat_rad)
        dy_m = dlat * R_EARTH_M

        distance_m = np.hypot(dx_m, dy_m)
        if distance_m < 1e-9:
            return np.array([0.0, 0.0])

        bearing_deg = np.degrees(np.arctan2(dx_m, dy_m)) % 360.0
        bearing_rad = np.radians(bearing_deg)

        rang_km = distance_m / range_scale
        x = rang_km * np.cos(bearing_rad)
        y = rang_km * np.sin(bearing_rad)

        return np.array([x, y])

    def _load_reference_point(self):
        """Load the reference lat/lon and scale from limits.json."""
        if not os.path.exists(self.limits_path):
            print(f"Warning: limits.json not found at {self.limits_path}")
            self.ref = None
            return

        with open(self.limits_path, 'r') as f:
            ref_data = json.load(f)

        self.ref = (ref_data['ref_lat'], ref_data['ref_lon'], ref_data['range_scale'])
        print(f"Loaded reference point: lat={self.ref[0]}, lon={self.ref[1]}")

    def _get_ref_for_airport(self, airport_code: str) -> Tuple[float, float, float]:
        """Return (ref_lat, ref_lon, range_scale) for a given airport code."""
        limits_path = os.path.join(self.asset_dir, airport_code, 'limits.json')
        with open(limits_path, 'r') as f:
            ref_data = json.load(f)
        return (ref_data['ref_lat'], ref_data['ref_lon'], ref_data['range_scale'])

    # ------------------------------------------------------------------
    # Main evaluation entry point (vectorised)
    # ------------------------------------------------------------------

    def evaluate_prediction(
        self,
        batch: Dict[str, Any],
        ego_mu: torch.Tensor,           # (B, 1, T_total, M, D)
        ego_pred_scores: torch.Tensor,  # (B, 1, M)
        hist_len: int,
        safety_margin: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate predicted trajectories for all B samples in the batch.

        Vectorised pipeline:
            1. Collect start poses
            2. inv_transform_batch
            3. direct_wrapper_batch (same airport) or per-sample
            4. GeoSeries.to_crs on all points
            5. sindex.nearest on all points
            6. Slice results back per sample
        """
        scene = batch['scene_dict']
        sequences = scene['sequences']
        ego_agents = scene['ego_agent_id_test']
        B = ego_mu.shape[0]
        T_pred = ego_mu.shape[2] - hist_len

        if self.reference_gdf is None:
            return [self._get_empty_results(0) for _ in range(B)]

        # Collect per-sample inputs
        ego_agent_ids = [
            ego_agents[b].item() if torch.is_tensor(ego_agents) else int(ego_agents[b])
            for b in range(B)
        ]

        best_modes = ego_pred_scores[:, 0].argmax(dim=-1).cpu().numpy()

        future_rel = np.stack([
            ego_mu[b, 0, hist_len:, best_modes[b], :2].detach().cpu().numpy()
            for b in range(B)
        ], axis=0)

        start_abs = np.stack([
            sequences[b, ego_agent_ids[b], hist_len - 1, G.XY]
            .detach().cpu().numpy().flatten()
            for b in range(B)
        ], axis=0)

        start_heading = np.array([
            float(sequences[b, ego_agent_ids[b], hist_len - 1, G.HD]
                  .detach().cpu().numpy())
            for b in range(B)
        ])

        # Batched inverse transform
        traj_xy_abs = inv_transform_batch(future_rel, start_abs, start_heading)

        x = traj_xy_abs[:, :, 0]
        y = traj_xy_abs[:, :, 1]
        rang = np.sqrt(x ** 2 + y ** 2)
        bearing = np.degrees(np.arctan2(y, x))

        airport_codes = [scene['airport_id'][b] for b in range(B)]
        refs = [self._get_ref_for_airport(c) for c in airport_codes]

        # Geodesic conversion
        all_lats = np.empty((B, T_pred), dtype=np.float64)
        all_lons = np.empty((B, T_pred), dtype=np.float64)

        if len(set(airport_codes)) == 1:
            all_lats, all_lons = direct_wrapper_batch(
                self.geodesic, bearing, rang,
                refs[0][0], refs[0][1], refs[0][2]
            )
        else:
            for b in range(B):
                lats, lons = direct_wrapper(
                    self.geodesic, bearing[b], rang[b],
                    refs[b][0], refs[b][1], refs[b][2]
                )
                all_lats[b] = lats
                all_lons[b] = lons

        # Project all points to EPSG:3857 in one call
        flat_lons = all_lons.ravel()
        flat_lats = all_lats.ravel()

        points_proj = gpd.GeoSeries(
            [sg.Point(lon, lat) for lon, lat in zip(flat_lons, flat_lats)],
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")

        # Bulk nearest-neighbour query
        road_geoms = self.reference_gdf_proj.geometry.values
        widths = self.reference_gdf['width_m'].values

        distances = np.empty(B * T_pred, dtype=np.float64)
        is_on_road = np.empty(B * T_pred, dtype=bool)

        try:
            input_idx, tree_idx = self.spatial_index.nearest(
                points_proj, return_all=False
            )
            for qi, ti in zip(input_idx, tree_idx):
                dist = points_proj.iloc[qi].distance(road_geoms[ti])
                max_allowed = (widths[ti] / 2.0) * safety_margin
                distances[qi] = dist
                is_on_road[qi] = dist <= max_allowed

        except Exception:
            # Fallback for older geopandas
            for i, pt_proj in enumerate(points_proj):
                bbox = pt_proj.bounds
                candidates_idx = list(self.spatial_index.intersection(bbox))
                if not candidates_idx:
                    candidates_idx = list(range(len(self.reference_gdf_proj)))
                cand_geoms = road_geoms[candidates_idx]
                cand_widths = widths[candidates_idx]
                dists = np.array([pt_proj.distance(g) for g in cand_geoms])
                best = dists.argmin()
                distances[i] = dists[best]
                is_on_road[i] = distances[i] <= (cand_widths[best] / 2.0) * safety_margin

        # Slice results per sample
        return [
            self._compute_metrics(
                is_on_road[b * T_pred: (b + 1) * T_pred].tolist(),
                distances[b * T_pred: (b + 1) * T_pred].tolist(),
            )
            for b in range(B)
        ]

    # ------------------------------------------------------------------
    # Single-trajectory evaluation
    # ------------------------------------------------------------------

    def _evaluate_trajectory(
        self,
        trajectory_ll: List[Tuple[float, float]],
        safety_margin: float = 1.0,
    ) -> Dict[str, Any]:
        """Evaluate a single trajectory as a list of (lon, lat) points."""
        if self.reference_gdf is None or len(trajectory_ll) == 0:
            return self._get_empty_results(len(trajectory_ll))

        results = []
        distances = []

        for lon, lat in trajectory_ll:
            pt = sg.Point(lon, lat)
            pt_proj = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs("EPSG:3857")[0]

            bbox = pt_proj.bounds
            candidates_idx = list(self.spatial_index.intersection(bbox))
            if not candidates_idx:
                candidates = self.reference_gdf_proj
            else:
                candidates = self.reference_gdf_proj.iloc[candidates_idx]

            dists = candidates.geometry.distance(pt_proj)
            min_idx = dists.idxmin()
            min_dist = dists[min_idx]
            road_width = self.reference_gdf.loc[min_idx, 'width_m']
            max_allowed = (road_width / 2.0) * safety_margin

            results.append(min_dist <= max_allowed)
            distances.append(min_dist)

        return self._compute_metrics(results, distances)

    def _evaluate_trajectory_batch(
        self,
        trajectory_ll: List[Tuple[float, float]],
        safety_margin: float = 1.0,
    ) -> Dict[str, Any]:
        """Vectorised version of _evaluate_trajectory."""
        if self.reference_gdf is None or len(trajectory_ll) == 0:
            return self._get_empty_results(len(trajectory_ll))

        lons, lats = zip(*trajectory_ll)

        points_proj = gpd.GeoSeries(
            [sg.Point(lon, lat) for lon, lat in zip(lons, lats)],
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")

        road_geoms = self.reference_gdf_proj.geometry.values
        widths = self.reference_gdf['width_m'].values
        T = len(points_proj)
        distances = np.empty(T, dtype=np.float64)
        is_on_road = np.empty(T, dtype=bool)

        try:
            input_idx, tree_idx = self.spatial_index.nearest(
                points_proj, return_all=False
            )
            for qi, ti in zip(input_idx, tree_idx):
                dist = points_proj.iloc[qi].distance(road_geoms[ti])
                max_allowed = (widths[ti] / 2.0) * safety_margin
                distances[qi] = dist
                is_on_road[qi] = dist <= max_allowed

        except Exception:
            for i, pt_proj in enumerate(points_proj):
                bbox = pt_proj.bounds
                candidates_idx = list(self.spatial_index.intersection(bbox)) or \
                                 list(range(len(self.reference_gdf_proj)))
                cand_geoms = road_geoms[candidates_idx]
                cand_widths = widths[candidates_idx]
                dists = np.array([pt_proj.distance(g) for g in cand_geoms])
                best = dists.argmin()
                distances[i] = dists[best]
                is_on_road[i] = distances[i] <= (cand_widths[best] / 2.0) * safety_margin

        return self._compute_metrics(is_on_road.tolist(), distances.tolist())

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        is_on_taxiway: List[bool],
        distances: List[float],
    ) -> Dict[str, Any]:
        """Compute off-road metrics from per-point results."""
        is_off = [not r for r in is_on_taxiway]
        off_distances = [d for d, off in zip(distances, is_off) if off]

        off_road_rate = sum(is_off) / len(is_off) if is_off else 0
        max_off_distance = max(off_distances) if off_distances else 0
        mean_off_distance = np.mean(off_distances) if off_distances else 0

        critical_distances = [d for d in off_distances if d > 15.0]
        critical_rate = len(critical_distances) / len(is_off) if is_off else 0

        # Episode detection: consecutive off-road steps
        episodes = []
        current_episode = 0
        for off in is_off:
            if off:
                current_episode += 1
            elif current_episode > 0:
                episodes.append(current_episode)
                current_episode = 0
        if current_episode > 0:
            episodes.append(current_episode)

        return {
            'off_road_rate': off_road_rate,
            'max_off_road_distance_m': max_off_distance,
            'mean_off_road_distance_m': mean_off_distance,
            'critical_off_road_rate': critical_rate,
            'num_off_road_episodes': len(episodes),
            'mean_episode_duration_steps': np.mean(episodes) if episodes else 0,
            'total_points': len(is_on_taxiway),
            'valid_points': sum(is_on_taxiway),
            'per_point_is_off': is_off,
            'per_point_distances': distances,
        }

    def _get_empty_results(self, num_points: int = 0) -> Dict[str, Any]:
        """Return a zero-filled result dict when evaluation cannot run."""
        return {
            'off_road_rate': 0.0,
            'max_off_road_distance_m': 0.0,
            'mean_off_road_distance_m': 0.0,
            'critical_off_road_rate': 0.0,
            'num_off_road_episodes': 0,
            'mean_episode_duration_steps': 0.0,
            'total_points': num_points,
            'valid_points': 0,
            'per_point_is_off': [False] * num_points,
            'per_point_distances': [0.0] * num_points,
        }


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def evaluate_off_road(
    asset_dir: str,
    airport_code: str,
    batch: Dict[str, Any],
    ego_mu: torch.Tensor,
    ego_pred_scores: torch.Tensor,
    hist_len: int,
    safety_margin: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: constructs an OffRoadEvaluator and evaluates one batch.
    """
    evaluator = OffRoadEvaluator(asset_dir, airport_code)
    return evaluator.evaluate_prediction(
        batch, ego_mu, ego_pred_scores, hist_len, safety_margin
    )