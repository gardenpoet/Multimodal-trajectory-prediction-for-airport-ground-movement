import os
import json
import torch
import numpy as np
import pickle
import shapely.geometry as sg
import geopandas as gpd
from typing import List, Tuple, Dict, Any
from geographiclib.geodesic import Geodesic

from amelia_tf.utils import global_masks as G
from amelia_scenes.utils.transform_utils import xy_to_ll


class OffRoadEvaluator:
    """
    Off-road detection evaluator for predicted trajectories.
    Uses Amelia's semantic_graph.pkl as reference network.

    node_type semantics (confirmed from pkl map_infos):
        1 = hold_line
        3 = runway  (zone='runway')
        4 = taxiway
    """

    def __init__(self, asset_dir: str, airport_code: str):
        """
        Args:
            asset_dir:    Path to assets directory (contains per-airport subdirs).
            airport_code: ICAO airport code (e.g. 'kbos').
        """
        self.asset_dir = asset_dir
        self.airport_code = airport_code
        self.geodesic = Geodesic.WGS84

        self.limits_path = os.path.join(asset_dir, airport_code, 'limits.json')
        self.pkl_path    = os.path.join(asset_dir, airport_code, 'semantic_graph.pkl')

        self._load_reference_network()
        self._load_reference_point()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_reference_network(self):
        """Load semantic_graph.pkl and build a spatially-indexed GeoDataFrame."""
        if not os.path.exists(self.pkl_path):
            print(f"Warning: PKL file not found at {self.pkl_path}")
            self.reference_gdf = None
            return

        print(f"Loading reference network from: {self.pkl_path}")
        with open(self.pkl_path, 'rb') as f:
            amelia_data = pickle.load(f)

        G_nx = amelia_data['graph_networkx']

        edges      = []
        geometries = []

        for u, v, data in G_nx.edges(data=True):
            u_data = G_nx.nodes[u]
            v_data = G_nx.nodes[v]

            if 'x' not in u_data or 'y' not in u_data:
                continue

            line = sg.LineString([(u_data['x'], u_data['y']),
                                  (v_data['x'], v_data['y'])])
            geometries.append(line)

            edges.append({
                'u':          u,
                'v':          v,
                # node_type: 1=hold_line, 3=runway, 4=taxiway
                'node_type_u': u_data.get('node_type', 4),
                'node_type_v': v_data.get('node_type', 4),
                # explicit width (metres) from OSM data, may be None
                'width':       data.get('width'),
                'length':      data.get('length', 0),
                'bearing':     data.get('bearing', 0),
            })

        self.reference_gdf = gpd.GeoDataFrame(edges, geometry=geometries, crs="EPSG:4326")
        self.reference_gdf_proj = self.reference_gdf.to_crs("EPSG:3857")
        self.spatial_index = self.reference_gdf_proj.geometry.sindex

        # FAA standard widths as fallback when no explicit width is available
        DEFAULT_RUNWAY_WIDTH   = 45.0   # typical commercial runway
        DEFAULT_TAXIWAY_WIDTH  = 23.0   # FAA ADG-V: 75 ft ˜ 23 m
        DEFAULT_HOLD_LINE_WIDTH = 5.0   # hold-line marking

        def get_width(row):
            # Priority 1: explicit width encoded in the OSM edge
            if row['width'] is not None:
                try:
                    return float(row['width'])
                except (ValueError, TypeError):
                    pass
            # Priority 2: infer from node_type
            # Use the higher-priority type of the two endpoints
            node_type = max(row['node_type_u'], row['node_type_v'])
            if node_type == 3:    # runway
                return DEFAULT_RUNWAY_WIDTH
            elif node_type == 4:  # taxiway
                return DEFAULT_TAXIWAY_WIDTH
            elif node_type == 1:  # hold_line
                return DEFAULT_HOLD_LINE_WIDTH
            else:
                return DEFAULT_TAXIWAY_WIDTH

        self.reference_gdf['width_m'] = self.reference_gdf.apply(get_width, axis=1)

        print(f"Loaded {len(self.reference_gdf)} reference segments from PKL")
        for node_type, label in [(3, 'runway'), (4, 'taxiway'), (1, 'hold_line')]:
            mask = ((self.reference_gdf['node_type_u'] == node_type) |
                    (self.reference_gdf['node_type_v'] == node_type))
            count = mask.sum()
            if count > 0:
                w = self.reference_gdf.loc[mask, 'width_m'].mean()
                print(f"  node_type {node_type} ({label}): {count} edges, mean_width={w:.1f}m")

    def _load_reference_point(self):
        """Load lat/lon reference point and range scale from limits.json."""
        if not os.path.exists(self.limits_path):
            print(f"Warning: limits.json not found at {self.limits_path}")
            self.ref = None
            return

        with open(self.limits_path, 'r') as f:
            ref_data = json.load(f)

        self.ref = (ref_data['ref_lat'], ref_data['ref_lon'], ref_data['range_scale'])
        print(f"Loaded reference point: lat={self.ref[0]}, lon={self.ref[1]}")

    def _get_ref_for_airport(self, airport_code: str) -> Tuple[float, float, float]:
        """Return (ref_lat, ref_lon, range_scale) for the given airport."""
        limits_path = os.path.join(self.asset_dir, airport_code, 'limits.json')
        with open(limits_path, 'r') as f:
            ref_data = json.load(f)
        return (ref_data['ref_lat'], ref_data['ref_lon'], ref_data['range_scale'])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_prediction(self,
                            ego_mu: torch.Tensor,
                            ego_pred_scores: torch.Tensor,
                            sequences: torch.Tensor,
                            ego_agents,
                            airport_codes: List[str],
                            hist_len: int,
                            safety_margin: float = 1.0) -> List[Dict[str, Any]]:
        """
        Evaluate trajectories against the airport reference network.

        The caller is responsible for passing sequences, ego_agents, and
        airport_codes that are already aligned with ego_mu - no index
        remapping is performed here.

        Args:
            ego_mu:         (B, 1, T_total, M, D) predicted trajectories, or
                            (B, 1, T_total, 1, D) for a single GT trajectory.
            ego_pred_scores:(B, 1, M) mode probabilities.
                            Pass torch.ones(B, 1, 1) when evaluating GT.
            sequences:      (B, A, T, D) absolute-coordinate sequences for
                            exactly the B samples in ego_mu.
            ego_agents:     (B,) ego agent index for each of the B samples.
            airport_codes:  list of length B with the ICAO code per sample.
            hist_len:       number of history timesteps.
            safety_margin:  corridor half-width multiplier (1.0 = strict).

        Returns:
            List of per-sample result dicts (length B).
        """
        B = ego_mu.shape[0]

        if self.reference_gdf is None:
            return [self._get_empty_results(0) for _ in range(B)]

        all_results = []

        for b in range(B):
            ego_agent = (ego_agents[b].item()
                         if torch.is_tensor(ego_agents) else ego_agents[b])

            best_mode  = ego_pred_scores[b, 0].argmax().item()
            T_total    = ego_mu.shape[2]
            T_pred     = T_total - hist_len
            future_rel = ego_mu[b, 0, hist_len:, best_mode, :2]        # (T_pred, 2)
            future_rel_reshaped = future_rel.unsqueeze(0)               # (1, T_pred, 2)

            seq_sample    = sequences[b]                                # (A, T, D)
            start_abs     = seq_sample[ego_agent, hist_len - 1, G.XY].detach().cpu().numpy()
            start_heading = seq_sample[ego_agent, hist_len - 1, G.HD].detach().cpu().numpy()

            if start_abs.shape != (2,):
                start_abs = start_abs.flatten()
            if hasattr(start_heading, 'shape') and len(start_heading.shape) > 0:
                start_heading = start_heading.item()

            ref      = self._get_ref_for_airport(airport_codes[b])
            traj_ll  = xy_to_ll(future_rel_reshaped, start_abs, start_heading,
                                 ref, self.geodesic)                    # (1, T_pred, 2)

            lats = traj_ll[0, :, 0].cpu().numpy()
            lons = traj_ll[0, :, 1].cpu().numpy()
            trajectory_ll = [(lons[t], lats[t]) for t in range(T_pred)]

            result = self._evaluate_trajectory(trajectory_ll, safety_margin)
            all_results.append(result)

        return all_results

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate_trajectory(self,
                              trajectory_ll: List[Tuple[float, float]],
                              safety_margin: float = 1.0) -> Dict[str, Any]:
        """
        Check each point in trajectory_ll against the reference network.

        Args:
            trajectory_ll: list of (lon, lat) points.
            safety_margin: multiplier on half-width; >1.0 allows overshoot.

        Returns:
            Dict with off-road metrics.
        """
        if self.reference_gdf is None or len(trajectory_ll) == 0:
            return self._get_empty_results(len(trajectory_ll))

        results   = []
        distances = []

        for lon, lat in trajectory_ll:
            pt      = sg.Point(lon, lat)
            pt_proj = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs("EPSG:3857")[0]

            bbox           = pt_proj.bounds
            candidates_idx = list(self.spatial_index.intersection(bbox))
            candidates     = (self.reference_gdf_proj.iloc[candidates_idx]
                              if candidates_idx else self.reference_gdf_proj)

            dists       = candidates.geometry.distance(pt_proj)
            min_idx     = dists.idxmin()
            min_dist    = dists[min_idx]
            road_width  = self.reference_gdf.loc[min_idx, 'width_m']
            max_allowed = (road_width / 2.0) * safety_margin

            results.append(min_dist <= max_allowed)
            distances.append(min_dist)

        return self._compute_metrics(results, distances)

    def _compute_metrics(self,
                         is_on_road: List[bool],
                         distances: List[float]) -> Dict[str, Any]:
        """Aggregate per-point on/off-road flags into summary metrics."""
        is_off        = [not r for r in is_on_road]
        off_distances = [d for d, off in zip(distances, is_off) if off]

        n             = len(is_off)
        off_road_rate = sum(is_off) / n if n > 0 else 0.0

        max_off_dist  = max(off_distances)  if off_distances else 0.0
        mean_off_dist = float(np.mean(off_distances)) if off_distances else 0.0

        # Critical: distance > 15 m off the nearest road edge
        critical_distances = [d for d in off_distances if d > 15.0]
        critical_rate      = len(critical_distances) / n if n > 0 else 0.0

        # Episode detection (consecutive off-road points)
        episodes        = []
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
            'off_road_rate':             off_road_rate,
            'max_off_road_distance_m':   max_off_dist,
            'mean_off_road_distance_m':  mean_off_dist,
            'critical_off_road_rate':    critical_rate,
            'num_off_road_episodes':     len(episodes),
            'mean_episode_duration_steps': float(np.mean(episodes)) if episodes else 0.0,
            'total_points':              n,
            'valid_points':              sum(is_on_road),
            'per_point_is_off':          is_off,
            'per_point_distances':       distances,
        }

    def _get_empty_results(self, num_points: int = 0) -> Dict[str, Any]:
        """Return zeroed results when evaluation cannot be performed."""
        return {
            'off_road_rate':             0.0,
            'max_off_road_distance_m':   0.0,
            'mean_off_road_distance_m':  0.0,
            'critical_off_road_rate':    0.0,
            'num_off_road_episodes':     0,
            'mean_episode_duration_steps': 0.0,
            'total_points':              num_points,
            'valid_points':              0,
            'per_point_is_off':          [False] * num_points,
            'per_point_distances':       [0.0]   * num_points,
        }


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def evaluate_off_road(asset_dir: str,
                      airport_code: str,
                      ego_mu: torch.Tensor,
                      ego_pred_scores: torch.Tensor,
                      sequences: torch.Tensor,
                      ego_agents,
                      airport_codes: List[str],
                      hist_len: int,
                      safety_margin: float = 1.0) -> List[Dict[str, Any]]:
    """
    Convenience wrapper around OffRoadEvaluator.evaluate_prediction.

    Args:
        asset_dir:     Path to assets directory.
        airport_code:  ICAO code used to load the evaluator.
        ego_mu:        (B, 1, T_total, M, D).
        ego_pred_scores: (B, 1, M).
        sequences:     (B, A, T, D) absolute coordinates, aligned with ego_mu.
        ego_agents:    (B,) ego agent indices, aligned with ego_mu.
        airport_codes: list[B] ICAO codes, aligned with ego_mu.
        hist_len:      History length.
        safety_margin: Corridor half-width multiplier.

    Returns:
        List of per-sample result dicts.
    """
    evaluator = OffRoadEvaluator(asset_dir, airport_code)
    return evaluator.evaluate_prediction(
        ego_mu=ego_mu,
        ego_pred_scores=ego_pred_scores,
        sequences=sequences,
        ego_agents=ego_agents,
        airport_codes=airport_codes,
        hist_len=hist_len,
        safety_margin=safety_margin,
    )