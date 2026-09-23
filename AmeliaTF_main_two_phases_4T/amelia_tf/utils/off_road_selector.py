import os
import torch
import numpy as np
from typing import List, Tuple
from collections import deque

import geopandas as gpd
import shapely.geometry as sg
from geographiclib.geodesic import Geodesic

from amelia_tf.utils import global_masks as G
from amelia_tf.utils.modes import TURN_MODES_NAMES
from amelia_scenes.utils.transform_utils import (
    inv_transform_batch,
    direct_wrapper,
    direct_wrapper_batch,
)


class OffRoadSelector:
    """
    Post-processing module that selects the best trajectory hypothesis for each
    turn mode.

    Selection modes
    ---------------
    - 'phase' (NEW, default): Phase-aware *longitudinal* selection. Empirically
      the K hypotheses of a given mode differ ~2-4x more in their LONGITUDINAL
      extent (how far / how fast they travel) than in their LATERAL shape (which
      path they take). The lateral/path dimension is essentially solved by the
      model; the residual error is dominated by speed, which in airport ground
      ops is driven by *speed-phase transitions* (taxi -> runway = takeoff
      acceleration; runway -> taxi = landing deceleration). This mode reads the
      map zone (runway vs taxiway) + history speed to classify the phase and
      picks the hypothesis whose longitudinal extent matches that phase:
          runway_accel : pick the FARTHEST hypothesis (expect acceleration)
          runway_decel : pick the hypothesis that best CONTINUES history speed
          taxi         : pick the hypothesis that best CONTINUES history speed
      On a 25-sample dump this beat fixed-k=0 (0.043 -> 0.036 mean ADE) and beat
      the old heading-anchor selector (0.057), with the largest gain on runway
      takeoff samples (0.100 -> 0.079). It needs NO ground truth and is cheap
      (no BFS, no off-road geopandas pass).

    - 'ade_oracle': Select by minimum ADE to GT. Upper bound for any selector;
      uses ground truth, so for analysis/comparison only, NOT deployable.

    - 'random': Deterministic (seeded) uniform-random candidate per (sample,
      mode). Naive lower bound for analysis/comparison only, NOT deployable --
      contrasts against 'ade_oracle' (upper bound) and a learned scorer to show
      how much of the scorer's gain is genuine learned selection vs simply
      having more candidates to draw from regardless of which one is picked.

    - 'heading': Legacy heading-anchor + off-road combined score. Kept for
      comparison. (Known to underperform fixed-k=0 on this data; the anchor
      matching systematically prefers the most aggressively-turning hypothesis,
      which correlates with the WORST ADE.)

    Combined score for 'heading' (lower is better):
        score = heading_diff_deg + off_road_dist_epsilon * off_road_distance
                                  + off_road_epsilon * off_road_rate

    ------------------------------------------------------------------------------
    BUGFIX 1: Cumulative turn starting heading
    ------------------------------------------------------------------------------
    The candidate turn computation now prepends the starting heading (0 deg in rel
    frame) to the heading sequence. This correctly accounts for the turn from the
    initial heading to the first displacement step, which was previously missing.

    ------------------------------------------------------------------------------
    BUGFIX 2: ADE computation uses correct ego agent + absolute coordinates
    ------------------------------------------------------------------------------
    ADE now uses the correct ego agent ID (ego_agent_id[b]) and absolute
    coordinates (all_traj_abs) for predictions, compared against GT absolute
    coordinates from sequences. This fixes the coordinate-frame mismatch and
    agent selection bug that caused incorrect ADE values in debug output.

    ------------------------------------------------------------------------------
    BUGFIX 3: BFS cumulative turn accumulation (CRITICAL)
    ------------------------------------------------------------------------------
    cum_turn now accumulates step_turn (relative to previous heading) instead of
    edge_turn (relative to start_heading). For single-hop paths this is identical,
    but for multi-hop paths the previous code incorrectly accumulated
    (b1 - start) + (b2 - start) + (b3 - start) instead of
    (b1 - start) + (b2 - b1) + (b3 - b2) = b3 - start.

    ------------------------------------------------------------------------------
    BUGFIX 4: combined variable NameError in ade_oracle + debug mode
    ------------------------------------------------------------------------------
    combined is now defined in both branches of the selection_mode conditional,
    using a safe fallback (torch.zeros_like(heading_scores)) when not used.
    ------------------------------------------------------------------------------
    """

    # Mode index convention, matching TURN_MODES_NAMES / VALID_TURN_MODES
    MODE_TURN_LEFT = 0
    MODE_TURN_RIGHT = 1
    MODE_STRAIGHT = 2
    MODE_HOLD = 3

    # Maximum BFS steps to prevent infinite loops
    MAX_BFS_STEPS = 1000

    def __init__(
        self,
        evaluator,                       # OffRoadEvaluator instance (airport-specific)
        safety_margin: float = 1.0,
        min_search_radius_km: float = 0.25,   # floor: 250m, enough to reach junctions
        step_duration_s: float = 1.0,         # seconds per trajectory timestep
        heading_smooth_w: int = 5,            # smoothing window for cumulative turn
        off_road_epsilon: float = 10.0,       # weight of off_road_rate in combined score
        off_road_dist_epsilon: float = 0.1,   # weight of off_road_distance (metres)
        selection_mode: str = 'speed_sigma',   # 'learned' | 'speed_sigma' | 'phase' | 'ade_oracle' | 'random' | 'heading'
        # ---- phase-aware selection params ----
        runway_decel_speed_knots: float = 40.0,  # hist speed >= this on runway => landing rollout (decel)
        phase_forward_dist_km: float = 0.15,      # how far ahead to probe the map zone (~150 m)
        phase_zone_knn: int = 5,                  # # nearest nodes to inspect for a non-None zone
        # ---- speed_sigma selection params ----
        ss_speed_edges: tuple = (0.0, 3.0, 8.0, 15.0, 25.0, 40.0, 1e9),  # hist-speed bins (knots)
        ss_rank_table: dict = None,               # bin -> expected arc-rank; default set below
        ss_rank_window: int = 1,                  # consider base_rank +/- window, tie-break by sigma_end
        # ---- learned-scorer selection params ----
        scorer_path: str = None,                  # path to hyp_scorer.pkl (required for 'learned')
    ) -> None:
        self.evaluator = evaluator
        self.safety_margin = safety_margin
        self.geodesic = Geodesic.WGS84
        self.min_search_radius_km = min_search_radius_km
        self.step_duration_s = step_duration_s
        self.heading_smooth_w = heading_smooth_w
        self.off_road_epsilon = off_road_epsilon
        self.off_road_dist_epsilon = off_road_dist_epsilon
        self.selection_mode = selection_mode

        # phase-aware params
        self.runway_decel_speed_knots = runway_decel_speed_knots
        self.phase_forward_dist_km = phase_forward_dist_km
        self.phase_zone_knn = phase_zone_knn

        # speed_sigma params (data-derived on KMSY: history speed -> expected arc-rank,
        # then tie-break by model's end-step sigma within a +/- window).
        self.ss_speed_edges = list(ss_speed_edges)
        self.ss_rank_table = ss_rank_table if ss_rank_table is not None else {
            0: 1, 1: 2, 2: 2, 3: 2, 4: 3, 5: 3}
        self.ss_rank_window = ss_rank_window

        # learned-scorer: lazy-loaded HistGBT that predicts the oracle-by-ADE
        # hypothesis from closed-book geometric features. Trained offline by
        # train_scorer.py on the traj_model VAL-set dump.
        self.scorer_path = scorer_path
        self._scorer = None
        if self.selection_mode == 'learned':
            self._load_scorer()

        # Reference to the original graph from evaluator
        self.graph_nx = getattr(evaluator, 'graph_nx', None)

        # Lazy-loaded KD-tree for node lookup
        self._node_kdtree = None
        self._kdtree_node_ids = None
        self._kdtree_node_xys = None

    # ------------------------------------------------------------------
    # Graph utilities (directly on original graph)
    # ------------------------------------------------------------------

    def _ensure_kdtree(self) -> None:
        """Ensure KD-tree is built from original graph nodes."""
        if self._node_kdtree is not None:
            return

        from scipy.spatial import cKDTree

        node_ids = []
        node_xys = []

        for node in self.graph_nx.nodes():
            xy = self.evaluator._get_node_xy_from_graph(node)
            if xy is not None:
                node_ids.append(node)
                node_xys.append(xy)

        self._kdtree_node_ids = node_ids
        self._kdtree_node_xys = np.array(node_xys) if node_xys else np.empty((0, 2))

        if len(node_xys) > 0:
            self._node_kdtree = cKDTree(self._kdtree_node_xys)
        else:
            self._node_kdtree = None

    def _get_neighbors(self, node) -> set:
        """Get undirected neighbors of a node in the original graph."""
        if self.graph_nx is None:
            return set()
        return (set(self.graph_nx.successors(node)) |
                set(self.graph_nx.predecessors(node)))

    def _get_edge_bearing(self, from_node, to_node) -> float:
        """
        Get bearing from from_node to to_node in degrees (0=North, 90=East).
        Returns 0.0 if the edge does not exist.
        """
        if self.graph_nx is None:
            return 0.0

        if self.graph_nx.has_edge(from_node, to_node):
            data = list(self.graph_nx[from_node][to_node].values())[0]
            return data.get('bearing', 0.0)

        if self.graph_nx.has_edge(to_node, from_node):
            data = list(self.graph_nx[to_node][from_node].values())[0]
            return (data.get('bearing', 0.0) + 180.0) % 360.0

        return 0.0

    def _get_edge_length_km(self, from_node, to_node) -> float:
        """Get edge length in kilometers."""
        if self.graph_nx is None:
            return 0.0

        if self.graph_nx.has_edge(from_node, to_node):
            data = list(self.graph_nx[from_node][to_node].values())[0]
            return data.get('length', 0.0) / 1000.0

        if self.graph_nx.has_edge(to_node, from_node):
            data = list(self.graph_nx[to_node][from_node].values())[0]
            return data.get('length', 0.0) / 1000.0

        return 0.0

    def _is_dead_end(self, node) -> bool:
        """Check if a node is a dead-end (degree == 1)."""
        return len(self._get_neighbors(node)) == 1

    def _is_junction(self, node) -> bool:
        """Check if a node is a junction (degree >= 3)."""
        return len(self._get_neighbors(node)) >= 3

    # ------------------------------------------------------------------
    # Zone / phase utilities (NEW)
    # ------------------------------------------------------------------

    def _get_zone_at(self, xy: np.ndarray):
        """
        Return the 'zone' label of the graph node nearest to xy (local-km frame,
        same frame as start_abs / the KD-tree). Among the K nearest nodes, prefer
        the first whose zone is not None (a large fraction of nodes are unlabelled).
        Returns None if no labelled node is found.
        """
        self._ensure_kdtree()
        if self._node_kdtree is None or self.graph_nx is None:
            return None
        k = min(self.phase_zone_knn, len(self._kdtree_node_ids))
        if k <= 0:
            return None
        _, idx = self._node_kdtree.query(xy, k=k)
        for i in np.atleast_1d(idx):
            node = self._kdtree_node_ids[int(i)]
            z = self.graph_nx.nodes[node].get('zone')
            if z is not None:
                return z
        return None

    def _detect_phase(
        self,
        start_xy: np.ndarray,
        start_heading: float,
        mean_hist_speed_knots: float,
    ) -> str:
        """
        Classify the longitudinal speed phase from map zone + history speed.

        Logic:
          - on/approaching a runway zone (current or ~150 m ahead) AND low history
            speed  -> 'runway_accel' (takeoff roll: about to accelerate hard)
          - on a runway zone AND high history speed (>= runway_decel_speed_knots)
            -> 'runway_decel' (landing rollout: decelerating)
          - otherwise -> 'taxi' (history speed is a decent predictor)

        Uses only inference-available signals (zone + history). No ground truth.
        """
        cz = self._get_zone_at(start_xy)
        hr = np.radians(start_heading)
        # x = North component (cos), y = East component (sin) -- same convention
        # as _get_turn_anchor's forward projection.
        fwd = start_xy + self.phase_forward_dist_km * np.array([np.cos(hr), np.sin(hr)])
        fz = self._get_zone_at(fwd)

        on_runway = (cz == 'runway') or (fz == 'runway')
        if on_runway:
            if mean_hist_speed_knots >= self.runway_decel_speed_knots:
                return 'runway_decel'
            return 'runway_accel'
        return 'taxi'

    def _select_by_phase(
        self,
        ego_mu: torch.Tensor,           # (B, 1, T_total, M, K, D)
        hist_len: int,
        start_abs: np.ndarray,          # (B, 2) local-km
        start_heading: np.ndarray,      # (B,)
        hist_step_km: np.ndarray,       # (B,) mean per-step history displacement (km)
        mean_hist_speed_knots: np.ndarray,  # (B,)
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Phase-aware longitudinal hypothesis selection.

        For each (sample, mode) choose hypothesis k by longitudinal extent matched
        to the predicted speed phase:
          runway_accel : argmax future arc length  (expect acceleration -> far)
          runway_decel : argmin |future_per_step - history_per_step|  (continue/slow)
          taxi         : argmin |future_per_step - history_per_step|  (continue)

        Returns:
            selected_k_idx: (B, M) long tensor on CPU
            phases:         list of B phase strings (for logging)
        """
        B, A, T_total, M, K, D = ego_mu.shape
        T_pred = T_total - hist_len

        fut = ego_mu[:, 0, hist_len:, :, :, :2]                       # (B, T_pred, M, K, 2)
        step = torch.linalg.norm(fut[:, 1:] - fut[:, :-1], dim=-1)    # (B, T_pred-1, M, K)
        arc = step.sum(dim=1)                                         # (B, M, K) total path length
        mean_step = arc / max(T_pred - 1, 1)                          # (B, M, K) avg per-step speed

        arc_np = arc.detach().cpu().numpy()
        mean_step_np = mean_step.detach().cpu().numpy()

        selected = np.zeros((B, M), dtype=np.int64)
        phases: List[str] = []
        for b in range(B):
            phase = self._detect_phase(
                start_abs[b], float(start_heading[b]), float(mean_hist_speed_knots[b])
            )
            phases.append(phase)
            for m in range(M):
                if phase == 'runway_accel':
                    selected[b, m] = int(np.argmax(arc_np[b, m]))
                else:
                    # speed continuation: future per-step closest to history per-step
                    selected[b, m] = int(np.argmin(np.abs(mean_step_np[b, m] - hist_step_km[b])))

        return torch.from_numpy(selected).long(), phases

    def _select_by_speed_sigma(
        self,
        ego_mu: torch.Tensor,            # (B, 1, T_total, M, K, D)
        ego_sigma: torch.Tensor,         # (B, 1, T_total, M, K, D)
        hist_len: int,
        mean_hist_speed_knots: np.ndarray,   # (B,)
    ) -> torch.Tensor:
        """
        Speed-windowed, sigma-tie-broken longitudinal hypothesis selection.

        This is the best pure-rule selector found by offline analysis on KMSY.
        For each (sample, mode):
          1. rank the K hypotheses by future arc length (0=shortest .. K-1=longest);
          2. map history mean speed -> an expected arc-rank via ss_rank_table
             (slower history => pick a shorter hypothesis; faster => longer);
          3. candidate set = ranks {base-w, ..., base+w} clipped to [0, K-1];
          4. among candidates, pick the hypothesis with the SMALLEST end-step sigma
             (the model's own most-confident hypothesis within the plausible window).

        The speed window excludes implausible "how far" choices; model sigma then picks
        the most trustworthy hypothesis inside that window. Uses only inference-available
        signals (predicted trajectories, predicted sigma, history speed). No GT, no BFS,
        no off-road geopandas, no map zone.

        Returns:
            selected_k_idx: (B, M) long tensor on CPU.
        """
        B, A, T_total, M, K, D = ego_mu.shape

        fut = ego_mu[:, 0, hist_len:, :, :, :2]                       # (B, T_pred, M, K, 2)
        step = torch.linalg.norm(fut[:, 1:] - fut[:, :-1], dim=-1)    # (B, T_pred-1, M, K)
        arc = step.sum(dim=1)                                         # (B, M, K)
        # arc rank along K (0 = shortest)
        arc_rank = arc.argsort(dim=-1).argsort(dim=-1)                # (B, M, K)

        # end-step sigma magnitude per (B, M, K): model's own confidence at the horizon
        sig_end = torch.linalg.norm(ego_sigma[:, 0, -1, :, :, :2], dim=-1)  # (B, M, K)

        arc_rank_np = arc_rank.detach().cpu().numpy()
        sig_end_np = sig_end.detach().cpu().numpy()

        edges = np.asarray(self.ss_speed_edges)
        win = self.ss_rank_window
        selected = np.zeros((B, M), dtype=np.int64)
        for b in range(B):
            bin_id = int(np.digitize(mean_hist_speed_knots[b], edges) - 1)
            base = self.ss_rank_table.get(bin_id, K - 1)
            cand_ranks = [r for r in range(base - win, base + win + 1) if 0 <= r <= K - 1]
            for m in range(M):
                ks = [int(np.where(arc_rank_np[b, m] == r)[0][0]) for r in cand_ranks]
                selected[b, m] = ks[int(np.argmin([sig_end_np[b, m, k] for k in ks]))]

        return torch.from_numpy(selected).long()

    # ------------------------------------------------------------------
    # Learned hypothesis scorer (HistGBT trained offline on VAL dump)
    # ------------------------------------------------------------------

    def _load_scorer(self):
        """Lazy-load the pickled scorer payload (model + metadata)."""
        if self._scorer is not None:
            return
        if self.scorer_path is None:
            raise ValueError(
                "selection_mode='learned' requires scorer_path to point at a "
                "hyp_scorer.pkl produced by train_scorer.py.")
        import pickle
        with open(self.scorer_path, 'rb') as f:
            payload = pickle.load(f)
        self._scorer = payload
        # soft version check (sklearn pickles are version-sensitive)
        try:
            import sklearn
            if payload.get('sklearn_version') and payload['sklearn_version'] != sklearn.__version__:
                print(f"[OffRoadSelector/learned] WARNING: scorer trained with "
                      f"sklearn {payload['sklearn_version']}, runtime has "
                      f"{sklearn.__version__}. Predictions may differ; retrain if odd.")
        except Exception:
            pass

    def _select_by_learned(
        self,
        ego_mu: torch.Tensor,        # (B, 1, T_total, M, K, D)
        ego_sigma: torch.Tensor,     # (B, 1, T_total, M, K, D)
        seq_ego_np: np.ndarray,      # (B, T_total, 9) ego raw sequence
        hist_len: int,
    ) -> torch.Tensor:
        """
        Learned longitudinal selection: predict the oracle-by-ADE hypothesis from
        closed-book geometric features, per (sample, mode).

        Uses the SAME feature function as training (hyp_features.compute_hyp_features),
        so train/inference features never diverge. Returns (B, M) long tensor on CPU.
        """
        self._load_scorer()
        from amelia_tf.utils.hyp_features import compute_hyp_features

        B, A, T_total, M, K, D = ego_mu.shape
        mu_np = ego_mu[:, 0].detach().cpu().numpy()        # (B, T_total, M, K, D)
        sig_np = ego_sigma[:, 0].detach().cpu().numpy()    # (B, T_total, M, K, D)

        X = compute_hyp_features(mu_np, sig_np, seq_ego_np, hist_len)  # (B, M, F)
        clf = self._scorer['model']
        Xf = X.reshape(B * M, -1)

        # feature-count guard: catch silent feature drift between train & inference
        n_exp = self._scorer.get('n_features')
        if n_exp is not None and Xf.shape[1] != n_exp:
            raise ValueError(
                f"[learned] feature count mismatch: scorer expects {n_exp}, "
                f"got {Xf.shape[1]}. compute_hyp_features changed since training?")

        pred = clf.predict(Xf).reshape(B, M).astype(np.int64)
        # clip to valid hypothesis range, just in case
        pred = np.clip(pred, 0, K - 1)
        return torch.from_numpy(pred).long()

    # ------------------------------------------------------------------
    # BFS on original graph (used by 'heading' mode only)
    # ------------------------------------------------------------------

    def _bfs_on_original_graph(
        self,
        start_node: int,
        start_heading: float,
        mode_idx: int,
        remaining_budget_km: float,
        turn_to_forward: float,
        hist_cumulative_turn: float,
        turn_left_threshold: float,
        turn_right_threshold: float,
    ) -> List[float]:
        """
        Breadth-first search on the original graph to find all paths that
        reach the turn threshold. (Used only by 'heading' selection mode.)
        """
        queue = deque()
        queue.append((start_node, 0.0, 0.0, start_heading))

        visited = {start_node}
        crossing_buckets = {}

        step_count = 0

        while queue and step_count < self.MAX_BFS_STEPS:
            step_count += 1
            cur_node, dist_used, cum_turn, cur_heading = queue.popleft()

            for neighbor in self._get_neighbors(cur_node):
                if neighbor in visited:
                    continue

                out_bearing = self._get_edge_bearing(cur_node, neighbor)
                edge_length_km = self._get_edge_length_km(cur_node, neighbor)

                if edge_length_km <= 0:
                    continue

                step_turn = ((out_bearing - cur_heading + 180.0) % 360.0) - 180.0
                edge_turn = ((out_bearing - start_heading + 180.0) % 360.0) - 180.0

                # Reject sharp turns (>90 degrees). NOTE: this also rejects normal
                # ~90deg taxiway junctions, which is why turn anchors frequently
                # fall back to [0.0]. Kept as-is for 'heading' parity.
                if abs(step_turn) > 90.0:
                    continue

                new_dist = dist_used + edge_length_km
                if new_dist > remaining_budget_km:
                    continue

                new_cum_turn = cum_turn + step_turn
                full_turn = hist_cumulative_turn + turn_to_forward + new_cum_turn

                if mode_idx == self.MODE_TURN_LEFT:
                    if full_turn >= turn_left_threshold:
                        bucket = int(round(new_cum_turn / 10.0))
                        if bucket not in crossing_buckets or \
                                new_cum_turn < crossing_buckets[bucket]:
                            crossing_buckets[bucket] = new_cum_turn
                        continue
                else:  # MODE_TURN_RIGHT
                    if full_turn <= turn_right_threshold:
                        bucket = int(round(new_cum_turn / 10.0))
                        if bucket not in crossing_buckets or \
                                new_cum_turn > crossing_buckets[bucket]:
                            crossing_buckets[bucket] = new_cum_turn
                        continue

                visited.add(neighbor)
                queue.append((neighbor, new_dist, new_cum_turn, out_bearing))

        if crossing_buckets:
            return [a + turn_to_forward for a in crossing_buckets.values()]
        else:
            return None

    def _get_turn_anchor(
        self,
        start_pos: np.ndarray,
        start_heading: float,
        mode_idx: int,
        search_radius_km: float,
        hist_cumulative_turn: float = 0.0,
    ) -> List[float]:
        """Map-derived expected cumulative-turn anchors (used by 'heading' mode)."""
        if mode_idx in (self.MODE_HOLD, self.MODE_STRAIGHT):
            return [0.0]

        self._ensure_kdtree()

        if self.graph_nx is None or self._node_kdtree is None:
            return [0.0]

        turn_left_threshold = getattr(self.evaluator, 'turn_left_threshold', 25.0)
        turn_right_threshold = getattr(self.evaluator, 'turn_right_threshold', -25.0)

        heading_rad = np.radians(start_heading)
        projection_dist = min(search_radius_km * 0.3, 0.1)
        projected_xy = start_pos + projection_dist * np.array([
            np.cos(heading_rad),
            np.sin(heading_rad),
        ])

        dist, idx = self._node_kdtree.query(projected_xy, k=1)
        forward_node = self._kdtree_node_ids[idx]
        forward_node_xy = self._kdtree_node_xys[idx]
        forward_dist = float(dist)

        delta = forward_node_xy - start_pos
        bearing_to_forward = np.degrees(
            np.arctan2(delta[1], delta[0])
        ) % 360.0
        turn_to_forward = ((bearing_to_forward - start_heading + 180.0) % 360.0) - 180.0

        remaining_budget = max(
            search_radius_km - forward_dist,
            self.min_search_radius_km * 0.5,
        )
        initial_heading = (start_heading + turn_to_forward) % 360.0

        anchors = self._bfs_on_original_graph(
            start_node=forward_node,
            start_heading=initial_heading,
            mode_idx=mode_idx,
            remaining_budget_km=remaining_budget,
            turn_to_forward=turn_to_forward,
            hist_cumulative_turn=hist_cumulative_turn,
            turn_left_threshold=turn_left_threshold,
            turn_right_threshold=turn_right_threshold,
        )

        return anchors if anchors is not None else [0.0]

    # ------------------------------------------------------------------
    # Cumulative-turn computation (matches SceneProcessor)
    # ------------------------------------------------------------------

    def _compute_cumulative_turn(self, headings: np.ndarray) -> float:
        """Signed cumulative heading change, identical to SceneProcessor."""
        from scipy.ndimage import uniform_filter1d

        if len(headings) < 3:
            return 0.0

        hdg_rad = np.unwrap(np.deg2rad(headings.astype(float)))
        hdg_smooth = uniform_filter1d(hdg_rad, size=self.heading_smooth_w)
        hdg = np.rad2deg(hdg_smooth)

        total = 0.0
        for i in range(1, len(hdg)):
            step = ((hdg[i] - hdg[i - 1] + 180.0) % 360.0) - 180.0
            total += np.clip(step, -90.0, 90.0)
        return total

    def _cumulative_turn_from_xy(self, xy: np.ndarray) -> float:
        """Cumulative turn from a sequence of (x, y) positions."""
        if len(xy) < 2:
            return 0.0

        deltas = np.diff(xy, axis=0)
        step_len = np.linalg.norm(deltas, axis=-1)
        valid = step_len > 1e-4

        if valid.sum() < 1:
            return 0.0

        headings = np.degrees(np.arctan2(deltas[valid, 1], deltas[valid, 0]))
        headings_with_start = np.concatenate([[0.0], headings])
        return self._compute_cumulative_turn(headings_with_start)

    # ------------------------------------------------------------------
    # Vectorised off-road scoring (used by 'heading' mode / debug only)
    # ------------------------------------------------------------------

    def _score_all_off_road(
        self,
        all_lats: np.ndarray,
        all_lons: np.ndarray,
        safety_margin: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """off_road_rate and mean_off_road_distance for every (mode, hyp, sample)."""
        M, K, B, T_pred = all_lats.shape
        total_points = M * K * B * T_pred

        if self.evaluator.reference_gdf is None or total_points == 0:
            return (
                np.ones((M, K, B), dtype=np.float64),
                np.zeros((M, K, B), dtype=np.float64),
            )

        flat_lats = all_lats.ravel()
        flat_lons = all_lons.ravel()

        points_proj = gpd.GeoSeries(
            [sg.Point(lon, lat) for lon, lat in zip(flat_lons, flat_lats)],
            crs="EPSG:4326",
        ).to_crs("EPSG:3857")

        road_geoms = self.evaluator.reference_gdf_proj.geometry.values
        widths = self.evaluator.reference_gdf['width_m'].values

        distances = np.empty(total_points, dtype=np.float64)
        is_on_road = np.empty(total_points, dtype=bool)

        try:
            input_idx, tree_idx = self.evaluator.spatial_index.nearest(
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
                candidates_idx = list(self.evaluator.spatial_index.intersection(bbox))
                if not candidates_idx:
                    candidates_idx = list(range(len(self.evaluator.reference_gdf_proj)))
                cand_geoms = road_geoms[candidates_idx]
                cand_widths = widths[candidates_idx]
                dists = np.array([pt_proj.distance(g) for g in cand_geoms])
                best = dists.argmin()
                distances[i] = dists[best]
                is_on_road[i] = distances[i] <= (cand_widths[best] / 2.0) * safety_margin

        distances = distances.reshape(M, K, B, T_pred)
        is_on_road = is_on_road.reshape(M, K, B, T_pred)
        is_off = ~is_on_road

        off_road_rates = is_off.mean(axis=-1)

        masked_distances = np.where(is_off, distances, np.nan)
        with np.errstate(invalid='ignore'):
            off_road_distances = np.nanmean(masked_distances, axis=-1)
        off_road_distances = np.nan_to_num(off_road_distances, nan=0.0)

        return off_road_rates, off_road_distances

    # ------------------------------------------------------------------
    # Debug printing  (unchanged)
    # ------------------------------------------------------------------

    def _print_selection_debug(
        self, B, M, K, expected_turn, candidate_turns, matched_anchor,
        direction_penalties, heading_scores, off_road_rates, off_road_distances,
        combined, selected_k_idx, ade_all=None, start_headings=None,
        hist_cumulative_turns=None, search_radii=None, gt_labels=None,
        mode_scores=None, turn_feasibility=None, selection_mode='heading',
        max_samples=25,
    ) -> None:
        heading_np = heading_scores.cpu().numpy()
        rate_np = off_road_rates.cpu().numpy()
        dist_np = off_road_distances.cpu().numpy()
        combined_np = combined.cpu().numpy()
        selected_np = selected_k_idx.cpu().numpy()

        print("\n" + "-" * 130)
        print(f"Per-(mode, hypothesis) selection breakdown (mode: {selection_mode})")
        print("-" * 130)

        n_show = min(B, max_samples)
        for b in range(n_show):
            print(f"\n{'='*130}")
            print(f"Sample {b}:")
            print(f"{'='*130}")

            info_parts = []
            if start_headings is not None:
                info_parts.append(f"heading={start_headings[b]:+6.1f}deg")
            if hist_cumulative_turns is not None:
                info_parts.append(f"hist_turn={hist_cumulative_turns[b]:+7.2f}deg")
            if search_radii is not None:
                info_parts.append(f"search_radius={search_radii[b]*1000:6.1f}m")
            if gt_labels is not None:
                gt_label_val = gt_labels[b]
                if isinstance(gt_label_val, (np.ndarray, list)):
                    gt_label_str = str(gt_label_val.tolist()) if hasattr(gt_label_val, 'tolist') else str(gt_label_val)
                else:
                    gt_label_str = str(gt_label_val)
                info_parts.append(f"GT={gt_label_str}")
            if info_parts:
                print("  " + "  ".join(info_parts))

            if turn_feasibility is not None:
                feas = turn_feasibility[b]
                feas_str = "  ".join(f"{TURN_MODES_NAMES[m]}={feas[m]:.3f}" for m in range(M))
                print(f"  turn_feasibility: {feas_str}")

            if mode_scores is not None:
                scores_str = "  ".join(f"{TURN_MODES_NAMES[m]}={mode_scores[b, m]:.3f}" for m in range(M))
                print(f"  mode_scores: {scores_str}")

            anchors_row = "  anchors: " + "  ".join(
                f"{TURN_MODES_NAMES[m]}="
                + ("[" + ", ".join(f"{a:+.1f}" for a in expected_turn[b][m]) + "]"
                   if expected_turn[b][m] is not None else "None") + "deg"
                for m in range(M)
            )
            print(anchors_row)

            for m in range(M):
                mode_name = TURN_MODES_NAMES[m]
                chosen_k = int(selected_np[b, m])
                anchors_str = ("[" + ", ".join(f"{a:+.1f}" for a in expected_turn[b][m]) + "]"
                               if expected_turn[b][m] is not None else "None")
                print(f"\n  Mode {mode_name}  (map anchors: {anchors_str}deg)")

                if ade_all is not None:
                    print(f"    {'k':>3} {'cand_turn':>10} {'matched_anchor':>14} "
                          f"{'min_dist':>9} {'dir_pen':>8} {'hdg_score':>10} "
                          f"{'off_rate':>9} {'off_dist_m':>11} {'ADE':>10} {'combined':>10}  picked")
                else:
                    print(f"    {'k':>3} {'cand_turn':>10} {'matched_anchor':>14} "
                          f"{'min_dist':>9} {'dir_pen':>8} {'hdg_score':>10} "
                          f"{'off_rate':>9} {'off_dist_m':>11} {'combined':>10}  picked")

                for k in range(K):
                    marker = "  <==" if k == chosen_k else ""
                    min_dist = heading_np[b, m, k] - direction_penalties[b, m, k]
                    if ade_all is not None:
                        ade_val = ade_all[b, m, k]
                        print(f"    {k:>3} {candidate_turns[b, m, k]:>+10.2f} "
                              f"{matched_anchor[b, m, k]:>+14.2f} {min_dist:>9.2f} "
                              f"{direction_penalties[b, m, k]:>8.2f} {heading_np[b, m, k]:>10.2f} "
                              f"{rate_np[b, m, k]:>9.3f} {dist_np[b, m, k]:>11.2f} "
                              f"{ade_val:>10.4f} {combined_np[b, m, k]:>10.4f}{marker}")
                    else:
                        print(f"    {k:>3} {candidate_turns[b, m, k]:>+10.2f} "
                              f"{matched_anchor[b, m, k]:>+14.2f} {min_dist:>9.2f} "
                              f"{direction_penalties[b, m, k]:>8.2f} {heading_np[b, m, k]:>10.2f} "
                              f"{rate_np[b, m, k]:>9.3f} {dist_np[b, m, k]:>11.2f} "
                              f"{combined_np[b, m, k]:>10.4f}{marker}")

        if B > n_show:
            print(f"\n... ({B - n_show} more samples not shown)")

        heading_best = heading_np.argmin(axis=-1)
        offroad_best = rate_np.argmin(axis=-1)
        agree = (heading_best == offroad_best)
        print("\n" + "-" * 130)
        print(f"Heading-best vs off-road-best agreement: "
              f"{agree.sum()}/{agree.size} ({agree.mean():.1%})")
        print("-" * 130 + "\n")

    def _print_ade_comparison_debug(self, B, M, K, all_traj_abs, gt_future, selected_k_idx) -> None:
        ade_all = np.zeros((B, M, K), dtype=np.float64)
        for b in range(B):
            gt = gt_future[b]
            for m in range(M):
                for k in range(K):
                    ade_all[b, m, k] = np.mean(np.linalg.norm(all_traj_abs[m, k, b] - gt, axis=-1))

        best_ade_k = ade_all.argmin(axis=-1)
        best_ade_val = ade_all.min(axis=-1)
        sel_k = selected_k_idx.cpu().numpy()
        sel_ade = np.array([[ade_all[b, m, sel_k[b, m]] for m in range(M)] for b in range(B)])

        print("\n" + "=" * 80)
        print("ADE COMPARISON: Selector vs ADE-Optimal (absolute coordinates)")
        print("=" * 80)
        mode_names = ['TurnLeft', 'TurnRight', 'Straight', 'Hold']
        for m in range(M):
            match = (sel_k[:, m] == best_ade_k[:, m]).sum()
            print(f"\n  {mode_names[m]}:")
            print(f"    Match with ADE-optimal: {match}/{B} ({match/B*100:.1f}%)")
            print(f"    Mean ADE (selector):   {sel_ade[:, m].mean():.6f} km")
            print(f"    Mean ADE (optimal):    {best_ade_val[:, m].mean():.6f} km")
            print(f"    Mean gap:              {(sel_ade[:, m]-best_ade_val[:, m]).mean():.6f} km")

        total_match = (sel_k == best_ade_k).sum()
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Overall match rate: {total_match}/{B*M} ({total_match/(B*M)*100:.1f}%)")
        print(f"Overall mean ADE (selector): {sel_ade.mean():.6f} km")
        print(f"Overall mean ADE (optimal):  {best_ade_val.mean():.6f} km")
        print(f"Overall mean gap:            {sel_ade.mean()-best_ade_val.mean():.6f} km")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_best_hypothesis(
        self,
        batch: dict,
        ego_mu: torch.Tensor,           # (B, 1, T_total, M, K, D)
        ego_probs: torch.Tensor,        # (B, 1, M)
        hist_len: int,
        ego_sigma: torch.Tensor = None,  # (B, 1, T_total, M, K, D); required for 'speed_sigma'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each (sample, mode), pick the hypothesis k* based on selection_mode.

        Returns:
            selected_mu:    (B, 1, T_total, M, D) best hypothesis per mode.
            selected_k_idx: (B, M) index of chosen hypothesis per (sample, mode).
        """
        B, A, T_total, M, K, D = ego_mu.shape
        assert A == 1, "OffRoadSelector only handles the ego agent (A must be 1)."

        # Fast path: K == 1, nothing to select
        if K == 1:
            selected_mu = ego_mu.squeeze(4)
            selected_k_idx = torch.zeros(B, M, dtype=torch.long, device=ego_mu.device)
            return selected_mu, selected_k_idx

        debug_on = os.environ.get("DEBUG_OFFROAD_CHECK") == "1"

        # What does the current mode actually require?
        need_heading = (self.selection_mode == 'heading') or debug_on
        need_offroad = (self.selection_mode == 'heading') or debug_on
        need_abs = (self.selection_mode in ('ade_oracle', 'heading')) or debug_on

        # ------------------------------------------------------------------
        # Collect per-sample start poses / history from the batch
        # ------------------------------------------------------------------
        scene = batch['scene_dict']
        sequences = scene['sequences']
        # ego id: prefer the test-time key, fall back to the train-time key.
        # (On the val split used for scorer dumping, only 'ego_agent_id' may exist;
        #  the two are identical at eval time.)
        ego_agents = scene.get('ego_agent_id', scene.get('ego_agent_id'))
        if ego_agents is None:
            raise KeyError("scene_dict has neither 'ego_agent_id' nor 'ego_agent_id'.")

        ego_agent_ids = [
            ego_agents[b].item() if torch.is_tensor(ego_agents) else int(ego_agents[b])
            for b in range(B)
        ]
        airport_codes = [scene['airport_id'][b] for b in range(B)]
        same_airport = len(set(airport_codes)) == 1

        start_abs = np.stack([
            sequences[b, ego_agent_ids[b], hist_len - 1, G.XY].detach().cpu().numpy().flatten()
            for b in range(B)
        ], axis=0)                                                          # (B, 2)

        start_heading = np.array([
            float(sequences[b, ego_agent_ids[b], hist_len - 1, G.HD].detach().cpu().numpy())
            for b in range(B)
        ])                                                                  # (B,)

        T_pred = T_total - hist_len

        # History per-step displacement (km) and mean speed (knots) -- needed by phase.
        hist_step_km = np.zeros(B, dtype=np.float64)
        mean_hist_speed_knots = np.zeros(B, dtype=np.float64)
        for b in range(B):
            hxy = sequences[b, ego_agent_ids[b], :hist_len, G.XY].detach().cpu().numpy()
            dstep = np.linalg.norm(np.diff(hxy, axis=0), axis=1)
            hist_step_km[b] = float(dstep.mean()) if dstep.size else 0.0
            hist_speed_knots = sequences[b, ego_agent_ids[b], :hist_len, G.SEQ_IDX.Speed].detach().cpu().numpy()
            mean_hist_speed_knots[b] = float(np.mean(hist_speed_knots))

        KNOTS_TO_KMS = 0.000514444
        sample_search_radius_km = np.maximum(
            mean_hist_speed_knots * KNOTS_TO_KMS * T_pred * self.step_duration_s,
            self.min_search_radius_km,
        )

        # Pre-allocate score containers (kept zero unless filled).
        heading_scores = torch.full((B, M, K), float('inf'))
        off_road_rates = torch.zeros((B, M, K))
        off_road_distances = torch.zeros((B, M, K))
        all_traj_abs = np.zeros((M, K, B, T_pred, 2), dtype=np.float64)

        # ------------------------------------------------------------------
        # History cumulative turn + map anchors (heading mode / debug only)
        # ------------------------------------------------------------------
        hist_cumulative_turns = np.zeros(B, dtype=np.float64)
        expected_turn = [[None] * M for _ in range(B)]
        candidate_turns = np.zeros((B, M, K), dtype=np.float64) if debug_on else None
        matched_anchor = np.zeros((B, M, K), dtype=np.float64) if debug_on else None
        direction_penalties = np.zeros((B, M, K), dtype=np.float64) if debug_on else None

        if need_heading:
            for b in range(B):
                hist_headings = sequences[b, ego_agent_ids[b], :hist_len, G.HD].detach().cpu().numpy()
                hist_cumulative_turns[b] = self._compute_cumulative_turn(hist_headings)
                for m in range(M):
                    expected_turn[b][m] = self._get_turn_anchor(
                        start_pos=start_abs[b],
                        start_heading=start_heading[b],
                        mode_idx=m,
                        search_radius_km=sample_search_radius_km[b],
                        hist_cumulative_turn=hist_cumulative_turns[b],
                    )

        # ------------------------------------------------------------------
        # Absolute trajectories (+ heading scores + lat/lon) when required
        # ------------------------------------------------------------------
        if need_abs or need_heading or need_offroad:
            tl_thr = getattr(self.evaluator, 'turn_left_threshold', 25.0)
            tr_thr = getattr(self.evaluator, 'turn_right_threshold', -25.0)
            refs = [self.evaluator._get_ref_for_airport(c) for c in airport_codes]
            all_lats = np.empty((M, K, B, T_pred), dtype=np.float64) if need_offroad else None
            all_lons = np.empty((M, K, B, T_pred), dtype=np.float64) if need_offroad else None

            for m in range(M):
                for k in range(K):
                    future_rel = ego_mu[:, 0, hist_len:, m, k, :2].detach().cpu().numpy()
                    traj_xy_abs = inv_transform_batch(future_rel, start_abs, start_heading)
                    all_traj_abs[m, k] = traj_xy_abs

                    if need_heading:
                        for b in range(B):
                            candidate_turn = self._cumulative_turn_from_xy(future_rel[b])
                            anchors = expected_turn[b][m]
                            if anchors is not None:
                                diffs = [abs(candidate_turn - a) for a in anchors]
                                min_idx = int(np.argmin(diffs))
                                min_dist = diffs[min_idx]
                            else:
                                min_idx, min_dist = 0, 0.0

                            full_cand_turn = candidate_turn + hist_cumulative_turns[b]
                            if m == self.MODE_TURN_LEFT:
                                if full_cand_turn < 0:
                                    dir_penalty = max(abs(full_cand_turn) * 2.0, tl_thr)
                                elif full_cand_turn < tl_thr:
                                    dir_penalty = tl_thr - full_cand_turn
                                else:
                                    dir_penalty = 0.0
                            elif m == self.MODE_TURN_RIGHT:
                                if full_cand_turn > 0:
                                    dir_penalty = max(abs(full_cand_turn) * 2.0, abs(tr_thr))
                                elif full_cand_turn > tr_thr:
                                    dir_penalty = full_cand_turn - tr_thr
                                else:
                                    dir_penalty = 0.0
                            else:
                                dir_penalty = 0.0

                            heading_scores[b, m, k] = min_dist + dir_penalty
                            if debug_on:
                                candidate_turns[b, m, k] = candidate_turn
                                matched_anchor[b, m, k] = anchors[min_idx] if anchors is not None else float('nan')
                                direction_penalties[b, m, k] = dir_penalty

                    if need_offroad:
                        x = traj_xy_abs[:, :, 0]
                        y = traj_xy_abs[:, :, 1]
                        rang = np.sqrt(x ** 2 + y ** 2)
                        bearing = np.degrees(np.arctan2(y, x))
                        if same_airport:
                            lats_mk, lons_mk = direct_wrapper_batch(
                                self.geodesic, bearing, rang, refs[0][0], refs[0][1], refs[0][2])
                        else:
                            lats_mk = np.empty((B, T_pred), dtype=np.float64)
                            lons_mk = np.empty((B, T_pred), dtype=np.float64)
                            for b in range(B):
                                lb, lo = direct_wrapper(
                                    self.geodesic, bearing[b], rang[b], refs[b][0], refs[b][1], refs[b][2])
                                lats_mk[b] = lb
                                lons_mk[b] = lo
                        all_lats[m, k] = lats_mk
                        all_lons[m, k] = lons_mk

            if need_offroad:
                orr, ord_ = self._score_all_off_road(all_lats, all_lons, self.safety_margin)
                off_road_rates = torch.from_numpy(orr).permute(2, 0, 1).float()
                off_road_distances = torch.from_numpy(ord_).permute(2, 0, 1).float()

        # GT absolute trajectories (ade_oracle / debug only)
        gt_future_abs = None
        if self.selection_mode == 'ade_oracle' or debug_on:
            gt_future_abs = np.zeros((B, T_pred, 2), dtype=np.float64)
            for b in range(B):
                gt_future_abs[b] = sequences[b, ego_agent_ids[b], hist_len:, G.XY].detach().cpu().numpy()

        # ------------------------------------------------------------------
        # Selection
        # ------------------------------------------------------------------
        phases = None
        combined_for_debug = torch.zeros_like(heading_scores)

        if self.selection_mode == 'learned':
            if ego_sigma is None:
                raise ValueError(
                    "selection_mode='learned' requires ego_sigma to be passed to "
                    "select_best_hypothesis(...). Pass ego_sigma=ego_sigma at the "
                    "test_step call site (still 6-D full-K there).")
            # ego row of the raw sequence: (B, T_total, 9)
            seq_ego_np = np.stack([
                sequences[b, ego_agent_ids[b]].detach().cpu().numpy()
                for b in range(B)
            ], axis=0)
            selected_k_idx = self._select_by_learned(
                ego_mu, ego_sigma, seq_ego_np, hist_len)

        elif self.selection_mode == 'speed_sigma':
            if ego_sigma is None:
                raise ValueError(
                    "selection_mode='speed_sigma' requires ego_sigma to be passed "
                    "to select_best_hypothesis(...). At the test_step call site, "
                    "ego_sigma is still 6-D full-K (B,1,T,M,K,D); pass it as "
                    "ego_sigma=ego_sigma.")
            selected_k_idx = self._select_by_speed_sigma(
                ego_mu, ego_sigma, hist_len, mean_hist_speed_knots)

        elif self.selection_mode == 'phase':
            selected_k_idx, phases = self._select_by_phase(
                ego_mu, hist_len, start_abs, start_heading,
                hist_step_km, mean_hist_speed_knots)

        elif self.selection_mode == 'ade_oracle':
            ade_all_oracle = np.zeros((B, M, K), dtype=np.float64)
            for b in range(B):
                gt = gt_future_abs[b]
                for m in range(M):
                    for k in range(K):
                        ade_all_oracle[b, m, k] = np.mean(
                            np.linalg.norm(all_traj_abs[m, k, b] - gt, axis=-1))
            selected_k_idx = torch.from_numpy(ade_all_oracle.argmin(axis=-1)).long()

        elif self.selection_mode == 'random':
            rng = np.random.default_rng(42)
            selected_k_idx = torch.from_numpy(rng.integers(0, K, size=(B, M))).long()

        elif self.selection_mode == 'heading':
            combined = (heading_scores
                        + self.off_road_dist_epsilon * off_road_distances
                        + self.off_road_epsilon * off_road_rates)
            selected_k_idx = combined.argmin(dim=-1)
            combined_for_debug = combined

        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode!r}. "
                             f"Expected 'learned', 'speed_sigma', 'phase', 'ade_oracle', "
                             f"'random', or 'heading'.")

        selected_k_idx = selected_k_idx.long()

        # Optional phase logging
        if phases is not None and os.environ.get("DEBUG_PHASE_CHECK") == "1":
            from collections import Counter
            print(f"[OffRoadSelector/phase] phase distribution: {dict(Counter(phases))}")

        # ------------------------------------------------------------------
        # Debug dumps (need heading/off-road/ADE all populated -> debug_on path)
        # ------------------------------------------------------------------
        if debug_on:
            ade_all = np.zeros((B, M, K), dtype=np.float64)
            for b in range(B):
                gt = gt_future_abs[b]
                for m in range(M):
                    for k in range(K):
                        ade_all[b, m, k] = np.mean(np.linalg.norm(all_traj_abs[m, k, b] - gt, axis=-1))

            gt_labels = scene.get('mode_labels', None)
            if gt_labels is not None and torch.is_tensor(gt_labels):
                if gt_labels.dim() == 3:
                    tmp = []
                    for b in range(B):
                        eid = ego_agent_ids[b]
                        tmp.append(gt_labels[b, eid].detach().cpu().numpy())
                    gt_labels = tmp
                else:
                    gt_labels = gt_labels.detach().cpu().numpy()
            elif isinstance(gt_labels, np.ndarray):
                gt_labels = gt_labels.tolist()

            turn_feasibility = scene.get('turn_feasibility', None)
            if turn_feasibility is not None and torch.is_tensor(turn_feasibility):
                feas_list = []
                for b in range(B):
                    eid = ego_agent_ids[b]
                    feas_list.append(turn_feasibility[b, eid].detach().cpu().numpy())
                turn_feasibility = np.array(feas_list)

            mode_scores = ego_probs[:, 0, :].detach().cpu().numpy()

            self._print_selection_debug(
                B=B, M=M, K=K, expected_turn=expected_turn,
                candidate_turns=candidate_turns, matched_anchor=matched_anchor,
                direction_penalties=direction_penalties, heading_scores=heading_scores,
                off_road_rates=off_road_rates, off_road_distances=off_road_distances,
                combined=combined_for_debug, selected_k_idx=selected_k_idx,
                ade_all=ade_all, start_headings=start_heading,
                hist_cumulative_turns=hist_cumulative_turns,
                search_radii=sample_search_radius_km, gt_labels=gt_labels,
                mode_scores=mode_scores, turn_feasibility=turn_feasibility,
                selection_mode=self.selection_mode)

            self._print_ade_comparison_debug(
                B=B, M=M, K=K, all_traj_abs=all_traj_abs,
                gt_future=gt_future_abs, selected_k_idx=selected_k_idx)

        selected_k_idx = selected_k_idx.to(ego_mu.device)

        # ------------------------------------------------------------------
        # Gather the winning hypothesis
        # ------------------------------------------------------------------
        idx_exp = selected_k_idx[:, None, None, :, None, None].expand(
            B, 1, T_total, M, 1, D)
        selected_mu = ego_mu.gather(4, idx_exp).squeeze(4)

        return selected_mu, selected_k_idx