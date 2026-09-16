import os
import json
import math
import pickle
import random
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
import amelia_scenes.utils.common as C
import amelia_scenes.utils.dataset as D
import amelia_scenes.utils.global_masks as G

from tqdm import tqdm
from easydict import EasyDict
from typing import Tuple, List, Optional
from joblib import Parallel, delayed
from collections import deque

from amelia_scenes.scoring.kinematic import compute_kinematic_scores
from amelia_scenes.scoring.critical import compute_simple_scene_critical
from amelia_scenes.scoring.interactive import compute_interactive_scores
from amelia_scenes.scoring.crowdedness import compute_simple_scene_crowdedness


class SceneProcessor:
    """Dataset class for pre-processing airport surface movement data into scenes."""

    # Cap on BFS expansions, guarding against pathological graphs
    MAX_BFS_STEPS = 1000

    def __init__(self, config: EasyDict) -> None:
        super(SceneProcessor, self).__init__()

        self.airport = config.airport
        self.in_data_dir = os.path.join(config.in_data_dir, self.airport)
        self.out_data_dir = os.path.join(config.out_data_dir, self.airport)
        os.makedirs(self.out_data_dir, exist_ok=True)
        self.blacklist_dir = os.path.join(config.out_data_dir, 'blacklist')
        os.makedirs(self.blacklist_dir, exist_ok=True)
        self.out_data_summary_dir = os.path.join(config.out_summary_dir, self.airport)
        os.makedirs(self.out_data_summary_dir, exist_ok=True)

        self.parallel        = config.parallel
        self.overwrite       = config.overwrite
        self.add_scores_meta = config.add_scores_meta

        self.seed            = config.seed
        self.extend          = True
        self.seq_extent      = 1

        self.pred_lens        = config.pred_lens
        self.pred_len         = max(self.pred_lens)
        self.hist_len         = config.hist_len
        self.seq_len          = self.hist_len + self.pred_len
        self.skip             = config.skip
        self.min_agents       = config.min_agents
        self.max_agents       = config.max_agents
        self.min_valid_points = config.min_valid_points
        self.n_jobs           = config.jobs

        self.turn_label_method = 'trajectory'

        # ------------------------------------------------------------------
        # Pre-computed thresholds switch
        # If True, use hard-coded thresholds for specific airports to save time
        # ------------------------------------------------------------------
        self.use_precomputed_thresholds = getattr(config, 'use_precomputed_thresholds', False)

        # ------------------------------------------------------------------
        # Runway filtering: agents taxiing on a runway are dropped entirely.
        # threshold in km (0.033 km = 33 m ~ runway half-width).
        # ------------------------------------------------------------------
        self.filter_runway     = getattr(config, 'filter_runway', True)
        self.runway_thr_km     = getattr(config, 'runway_thr_km', 0.033)
        # An aircraft counts as taxiing ALONG a runway only if it spends at
        # least this many points in the runway band AND is heading roughly
        # parallel to it. Crossings are kept (see _agent_on_runway).
        self.runway_min_pts    = getattr(config, 'runway_min_pts', 5)
        self.runway_par_deg    = getattr(config, 'runway_parallel_deg', 30.0)
        self.runway_par_frac   = getattr(config, 'runway_parallel_frac', 0.5)
        # Runway edges are excluded from the turn-feasibility BFS: the dataset
        # has no aircraft taxiing along a runway, so a path that turns onto one
        # is not a manoeuvre any retained aircraft performs. Verified safe on
        # this map: every runway/taxiway intersection carries >= 2 taxiway
        # edges, so crossings stay traversable after the removal.
        self.exclude_runway_edges = getattr(config, 'exclude_runway_edges', True)

        limits_file = os.path.join(config.assets_dir, self.airport, 'limits.json')
        with open(limits_file, 'r') as fp:
            self.ref_data = EasyDict(json.load(fp))

        # Load graph data
        graph_data_dir      = os.path.join(config.graph_data_dir, self.airport)
        print(f"Loading graph data from: {graph_data_dir}")
        pickle_map_filepath = os.path.join(graph_data_dir, "semantic_graph.pkl")
        with open(pickle_map_filepath, 'rb') as f:
            graph_pickle       = pickle.load(f)
            self.hold_lines    = graph_pickle['hold_lines'][:, 2:4]
            self.graph_nx      = graph_pickle['graph_networkx']
            self.all_polylines = graph_pickle['map_infos']['all_polylines']

        # ------------------------------------------------------------------
        # Runway segments (polyline type == 3, i.e. thr_id).
        # Verified: these are the only polylines with exactly two orientations
        # (~15 deg and ~105 deg), matching the two runways of the airport.
        # Columns [2,3] = start (x, y) and [6,7] = end (x, y), projected km,
        # with x = North and y = East.
        # ------------------------------------------------------------------
        _rw = self.all_polylines[self.all_polylines[:, 8] == 3]
        self._runway_seg = _rw[:, [2, 3, 6, 7]].astype(np.float64)   # (M,4)
        # compass bearing of every runway segment, for the parallel test
        _dx = self._runway_seg[:, 2] - self._runway_seg[:, 0]
        _dy = self._runway_seg[:, 3] - self._runway_seg[:, 1]
        self._runway_bear = np.degrees(np.arctan2(_dy, _dx)) % 360.0   # (M,)
        print(f"Runway segments (polyline type=3): {len(self._runway_seg)} "
              f"| filter_runway={self.filter_runway} thr={self.runway_thr_km*1000:.0f} m")

        # ------------------------------------------------------------------
        # Deviation filtering: drop agents whose MAXIMUM deviation from the
        # taxiway centreline exceeds a per-mode threshold (option B, strict:
        # any single point over the mode's mean deviation drops the whole
        # trajectory). Keeps only trajectories that hug a centreline, to raise
        # data precision. Uses the projected x,y (km) frame, same as runway
        # filtering. Thresholds are the per-mode mean deviations from the prior
        # diagnostic (km; range_scale 1000 so 1 local-xy unit = 1 km, i.e. the
        # diagnostic mean in local-xy units is already in km). Taxiway polylines
        # are type == 4, columns [2,3]/[6,7] = start/end (x, y) projected km.
        # ------------------------------------------------------------------
        _tw = self.all_polylines[self.all_polylines[:, 8] == 4]
        self._taxiway_seg = _tw[:, [2, 3, 6, 7]].astype(np.float64)   # (M,4)
        self.filter_deviation = getattr(config, 'filter_deviation', True)
        _default_dev = {'TurnLeft': 0.0137, 'TurnRight': 0.0100,
                        'Straight': 0.0137, 'Hold': 0.0127}
        self.deviation_thr_km = dict(getattr(config, 'deviation_thr_km',
                                             _default_dev))
        print(f"Taxiway segments (polyline type=4): {len(self._taxiway_seg)} "
              f"| filter_deviation={self.filter_deviation} "
              f"thr_km={self.deviation_thr_km}")

        # osmid -> (u_node, v_node) mapping
        self._osmid_to_edge = {
            int(data['osmid']): (u, v)
            for u, v, data in self.graph_nx.edges(data=True)
        }

        # Type of every graph edge, taken from the type of its polyline
        # (1 = hold line, 3 = runway, 4 = taxiway).
        self._build_edge_types()

        # Projected (x, y) km position of every graph node. The node 'x'/'y'
        # attributes hold lon/lat, not the projected frame the trajectories
        # live in, so positions are taken from the incident edge polylines.
        self._build_node_xy_lookup()

        # KD-tree over ALL original graph nodes (see _build_node_kdtree).
        self._build_node_kdtree()

        self.blacklist = []
        blacklist_file = os.path.join(self.blacklist_dir, f"{self.airport}.txt")
        if os.path.exists(blacklist_file) and not self.overwrite:
            with open(blacklist_file, 'r') as f:
                self.blacklist = f.read().splitlines()

        file_list = os.listdir(self.in_data_dir)
        for duplicate in list(set(self.blacklist) & set(file_list)):
            file_list.remove(duplicate)
        self.data_files = [
            os.path.join(self.in_data_dir, f)
            for f in file_list if f.endswith('.csv')
        ]
        random.seed(self.seed)
        random.shuffle(self.data_files)
        self.data_files = self.data_files[
            :int(len(self.data_files) * config.perc_process)]

        # Compute speed/acceleration thresholds from trajectory data.
        # Also computes turn thresholds via GMM on prediction-segment
        # cumulative heading changes (completely label-free).
        self._compute_global_thresholds()

    # ------------------------------------------------------------------
    # Runway filtering
    # ------------------------------------------------------------------

    def _agent_on_runway(self, agent_seq: np.ndarray,
                         pad_front: int, pad_end: int) -> bool:
        """Return True if the agent taxis ALONG a runway.

        Only aircraft travelling along a runway are dropped. They line up,
        accelerate for take-off or decelerate after landing, so their speed
        varies far more than normal taxiing and their future is much harder to
        predict. An altitude mask cannot catch them: taxiing on a runway keeps
        an almost constant altitude, so only the position and heading relative
        to the runway geometry can.

        Aircraft merely CROSSING a runway between taxiways are kept. At a busy
        airport such as KBOS the runways criss-cross the whole field and 17.7%
        of the taxiway network itself lies within the runway band, so dropping
        every trajectory that touches a runway would remove ~46% of the data,
        almost all of it ordinary taxiing.

        Two conditions must both hold:
          - at least `runway_min_pts` points lie within `runway_thr_km` of a
            runway centreline (a perpendicular crossing spends only a handful
            of timesteps in that band), and
          - among those points, a fraction of at least `runway_parallel_frac`
            have a heading within `runway_parallel_deg` of that runway
            (a crossing is roughly perpendicular, not parallel).

        Distances use the projected x,y frame (km), so the threshold is a true
        metric distance. Fully vectorised: (T points) x (M runway segments).
        """
        if not self.filter_runway or len(self._runway_seg) == 0:
            return False

        P = np.stack([
            agent_seq[pad_front:pad_end, G.RAW_IDX.x].astype(float),
            agent_seq[pad_front:pad_end, G.RAW_IDX.y].astype(float),
        ], axis=-1)                                       # (T,2)

        a   = self._runway_seg[:, 0:2]                    # (M,2) segment start
        b   = self._runway_seg[:, 2:4]                    # (M,2) segment end
        ab  = b - a                                       # (M,2)
        ab2 = np.clip((ab * ab).sum(1), 1e-12, None)      # (M,)

        ap   = P[:, None, :] - a[None]                    # (T,M,2)
        t    = np.clip((ap * ab[None]).sum(-1) / ab2[None], 0.0, 1.0)   # (T,M)
        proj = a[None] + t[..., None] * ab[None]          # (T,M,2)
        d    = np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1))           # (T,M)

        dmin = d.min(axis=1)                              # (T,) nearest runway
        near = dmin < self.runway_thr_km                  # (T,) inside the band
        if int(near.sum()) < self.runway_min_pts:
            return False                                  # barely touches it

        # heading of the in-band points vs the runway they are nearest to
        seg_idx = d.argmin(axis=1)[near]                              # (n,)
        rw_bear = self._runway_bear[seg_idx]                          # (n,)
        hdg     = agent_seq[pad_front:pad_end,
                            G.RAW_IDX.Heading].astype(float)[near]    # (n,)

        # undirected angle: taxiing either way along the runway counts
        off = np.abs((hdg - rw_bear + 180.0) % 360.0 - 180.0)         # (n,) 0..180
        off = np.minimum(off, 180.0 - off)                            # 0..90
        parallel_frac = float((off < self.runway_par_deg).mean())

        return parallel_frac >= self.runway_par_frac

    def _agent_deviates(self, agent_seq: np.ndarray,
                        pad_front: int, pad_end: int,
                        turn_label: str) -> bool:
        """Return True if the agent's MAXIMUM deviation from the taxiway
        centreline exceeds its per-mode threshold (option B: any single point
        over the mode's mean deviation drops the whole trajectory).

        Distances use the projected x,y frame (km), matching runway filtering.
        Fully vectorised: (T points) x (M taxiway segments).
        """
        if not self.filter_deviation or len(self._taxiway_seg) == 0:
            return False
        thr = self.deviation_thr_km.get(turn_label, None)
        if thr is None:
            return False

        P = np.stack([
            agent_seq[pad_front:pad_end, G.RAW_IDX.x].astype(float),
            agent_seq[pad_front:pad_end, G.RAW_IDX.y].astype(float),
        ], axis=-1)                                       # (T,2)

        a   = self._taxiway_seg[:, 0:2]                   # (M,2) segment start
        b   = self._taxiway_seg[:, 2:4]                   # (M,2) segment end
        ab  = b - a                                       # (M,2)
        ab2 = np.clip((ab * ab).sum(1), 1e-12, None)      # (M,)

        ap   = P[:, None, :] - a[None]                    # (T,M,2)
        t    = np.clip((ap * ab[None]).sum(-1) / ab2[None], 0.0, 1.0)   # (T,M)
        proj = a[None] + t[..., None] * ab[None]          # (T,M,2)
        d    = np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1))           # (T,M)

        dmin = d.min(axis=1)                              # (T,) nearest taxiway
        return float(dmin.max()) > thr                    # max deviation over thr

    # ------------------------------------------------------------------
    # Bidirectional graph helpers
    # ------------------------------------------------------------------

    def _build_edge_types(self) -> None:
        """Record the polyline type of every graph edge.

        Types follow the polyline convention: 1 = hold line, 3 = runway
        (thr_id), 4 = taxiway. Verified on this map by orientation: the type-3
        segments align with exactly the runway bearings, while type-4 spreads
        over every heading.
        """
        oid2type = {int(r[9]): int(r[8]) for r in self.all_polylines}
        self._edge_type = {}
        for u, v, data in self.graph_nx.edges(data=True):
            self._edge_type[(u, v)] = oid2type.get(int(data['osmid']))
        n_rw = sum(1 for t in self._edge_type.values() if t == 3)
        print(f"Edge types indexed: {len(self._edge_type)} edges "
              f"({n_rw} runway) | exclude_runway_edges={self.exclude_runway_edges}")

    def _is_runway_edge(self, u, v) -> bool:
        """True if the edge between u and v runs along a runway."""
        t = self._edge_type.get((u, v))
        if t is None:
            t = self._edge_type.get((v, u))
        return t == 3

    def _node_touches_taxiway(self, node) -> bool:
        """True if the node has at least one non-runway edge."""
        for nb in self._get_neighbors_undirected(node):
            if not self._is_runway_edge(node, nb):
                return True
        return False

    def _build_node_xy_lookup(self) -> None:
        """Map every graph node to its projected (x, y) position in km.

        The node attributes carry lon/lat, which is not the frame the agent
        trajectories use, so the position is read from the polyline of an
        incident edge: the polyline of edge (u, v) starts at u and ends at v.
        Built once in O(edges) rather than rescanning all polylines per node.
        """
        ids = self.all_polylines[:, 9].astype(int)
        first_row, last_row = {}, {}
        for i, oid in enumerate(ids):
            oid = int(oid)
            if oid not in first_row:
                first_row[oid] = i
            last_row[oid] = i

        self._node_xy = {}
        for u, v, data in self.graph_nx.edges(data=True):
            oid = int(data['osmid'])
            if oid in first_row:
                self._node_xy.setdefault(u, self.all_polylines[first_row[oid], [2, 3]])
            if oid in last_row:
                self._node_xy.setdefault(v, self.all_polylines[last_row[oid], [6, 7]])
        print(f"Node xy lookup: {len(self._node_xy)}/"
              f"{self.graph_nx.number_of_nodes()} nodes positioned")

    def _get_edge_length_km(self, from_node, to_node) -> float:
        """Length (km) of the edge between two nodes, trying both directions.

        The graph stores 'length' in metres on every edge, so no geometry is
        recomputed here.
        """
        if self.graph_nx.has_edge(from_node, to_node):
            d = list(self.graph_nx[from_node][to_node].values())[0]
            return float(d.get('length', 0.0)) / 1000.0
        if self.graph_nx.has_edge(to_node, from_node):
            d = list(self.graph_nx[to_node][from_node].values())[0]
            return float(d.get('length', 0.0)) / 1000.0
        return 0.0

    def _get_neighbors_undirected(self, node) -> set:
        """Return unique neighbours of a node, merging both edge directions.

        The airport graph stores every road as two directed edges (A->B and
        B->A). Merging successors and predecessors gives the true undirected
        neighbour set, which is needed to correctly identify intermediate nodes
        (degree 2) vs real junctions (degree >= 3).
        """
        return (set(self.graph_nx.successors(node)) |
                set(self.graph_nx.predecessors(node)))

    def _is_junction(self, node) -> bool:
        """Return True if node is a real fork/merge point (3+ unique neighbours)."""
        return len(self._get_neighbors_undirected(node)) >= 3

    def _get_edge_bearing(self, from_node, to_node, prev_bearing=None):
        """Return the travel bearing from from_node to to_node.

        If the forward edge does not exist the reverse edge bearing + 180 is
        used. If prev_bearing is provided, any candidate that differs from it
        by more than 150 is rejected (aircraft cannot reverse abruptly).
        """
        if self.graph_nx.has_edge(from_node, to_node):
            b = list(self.graph_nx[from_node][to_node].values())[0]['bearing']
        elif self.graph_nx.has_edge(to_node, from_node):
            b = (list(self.graph_nx[to_node][from_node].values())[0]['bearing']
                 + 180.0) % 360.0
        else:
            return None

        if prev_bearing is not None:
            if abs(self._signed_bearing_diff(b, prev_bearing)) > 150.0:
                return None
        return b

    def _build_node_kdtree(self):
        """Build a KD-tree over ALL original graph nodes.

        The simplified graph (collapsing degree-2 chains into semantic edges)
        is deliberately NOT used. Collapsing leaves only the junctions and
        dead-ends (~12% of the nodes) in the tree, so for an aircraft sitting
        mid-chain the nearest junction frequently belongs to a DIFFERENT chain
        (measured: ~31% of positions), which makes the forward endpoint and
        therefore the whole BFS unreliable. Searching the original graph node
        by node, as the rule-based selector does, avoids this entirely.
        """
        node_ids, node_xys = [], []
        n_rw_only = 0
        for node in self.graph_nx.nodes():
            xy = self._node_xy.get(node)
            if xy is None:
                continue
            # A node reachable only along a runway is never a valid forward
            # node once runway edges are excluded: the BFS would start there
            # with nothing to expand. Crossing points keep their taxiway edges
            # and so remain in the tree.
            if self.exclude_runway_edges and not self._node_touches_taxiway(node):
                n_rw_only += 1
                continue
            node_ids.append(node)
            node_xys.append(xy)

        self._kdtree_node_ids = node_ids
        self._kdtree_node_xys = np.array(node_xys) if node_xys else np.empty((0, 2))
        self._node_kdtree = cKDTree(self._kdtree_node_xys) if node_xys else None

        print(f"Node KD-tree: {len(node_ids)} nodes indexed "
              f"(full graph, {n_rw_only} runway-only nodes excluded).")

    # ------------------------------------------------------------------
    # Threshold computation
    # ------------------------------------------------------------------

    @staticmethod
    def _signed_bearing_diff(b_new: float, b_old: float) -> float:
        """Signed angular difference in (-180, 180], wrapping-safe."""
        return ((b_new - b_old + 180.0) % 360.0) - 180.0

    @staticmethod
    def _xy_to_bearing(dx: float, dy: float) -> float:
        """Compass bearing (deg) of the vector (dx, dy) in the projected frame.

        In this projection x = North and y = East, so the compass bearing
        (measured from North, clockwise towards East) is atan2(East, North)
        = atan2(dy, dx). Verified against the 'bearing' attribute stored on
        the graph edges, which matches this convention exactly.
        """
        return math.degrees(math.atan2(dy, dx)) % 360.0

    @staticmethod
    def _compute_cumulative_turn(headings: np.ndarray,
                                 smooth_w: int = 5) -> float:
        """Cumulative signed heading change over a heading sequence.

        Headings are unwrapped in radians before smoothing so that 0/360
        crossings are handled correctly (e.g. 358->2 is +4, not -356).
        Steps are clamped to [-90, 90] to suppress GPS noise spikes.
        """
        if len(headings) < 3:
            return 0.0
        hdg_rad    = np.unwrap(np.deg2rad(headings.astype(float)))
        hdg_smooth = uniform_filter1d(hdg_rad, size=smooth_w)
        hdg        = np.rad2deg(hdg_smooth)
        total = 0.0
        for i in range(1, len(hdg)):
            step   = SceneProcessor._signed_bearing_diff(hdg[i], hdg[i - 1])
            total += np.clip(step, -90.0, 90.0)
        return total

    def _fit_gmm_thresholds(self, pred_turns: np.ndarray):
        """Derive turn thresholds from prediction-segment turn distribution.

        Fits GMM with n=1,2,3 components (BIC-selected) to the signed
        cumulative turn distribution. Thresholds are set at midpoints between
        component means.

        Single-peak fallback: threshold = max(3 * sigma_straight, 25 deg).

        Parameters
        ----------
        pred_turns : np.ndarray
            Signed cumulative turn angles over prediction segments (degrees).

        Returns
        -------
        turn_left_threshold  : float  (positive, degrees)
        turn_right_threshold : float  (negative, degrees)
        """
        X = pred_turns.reshape(-1, 1)

        # Fit GMM for n = 1, 2, 3 and select by BIC
        best_bic = np.inf
        best_gmm = None
        best_n   = 1
        for n in [1, 2, 3]:
            try:
                gmm = GaussianMixture(n_components=n, random_state=42,
                                      max_iter=300)
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    best_n   = n
            except Exception:
                continue

        print(f"\n[GMM turn threshold] best n_components={best_n}  BIC={best_bic:.1f}")

        # Sort components by mean (needed before fallback definition)
        means   = best_gmm.means_.flatten()
        stds    = np.sqrt(best_gmm.covariances_.flatten())
        weights = best_gmm.weights_.flatten()
        order   = np.argsort(means)
        means_s = means[order]
        stds_s  = stds[order]

        # Straight component: the one with the smallest std
        straight_comp = int(np.argmin(stds_s))

        # Fallback: use the std of the GMM straight component to estimate
        # straight-segment heading noise.
        def _mad_fallback():
            straight_std = float(stds_s[straight_comp])
            thr          = max(3.0 * straight_std, 25.0)   # minimum 25 deg floor
            print(f"  Fallback (GMM straight std): "
                  f"sigma_straight={straight_std:.1f} degrees  threshold=+/-{thr:.1f} degrees")
            return thr, -thr

        if best_n == 1 or best_gmm is None:
            tl, tr = _mad_fallback()
            print(f"  turn_left_threshold  = +{tl:.1f} degrees")
            print(f"  turn_right_threshold = {tr:.1f} degrees")
            self._plot_gmm_fit(
                pred_turns, means_s, stds_s, weights[order],
                tl, tr, best_n,
                out_path=f'gmm_threshold_{self.airport}_pred{self.pred_len}.png')
            return tl, tr

        # -- Component quality checks (pseudocode: 80 deg, 25 deg) ----------
        # A genuine turn cluster should be compact (std < MAX_TURN_STD).
        # If any non-straight component has an excessively large std it is
        # just fitting the long tail of a unimodal distribution.
        # Also, a turn component must be sufficiently far from straight.
        MAX_TURN_STD = 80.0      # degrees; genuine turn clusters are tighter
        MIN_TURN_SEP = 25.0      # degrees; turn component must be at least 25 deg from straight

        # Check every non-straight component for quality
        poor_quality = False
        for ci in range(len(means_s)):
            if ci == straight_comp:
                continue
            if stds_s[ci] > MAX_TURN_STD:
                print(f"  Component {ci} has std={stds_s[ci]:.1f} degrees > "
                      f"{MAX_TURN_STD} degrees -> not a genuine turn cluster")
                poor_quality = True
                break
            if abs(means_s[ci] - means_s[straight_comp]) < MIN_TURN_SEP:
                print(f"  Component {ci} has mean={means_s[ci]:.1f} degrees, "
                      f"too close to straight component (sep={abs(means_s[ci] - means_s[straight_comp]):.1f} < {MIN_TURN_SEP})")
                poor_quality = True
                break

        if poor_quality:
            for i, (m, s, w) in enumerate(zip(
                    means_s, stds_s, weights[order])):
                print(f"  component {i}: mean={m:+.1f} degrees  std={s:.1f} degrees  "
                      f"weight={w:.2f}")
            tl, tr = _mad_fallback()
            print(f"  turn_left_threshold  = +{tl:.1f} degrees")
            print(f"  turn_right_threshold = {tr:.1f} degrees")
            self._plot_gmm_fit(
                pred_turns, means_s, stds_s, weights[order],
                tl, tr, best_n,
                out_path=f'gmm_threshold_{self.airport}_pred{self.pred_len}.png')
            return tl, tr

        # -- Compute thresholds as midpoints between component means ----------
        if best_n == 2:
            # Two components: one negative (right-turn) and one positive
            # (straight or left-turn), or straight + left.
            # Use midpoint between means.
            mu_mid = (means_s[0] + means_s[1]) / 2.0
            
            if mu_mid >= 0:
                # Components are both on positive side -> use midpoint for left,
                # fallback for right
                tl = mu_mid
                _, tr = _mad_fallback()
            elif mu_mid < 0:
                # Components are both on negative side -> use midpoint for right,
                # fallback for left
                tr = mu_mid
                tl, _ = _mad_fallback()
            else:
                # One positive, one negative -> symmetric thresholds
                tl = abs(mu_mid)
                tr = -abs(mu_mid)

            # Ensure left threshold is positive, right is negative
            tl = abs(tl)
            tr = -abs(tr)

        else:   # best_n == 3: left-turn / straight / right-turn
            # Thresholds at midpoints between adjacent means
            tr = (means_s[0] + means_s[1]) / 2.0   # between right-turn and straight
            tl = (means_s[1] + means_s[2]) / 2.0   # between straight and left-turn
            
            # Ensure left threshold is positive, right is negative
            tl = abs(tl)
            tr = -abs(tr)

            # If thresholds are too small, fallback
            if tl < 25.0 or abs(tr) < 25.0:
                print(f"  Thresholds too small (tl={tl:.1f}, tr={tr:.1f}), falling back")
                tl, tr = _mad_fallback()

        for i, (m, s, w) in enumerate(zip(
                means_s, stds_s, weights[order])):
            print(f"  component {i}: mean={m:+.1f} degrees  std={s:.1f} degrees  weight={w:.2f}")
        print(f"  turn_left_threshold  = +{tl:.1f} degrees")
        print(f"  turn_right_threshold = {tr:.1f} degrees")

        # Save GMM fit visualisation
        self._plot_gmm_fit(
            pred_turns, means_s, stds_s, weights[order],
            tl, tr, best_n,
            out_path=f'gmm_threshold_{self.airport}_pred{self.pred_len}.png')

        return float(tl), float(tr)

    def _plot_gmm_fit(
            self,
            pred_turns: np.ndarray,
            means_s: np.ndarray,
            stds_s: np.ndarray,
            weights_s: np.ndarray,
            tl: float,
            tr: float,
            best_n: int,
            out_path: str = 'gmm_threshold.png',
    ) -> None:
        """Plot the GMM fit over the full-sequence turn angle distribution.

        Shows the empirical histogram, each Gaussian component, the total
        mixture density, and the derived turn/straight thresholds.

        Parameters
        ----------
        pred_turns : np.ndarray  raw cumulative turn angles (degrees)
        means_s    : sorted component means
        stds_s     : sorted component stds
        weights_s  : sorted component weights
        tl, tr     : turn_left / turn_right thresholds (positive / negative)
        best_n     : number of GMM components selected by BIC
        out_path   : output image file path
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0D0D1A')

        COLORS = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']
        bins   = np.linspace(-180, 180, 73)   # 5 degrees bins
        x_plot = np.linspace(-180, 180, 1000)

        for ax in axes:
            ax.set_facecolor('#1A1A2E')
            ax.tick_params(colors='white', labelsize=8)
            ax.set_xlabel('Cumulative turn (degrees)', color='white', fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444466')

        # -- Left panel: signed distribution ------------------------------
        ax0 = axes[0]
        ax0.hist(np.clip(pred_turns, -180, 180), bins=bins,
                 color='#4A90D9', edgecolor='none', alpha=0.55,
                 density=True, label='Empirical')

        pdf_total = np.zeros_like(x_plot)
        for i, (m, s, w) in enumerate(zip(means_s, stds_s, weights_s)):
            pdf_i     = w * scipy_norm.pdf(x_plot, m, s)
            pdf_total += pdf_i
            label_i   = (f'Component {i}'
                         f' mu={m:+.1f} degrees, s={s:.1f} degrees, w={w:.2f}')
            ax0.plot(x_plot, pdf_i, color=COLORS[i % len(COLORS)],
                     lw=1.8, ls='--', label=label_i)

        ax0.plot(x_plot, pdf_total, color='white', lw=2.0, ls='-',
                 label='GMM total')

        # Threshold lines
        ax0.axvline( tl, color='#F39C12', lw=2.0, ls=':',
                    label=f'TurnLeft  ={tl:+.1f} degrees')
        ax0.axvline( tr, color='#F39C12', lw=2.0, ls=':',
                    label=f'TurnRight ={tr:+.1f} degrees')
        ax0.axvline(0,  color='white',   lw=0.8, alpha=0.4)

        ax0.set_xlim(-180, 180)
        ax0.set_ylabel('Density', color='white', fontsize=9)
        ax0.set_title(f'GMM fit (n={best_n})  -  signed turn distribution',
                      color='white', fontsize=10, pad=8)
        ax0.legend(facecolor='#1A1A2E', edgecolor='#444466',
                   labelcolor='white', fontsize=7, loc='upper right')

        # -- Right panel: absolute |turn| distribution ---------------------
        ax1 = axes[1]
        abs_turns = np.abs(pred_turns)
        bins_abs  = np.linspace(0, 180, 37)
        ax1.hist(np.clip(abs_turns, 0, 180), bins=bins_abs,
                 color='#4A90D9', edgecolor='none', alpha=0.55,
                 density=True, label='|turn| empirical')

        x_abs     = np.linspace(0, 180, 500)
        pdf_abs   = np.zeros_like(x_abs)
        for i, (m, s, w) in enumerate(zip(means_s, stds_s, weights_s)):
            # Folded normal approximation for |turn|
            pdf_fold  = w * (scipy_norm.pdf(x_abs,  m, s) +
                             scipy_norm.pdf(x_abs, -m, s))
            pdf_abs  += pdf_fold
            ax1.plot(x_abs, pdf_fold, color=COLORS[i % len(COLORS)],
                     lw=1.8, ls='--', label=f'Component {i}')

        ax1.plot(x_abs, pdf_abs, color='white', lw=2.0, label='GMM total')
        ax1.axvline(tl, color='#F39C12', lw=2.0, ls=':',
                    label=f'Threshold = +/-{tl:.1f} degrees')

        straight_std = stds_s[int(np.argmin(stds_s))]
        ax1.axvline(3 * straight_std, color='#E74C3C', lw=1.5, ls='--',
                    alpha=0.7, label=f'3s straight = {3*straight_std:.1f} degrees')

        ax1.set_xlim(0, 180)
        ax1.set_ylabel('Density', color='white', fontsize=9)
        ax1.set_title('|turn| distribution + threshold',
                      color='white', fontsize=10, pad=8)
        ax1.legend(facecolor='#1A1A2E', edgecolor='#444466',
                   labelcolor='white', fontsize=7, loc='upper right')

        # -- Stats box -----------------------------------------------------
        stats = (f'n={len(pred_turns):,}  '
                 f'p50={np.percentile(abs_turns,50):.1f} degrees  '
                 f'p90={np.percentile(abs_turns,90):.1f} degrees  '
                 f'p99={np.percentile(abs_turns,99):.1f} degrees')
        fig.text(0.5, 0.01, stats, ha='center', color='#AAAACC',
                 fontsize=8)

        fig.suptitle(
            f'GMM Turn Threshold  |  airport={self.airport.upper()}  '
            f'|  thr=+/-{tl:.1f} degrees',
            color='white', fontsize=12, y=1.02)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  GMM plot saved to: {out_path}")

    def _compute_global_thresholds(self):
        """Compute data-driven thresholds from the dataset.
        
        If use_precomputed_thresholds is True, use hard-coded thresholds
        for specific airports to avoid re-computing.
        """
        # Fixed speed label thresholds (not used in downstream tasks)
        self.accel_th = 0.05    # knots/s
        self.decel_th = -0.05   # knots/s

        # ------------------------------------------------------------------
        # Pre-computed thresholds for specific airports (when switch is ON)
        # Turn thresholds follow pseudocode logic: +/-25 deg
        # Hold thresholds from previous runs
        # ------------------------------------------------------------------
        if self.use_precomputed_thresholds:
            if self.airport.lower() == 'klax':
                self.turn_left_threshold = 25.0
                self.turn_right_threshold = -25.0
                self.hold_speed_threshold = 0.6622
                print(f"\n[thresholds] Using pre-computed thresholds for KLAX:")
                print(f"  turn_left_threshold  = +{self.turn_left_threshold:.2f} degrees")
                print(f"  turn_right_threshold = {self.turn_right_threshold:.2f} degrees")
                print(f"  hold_speed_threshold = {self.hold_speed_threshold:.4f}")
                return
            
            if self.airport.lower() == 'kmsy':
                self.turn_left_threshold = 25.0
                self.turn_right_threshold = -25.0
                self.hold_speed_threshold = 0.1344
                print(f"\n[thresholds] Using pre-computed thresholds for KMSY:")
                print(f"  turn_left_threshold  = +{self.turn_left_threshold:.2f} degrees")
                print(f"  turn_right_threshold = {self.turn_right_threshold:.2f} degrees")
                print(f"  hold_speed_threshold = {self.hold_speed_threshold:.4f}")
                return

        # ------------------------------------------------------------------
        # For other airports (or when precomputed switch is OFF):
        # compute thresholds from data
        # ------------------------------------------------------------------
        print("Computing global thresholds ...")

        all_pred_turns = []
        all_speeds     = []
        n_rw_skipped   = 0
        n_seen         = 0

        for f in tqdm(self.data_files, desc="Sampling statistics"):
            try:
                data       = pd.read_csv(f)
                frames     = data.Frame.unique().tolist()
                frame_data = [data[data.Frame == fr] for fr in frames]
                num_sequences = max(
                    1, int(math.ceil((len(frames) - self.seq_len + 1) / self.skip)))

                for i in range(0, num_sequences * self.skip + 1, self.skip):
                    seq = np.concatenate(
                        frame_data[i:i + self.seq_len], axis=0)

                    for agent_id in np.unique(seq[:, G.RAW_IDX.ID]):
                        agent_seq = seq[seq[:, G.RAW_IDX.ID] == agent_id]
                        if len(agent_seq) < self.seq_len:
                            continue

                        speed = agent_seq[:, G.RAW_IDX.Speed].astype(float)
                        if np.allclose(speed, 0):
                            continue

                        n_seen += 1
                        # keep the statistics consistent with the filtered data
                        if self._agent_on_runway(agent_seq, 0, self.seq_len):
                            n_rw_skipped += 1
                            continue

                        # Full sequence heading for turn threshold.
                        # Using the full sequence (history + prediction) gives
                        # a larger cumulative turn, making the distribution more
                        # likely to exhibit genuine multi-modal structure so that
                        # the GMM can find a meaningful threshold.
                        hdg_full = agent_seq[
                            :self.seq_len,
                            G.RAW_IDX.Heading].astype(float)
                        if len(hdg_full) >= 3:
                            all_pred_turns.append(
                                self._compute_cumulative_turn(hdg_full))

                        # Full-sequence mean speed for hold threshold,
                        # consistent with the full-sequence Hold detection.
                        all_speeds.append(np.mean(speed[:self.seq_len]))

            except Exception as e:
                print(f"Warning: file skipped - {e}")

        if self.filter_runway and n_seen > 0:
            print(f"[runway] threshold stats: skipped {n_rw_skipped}/{n_seen} "
                  f"({100.0 * n_rw_skipped / n_seen:.1f}%) runway-taxiing agents")

        all_pred_turns = np.array(all_pred_turns)
        all_speeds     = np.array(all_speeds)

        # Turn thresholds via GMM (label-free)
        self.turn_left_threshold, self.turn_right_threshold = \
            self._fit_gmm_thresholds(all_pred_turns)

        # Hold threshold: full-sequence mean speed below p5
        self.hold_speed_threshold = float(np.percentile(all_speeds, 5))

        print("Learned thresholds:")
        print(f"  turn_left_threshold  = +{self.turn_left_threshold:.2f} degrees")
        print(f"  turn_right_threshold = {self.turn_right_threshold:.2f} degrees")
        print(f"  accel_th             = {self.accel_th:.4f} (fixed)")
        print(f"  decel_th             = {self.decel_th:.4f} (fixed)")
        print(f"  hold_speed_threshold = {self.hold_speed_threshold:.4f}")

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def process_data(self) -> None:
        """Process all CSV files and write scenario-level pickle shards."""
        print(f"Processing data for airport {self.airport.upper()}.")
        total_rw_skipped, total_agents = 0, 0
        total_dev_seen, total_dev_skipped = 0, 0

        if self.parallel:
            scenes = Parallel(n_jobs=self.n_jobs)(
                delayed(self.process_file)(f) for f in tqdm(self.data_files))
            for res in scenes:
                if res is None:
                    continue
                bl, n_skip, n_tot, dev_seen, dev_skipped = res
                self.blacklist   += bl
                total_rw_skipped += n_skip
                total_agents     += n_tot
                total_dev_seen   += dev_seen
                total_dev_skipped += dev_skipped
        else:
            for f in tqdm(self.data_files):
                res = self.process_file(f)
                if res is None:
                    continue
                bl, n_skip, n_tot, dev_seen, dev_skipped = res
                self.blacklist   += bl
                total_rw_skipped += n_skip
                total_agents     += n_tot
                total_dev_seen   += dev_seen
                total_dev_skipped += dev_skipped

        if self.filter_runway:
            pct = 100.0 * total_rw_skipped / max(total_agents, 1)
            print("=" * 70)
            print(f"[runway] TOTAL skipped {total_rw_skipped}/{total_agents} "
                  f"({pct:.1f}%) runway-taxiing agents "
                  f"(threshold {self.runway_thr_km*1000:.0f} m)")
            print("=" * 70)

        if self.filter_deviation and total_dev_seen > 0:
            pct = 100.0 * total_dev_skipped / max(total_dev_seen, 1)
            print("=" * 70)
            print(f"[deviation] TOTAL dropped {total_dev_skipped}/{total_dev_seen} "
                  f"({pct:.1f}%) agents over per-mode deviation threshold")
            print("=" * 70)

        blacklist_file = os.path.join(self.blacklist_dir, f'{self.airport}.txt')
        with open(blacklist_file, 'w') as fp:
            fp.write('\n'.join(self.blacklist))

    def process_file(self, f: str):
        """Process a single CSV data file.

        Returns
        -------
        (blacklist, n_runway_skipped, n_agents_seen, n_dev_seen, n_dev_skipped) or None
        """
        print(f"Processing file: {f}")
        base_name  = f.split('/')[-1]
        shard_name = base_name.split('.')[0]
        airport_id = base_name.split('_')[0].lower()
        file_time  = D._get_file_timestamp(base_name)
        data_dir   = os.path.join(self.out_data_dir, shard_name)

        if not self.overwrite and (
                os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0):
            return None

        data       = pd.read_csv(f)
        frames     = data.Frame.unique().tolist()
        frame_data = [data[data.Frame == fr] for fr in frames]

        blacklist     = []
        f_rw_skipped  = 0
        f_agents_seen = 0
        total_dev_seen = 0
        total_dev_skipped = 0
        num_sequences = int(math.ceil(
            (len(frames) - self.seq_len + 1) / self.skip))
        if num_sequences < 1:
            blacklist.append(f.removeprefix(self.in_data_dir + '/'))
            return blacklist, f_rw_skipped, f_agents_seen, total_dev_seen, total_dev_skipped

        os.makedirs(data_dir, exist_ok=True)
        valid_seq = 0

        for i in range(0, num_sequences * self.skip + 1, self.skip):
            scenario_id = str(valid_seq).zfill(6)
            result = self.process_seq(
                frame_data=frame_data, frames=frames,
                seq_idx=i, airport_id=airport_id)

            (seq, agent_id, agent_type, agent_valid, agent_mask,
             mode_labels, rule_based_encoding, turn_feasibility,
             n_skip, n_seen, dev_seen, dev_skipped) = result

            f_rw_skipped += n_skip
            f_agents_seen += n_seen
            total_dev_seen += dev_seen
            total_dev_skipped += dev_skipped

            if seq is None:
                continue

            num_agents, _, _ = seq.shape
            time_meta = D._process_timestamp(
                scene_ts=file_time, frame_idx=i, airport_code=airport_id)

            scene = {
                'scenario_id':         scenario_id,
                'num_agents':          num_agents,
                'airport_id':          airport_id,
                'agent_sequences':     seq,
                'agent_ids':           agent_id,
                'agent_types':         agent_type,
                'agent_masks':         agent_mask,
                'agent_valid':         agent_valid,
                'time_meta':           time_meta,
                'mode_labels':         mode_labels,
                'rule_based_encoding': rule_based_encoding,
                'turn_feasibility':    turn_feasibility,
                'encoding_info': {
                    'rule_based_columns': (
                        list(rule_based_encoding.columns)
                        if rule_based_encoding is not None else []),
                    'turn_feasibility_columns': (
                        list(turn_feasibility.columns)
                        if turn_feasibility is not None else []),
                    'encoding_method': self.turn_label_method,
                },
            }
            scene['meta'] = (self.process_scores(scene)
                             if self.add_scores_meta else None)

            scene_filepath = os.path.join(
                data_dir, f"{scenario_id}_n-{num_agents}.pkl")
            with open(scene_filepath, 'wb') as fp:
                pickle.dump(scene, fp, protocol=pickle.HIGHEST_PROTOCOL)

            valid_seq += 1

        if len(os.listdir(data_dir)) == 0:
            blacklist.append(f.removeprefix(self.in_data_dir + '/'))
            os.rmdir(data_dir)

        if self.filter_runway and f_agents_seen > 0:
            print(f"  [runway] {base_name}: skipped {f_rw_skipped}/{f_agents_seen} "
                  f"({100.0 * f_rw_skipped / f_agents_seen:.1f}%) agents on runway")
        return blacklist, f_rw_skipped, f_agents_seen, total_dev_seen, total_dev_skipped

    def process_seq(self, frame_data, frames, seq_idx, airport_id):
        """Process all valid agent sequences and assign taxiing mode labels.

        Agents whose trajectory passes over a runway are dropped entirely, so
        they never enter the scene and can never be selected as the ego agent.

        Returns
        -------
        tuple : (seq, agent_ids, agent_types, agent_valid, agent_masks,
                 mode_labels, rule_based_encoding, turn_feasibility,
                 n_runway_skipped, n_agents_seen,
                 n_dev_seen, n_dev_skipped)
        """
        none_outs = (None,) * 8
        seq_data  = np.concatenate(
            frame_data[seq_idx:seq_idx + self.seq_len], axis=0)

        n_rw_skipped, n_agents_seen = 0, 0
        n_dev_seen, n_dev_skipped = 0, 0

        if math.isclose(seq_data[:, G.RAW_IDX.Speed].sum(), 0):
            return none_outs + (n_rw_skipped, n_agents_seen, n_dev_seen, n_dev_skipped)
        if not np.isin(
                seq_data[:, G.RAW_IDX.Type].astype(int), C.AIRCRAFT).sum():
            return none_outs + (n_rw_skipped, n_agents_seen, n_dev_seen, n_dev_skipped)

        unique_agents = np.unique(seq_data[:, G.RAW_IDX.ID])
        num_agents    = len(unique_agents)
        if num_agents < self.min_agents or num_agents > self.max_agents:
            return none_outs + (n_rw_skipped, n_agents_seen, n_dev_seen, n_dev_skipped)

        num_considered      = 0
        seq                 = np.zeros((num_agents, self.seq_len, G.DIM))
        agent_masks         = np.zeros((num_agents, self.seq_len)).astype(bool)
        agent_id_list       = []
        agent_type_list     = []
        valid_agent_list    = []
        mode_labels         = []
        possible_turns_list = []
        alt_idx             = G.RAW_IDX.Altitude

        for _, agent_id in enumerate(unique_agents):
            agent_seq = seq_data[seq_data[:, 1] == agent_id]
            pad_front = frames.index(agent_seq[0, 0]) - seq_idx
            pad_end   = frames.index(agent_seq[-1, 0]) - seq_idx + 1
            if pad_end - pad_front != self.seq_len:
                continue

            n_agents_seen += 1

            # ---- drop agents taxiing on a runway -------------------------
            # Must happen BEFORE any list is appended, so that the indices of
            # agent_id_list / agent_type_list / valid_agent_list / mode_labels
            # stay aligned with num_considered.
            if self._agent_on_runway(agent_seq, pad_front, pad_end):
                n_rw_skipped += 1
                continue

            mx = self.ref_data.limits.Altitude.max
            mn = self.ref_data.limits.Altitude.min
            agent_seq[:, alt_idx] = (agent_seq[:, alt_idx] - mn) / (mx - mn)

            agent_id_list.append(int(agent_id))
            agent_type_list.append(int(agent_seq[0, G.RAW_IDX.Type]))

            mask = agent_seq[:, G.RAW_IDX.Interp] == '[ORG]'
            agent_seq[mask,  G.RAW_IDX.Interp] = 1.0
            agent_seq[~mask, G.RAW_IDX.Interp] = 0.0

            valid = mask[:self.hist_len].sum() >= self.min_valid_points
            if valid:
                for t in self.pred_lens:
                    if (mask[self.hist_len:self.hist_len + t].sum()
                            < self.min_valid_points):
                        valid = False
                        break
            valid_agent_list.append(valid)

            agent_seq = C.impute(agent_seq, self.seq_len)

            mode_label, possible_turns = self._compute_taxiing_mode(
                agent_seq, pad_front, pad_end)

            # ---- drop agents deviating too far from taxiway centreline ---
            # Must happen AFTER the mode is known (per-mode threshold) but
            # BEFORE any seq write, so num_considered stays aligned. The three
            # appends already made above for this agent (id / type / valid) are
            # undone with pop(); mode_labels / possible_turns_list are appended
            # later so need no undo.
            if self.filter_deviation:
                n_dev_seen += 1
                _turn = mode_label.split('_', 1)[0] if '_' in mode_label else mode_label
                if self._agent_deviates(agent_seq, pad_front, pad_end, _turn):
                    n_dev_skipped += 1
                    agent_id_list.pop()
                    agent_type_list.pop()
                    valid_agent_list.pop()
                    continue

            valid_mask = agent_seq[:, G.RAW_IDX.Interp].astype(bool)
            agent_masks[num_considered, pad_front:pad_end] = valid_mask

            agent_seq = agent_seq[:, G.RAW_SEQ_MASK]
            seq[num_considered, pad_front:pad_end] = agent_seq[:, G.SEQ_ORDER]
            num_considered += 1

            mode_labels.append(mode_label)
            possible_turns_list.append(possible_turns)

        valid_agent_list = np.asarray(valid_agent_list)
        if valid_agent_list.sum() == 0 or num_considered < self.min_agents:
            return none_outs + (n_rw_skipped, n_agents_seen, n_dev_seen, n_dev_skipped)

        rule_based_encoding = self._compute_separate_encoding(mode_labels)
        turn_feasibility    = self._compute_turn_feasibility(
            possible_turns_list)

        return (seq[:num_considered], agent_id_list, agent_type_list,
                valid_agent_list, agent_masks[:num_considered], mode_labels,
                rule_based_encoding, turn_feasibility,
                n_rw_skipped, n_agents_seen,
                n_dev_seen, n_dev_skipped)

    # ------------------------------------------------------------------
    # Taxiing mode labeling (trajectory-only, prediction segment)
    # ------------------------------------------------------------------

    def _compute_taxiing_mode(self, agent_seq, pad_front, pad_end):
        """Compute taxiing mode label from prediction-segment kinematics.

        Turn label
        ----------
        Based on cumulative heading change over the prediction segment only
        (frames hist_len : seq_len). Heading is unwrapped before smoothing
        to handle 0/360 wrap-around correctly.
        Thresholds are data-driven (GMM on prediction-segment distribution).

        Hold detection
        --------------
        If the mean speed over the prediction segment is below
        hold_speed_threshold (p5 of non-static agent speeds), the agent is
        classified as Hold regardless of heading change.

        Speed label
        -----------
        Based on mean acceleration over the full sequence (unchanged from
        the original implementation to maintain output format compatibility).

        Returns
        -------
        mode_label   : str  e.g. "TurnLeft_Accel"
        possible_turns : set  topology-derived feasibility (from map BFS)
        """
        all_possible = {'TurnLeft', 'TurnRight', 'Straight', 'Hold'}
        try:
            spd_full = agent_seq[pad_front:pad_end,
                                 G.RAW_IDX.Speed].astype(float)
            spd_pred = agent_seq[
                pad_front + self.hist_len:pad_end,
                G.RAW_IDX.Speed].astype(float)
            hdg_full = agent_seq[
                pad_front:pad_end,
                G.RAW_IDX.Heading].astype(float)

            # Hold: full-sequence mean speed below threshold.
            # Using the full sequence is consistent with how hold_speed_threshold
            # is derived (p5 of full-sequence mean speeds).
            if (np.allclose(spd_full, 0) or
                    spd_full.mean() <= self.hold_speed_threshold):
                possible_turns = self._get_turn_feasibility_from_map(
                    agent_seq, pad_front)
                possible_turns.add('Hold')
                return "Hold_Hold", possible_turns

            # Cumulative turn over full sequence (history + prediction).
            # Using the full sequence is consistent with how the threshold
            # was derived and captures the complete turning behaviour.
            net_turn = self._compute_cumulative_turn(hdg_full)

            if net_turn >= self.turn_left_threshold:
                turn_label = "TurnLeft"
            elif net_turn <= self.turn_right_threshold:
                turn_label = "TurnRight"
            else:
                turn_label = "Straight"

            # Speed label from full sequence acceleration
            mean_accel = np.mean(np.gradient(spd_full))
            if mean_accel > self.accel_th:
                speed_label = "Accel"
            elif mean_accel < self.decel_th:
                speed_label = "Decel"
            else:
                speed_label = "Normal"

            # Turn feasibility from map BFS
            possible_turns = self._get_turn_feasibility_from_map(
                agent_seq, pad_front)
            # Ground-truth label is always feasible
            possible_turns.add(turn_label)
            possible_turns.add('Hold')

            return f"{turn_label}_{speed_label}", possible_turns

        except Exception as e:
            print(f"Error computing taxiing mode: {e}")
            return "Unknown_Unknown", all_possible

    # ------------------------------------------------------------------
    # Turn feasibility from map (BFS on the original graph)
    # ------------------------------------------------------------------

    def _get_turn_feasibility_from_map(self, agent_seq, pad_front) -> set:
        """Determine topologically feasible turn modes via BFS on the ORIGINAL graph.

        Why the original graph
        ----------------------
        An earlier version ran the BFS on a simplified graph in which degree-2
        chains were collapsed into semantic edges. That left only junctions and
        dead-ends (~12% of nodes) available for the nearest-node lookup, so for
        an aircraft mid-chain the nearest junction often belonged to a different
        chain and the search started from the wrong place. Walking the original
        graph edge by edge, as the rule-based selector does, removes that failure
        mode and also lets the BFS use the 'length' and 'bearing' attributes the
        graph already carries on every edge.

        Forward node
        ------------
        The aircraft position is projected a short distance along its current
        heading and snapped to the nearest node, so the node picked is the one
        the aircraft is heading TOWARD and no U-turn is implied.

        Distance budget
        ---------------
        mean history speed (km/s) * pred_len, i.e. how far the aircraft travels
        in the prediction horizon at its current speed, minus the distance
        already committed to reach the forward node. A 200 m floor keeps slow
        aircraft from getting a zero budget.

        Cumulative turn
        ---------------
        cum_turn accumulates step_turn, measured against the PREVIOUS heading.
        Accumulating turns measured against the initial heading (as an earlier
        version did) double counts on multi-hop paths: for headings b1, b2, b3 it
        gives (b1-s) + (b2-s) + (b3-s) instead of (b1-s) + (b2-b1) + (b3-b2).

        BFS state: (node, dist_used, cum_turn, cur_heading)

        Parameters
        ----------
        agent_seq : np.ndarray  shape (seq_len, DIM)
        pad_front : int

        Returns
        -------
        set : subset of {'TurnLeft', 'TurnRight', 'Straight', 'Hold'}
        """
        all_possible = {'TurnLeft', 'TurnRight', 'Straight', 'Hold'}

        if self._node_kdtree is None:
            return all_possible

        try:
            last_hist_idx = pad_front + self.hist_len - 1

            agent_xy = np.array([
                float(agent_seq[last_hist_idx, G.RAW_IDX.x]),
                float(agent_seq[last_hist_idx, G.RAW_IDX.y]),
            ])
            cur_heading = float(agent_seq[last_hist_idx, G.RAW_IDX.Heading])

            # Distance budget: mean history speed (km/s) * pred_len (s)
            KNOTS_TO_KMS   = 0.000514444   # 1 knot = 1.852/3600 km/s
            spd            = agent_seq[
                pad_front:pad_front + self.hist_len,
                G.RAW_IDX.Speed].astype(float)
            mean_speed_kms = np.mean(spd) * KNOTS_TO_KMS
            dist_budget    = mean_speed_kms * self.pred_len
            dist_budget    = max(dist_budget, 0.2)   # minimum 200 m

            # -- Forward node ----------------------------------------------
            # Project ahead along the heading, then snap. x = North, y = East,
            # so a compass heading projects to (cos, sin) in that order.
            hr           = math.radians(cur_heading)
            proj_dist    = min(dist_budget * 0.3, 0.1)
            projected_xy = agent_xy + proj_dist * np.array([math.cos(hr),
                                                            math.sin(hr)])
            dist, idx    = self._node_kdtree.query(projected_xy, k=1)
            forward_node    = self._kdtree_node_ids[int(idx)]
            forward_node_xy = self._kdtree_node_xys[int(idx)]

            # Turn needed to reach the forward node. Compass bearing is
            # atan2(East, North) = atan2(dy, dx); the reverse argument order
            # returns an angle off the East axis and is not comparable with a
            # heading.
            delta              = forward_node_xy - agent_xy
            bearing_to_forward = self._xy_to_bearing(delta[0], delta[1])
            turn_to_forward    = self._signed_bearing_diff(bearing_to_forward,
                                                           cur_heading)

            forward_dist     = float(np.linalg.norm(forward_node_xy - agent_xy))
            remaining_budget = max(dist_budget - forward_dist, 0.0)
            initial_heading  = (cur_heading + turn_to_forward) % 360.0

            # -- BFS on the original graph ---------------------------------
            def _turn_bucket(turn: float) -> int:
                return int(round(turn / 10.0))

            queue    = deque([(forward_node, 0.0, turn_to_forward, initial_heading)])
            visited  = {(forward_node, _turn_bucket(turn_to_forward))}
            possible = {'Hold'}

            # The aircraft is already heading toward the forward node, so
            # continuing straight is always reachable.
            possible.add('Straight')
            if turn_to_forward >= self.turn_left_threshold:
                possible.add('TurnLeft')
            elif turn_to_forward <= self.turn_right_threshold:
                possible.add('TurnRight')

            steps = 0
            while queue and steps < self.MAX_BFS_STEPS:
                steps += 1
                cur_node, dist_used, cum_turn, heading = queue.popleft()

                for nb in self._get_neighbors_undirected(cur_node):
                    # no aircraft in the dataset travels along a runway
                    if self.exclude_runway_edges and self._is_runway_edge(cur_node, nb):
                        continue
                    out_bearing = self._get_edge_bearing(cur_node, nb)
                    if out_bearing is None:
                        continue
                    edge_km = self._get_edge_length_km(cur_node, nb)
                    if edge_km <= 0:
                        continue

                    # turn taken at this node, relative to the CURRENT heading
                    step_turn = self._signed_bearing_diff(out_bearing, heading)
                    if abs(step_turn) >= 150.0:      # no U-turns
                        continue

                    new_dist = dist_used + edge_km
                    if new_dist > remaining_budget:
                        continue

                    new_cum_turn = cum_turn + step_turn
                    state = (nb, _turn_bucket(new_cum_turn))
                    if state in visited:
                        continue
                    visited.add(state)
                    queue.append((nb, new_dist, new_cum_turn, out_bearing))

                    if new_cum_turn >= self.turn_left_threshold:
                        possible.add('TurnLeft')
                    elif new_cum_turn <= self.turn_right_threshold:
                        possible.add('TurnRight')
                    else:
                        possible.add('Straight')

            return possible

        except Exception as e:
            print(f"Warning: turn feasibility BFS failed - {e}")
            return all_possible

    # ------------------------------------------------------------------
    # Encoding (output format unchanged)
    # ------------------------------------------------------------------

    def _compute_separate_encoding(self, taxiing_modes: list) -> pd.DataFrame:
        """One-hot encode taxiing modes into separate turn and speed vectors.

        Output format is identical to the original implementation:
        8 columns: turn_TurnLeft, turn_TurnRight, turn_Straight, turn_Hold,
                   speed_Accel, speed_Decel, speed_Normal, speed_Hold
        """
        turn_modes_list  = ['TurnLeft', 'TurnRight', 'Straight', 'Hold']
        speed_modes_list = ['Accel', 'Decel', 'Normal', 'Hold']
        turn_modes, speed_modes = [], []

        for mode in taxiing_modes:
            if '_' in mode:
                turn, speed = mode.split('_', 1)
                turn_modes.append(turn)
                speed_modes.append(speed)
            else:
                turn_modes.append('Unknown')
                speed_modes.append('Unknown')

        df_turn  = pd.DataFrame({'turn_mode': pd.Categorical(
            turn_modes,  categories=turn_modes_list)})
        df_speed = pd.DataFrame({'speed_mode': pd.Categorical(
            speed_modes, categories=speed_modes_list)})

        separate_encoded = pd.concat([
            pd.get_dummies(df_turn['turn_mode'],  prefix='turn'),
            pd.get_dummies(df_speed['speed_mode'], prefix='speed'),
        ], axis=1)

        return separate_encoded

    def _compute_turn_feasibility(
            self, possible_turns_list: list) -> pd.DataFrame:
        """Build binary feasibility matrix from per-agent topology sets.

        Output format is identical to the original implementation:
        4 columns: feasible_TurnLeft, feasible_TurnRight,
                   feasible_Straight, feasible_Hold
        """
        turn_modes_list = ['TurnLeft', 'TurnRight', 'Straight', 'Hold']
        rows = [
            {f'feasible_{m}': int(m in pt) for m in turn_modes_list}
            for pt in possible_turns_list
        ]
        feasibility = pd.DataFrame(
            rows, columns=[f'feasible_{m}' for m in turn_modes_list])

        return feasibility

    # ------------------------------------------------------------------
    # Scoring (unchanged)
    # ------------------------------------------------------------------

    def process_scores(self, scene):
        """Compute kinematic and interactive scores for all valid agents."""
        crowd_scene_score                   = compute_simple_scene_crowdedness(
            scene, self.max_agents)
        kin_agents_scores, kin_scene_score  = compute_kinematic_scores(
            scene, self.hold_lines)
        int_agents_scores, int_scene_score  = compute_interactive_scores(
            scene, self.hold_lines)
        crit_agent_scores, crit_scene_score = compute_simple_scene_critical(
            agent_scores_list=[
                kin_agents_scores.copy(), int_agents_scores].copy(),
            scene_score_list=[
                crowd_scene_score.copy(),
                kin_scene_score.copy(),
                int_scene_score.copy()])
        return {
            'agent_scores': {
                'kinematic':   kin_agents_scores,
                'interactive': int_agents_scores,
                'critical':    crit_agent_scores,
            },
            'agent_order': {
                'random':      C.get_random_order(
                    scene['num_agents'], scene['agent_valid'], self.seed),
                'interactive': C.get_sorted_order(int_agents_scores),
                'critical':    C.get_sorted_order(crit_agent_scores),
            },
            'scene_scores': {
                'crowdedness': crowd_scene_score,
                'kinematic':   kin_scene_score,
                'interactive': int_scene_score,
                'critical':    crit_scene_score,
            },
        }