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

from amelia_scenes.scoring.kinematic import compute_kinematic_scores
from amelia_scenes.scoring.critical import compute_simple_scene_critical
from amelia_scenes.scoring.interactive import compute_interactive_scores
from amelia_scenes.scoring.crowdedness import compute_simple_scene_crowdedness


class SceneProcessor:
    """Dataset class for pre-processing airport surface movement data into scenes."""

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

        # osmid -> (u_node, v_node) mapping
        self._osmid_to_edge = {
            int(data['osmid']): (u, v)
            for u, v, data in self.graph_nx.edges(data=True)
        }

        # Build simplified graph for turn feasibility BFS.
        # No threshold needed — intermediate nodes are identified purely by
        # undirected neighbour count (== 2).
        self._simplified_edges, self._node_to_sem_edge = self._build_simplified_graph()

        # Build KD-tree over simplified graph node positions for fast
        # nearest-node lookup (used by turn feasibility, not trajectory snap).
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
        dd = 1

    # ------------------------------------------------------------------
    # Bidirectional graph helpers
    # ------------------------------------------------------------------

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

        If the forward edge does not exist the reverse edge bearing + 180° is
        used. If prev_bearing is provided, any candidate that differs from it
        by more than 150° is rejected (aircraft cannot reverse abruptly).
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

    def _get_node_xy(self, node):
        """Return projected (x, y) coordinates of a node in km, or None."""
        for u, v, data in self.graph_nx.edges(node, data=True):
            osmid = int(data['osmid'])
            mask  = self.all_polylines[:, 9].astype(int) == osmid
            if mask.any():
                idx = np.where(mask)[0][0]
                return self.all_polylines[idx, [2, 3]]
        for u, v, data in self.graph_nx.in_edges(node, data=True):
            osmid = int(data['osmid'])
            mask  = self.all_polylines[:, 9].astype(int) == osmid
            if mask.any():
                idx = np.where(mask)[0][0]
                return self.all_polylines[idx, [6, 7]]
        return None

    # ------------------------------------------------------------------
    # Simplified graph construction
    # ------------------------------------------------------------------

    def _build_simplified_graph(self):
        """Collapse intermediate nodes (degree 2) into semantic edges.

        No bearing threshold is used — the decision is based purely on
        undirected neighbour count. Each semantic edge records:
            u, v         : endpoint junction/dead-end node IDs
            bearing_in   : entering bearing (degrees)
            bearing_out  : leaving bearing (degrees)
            net_turn     : accumulated bearing change (degrees)
            nodes        : ordered list of all node IDs in the chain
            xy_start     : projected (x, y) of u (km)
            xy_end       : projected (x, y) of v (km)
            length       : Euclidean length of the semantic edge (km)

        Returns
        -------
        simplified_edges : list of dict
        node_to_sem_edge : dict  (intermediate node -> its semantic edge)
        """
        simplified_edges  = []
        visited_edge_keys = set()

        for start_node in self.graph_nx.nodes():
            n_nb = len(self._get_neighbors_undirected(start_node))
            if n_nb != 1 and not self._is_junction(start_node):
                continue

            for next_node in self._get_neighbors_undirected(start_node):
                edge_key = (min(start_node, next_node),
                            max(start_node, next_node))
                if edge_key in visited_edge_keys:
                    continue

                bearing_in = self._get_edge_bearing(start_node, next_node)
                if bearing_in is None:
                    continue

                cur_bearing = bearing_in
                cur_node    = next_node
                prev_node   = start_node
                net_turn    = 0.0
                nodes       = [start_node, next_node]

                # Walk forward, absorbing intermediate (degree-2) nodes
                while True:
                    neighbors = self._get_neighbors_undirected(cur_node)
                    if self._is_junction(cur_node) or len(neighbors) == 1:
                        break

                    next_candidates = neighbors - {prev_node}
                    if not next_candidates:
                        break
                    nxt = next_candidates.pop()

                    next_bearing = self._get_edge_bearing(
                        cur_node, nxt, prev_bearing=cur_bearing)
                    if next_bearing is None:
                        break

                    diff        = self._signed_bearing_diff(next_bearing, cur_bearing)
                    net_turn   += diff
                    cur_bearing = next_bearing
                    prev_node   = cur_node
                    cur_node    = nxt
                    nodes.append(nxt)

                xy_start = self._get_node_xy(start_node)
                xy_end   = self._get_node_xy(cur_node)

                edge_key = (min(start_node, cur_node),
                            max(start_node, cur_node))
                if edge_key not in visited_edge_keys:
                    visited_edge_keys.add(edge_key)

                    length = (float(np.linalg.norm(xy_end - xy_start))
                              if xy_start is not None and xy_end is not None
                              else 0.0)

                    simplified_edges.append({
                        'u':           start_node,
                        'v':           cur_node,
                        'bearing_in':  bearing_in,
                        'bearing_out': cur_bearing,
                        'net_turn':    net_turn,
                        'nodes':       nodes,
                        'xy_start':    xy_start,
                        'xy_end':      xy_end,
                        'length':      length,
                    })

        node_to_sem_edge = {}
        for se in simplified_edges:
            for node in se['nodes'][1:-1]:
                node_to_sem_edge[node] = se

        print(f"Simplified graph: {len(simplified_edges)} semantic edges "
              f"(original: {self.graph_nx.number_of_nodes()} nodes, "
              f"{self.graph_nx.number_of_edges()} edges)")
        return simplified_edges, node_to_sem_edge

    def _build_node_kdtree(self):
        """Build a KD-tree over junction/endpoint node positions.

        Used for fast nearest-node lookup in turn feasibility computation.
        Only nodes that are endpoints of semantic edges (real junctions or
        dead-ends) are included.
        """
        node_ids = []
        node_xys = []

        seen = set()
        for se in self._simplified_edges:
            for node, xy in [(se['u'], se['xy_start']),
                             (se['v'], se['xy_end'])]:
                if node not in seen and xy is not None:
                    seen.add(node)
                    node_ids.append(node)
                    node_xys.append(xy)

        self._kdtree_node_ids = node_ids
        self._kdtree_node_xys = np.array(node_xys) if node_xys else np.empty((0, 2))
        if len(node_xys) > 0:
            self._node_kdtree = cKDTree(self._kdtree_node_xys)
        else:
            self._node_kdtree = None

        print(f"Node KD-tree: {len(node_ids)} junction/endpoint nodes indexed.")

    # ------------------------------------------------------------------
    # Threshold computation
    # ------------------------------------------------------------------

    @staticmethod
    def _signed_bearing_diff(b_new: float, b_old: float) -> float:
        """Signed angular difference in (-180, 180], wrapping-safe."""
        return ((b_new - b_old + 180.0) % 360.0) - 180.0

    @staticmethod
    def _compute_cumulative_turn(headings: np.ndarray,
                                 smooth_w: int = 5) -> float:
        """Cumulative signed heading change over a heading sequence.

        Headings are unwrapped in radians before smoothing so that 0°/360°
        crossings are handled correctly (e.g. 358°->2° is +4°, not -356°).
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
        cumulative turn distribution. Crossing points between adjacent
        ordered components give the left/right turn thresholds.

        Single-peak fallback: threshold = 3 * robust_sigma (MAD-based),
        applied symmetrically to both directions.

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
        # straight-segment heading noise. This is more precise than MAD on
        # the full distribution because the GMM has already separated the
        # straight cluster from the long-tailed turn data.
        # Threshold = 3 * sigma_straight → ~0.3% misclassification rate.
        def _mad_fallback():
            straight_std = float(stds_s[straight_comp])
            thr          = max(3.0 * straight_std, 5.0)   # minimum 5° floor
            print(f"  Fallback (GMM straight std): "
                  f"sigma_straight={straight_std:.1f}°  threshold=±{thr:.1f}°")
            return thr, -thr

        if best_n == 1 or best_gmm is None:
            tl, tr = _mad_fallback()
            print(f"  turn_left_threshold  = +{tl:.1f}°")
            print(f"  turn_right_threshold = {tr:.1f}°")
            self._plot_gmm_fit(
                pred_turns, means_s, stds_s, weights[order],
                tl, tr, best_n,
                out_path=f'gmm_threshold_{self.airport}.png')
            return tl, tr

        # ── Component quality checks ──────────────────────────────────────
        # A genuine turn cluster should be compact (std < MAX_TURN_STD).
        # If any non-straight component has an excessively large std it is
        # just fitting the long tail of a unimodal distribution, not a real
        # second peak. In that case fall back to the GMM straight std method.
        MAX_TURN_STD = 30.0   # degrees; genuine turn clusters are tighter

        # Check every non-straight component for quality
        poor_quality = False
        for ci in range(len(means_s)):
            if ci == straight_comp:
                continue
            if stds_s[ci] > MAX_TURN_STD:
                print(f"  Component {ci} has std={stds_s[ci]:.1f}° > "
                      f"{MAX_TURN_STD}° → not a genuine turn cluster")
                poor_quality = True
                break

        if poor_quality:
            for i, (m, s, w) in enumerate(zip(
                    means_s, stds_s, weights[order])):
                print(f"  component {i}: mean={m:+.1f}°  std={s:.1f}°  "
                      f"weight={w:.2f}")
            tl, tr = _mad_fallback()
            print(f"  turn_left_threshold  = +{tl:.1f}°")
            print(f"  turn_right_threshold = {tr:.1f}°")
            self._plot_gmm_fit(
                pred_turns, means_s, stds_s, weights[order],
                tl, tr, best_n,
                out_path=f'gmm_threshold_{self.airport}.png')
            return tl, tr

        # Check if components are well-separated (min gap > 5°)
        gaps = np.diff(means_s)
        if np.any(gaps < 5.0):
            tl, tr = _mad_fallback()
            print(f"  turn_left_threshold  = +{tl:.1f}°")
            print(f"  turn_right_threshold = {tr:.1f}°")
            self._plot_gmm_fit(
                pred_turns, means_s, stds_s, weights[order],
                tl, tr, best_n,
                out_path=f'gmm_threshold_{self.airport}.png')
            return tl, tr

        # Find crossing points between adjacent ordered components
        x_scan = np.linspace(-180, 180, 7201).reshape(-1, 1)
        probs  = best_gmm.predict_proba(x_scan)
        # Reorder columns to match sorted means
        probs  = probs[:, order]

        def _find_crossing(comp_a: int, comp_b: int) -> Optional[float]:
            """Find x where P(comp_a) == P(comp_b), searching between their means."""
            diff  = probs[:, comp_a] - probs[:, comp_b]
            signs = np.sign(diff)
            cross = np.where(np.diff(signs) != 0)[0]
            if len(cross) == 0:
                return None
            # Pick the crossing closest to the midpoint of the two means
            mid = (means_s[comp_a] + means_s[comp_b]) / 2.0
            best = min(cross, key=lambda i: abs(x_scan[i][0] - mid))
            return float(x_scan[best][0])

        if best_n == 2:
            # Two components: one negative (right-turn) and one positive
            # (straight or left-turn), or straight + left.
            crossing = _find_crossing(0, 1)
            if crossing is None:
                tl, tr = _mad_fallback()
            elif crossing >= 0:
                # Components are both on positive side → use MAD for right
                tl = crossing
                _, tr = _mad_fallback()
            elif crossing < 0:
                # Components are both on negative side → use MAD for left
                tr = crossing
                tl, _ = _mad_fallback()
            else:
                tl = abs(crossing)
                tr = crossing

            # Ensure left threshold is positive, right is negative
            tl = abs(tl)
            tr = -abs(tr)

        else:   # best_n == 3: left-turn / straight / right-turn
            # Crossing between component 0 (most negative) and 1 (middle)
            tr_crossing = _find_crossing(0, 1)
            # Crossing between component 1 (middle) and 2 (most positive)
            tl_crossing = _find_crossing(1, 2)

            if tr_crossing is None or tl_crossing is None:
                tl, tr = _mad_fallback()
            else:
                tr = float(tr_crossing)   # negative
                tl = float(tl_crossing)   # positive
                if tl <= 0:
                    tl, _ = _mad_fallback()
                if tr >= 0:
                    _, tr = _mad_fallback()

        for i, (m, s, w) in enumerate(zip(
                means_s, stds_s, weights[order])):
            print(f"  component {i}: mean={m:+.1f}°  std={s:.1f}°  weight={w:.2f}")
        print(f"  turn_left_threshold  = +{tl:.1f}°")
        print(f"  turn_right_threshold = {tr:.1f}°")

        # Save GMM fit visualisation
        self._plot_gmm_fit(
            pred_turns, means_s, stds_s, weights[order],
            tl, tr, best_n,
            out_path=f'gmm_threshold_{self.airport}.png')

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
        bins   = np.linspace(-180, 180, 73)   # 5° bins
        x_plot = np.linspace(-180, 180, 1000)

        for ax in axes:
            ax.set_facecolor('#1A1A2E')
            ax.tick_params(colors='white', labelsize=8)
            ax.set_xlabel('Cumulative turn (°)', color='white', fontsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444466')

        # ── Left panel: signed distribution ──────────────────────────────
        ax0 = axes[0]
        ax0.hist(np.clip(pred_turns, -180, 180), bins=bins,
                 color='#4A90D9', edgecolor='none', alpha=0.55,
                 density=True, label='Empirical')

        pdf_total = np.zeros_like(x_plot)
        for i, (m, s, w) in enumerate(zip(means_s, stds_s, weights_s)):
            pdf_i     = w * scipy_norm.pdf(x_plot, m, s)
            pdf_total += pdf_i
            label_i   = (f'Component {i}'
                         f'μ={m:+.1f}°, σ={s:.1f}°, w={w:.2f}')
            ax0.plot(x_plot, pdf_i, color=COLORS[i % len(COLORS)],
                     lw=1.8, ls='--', label=label_i)

        ax0.plot(x_plot, pdf_total, color='white', lw=2.0, ls='-',
                 label='GMM total')

        # Threshold lines
        ax0.axvline( tl, color='#F39C12', lw=2.0, ls=':',
                    label=f'TurnLeft  ={tl:+.1f}°')
        ax0.axvline( tr, color='#F39C12', lw=2.0, ls=':',
                    label=f'TurnRight ={tr:+.1f}°')
        ax0.axvline(0,  color='white',   lw=0.8, alpha=0.4)

        ax0.set_xlim(-180, 180)
        ax0.set_ylabel('Density', color='white', fontsize=9)
        ax0.set_title(f'GMM fit (n={best_n})  —  signed turn distribution',
                      color='white', fontsize=10, pad=8)
        ax0.legend(facecolor='#1A1A2E', edgecolor='#444466',
                   labelcolor='white', fontsize=7, loc='upper right')

        # ── Right panel: absolute |turn| distribution ─────────────────────
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
                    label=f'Threshold = ±{tl:.1f}°')

        straight_std = stds_s[int(np.argmin(stds_s))]
        ax1.axvline(3 * straight_std, color='#E74C3C', lw=1.5, ls='--',
                    alpha=0.7, label=f'3σ straight = {3*straight_std:.1f}°')

        ax1.set_xlim(0, 180)
        ax1.set_ylabel('Density', color='white', fontsize=9)
        ax1.set_title('|turn| distribution + threshold',
                      color='white', fontsize=10, pad=8)
        ax1.legend(facecolor='#1A1A2E', edgecolor='#444466',
                   labelcolor='white', fontsize=7, loc='upper right')

        # ── Stats box ─────────────────────────────────────────────────────
        stats = (f'n={len(pred_turns):,}  '
                 f'p50={np.percentile(abs_turns,50):.1f}°  '
                 f'p90={np.percentile(abs_turns,90):.1f}°  '
                 f'p99={np.percentile(abs_turns,99):.1f}°')
        fig.text(0.5, 0.01, stats, ha='center', color='#AAAACC',
                 fontsize=8)

        fig.suptitle(
            f'GMM Turn Threshold  |  airport={self.airport.upper()}  '
            f'|  thr=±{tl:.1f}°',
            color='white', fontsize=12, y=1.02)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  GMM plot saved to: {out_path}")

    def _compute_global_thresholds(self):
        """Compute data-driven thresholds from the dataset.

        Turn thresholds (turn_left_threshold, turn_right_threshold):
            Derived label-free from the prediction-segment cumulative turn
            distribution using an ordered GMM (BIC-selected n=1,2,3).
            Fallback to 3 * GMM-straight-std if no genuine turn clusters found.

        Speed label thresholds (accel_th, decel_th):
            Fixed values — speed label is not critical for downstream tasks
            so these are set directly to avoid scanning all files for
            acceleration statistics.

        Hold threshold (hold_speed_threshold):
            5th percentile of mean prediction-segment speed across non-static
            agent sequences. An agent whose prediction-segment mean speed falls
            below this value is classified as Hold.
        """
        print("Computing global thresholds ...")

        # Fixed speed label thresholds (not used in downstream tasks)
        self.accel_th = 0.05    # knots/s
        self.decel_th = -0.05   # knots/s

        all_pred_turns = []
        all_speeds     = []

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
                print(f"Warning: file skipped — {e}")

        all_pred_turns = np.array(all_pred_turns)
        all_speeds     = np.array(all_speeds)

        # Turn thresholds via GMM (label-free)
        self.turn_left_threshold, self.turn_right_threshold = \
            self._fit_gmm_thresholds(all_pred_turns)

        # Hold threshold: full-sequence mean speed below p5
        self.hold_speed_threshold = float(np.percentile(all_speeds, 5))

        print("Learned thresholds:")
        print(f"  turn_left_threshold  = +{self.turn_left_threshold:.2f}°")
        print(f"  turn_right_threshold = {self.turn_right_threshold:.2f}°")
        print(f"  accel_th             = {self.accel_th:.4f} (fixed)")
        print(f"  decel_th             = {self.decel_th:.4f} (fixed)")
        print(f"  hold_speed_threshold = {self.hold_speed_threshold:.4f}")

    # ------------------------------------------------------------------
    # Processing pipeline
    # ------------------------------------------------------------------

    def process_data(self) -> None:
        """Process all CSV files and write scenario-level pickle shards."""
        print(f"Processing data for airport {self.airport.upper()}.")
        if self.parallel:
            scenes = Parallel(n_jobs=self.n_jobs)(
                delayed(self.process_file)(f) for f in tqdm(self.data_files))
            for res in scenes:
                if res is not None:
                    self.blacklist += res
        else:
            for f in tqdm(self.data_files):
                res = self.process_file(f)
                if res is not None:
                    self.blacklist += res

        blacklist_file = os.path.join(self.blacklist_dir, f'{self.airport}.txt')
        with open(blacklist_file, 'w') as fp:
            fp.write('\n'.join(self.blacklist))

    def process_file(self, f: str):
        """Process a single CSV data file."""
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
        num_sequences = int(math.ceil(
            (len(frames) - self.seq_len + 1) / self.skip))
        if num_sequences < 1:
            blacklist.append(f.removeprefix(self.in_data_dir + '/'))
            return blacklist

        os.makedirs(data_dir, exist_ok=True)
        valid_seq = 0

        for i in range(0, num_sequences * self.skip + 1, self.skip):
            scenario_id = str(valid_seq).zfill(6)
            result = self.process_seq(
                frame_data=frame_data, frames=frames,
                seq_idx=i, airport_id=airport_id)
            if result[0] is None:
                continue

            (seq, agent_id, agent_type, agent_valid, agent_mask,
             mode_labels, rule_based_encoding, turn_feasibility) = result

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
        return blacklist

    def process_seq(self, frame_data, frames, seq_idx, airport_id):
        """Process all valid agent sequences and assign taxiing mode labels."""
        none_outs = (None,) * 8
        seq_data  = np.concatenate(
            frame_data[seq_idx:seq_idx + self.seq_len], axis=0)

        if math.isclose(seq_data[:, G.RAW_IDX.Speed].sum(), 0):
            return none_outs
        if not np.isin(
                seq_data[:, G.RAW_IDX.Type].astype(int), C.AIRCRAFT).sum():
            return none_outs

        unique_agents = np.unique(seq_data[:, G.RAW_IDX.ID])
        num_agents    = len(unique_agents)
        if num_agents < self.min_agents or num_agents > self.max_agents:
            return none_outs

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

            mx = self.ref_data.limits.Altitude.max
            mn = self.ref_data.limits.Altitude.min
            agent_seq[:, alt_idx] = (agent_seq[:, alt_idx] - mn) / (mx - mn)
            print(agent_seq[:, alt_idx])
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

            valid_mask = agent_seq[:, G.RAW_IDX.Interp].astype(bool)
            agent_masks[num_considered, pad_front:pad_end] = valid_mask

            agent_seq = agent_seq[:, G.RAW_SEQ_MASK]
            seq[num_considered, pad_front:pad_end] = agent_seq[:, G.SEQ_ORDER]
            num_considered += 1

            mode_labels.append(mode_label)
            possible_turns_list.append(possible_turns)

        valid_agent_list = np.asarray(valid_agent_list)
        if valid_agent_list.sum() == 0 or num_considered < self.min_agents:
            return none_outs

        rule_based_encoding = self._compute_separate_encoding(mode_labels)
        turn_feasibility    = self._compute_turn_feasibility(
            possible_turns_list)

        return (seq[:num_considered], agent_id_list, agent_type_list,
                valid_agent_list, agent_masks[:num_considered], mode_labels,
                rule_based_encoding, turn_feasibility)

    # ------------------------------------------------------------------
    # Taxiing mode labeling (trajectory-only, prediction segment)
    # ------------------------------------------------------------------

    def _compute_taxiing_mode(self, agent_seq, pad_front, pad_end):
        """Compute taxiing mode label from prediction-segment kinematics.

        Turn label
        ----------
        Based on cumulative heading change over the prediction segment only
        (frames hist_len : seq_len). Heading is unwrapped before smoothing
        to handle 0/360° wrap-around correctly.
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
    # Turn feasibility from map (BFS on simplified graph)
    # ------------------------------------------------------------------

    def _get_turn_feasibility_from_map(self, agent_seq, pad_front) -> set:
        """Determine topologically feasible turn modes via BFS on simplified graph.

        The agent is assumed to be travelling along a semantic edge at the last
        history frame. The BFS starts from the FORWARD endpoint of that edge
        (the endpoint the agent is heading toward), so the agent's travel
        direction is respected and no U-turn is implied.

        The forward endpoint is identified by finding the nearest two junction
        nodes, selecting the semantic edge that connects them, and choosing the
        endpoint whose direction from the agent is consistent with cur_heading.
        The distance from the agent to that forward endpoint is subtracted from
        the total distance budget before BFS begins, reflecting the distance
        already "committed" to reach the next junction.

        Turn classification uses CUMULATIVE bearing change along each BFS path
        relative to cur_heading, consistent with _compute_taxiing_mode. A path
        is only classified as TurnLeft / TurnRight once its accumulated bearing
        change reaches turn_left_threshold / turn_right_threshold.

        BFS state: (node, dist_used, cumulative_turn)

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

            agent_xy    = np.array([
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

            # ── Find forward endpoint ─────────────────────────────────────
            # Query the two nearest junction nodes. The agent lies on the
            # semantic edge between them (or near it). We pick the endpoint
            # whose direction from the agent is consistent with cur_heading
            # (i.e. the one the agent is heading TOWARD), avoiding a U-turn.
            k         = min(2, len(self._kdtree_node_ids))
            dists, idxs = self._node_kdtree.query(agent_xy, k=k)

            forward_node    = None
            forward_dist    = None

            for nn_idx, nn_dist in zip(idxs, dists):
                node    = self._kdtree_node_ids[nn_idx]
                node_xy = self._kdtree_node_xys[nn_idx]

                # Bearing from agent to this candidate node
                delta   = node_xy - agent_xy
                bearing_to_node = math.degrees(
                    math.atan2(delta[0], delta[1])) % 360.0

                # Choose the node the agent is heading toward:
                # bearing to node must be within 90° of cur_heading.
                angle_diff = abs(self._signed_bearing_diff(
                    bearing_to_node, cur_heading))

                if angle_diff <= 90.0:
                    forward_node = node
                    forward_dist = float(nn_dist)
                    break

            # Fallback: if no forward node found (e.g. agent at a junction),
            # use the nearest node regardless of direction.
            if forward_node is None:
                forward_node = self._kdtree_node_ids[idxs[0]]
                forward_dist = float(dists[0])

            # Subtract the distance to the forward endpoint from the budget.
            # This reflects the distance already committed to reach the next
            # junction. The BFS then uses the remaining budget to explore
            # further junctions beyond that point.
            remaining_budget = dist_budget - forward_dist
            if remaining_budget <= 0:
                # Agent can barely reach the next junction; still record the
                # turn options available at that junction.
                remaining_budget = 0.0

            # ── Build adjacency list ──────────────────────────────────────
            adj = {}
            for se in self._simplified_edges:
                u, v = se['u'], se['v']
                adj.setdefault(u, []).append((v, se))
                adj.setdefault(v, []).append((u, se))

            # ── BFS ───────────────────────────────────────────────────────
            # State: (node, dist_used_so_far, cumulative_turn_so_far)
            # visited keyed by (node, turn_bucket) so that paths reaching the
            # same node via very different accumulated turns are both explored.
            def _turn_bucket(turn: float) -> int:
                return int(round(turn / 10.0))

            queue    = [(forward_node, 0.0, 0.0)]
            visited  = {(forward_node, 0)}
            possible = {'Hold'}

            # Classify the forward node itself: the agent is heading there,
            # so record Straight as reachable at minimum.
            possible.add('Straight')

            while queue:
                cur_node, dist_used, cum_turn = queue.pop(0)

                for neighbour, se in adj.get(cur_node, []):
                    # Outgoing bearing from cur_node along this semantic edge
                    if se['u'] == cur_node:
                        out_bearing = se['bearing_in']
                    else:
                        out_bearing = (se['bearing_in'] + 180.0) % 360.0

                    # Cumulative bearing change relative to agent's cur_heading
                    edge_turn    = self._signed_bearing_diff(
                        out_bearing, cur_heading)

                    # Exclude U-turns (not valid airport manoeuvres)
                    if abs(edge_turn) >= 150.0:
                        continue

                    new_dist = dist_used + se['length']
                    if new_dist > remaining_budget:
                        continue

                    new_cum_turn = cum_turn + edge_turn
                    state        = (neighbour, _turn_bucket(new_cum_turn))
                    if state in visited:
                        continue
                    visited.add(state)
                    queue.append((neighbour, new_dist, new_cum_turn))

                    # Classify this path by its cumulative turn
                    if new_cum_turn >= self.turn_left_threshold:
                        possible.add('TurnLeft')
                    elif new_cum_turn <= self.turn_right_threshold:
                        possible.add('TurnRight')
                    else:
                        possible.add('Straight')

            return possible

        except Exception as e:
            print(f"Warning: turn feasibility BFS failed — {e}")
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

        # if taxiing_modes:
        #     print(f"Processed {len(taxiing_modes)} taxiing modes")
        #     print("Turn distribution:",
        #           df_turn['turn_mode'].value_counts(dropna=False).to_dict())
        #     print("Speed distribution:",
        #           df_speed['speed_mode'].value_counts(dropna=False).to_dict())

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

        if possible_turns_list:
            print("Turn feasibility totals:", feasibility.sum().to_dict())

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