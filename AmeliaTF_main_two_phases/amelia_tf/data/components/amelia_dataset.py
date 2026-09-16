import json
import os
import numpy as np
import pickle
import random
import torch
import time

import amelia_tf.utils.data_utils as D
import amelia_scenes.utils.transform_utils as T
import amelia_scenes.utils.global_masks as G

from easydict import EasyDict
from math import radians, sin, cos
from typing import Dict, List

from amelia_tf.data.components.base_dataset import BaseDataset
from amelia_tf.utils import pylogger

log = pylogger.get_pylogger(__name__)


class AmeliaDataset(BaseDataset):
    """ Dataset class for post-processing the SWIM data (i.e., methods required by __getitem__()). """

    def __init__(self, config: EasyDict) -> None:
        super(AmeliaDataset, self).__init__(config=config)

    def prepare_data(self) -> None:
        """ Prepares data for sharding. """
        log.info("Preparing data for training.")

        self.semantic_maps = {}
        self.semantic_pkl = {}
        self.limits = {}
        self.ref_data = {}
        self.scenario_list = {}
        self.hold_lines = {}
        self.runway_segments = {}   # per-airport runway (thr, type==3) segments (lat1,lon1,lat2,lon2)
        self.data_files = []

        airports = list(set([f.split('/')[0] for f in self.split_list]))

        for airport in airports:
            graph_file = os.path.normpath(os.path.join(
                self.context_dir, airport, 'semantic_graph.pkl'))
            with open(graph_file, 'rb') as f:
                temp_dict = pickle.load(f)
                self.semantic_pkl[airport] = temp_dict
                self.semantic_maps[airport] = temp_dict['map_infos']['all_polylines'][:, G.MAP_IDX]
                self.hold_lines[airport] = temp_dict['hold_lines']
                # runway segments: polylines whose type (column index 8) == 3 (thr_id)
                # each row: [lat1, lon1, ..., lat2, lon2, ...]; keep endpoints (lat/lon)
                _pl = temp_dict['map_infos']['all_polylines']
                _rw = _pl[_pl[:, 8] == 3][:, [0, 1, 4, 5]].astype('float32')
                self.runway_segments[airport] = _rw   # (M,4) lat1,lon1,lat2,lon2

            limits_file = os.path.join(self.assets_dir, airport, 'limits.json')
            with open(limits_file, 'r') as fp:
                self.ref_data[airport] = EasyDict(json.load(fp))

            self.limits[airport] = (
                self.ref_data[airport].espg_4326.north,
                self.ref_data[airport].espg_4326.east,
                self.ref_data[airport].espg_4326.south,
                self.ref_data[airport].espg_4326.west
            )
            scenario_files = D.get_filtered_list(
                airport, self.in_data_dir, self.split_list,
                self.min_agents, self.max_agents)

            self.scenario_list[airport] = scenario_files
            print(f"[INFO] airport {airport}: loaded {len(scenario_files)} original files")

        files_per_airport = min(len(f) for _, f in self.scenario_list.items())
        balanced_list = []

        print(f"[INFO] Balancing data:")
        for airport, files in self.scenario_list.items():
            random.seed(self.seed)
            random.shuffle(files)
            selected_files = files[:files_per_airport]
            balanced_list += selected_files
            print(f"[INFO] Selected {len(selected_files)}/{len(files)} scenes from {airport}")

        self.scenario_list = balanced_list
        print(f"[INFO] Data preparation completed: {len(self.scenario_list)} total scenes")

    def collate_batch(self, batch_data: Dict) -> Dict:
        """ Collate function prepares tensor data and adds padding where necessary. """
        batch_size = len(batch_data)
        key_to_list = {}
        for key in batch_data[0].keys():
            key_to_list[key] = [batch_data[idx][key] for idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            if key in ['scenario_id', 'airport_id', 'ego_agent_id', 'ego_agent_id_test', 'num_agents']:
                input_dict[key] = np.asarray(val_list)
            elif key in ['sequences', 'rel_sequences']:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = D.merge_seq3d_by_padding(
                    val_list, max_pad=self.k_agents)
            elif key in ['agent_masks']:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = D.merge_seq2d_by_padding(
                    val_list, max_pad=self.k_agents)
            elif key in ['context', 'adjacency']:
                input_dict[key] = None
                if not val_list[0] is None:
                    val_list = [torch.from_numpy(x).type(
                        torch.FloatTensor) for x in val_list]
                    input_dict[key] = D.merge_seq3d_by_padding(
                        val_list, max_pad=self.k_agents)
            elif key in ['mode_labels']:
                input_dict[key] = val_list
            elif key in ['rule_based_encoding']:
                val_list = [torch.from_numpy(x).type(torch.FloatTensor) for x in val_list]
                input_dict[key] = D.merge_seq2d_by_padding(
                    val_list, max_pad=self.k_agents)
            elif key in ['turn_feasibility']:
                # turn_feasibility: (A, 4) float32 tensor per scene.
                # Pad to k_agents along the agent dimension.
                
                # Fix: Filter out None values
                filtered_list = []
                for x in val_list:
                    if x is None:
                        # Create default turn_feasibility matrix for None values
                        # Default: all turns disallowed, or adjust based on business logic
                        # Note: A represents number of agents, need to know A's size here
                        # Option 1: Skip None samples if batch has other valid samples
                        continue
                        # Option 2: Create default tensor (requires knowing number of agents)
                        # default_turn = np.zeros((num_agents, 4), dtype=np.float32)
                        # filtered_list.append(default_turn)
                    else:
                        filtered_list.append(torch.from_numpy(x).type(torch.FloatTensor))
                
                if len(filtered_list) > 0:
                    input_dict[key] = D.merge_seq2d_by_padding(
                        filtered_list, max_pad=self.k_agents)
                else:
                    # If all samples are None, create empty tensor or skip
                    input_dict[key] = torch.zeros((0, self.k_agents, 4), dtype=torch.float32)
            else:
                val_list = [torch.from_numpy(np.asarray(x)) for x in val_list]
                input_dict[key] = D.merge_seq1d_by_padding(val_list)

        return {
            'batch_size': batch_size, 'scene_dict': input_dict, 'strategy': self.sampling_strategy
        }

    def compute_z_variation_mask(self, sequences: np.array, threshold: float) -> np.array:
        """
        Computes mask for agents based on z-direction variation.
        
        Args:
            sequences: Array of shape (num_agents, timesteps, features)
            threshold: Threshold for z variation. Agents with variation > threshold are masked out.
        
        Returns:
            mask: Boolean array of shape (num_agents,), True for agents to keep, False to mask
        """
        # Get the index for z coordinate
        # Try to get Z index from G.SEQ_IDX if available, otherwise assume index 2
        if hasattr(G.SEQ_IDX, 'Z'):
            z_idx = G.SEQ_IDX.Z
        else:
            # Z is typically the 3rd dimension (index 2) in XYZ
            z_idx = 2
        
        # Extract z positions: shape (num_agents, timesteps)
        z_positions = sequences[:, :, z_idx]
        
        # Compute z variation as the range (max - min) over time
        z_range = np.max(z_positions, axis=1) - np.min(z_positions, axis=1)
        z_variation = z_range
        
        # Check if we should invert the masking logic
        invert_mask = False
        
        if invert_mask:
            # Invert logic: mask agents with small z variation
            mask = z_variation >= threshold
        else:
            # Default logic: mask agents with large z variation
            mask = z_variation <= threshold
        
        # Optionally mask out agents with completely static z (zero variation)
        mask_static_z =  False
        if mask_static_z:
            static_mask = z_variation > 0
            mask = np.logical_and(mask, static_mask)
        
        return mask

    def compute_runway_mask(self, sequences: np.array, airport_id: str,
                            thr_deg: float = 3e-4, min_pts: int = 1,
                            ego_agent: int = None) -> np.array:
        """
        Mask out agents whose trajectory passes over a runway.

        An agent is masked (False) if at least `min_pts` of its trajectory
        points lie within `thr_deg` degrees (~ thr_deg*111km) of any runway
        segment (semantic map polylines with type==3, i.e. thr_id).

        Optimised: fully vectorised distance computation, and if `ego_agent`
        is given only that agent is checked (evaluation only looks at the ego,
        so masking non-ego agents has no effect on the metrics). This makes it
        ~num_agents x faster.

        Args:
            sequences: (num_agents, timesteps, features) with lat/lon per point.
            airport_id: airport code to look up runway segments.
            thr_deg: distance threshold in degrees (3e-4 deg ~ 33 m).
            min_pts: number of on-runway points required to mask the agent.
            ego_agent: if given, only this agent is checked (others kept).

        Returns:
            mask: (num_agents,) bool, True = keep, False = on runway (mask out).
        """
        rw = self.runway_segments.get(airport_id, None)
        num_agents = sequences.shape[0]
        keep = np.ones(num_agents, dtype=bool)
        if rw is None or len(rw) == 0:
            return keep

        lat_idx = getattr(G.SEQ_IDX, 'Lat', 0)
        lon_idx = getattr(G.SEQ_IDX, 'Lon', 1)

        a = rw[:, 0:2]                                     # (M,2)
        b = rw[:, 2:4]                                     # (M,2)
        ab = b - a                                         # (M,2)
        ab2 = np.clip((ab * ab).sum(1), 1e-12, None)       # (M,)

        # points to check: only ego if given, else all agents
        if ego_agent is not None:
            idxs = [ego_agent]
        else:
            idxs = range(num_agents)

        pts = np.stack([sequences[:, :, lat_idx],
                        sequences[:, :, lon_idx]], axis=-1)   # (N,T,2)

        for i in idxs:
            P = pts[i]                                     # (T,2)
            # vectorised point-to-segment distance: (T points) x (M segments)
            ap = P[:, None, :] - a[None]                   # (T,M,2)
            t = np.clip((ap * ab[None]).sum(-1) / ab2[None], 0.0, 1.0)  # (T,M)
            proj = a[None] + t[..., None] * ab[None]       # (T,M,2)
            d = np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1))  # (T,M)
            dmin = d.min(1)                                # (T,) nearest runway per point
            if int((dmin < thr_deg).sum()) >= min_pts:
                keep[i] = False
        return keep

    def transform_sequences(self, sequences: np.array, ego_agent_id: int = 0) -> np.array:
        """ Transforms the scene w.r.t. the ego_agent's reference frame. """
        num_agents, timesteps, _ = sequences.shape
        rel_sequence = np.zeros(shape=(num_agents, timesteps, 7))

        ego_heading = radians(
            sequences[ego_agent_id, self.curr_timestep, G.SEQ_IDX.Heading])

        R = np.array(
            [[cos(ego_heading), -sin(ego_heading), 0.0],
             [sin(ego_heading),  cos(ego_heading), 0.0],
             [0.0,               0.0,              1.0]])
        R = np.repeat(R.reshape(1, 3, 3), num_agents, axis=0)

        rel_xyz = sequences[:, :, G.XYZ] - \
            sequences[ego_agent_id, self.curr_timestep, G.XYZ]
        rel_sequence[:, :, :3] = np.matmul(rel_xyz, R)

        headings = sequences[:, :, G.SEQ_IDX.Heading]
        ego_heading = sequences[ego_agent_id, self.curr_timestep, G.SEQ_IDX.Heading]
        rel_sequence[:, :, 3] = T.wrap_angle(headings - ego_heading)

        KNOTS_TO_KMS = 0.000514444
        headings_rad = np.radians(sequences[:, :, G.SEQ_IDX.Heading])
        speeds = sequences[:, :, G.SEQ_IDX.Speed] * KNOTS_TO_KMS

        vx_abs = speeds * np.cos(headings_rad)
        vy_abs = speeds * np.sin(headings_rad)
        velocity_vectors = np.stack([vx_abs, vy_abs, np.zeros_like(vx_abs)], axis=-1)

        rel_velocity = np.matmul(velocity_vectors, R)
        rel_sequence[:, :, 4] = rel_velocity[:, :, 0]
        rel_sequence[:, :, 5] = rel_velocity[:, :, 1]
        rel_sequence[:, :, 6] = rel_velocity[:, :, 2]
        return rel_sequence

    def transform_context(
        self, semantic_map: np.array, sequences: np.array, rel_sequences: np.array,
        ego_agent: int, limits: list
    ) -> np.array:
        """ Generates the map context for a given sequence and ego agent ID. """
        ego_position = sequences[ego_agent, self.curr_timestep, G.XY]
        ego_heading = radians(
            sequences[ego_agent, self.curr_timestep, G.SEQ_IDX.Heading])
        semantic_map, adjacency = D.compute_local_context_from_ego_agent(
            semantic_map, ego_position, ego_heading, rel_sequences, self.curr_timestep,
            self.num_polylines, self.debug, ego_id=ego_agent, limits=limits
        )
        return semantic_map, adjacency

    def transform_scene_data(self, scene_data: Dict, random_ego: bool = True, ego_agent_id: int = 0) -> Dict:
        """ Transforms scene's global data to the ego-agent's reference frame. """
        MODE_MAP = {
            "TurnLeft_Accel":   0,
            "TurnLeft_Decel":   1,
            "TurnLeft_Normal":  2,
            "TurnLeft_Hold":    3,

            "TurnRight_Accel":  4,
            "TurnRight_Decel":  5,
            "TurnRight_Normal": 6,
            "TurnRight_Hold":   7,

            "Straight_Accel":   8,
            "Straight_Decel":   9,
            "Straight_Normal":  10,
            "Straight_Hold":    11,

            "Hold_Accel":       12,
            "Hold_Decel":       13,
            "Hold_Normal":      14,
            "Hold_Hold":        15,
        }

        sequences   = scene_data['agent_sequences']
        agent_masks = scene_data['agent_masks']
        airport_id  = scene_data['airport_id']

        agents_in_scene = scene_data['meta']['agent_order'][self.sampling_strategy][:self.k_agents]
        num_agents = len(agents_in_scene)
        if random_ego:
            ego_agent = random.randint(a=0, b=num_agents-1)

        else:
            ego_agent = 0
            
        ego_agent_test = ego_agent

        sequences   = sequences[agents_in_scene]
        agent_masks = agent_masks[agents_in_scene]

        # Compute z-variation mask based on z-direction changes
        #z_threshold = 0.020
        #z_variation_mask = self.compute_z_variation_mask(sequences, z_threshold)

        # Compute runway mask: agents whose trajectory passes over a runway
        # (semantic map type==3) are masked out. Keep=True, on-runway=False.
        #runway_mask = self.compute_runway_mask(sequences, airport_id,
        #                                       thr_deg=3e-4, min_pts=1,
        #                                       ego_agent=ego_agent)

        # Per-agent keep mask = keep only if (low z-variation) AND (not on runway)
        #agent_keep = np.logical_and(z_variation_mask, runway_mask)   # (num_agents,)

        # Combine with existing agent masks
        #if len(agent_masks.shape) == 1:
            # If agent_masks is 1D (per-agent), expand to time dimension
        #    combined_mask = np.logical_and(agent_masks[:, None], agent_keep[:, None])
        #    combined_mask = combined_mask.squeeze()
        #else:
            # If agent_masks is 2D (per-agent per-timestep), element-wise AND
        #    combined_mask = np.logical_and(agent_masks, agent_keep[:, None])

        # Update agent_masks with combined mask
        #agent_masks = combined_mask.astype(bool)

        rel_sequences = self.transform_sequences(sequences, ego_agent)

        if self.encode_interp_flag:
            rel_sequences = np.concatenate(
                (rel_sequences, agent_masks[..., None]), axis=-1)

        if self.encode_agent_type:
            agent_types = np.asarray(scene_data['agent_types'])
            agent_types = agent_types[agents_in_scene, :, :]
            index = np.arange(agent_types.size)
            agent_types_onehot = np.zeros(
                shape=(agent_types.shape[0], 1, self.num_agent_types))
            agent_types_onehot[index, 0, agent_types] = 1
            agent_types_onehot = np.tile(agent_types_onehot, (1, self.seq_len, 1))
            rel_sequences = np.concatenate(
                (rel_sequences, agent_types_onehot), axis=-1)

        context_map, adjacency = None, None
        if self.add_context:
            context_map, adjacency = self.transform_context(
                self.semantic_maps[airport_id], sequences, rel_sequences, ego_agent,
                self.limits[airport_id]
            )

        agent_types = np.asarray(scene_data['agent_types'])
        agent_types = agent_types[agents_in_scene]

        # Extract and index mode labels for the selected agents
        mode_labels = scene_data.get('mode_labels', None)
        if mode_labels is not None:
            if isinstance(mode_labels, list):
                mode_labels = [MODE_MAP[m] for m in mode_labels]
                mode_labels = np.asarray(mode_labels)[agents_in_scene]
            else:
                mode_labels = np.array([MODE_MAP[mode_labels]])
                mode_labels = mode_labels[agents_in_scene]

        # Extract and index rule-based encoding for the selected agents
        rule_based_encoding = scene_data.get('rule_based_encoding', None)
        if rule_based_encoding is not None:
            rule_based_encoding = np.asarray(rule_based_encoding)[agents_in_scene]

        # Extract and index turn feasibility for the selected agents.
        # turn_feasibility is stored as a pd.DataFrame (N_agents, 4); convert to
        # float32 ndarray (A, 4) and index to the selected agents.
        # Column order: [feasible_TurnLeft, feasible_TurnRight,
        #                feasible_Straight, feasible_Hold]
        turn_feasibility = scene_data.get('turn_feasibility', None)
        if turn_feasibility is not None:
            turn_feasibility = np.asarray(turn_feasibility, dtype=np.float32)[agents_in_scene]

        return {
            'scenario_id':       scene_data['scenario_id'],
            'airport_id':        airport_id,
            'agent_ids':         scene_data['agent_ids'],
            'agent_types':       agent_types,
            'agent_masks':       agent_masks,
            'ego_agent_id':      ego_agent,
            'ego_agent_id_test': ego_agent_test,
            'num_agents':        sequences.shape[0],
            'sequences':         sequences,
            'rel_sequences':     rel_sequences,
            'agents_in_scene':   agents_in_scene,
            'context':           context_map,
            'adjacency':         adjacency,
            'mode_labels':       mode_labels,
            'rule_based_encoding': rule_based_encoding,
            'turn_feasibility':  turn_feasibility
        }

    def transform_scene_data_bench(
        self, scene_data: Dict, agents_in_scene: list, ego_agent: int
    ) -> Dict:
        """ Transforms scene's global data to the ego-agent's reference frame (benchmark). """
        sequences   = scene_data['agent_sequences']
        agent_masks = scene_data['agent_masks']
        airport_id  = scene_data['airport_id']

        sequences   = sequences[agents_in_scene]
        agent_masks = agent_masks[agents_in_scene]

        # Compute z-variation mask based on z-direction changes for benchmark data
        z_threshold = getattr(self.config, 'z_variation_threshold', 0.020)
        z_variation_mask = self.compute_z_variation_mask(sequences, z_threshold)
        
        # Combine z-variation mask with existing agent masks
        if len(agent_masks.shape) == 1:
            combined_mask = np.logical_and(agent_masks[:, None], z_variation_mask[:, None])
            combined_mask = combined_mask.squeeze()
        else:
            combined_mask = np.logical_and(agent_masks, z_variation_mask[:, None])
        
        agent_masks = combined_mask.astype(bool)

        rel_sequences = self.transform_sequences(sequences, ego_agent)

        context_map, adjacency = None, None
        context_map, adjacency = self.transform_context(
            self.semantic_maps[airport_id], sequences, rel_sequences, ego_agent,
            self.limits[airport_id]
        )

        agent_types = np.asarray(scene_data['agent_types'])
        agent_types = agent_types[agents_in_scene]

        # Extract turn feasibility for benchmark (no mode labels needed)
        turn_feasibility = scene_data.get('turn_feasibility', None)
        if turn_feasibility is not None:
            turn_feasibility = np.asarray(turn_feasibility, dtype=np.float32)[agents_in_scene]

        return {
            'scenario_id':      scene_data['scenario_id'],
            'airport_id':       airport_id,
            'agent_ids':        scene_data['agent_ids'],
            'agent_types':      agent_types,
            'agent_masks':      agent_masks,
            'ego_agent_id':     ego_agent,
            'num_agents':       sequences.shape[0],
            'sequences':        sequences,
            'rel_sequences':    rel_sequences,
            'agents_in_scene':  agents_in_scene,
            'context':          context_map,
            'adjacency':        adjacency,
            'turn_feasibility': turn_feasibility,
            'z_variation_mask': z_variation_mask,  # Store for debugging/info
        }

    def __len__(self):
        return len(self.scenario_list)

    def __getitem__(self, index):
        """ Loads and transforms the scene at the given index. """
        item = self.scenario_list[index]
        with open(str(item), 'rb') as f:
            data = pickle.load(f)
        return self.transform_scene_data(data)