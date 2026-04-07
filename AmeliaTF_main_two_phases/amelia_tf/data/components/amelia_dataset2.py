import json
import os
import numpy as np
import pickle
import random
import torch
from enum import IntEnum
from typing import Dict, List, Tuple, Optional

import amelia_tf.utils.data_utils as D
import amelia_scenes.utils.transform_utils as T
import amelia_scenes.utils.global_masks as G

from easydict import EasyDict
from math import radians, sin, cos

from amelia_tf.data.components.base_dataset import BaseDataset
from amelia_tf.utils import pylogger

log = pylogger.get_pylogger(__name__)


class TaxiMode(IntEnum):
    """Taxiing maneuver modes"""
    STRAIGHT = 0    # Go straight
    LEFT_TURN = 1   # Turn left
    RIGHT_TURN = 2  # Turn right
    HOLD = 3        # Hold/wait


class AmeliaDataset(BaseDataset):
    """ Dataset class for post-processing the SWIM data with taxi mode prediction features. """

    def __init__(self, config: EasyDict) -> None:
        """ Inherits base methods from BaseDataset.

        Inputs
        ------
            config[EasyDict]: dictionary containing configuration parameters needed to process the
            airport trajectory data.
        """
        super(AmeliaDataset, self).__init__(config=config)
        
        # Taxi mode feature configuration
        self.add_taxi_mode_features = config.get('add_taxi_mode_features', False)
        self.intersection_search_distance = config.get('intersection_search_distance', 300.0)
        self.max_intersections_ahead = config.get('max_intersections_ahead', 3)
        self.straight_angle_threshold = config.get('straight_angle_threshold', np.radians(30))
        self.turn_angle_threshold = config.get('turn_angle_threshold', np.radians(30))
        self.max_turn_angle = config.get('max_turn_angle', np.radians(90))
        
        # Cache for intersection graphs per airport
        self.intersection_graphs = {}
        
        # Cache for current intersection features (used in __getitem__)
        self._current_intersection_features = None

    def prepare_data(self) -> None:
        """ Prepares data for sharding: loads the graphs and limit files, and prepares output
        directories and input files. Also builds intersection graphs for each airport.
        """
        log.info("Preparing data for training.")

        self.semantic_maps = {}
        self.semantic_pkl = {}
        self.limits = {}
        self.ref_data = {}
        self.scenario_list = {}
        self.hold_lines = {}
        self.data_files = []

        airports = list(set([f.split('\\')[0] for f in self.split_list]))
        for airport in airports:
            graph_file = os.path.normpath(os.path.join(
                self.context_dir, airport, 'semantic_graph.pkl'))
            with open(graph_file, 'rb') as f:
                temp_dict = pickle.load(f)
                self.semantic_pkl[airport] = temp_dict
                self.semantic_maps[airport] = temp_dict['map_infos']['all_polylines'][:, G.MAP_IDX]
                self.hold_lines[airport] = temp_dict['hold_lines']
            
            # Build intersection graph for this airport
            if self.add_taxi_mode_features:
                self.intersection_graphs[airport] = self._build_intersection_graph(
                    self.semantic_maps[airport]
                )

            limits_file = os.path.join(self.assets_dir, airport, 'limits.json')
            with open(limits_file, 'r') as fp:
                self.ref_data[airport] = EasyDict(json.load(fp))

            self.limits[airport] = (
                self.ref_data[airport].espg_4326.north,
                self.ref_data[airport].espg_4326.east,
                self.ref_data[airport].espg_4326.south,
                self.ref_data[airport].espg_4326.west
            )

            self.scenario_list[airport] = D.get_filtered_list(
                airport, self.in_data_dir, self.split_list, self.min_agents, self.max_agents)

        # TODO: find cleaner solution to this:
        # Balance the number of scenarios in the multi-airport setting. For some airports, scenario
        # rejection is more marked, but for a 'fair' comparison I'm trying to keep the balance
        # between airport. Also, if an airport gets too many rejections the number of files that get
        # in per airport will depend on it, which is also not great.
        files_per_airport = min(len(f) for _, f in self.scenario_list.items())
        balanced_list = []
        for airport, files in self.scenario_list.items():
            random.seed(self.seed)
            random.shuffle(files)
            balanced_list += files[:files_per_airport]
        self.scenario_list = balanced_list

    def _build_intersection_graph(self, semantic_map: np.array) -> Dict:
        """Build intersection graph from vectorized edge representation.
        
        Each edge is represented as: [start_x, start_y, start_z, start_heading,
                                       end_x, end_y, end_z, end_heading,
                                       semantic_one_hot]
        
        Args:
            semantic_map: Array of shape [num_edges, 11] containing edge vectors
        
        Returns:
            Dictionary containing:
                - nodes: {node_id: {'position': [x,y], 'edges': [edge_ids], 'is_intersection': bool}}
                - edges: {edge_id: {'start': node_id, 'end': node_id, 'type': int, 'direction': float}}
                - node_positions: list of node coordinates for quick lookup
        """
        # Node clustering threshold (meters)
        NODE_CLUSTER_THRESHOLD = 5.0
        
        nodes = {}  # node_id -> {'position': [x,y], 'edges': set(), 'is_intersection': False}
        edges = {}  # edge_id -> {'start': node_id, 'end': node_id, 'type': int, 'direction': float}
        node_positions = []  # List of (x, y) for distance calculations
        
        def _find_or_create_node(position: np.array) -> int:
            """Find existing node within threshold or create new one."""
            for node_id, node in nodes.items():
                dist = np.linalg.norm(node['position'] - position)
                if dist < NODE_CLUSTER_THRESHOLD:
                    return node_id
            
            # Create new node
            node_id = len(nodes)
            nodes[node_id] = {
                'position': position.copy(),
                'edges': set(),
                'is_intersection': False
            }
            node_positions.append(position)
            return node_id
        
        # Process each edge
        for edge_id in range(semantic_map.shape[0]):
            edge_vec = semantic_map[edge_id]
            
            # Extract start and end positions
            start_pos = edge_vec[0:2]  # x, y
            end_pos = edge_vec[4:6]    # x, y
            
            # Get semantic type (one-hot to class index)
            semantic_type = np.argmax(edge_vec[8:11])
            
            # Find or create nodes
            start_node = _find_or_create_node(start_pos)
            end_node = _find_or_create_node(end_pos)
            
            # Compute edge direction (from start to end)
            direction = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
            
            # Store edge info
            edges[edge_id] = {
                'start': start_node,
                'end': end_node,
                'type': semantic_type,
                'direction': direction
            }
            
            # Update node edge connections
            nodes[start_node]['edges'].add(edge_id)
            nodes[end_node]['edges'].add(edge_id)
        
        # Mark intersections (nodes with >= 3 edges)
        for node_id in nodes:
            nodes[node_id]['is_intersection'] = len(nodes[node_id]['edges']) >= 3
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_positions': np.array(node_positions)
        }

    def _find_current_edge(self, 
                           graph: Dict, 
                           position: np.array, 
                           heading: float,
                           max_distance: float = 10.0) -> Optional[int]:
        """Find the edge the agent is currently on based on position and heading.
        
        Args:
            graph: Intersection graph
            position: Current position [x, y] in global coordinates
            heading: Current heading in radians
            max_distance: Maximum distance to consider an edge
        
        Returns:
            Edge ID if found, None otherwise
        """
        best_edge_id = None
        best_score = float('inf')
        
        for edge_id, edge in graph['edges'].items():
            start_pos = graph['nodes'][edge['start']]['position']
            end_pos = graph['nodes'][edge['end']]['position']
            
            # Compute distance to line segment
            line_vec = end_pos - start_pos
            point_vec = position - start_pos
            line_len = np.linalg.norm(line_vec)
            
            if line_len < 1e-6:
                continue
                
            # Project point onto line
            t = np.dot(point_vec, line_vec) / (line_len * line_len)
            t = np.clip(t, 0.0, 1.0)
            closest_point = start_pos + t * line_vec
            distance = np.linalg.norm(position - closest_point)
            
            if distance > max_distance:
                continue
            
            # Compute alignment with heading
            edge_dir = edge['direction']
            angle_diff = abs(T.wrap_angle(edge_dir - heading))
            alignment_score = min(angle_diff, np.pi - angle_diff)
            
            # Combined score (prioritize alignment over distance)
            score = distance + 0.5 * alignment_score
            
            if score < best_score:
                best_score = score
                best_edge_id = edge_id
        
        return best_edge_id

    def _get_next_intersections(self,
                                graph: Dict,
                                start_position: np.array,
                                start_heading: float,
                                start_edge_id: Optional[int] = None,
                                max_distance: float = 300.0,
                                max_count: int = 3) -> List[Dict]:
        """Find intersections ahead along the path.
        
        Args:
            graph: Intersection graph
            start_position: Starting position
            start_heading: Starting heading
            start_edge_id: Current edge ID (if known)
            max_distance: Maximum search distance
            max_count: Maximum number of intersections to return
        
        Returns:
            List of intersection features, each containing:
                - distance: Distance from start to this intersection
                - position: Intersection position
                - outgoing_directions: List of possible directions from this intersection
                - num_outgoing: Number of outgoing paths
                - has_straight: Whether straight path exists
                - has_left: Whether left turn exists
                - has_right: Whether right turn exists
        """
        intersections = []
        
        # Find current edge if not provided
        if start_edge_id is None:
            start_edge_id = self._find_current_edge(graph, start_position, start_heading)
            if start_edge_id is None:
                return intersections
        
        # Navigate along the path
        current_edge_id = start_edge_id
        current_node = self._get_next_node(graph, current_edge_id, start_heading)
        distance_traveled = 0.0
        
        for _ in range(max_count * 2):  # Safety limit
            node = graph['nodes'][current_node]
            
            if node['is_intersection']:
                node_pos = node['position']
                total_distance = distance_traveled + np.linalg.norm(node_pos - start_position)
                
                if total_distance > max_distance:
                    break
                
                # Get possible directions from this intersection
                outgoing = self._get_outgoing_directions(graph, current_node, current_edge_id)
                outgoing_dirs = [d for d, _ in outgoing]
                outgoing_types = [t for _, t in outgoing]
                
                # Classify turn types
                has_straight = self._has_straight_direction(outgoing_dirs, start_heading)
                has_left = self._has_left_turn(outgoing_dirs, start_heading)
                has_right = self._has_right_turn(outgoing_dirs, start_heading)
                
                intersections.append({
                    'distance': total_distance,
                    'position': node_pos,
                    'outgoing_directions': outgoing_dirs,
                    'outgoing_types': outgoing_types,
                    'num_outgoing': len(outgoing),
                    'has_straight': has_straight,
                    'has_left': has_left,
                    'has_right': has_right,
                })
                
                # Find straight path to continue search
                straight_edge = self._get_straight_edge(graph, current_node, current_edge_id, start_heading)
                if straight_edge is None:
                    break
                
                current_edge_id = straight_edge
                current_node = self._get_next_node(graph, current_edge_id, current_node)
                distance_traveled = total_distance
            else:
                # Not an intersection, continue to next node
                next_node = self._get_other_node(graph, current_edge_id, current_node)
                edge_length = self._get_edge_length(graph, current_edge_id)
                distance_traveled += edge_length
                current_node = next_node
        
        return intersections[:max_count]

    def _get_next_node(self, graph: Dict, edge_id: int, heading: float) -> int:
        """Get the node that is forward along the edge given heading."""
        edge = graph['edges'][edge_id]
        start_pos = graph['nodes'][edge['start']]['position']
        end_pos = graph['nodes'][edge['end']]['position']
        
        # Direction from start to end
        edge_dir = edge['direction']
        angle_diff = T.wrap_angle(edge_dir - heading)
        
        # If heading is roughly aligned with edge direction, go to end
        if abs(angle_diff) < np.pi / 2:
            return edge['end']
        else:
            return edge['start']

    def _get_other_node(self, graph: Dict, edge_id: int, current_node: int) -> int:
        """Get the other node of an edge."""
        edge = graph['edges'][edge_id]
        if edge['start'] == current_node:
            return edge['end']
        else:
            return edge['start']

    def _get_edge_length(self, graph: Dict, edge_id: int) -> float:
        """Compute the length of an edge."""
        edge = graph['edges'][edge_id]
        start_pos = graph['nodes'][edge['start']]['position']
        end_pos = graph['nodes'][edge['end']]['position']
        return np.linalg.norm(end_pos - start_pos)

    def _get_outgoing_directions(self, 
                                 graph: Dict, 
                                 node_id: int, 
                                 incoming_edge_id: int) -> List[Tuple[float, int]]:
        """Get all possible directions from a node excluding the incoming edge."""
        node = graph['nodes'][node_id]
        directions = []
        
        for edge_id in node['edges']:
            if edge_id == incoming_edge_id:
                continue
            
            edge = graph['edges'][edge_id]
            if edge['start'] == node_id:
                directions.append((edge['direction'], edge['type']))
            else:
                # Edge points into the node, so outgoing direction is opposite
                opposite_dir = T.wrap_angle(edge['direction'] + np.pi)
                directions.append((opposite_dir, edge['type']))
        
        return directions

    def _get_straight_edge(self, 
                          graph: Dict, 
                          node_id: int, 
                          incoming_edge_id: int,
                          heading: float) -> Optional[int]:
        """Find the edge that goes straight through the intersection."""
        outgoing = self._get_outgoing_directions(graph, node_id, incoming_edge_id)
        
        best_edge_id = None
        best_angle_diff = float('inf')
        
        for edge_id, (direction, _) in outgoing:
            angle_diff = abs(T.wrap_angle(direction - heading))
            if angle_diff < self.straight_angle_threshold and angle_diff < best_angle_diff:
                best_angle_diff = angle_diff
                # Need to find the actual edge ID
                for eid in graph['nodes'][node_id]['edges']:
                    if eid != incoming_edge_id:
                        e = graph['edges'][eid]
                        if (e['start'] == node_id and abs(T.wrap_angle(e['direction'] - direction)) < 0.1) or \
                           (e['end'] == node_id and abs(T.wrap_angle(e['direction'] + np.pi - direction)) < 0.1):
                            best_edge_id = eid
                            break
        
        return best_edge_id

    def _has_straight_direction(self, directions: List[float], heading: float) -> bool:
        """Check if there is a straight path."""
        for d in directions:
            angle_diff = abs(T.wrap_angle(d - heading))
            if angle_diff < self.straight_angle_threshold:
                return True
        return False

    def _has_left_turn(self, directions: List[float], heading: float) -> bool:
        """Check if there is a left turn option."""
        for d in directions:
            angle_diff = T.wrap_angle(d - heading)
            if -self.max_turn_angle <= angle_diff <= -self.turn_angle_threshold:
                return True
        return False

    def _has_right_turn(self, directions: List[float], heading: float) -> bool:
        """Check if there is a right turn option."""
        for d in directions:
            angle_diff = T.wrap_angle(d - heading)
            if self.turn_angle_threshold <= angle_diff <= self.max_turn_angle:
                return True
        return False

    def _compute_available_modes(self, intersections: List[Dict]) -> List[int]:
        """Compute all possible taxi modes based on intersections ahead."""
        modes = set([TaxiMode.HOLD])  # Always possible to hold
        
        for inter in intersections:
            if inter['has_straight']:
                modes.add(TaxiMode.STRAIGHT)
            if inter['has_left']:
                modes.add(TaxiMode.LEFT_TURN)
            if inter['has_right']:
                modes.add(TaxiMode.RIGHT_TURN)
        
        return list(modes)

    def _compute_intersection_features(self,
                                        position: np.array,
                                        heading: float,
                                        airport_id: str) -> Dict:
        """Compute intersection-based taxi mode features.
        
        Args:
            position: Current position [x, y] in global coordinates
            heading: Current heading in radians
            airport_id: Airport identifier
        
        Returns:
            Dictionary containing intersection features and available taxi modes
        """
        graph = self.intersection_graphs.get(airport_id)
        if graph is None:
            return {
                'intersections': [],
                'num_intersections': 0,
                'available_modes': [TaxiMode.STRAIGHT, TaxiMode.LEFT_TURN, 
                                   TaxiMode.RIGHT_TURN, TaxiMode.HOLD]
            }
        
        # Find intersections ahead
        intersections = self._get_next_intersections(
            graph, position, heading, 
            max_distance=self.intersection_search_distance,
            max_count=self.max_intersections_ahead
        )
        
        # Encode intersections as fixed-length features (max 3 intersections)
        intersection_features = {}
        for i in range(self.max_intersections_ahead):
            if i < len(intersections):
                inter = intersections[i]
                intersection_features[f'inter_{i}_dist'] = inter['distance']
                intersection_features[f'inter_{i}_num_outgoing'] = inter['num_outgoing']
                intersection_features[f'inter_{i}_has_straight'] = 1 if inter['has_straight'] else 0
                intersection_features[f'inter_{i}_has_left'] = 1 if inter['has_left'] else 0
                intersection_features[f'inter_{i}_has_right'] = 1 if inter['has_right'] else 0
            else:
                intersection_features[f'inter_{i}_dist'] = -1.0
                intersection_features[f'inter_{i}_num_outgoing'] = 0
                intersection_features[f'inter_{i}_has_straight'] = 0
                intersection_features[f'inter_{i}_has_left'] = 0
                intersection_features[f'inter_{i}_has_right'] = 0
        
        intersection_features['num_intersections'] = len(intersections)
        intersection_features['available_modes'] = self._compute_available_modes(intersections)
        intersection_features['raw_intersections'] = intersections
        
        return intersection_features

    def collate_batch(self, batch_data: Dict) -> Dict:
        """ Collate function prepares tensor data and adds padding where necessary.

        Input
        -----
            batch_data[Dict]: tuple containing the scenes to prepare.

        Output
        ------
            batch_data[Dict]: dictionary containing the prepared batch.
        """
        batch_size = len(batch_data)
        key_to_list = {}
        for key in batch_data[0].keys():
            key_to_list[key] = [batch_data[idx][key]
                                for idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            if key in ['scenario_id', 'airport_id', 'ego_agent_id', 'num_agents']:
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
            elif key in ['intersection_features']:
                # Handle intersection features (dictionary)
                input_dict[key] = val_list
            else:
                val_list = [torch.from_numpy(np.asarray(x)) for x in val_list]
                input_dict[key] = D.merge_seq1d_by_padding(val_list)

        return {
            'batch_size': batch_size, 'scene_dict': input_dict, 'strategy': self.sampling_strategy
        }

    def transform_sequences(self, sequences: np.array, ego_agent_id: int = 0) -> np.array:
        """ Transforms the scene w.r.t. the ego_agent's reference frame.

        Inputs
        ------
            sequences[np.array]: numpy array containing scene information in absolute coordinates.
            ego_agent_id[int]: index of the agent chosen to be the ego-agent (NOTE: currently, this
            is done randomly during sharding.)

        Output
        ------
            rel_seq[np.array]: numpy array containing the trajectory sequences in the ego-agent's
            reference frame.
        """
        num_agents, timesteps, _ = sequences.shape
        rel_sequence = np.zeros(
            shape=(num_agents, timesteps, 4))  # [x, y, z, heading]

        # the ego-agent's heading at the 'current time step'
        ego_heading = radians(
            sequences[ego_agent_id, self.curr_timestep, G.SEQ_IDX.Heading])

        R = np.array(
            [[cos(ego_heading), -sin(ego_heading), 0.0],
             [sin(ego_heading),  cos(ego_heading), 0.0],
             [0.0,               0.0, 1.0]])
        R = np.repeat(R.reshape(1, 3, 3), num_agents, axis=0)

        rel_xyz = sequences[:, :, G.XYZ] - \
            sequences[ego_agent_id, self.curr_timestep, G.XYZ]
        rel_sequence[:, :, :3] = np.matmul(rel_xyz, R)

        headings = sequences[:, :, G.SEQ_IDX.Heading]
        ego_heading = sequences[ego_agent_id,
                                self.curr_timestep, G.SEQ_IDX.Heading]

        # wrap the angle
        rel_sequence[:, :, -1] = T.wrap_angle(headings - ego_heading)
        return rel_sequence

    def transform_context(
        self, semantic_map: np.array, sequences: np.array, rel_sequences: np.array, ego_agent: int,
        limits: list
    ) -> np.array:
        """ Generates the map context for a given sequence and ego agent ID. Using the polyline
        representaton of the map.

        Inputs
        ------
            semantic_map[np.array]: numpy array containing the vectorized representation of the map.
            sequences[np.array]: numpy array with the scene information in global frame.
            rel_sequences[np.array]: numpy array with the scene information in the ego-agent's frame.
            ego_agent[float]: index of the ego agent.
            limits[list]: list containing the global limits in latitude and longitude of the map.

        Outputs
        -------
            semantic_map[np.array(N, #Polylines, D = 11)]: the corresponding scene's map information
            in relative frame.
        """
        ego_position = sequences[ego_agent, self.curr_timestep, G.XY]
        ego_heading = radians(
            sequences[ego_agent, self.curr_timestep, G.SEQ_IDX.Heading])
        semantic_map, adjacency = D.compute_local_context_from_ego_agent(
            semantic_map, ego_position, ego_heading, rel_sequences, self.curr_timestep,
            self.num_polylines, self.debug, ego_id=ego_agent, limits=limits
        )
        return semantic_map, adjacency

    def transform_scene_data(self, scene_data: Dict, random_ego: bool = True, ego_agent_id: int = 0) -> Dict:
        """ Transforms scene's global data to the ego-agent's reference frame.

        Input
        -----
            scene_data[Dict]: a dictionary containing the pre-processed scene information.

        Output
        ------
            scene_dict[Dict]: the transformed scene data.
        """
        sequences = scene_data['agent_sequences']
        agent_masks = scene_data['agent_masks']
        airport_id = scene_data['airport_id']

        # get the ego agent id from the scene data
        agents_in_scene = scene_data['meta']['agent_order'][self.sampling_strategy][:self.k_agents]
        num_agents = len(agents_in_scene)
        if random_ego:
            random.seed(self.seed)
            ego_agent = random.randint(a=0, b=num_agents-1)

        elif not ego_agent_id:
            # most critical agent
            ego_agent = 0
        elif ego_agent_id in scene_data['agent_ids']:
            # get the index of the ego agent in the scene data
            ego_agent_idx = scene_data['agent_ids'].index(ego_agent_id)
            if not ego_agent_idx in agents_in_scene:
                agents_in_scene = np.append([ego_agent_idx], agents_in_scene)
                agents_in_scene = agents_in_scene[:self.k_agents]
                ego_agent = 0
            else:
                ego_agent = np.where(agents_in_scene == ego_agent_idx)[0][0]
        else:
            raise ValueError(f"Ego agent {ego_agent_id} not in scene data!")

        # set ego agent to the first agent in the scene
        if ego_agent != 0:
            # set ego agent in the first position and shift the rest
            agents_in_scene = np.append([agents_in_scene[ego_agent]], np.delete(agents_in_scene, ego_agent))
            ego_agent = 0

        # Slice the number of agents from the sequence and define random ego agent
        sequences = sequences[agents_in_scene]
        agent_masks = agent_masks[agents_in_scene]

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
            agent_types_onehot = np.tile(
                agent_types_onehot, (1, self.seq_len, 1))
            rel_sequences = np.concatenate(
                (rel_sequences, agent_types_onehot), axis=-1)

        # Transform context (original functionality)
        context_map, adjacency = None, None
        if self.add_context:
            context_map, adjacency = self.transform_context(
                self.semantic_maps[airport_id], sequences, rel_sequences, ego_agent,
                self.limits[airport_id]
            )
        
        # Compute intersection-based taxi mode features
        intersection_features = None
        if self.add_taxi_mode_features and airport_id in self.intersection_graphs:
            ego_position = sequences[ego_agent, self.curr_timestep, G.XY]
            ego_heading = sequences[ego_agent, self.curr_timestep, G.SEQ_IDX.Heading]
            
            intersection_features = self._compute_intersection_features(
                ego_position, radians(ego_heading), airport_id
            )
        
        agent_types = np.asarray(scene_data['agent_types'])
        agent_types = agent_types[agents_in_scene]

        result = {
            'scenario_id': scene_data['scenario_id'],
            'airport_id': airport_id,
            'agent_ids': scene_data['agent_ids'],
            'agent_types': agent_types,
            'agent_masks': agent_masks,
            'ego_agent_id': ego_agent,
            'num_agents': sequences.shape[0],
            'sequences': sequences,
            'rel_sequences': rel_sequences,
            'agents_in_scene': agents_in_scene,
            'context': context_map,
            'adjacency': adjacency,
        }
        
        # Add intersection features if computed
        if intersection_features is not None:
            result['intersection_features'] = intersection_features
            result['valid_taxi_modes'] = intersection_features['available_modes']
        
        return result

    def transform_scene_data_bench(self, scene_data: Dict, agents_in_scene: list, ego_agent: int) -> Dict:
        """ Transforms scene's global data to the ego-agent's reference frame.

        Input
        -----
            scene_data[Dict]: a dictionary containing the pre-processed scene information.

        Output
        ------
            scene_dict[Dict]: the transformed scene data.
        """
        sequences = scene_data['agent_sequences']
        agent_masks = scene_data['agent_masks']
        airport_id = scene_data['airport_id']

        # Slice the number of agents from the sequence and define random ego agent
        sequences = sequences[agents_in_scene]
        agent_masks = agent_masks[agents_in_scene]

        rel_sequences = self.transform_sequences(sequences, ego_agent)

        # TODO: debug
        context_map, adjacency = None, None
        context_map, adjacency = self.transform_context(
            self.semantic_maps[airport_id], sequences, rel_sequences, ego_agent,
            self.limits[airport_id]
        )
        agent_types = np.asarray(scene_data['agent_types'])
        agent_types = agent_types[agents_in_scene]

        return {
            'scenario_id': scene_data['scenario_id'],
            'airport_id': airport_id,
            'agent_ids': scene_data['agent_ids'],
            'agent_types': agent_types,
            'agent_masks': agent_masks,
            'ego_agent_id': ego_agent,
            'num_agents': sequences.shape[0],
            'sequences': sequences,
            'rel_sequences': rel_sequences,
            'agents_in_scene': agents_in_scene,
            'context': context_map,
            'adjacency': adjacency,
        }

    def __len__(self):
        return len(self.scenario_list)

    def __getitem__(self, index):
        """ Loads scene from given index and transforms it w.r.t. to its corresponding ego-agent."""
        with open(self.scenario_list[index], 'rb') as f:
            data = pickle.load(f)

        data = self.transform_scene_data(data)

        return data