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
        """ Inherits base methods from BaseDataset.

        Inputs
        ------
            config[EasyDict]: dictionary containing configuration parameters needed to process the
            airport trajectory data.
        """
        super(AmeliaDataset, self).__init__(config=config)
        
        # Packed data managers
        self.pack_managers = {}
        self.use_packed = False
        
        # ScenePackManager high-performance configuration parameters
        self.max_memory_cache = getattr(config, 'max_memory_cache', 200)    # Number of files in memory cache
        self.max_tar_cache = getattr(config, 'max_tar_cache', 5)           # Number of tar cache
        self.prefetch_size = getattr(config, 'prefetch_size', 20)          # Prefetch size
        self.pack_num_workers = getattr(config, 'pack_num_workers', 2)     # Number of prefetch threads
        self.enable_warmup = getattr(config, 'enable_warmup', True)       # Whether to enable warmup
        
        # Performance monitoring
        self.load_times = []
        self.last_stat_time = time.time()
        self.total_samples_loaded = 0
        
        # Whether to enable packed mode
        if hasattr(config, 'use_packed_scenes') and config.use_packed_scenes:
            if hasattr(config, 'scene_index_path'):
                self.packed_root = config.scene_index_path
                self.use_packed = True
                print(f"[INFO] Enabled high-performance packed mode, root directory: {self.packed_root}")
                print(f"[INFO] ScenePackManager configuration:")
                print(f"  Memory cache: {self.max_memory_cache} files")
                print(f"  Tar cache: {self.max_tar_cache} files")  
                print(f"  Prefetch size: {self.prefetch_size}")
                print(f"  Worker threads: {self.pack_num_workers}")
                print(f"  Warmup cache: {'Yes' if self.enable_warmup else 'No'}")
            else:
                print("[WARNING] No packed_root set, will use original files")
    
    def _get_scene_manager(self, airport):
        """Get scene manager (lazy loading)"""
        if not self.use_packed:
            return None
            
        # If already loaded, return directly
        if airport in self.pack_managers:
            return self.pack_managers[airport]
        
        # Load new manager
        index_path = os.path.join(self.packed_root, airport, "scene_index.json")
        
        if os.path.exists(index_path):
            try:
                # First try high-performance manager
                from amelia_tf.utils.scene_pack_manager import FastScenePackManager
                
                manager = FastScenePackManager(
                    index_path,
                    max_memory_cache=self.max_memory_cache,
                    max_tar_cache=self.max_tar_cache,
                    prefetch_size=self.prefetch_size,
                    num_workers=self.pack_num_workers
                )
                
                self.pack_managers[airport] = manager
                print(f"[INFO] Using high-performance manager: {airport}")
                
                # Warmup cache
                if self.enable_warmup:
                    try:
                        manager.warmup_cache(num_samples=min(100, manager.get_total_files()))
                    except:
                        print(f"[WARNING] Warmup cache failed, continuing...")
                
                return manager
                
            except ImportError:
                # Fallback to standard manager
                print(f"[WARNING] High-performance manager not available, using standard manager: {airport}")
                try:
                    from amelia_tf.utils.scene_pack_manager import ScenePackManager
                    manager = ScenePackManager(index_path)
                    self.pack_managers[airport] = manager
                    return manager
                except Exception as e:
                    print(f"[ERROR] Failed to load {airport}: {e}")
                    self.pack_managers[airport] = None
                    return None
            except Exception as e:
                print(f"[ERROR] Failed to load high-performance manager {airport}: {e}")
                self.pack_managers[airport] = None
                return None
        else:
            print(f"[ERROR] No packed files found for: {airport}")
            self.pack_managers[airport] = None
            return None

    def prepare_data(self) -> None:
        """ Prepares data for sharding: loads the graphs and limit files, and prepares output
        directories and input files.
        """
        log.info("Preparing data for training.")

        self.semantic_maps = {}
        self.semantic_pkl = {}
        self.limits = {}
        self.ref_data = {}
        self.scenario_list = {}
        self.hold_lines = {}
        self.data_files = []
        
        airports = list(set([f.split('/')[0] for f in self.split_list]))
        
        for airport in airports:
            # Load semantic map
            graph_file = os.path.normpath(os.path.join(
                self.context_dir, airport, 'semantic_graph.pkl'))
            with open(graph_file, 'rb') as f:
                temp_dict = pickle.load(f)
                self.semantic_pkl[airport] = temp_dict
                self.semantic_maps[airport] = temp_dict['map_infos']['all_polylines'][:, G.MAP_IDX]
                self.hold_lines[airport] = temp_dict['hold_lines']

            # Load limits file
            limits_file = os.path.join(self.assets_dir, airport, 'limits.json')
            with open(limits_file, 'r') as fp:
                self.ref_data[airport] = EasyDict(json.load(fp))

            self.limits[airport] = (
                self.ref_data[airport].espg_4326.north,
                self.ref_data[airport].espg_4326.east,
                self.ref_data[airport].espg_4326.south,
                self.ref_data[airport].espg_4326.west
            )
            
            manager = self._get_scene_manager(airport)

            if manager:
                # Extract scene names for current airport from split_list
                split_scene_names = self._extract_scene_names_from_split_list(airport)
                
                print(f"[INFO] airport {airport}: split_list specifies {len(split_scene_names)} scenes")
                
                if not split_scene_names:
                    print(f"[WARNING] No scenes found for {airport}, skipping")
                    self.scenario_list[airport] = []
                    continue
                
                # Use filter_by_agents_and_scenes method (if available)
                if hasattr(manager, 'filter_by_agents_and_scenes'):
                    indices = manager.filter_by_agents_and_scenes(
                        self.min_agents, self.max_agents, split_scene_names)
                else:
                    # Fallback method
                    all_indices = manager.filter_by_agents(self.min_agents, self.max_agents)
                    indices = []
                    for idx in all_indices:
                        file_info = manager.get_file_info_by_index(idx)
                        if file_info and file_info.get('scene_name') in split_scene_names:
                            indices.append(idx)
                
                print(f"[INFO] airport {airport}: found {len(indices)} qualified packed scenes")
                
                # Store index information
                airport_scenarios = []
                for idx in indices:
                    file_info = manager.get_file_info_by_index(idx)
                    if file_info:
                        airport_scenarios.append({
                            'type': 'packed',
                            'airport': airport,
                            'manager': manager,
                            'index_in_manager': idx,
                            'scene_name': file_info.get('scene_name', ''),
                            'arcname': file_info.get('arcname', ''),
                            'agents': file_info.get('agents', 0)
                        })
                
                self.scenario_list[airport] = airport_scenarios
                
            else:
                # Original file mode
                scenario_files = D.get_filtered_list(
                    airport, self.in_data_dir, self.split_list,
                    self.min_agents, self.max_agents)
                
                self.scenario_list[airport] = scenario_files
                print(f"[INFO] airport {airport}: loaded {len(scenario_files)} original files")
        
        # Balance data volume across different airports
        files_per_airport = min(len(f) for _, f in self.scenario_list.items())
        balanced_list = []
        
        print(f"[INFO] Balancing data:")
        for airport, files in self.scenario_list.items():
            random.seed(self.seed)
            random.shuffle(files)
            selected_files = files[:files_per_airport]
            balanced_list += selected_files
            print(f"[INFO] Selected {len(selected_files)}/{len(files)} scenes from {airport}")
        
        # Update to final list
        self.scenario_list = balanced_list
        
        print(f"[INFO] Data preparation completed: {len(self.scenario_list)} total scenes")
        
        # Print detailed statistics
        # self._print_detailed_statistics()

    def _extract_scene_names_from_split_list(self, airport: str) -> List[str]:
        """Extract scene names for specified airport from split_list"""
        airport_split_items = [item for item in self.split_list if item.startswith(f"{airport}/")]
        
        split_scene_names = []
        for item in airport_split_items:
            parts = item.split('/')
            if len(parts) > 1:
                scene_name = parts[1]
                # Remove possible extensions
                if scene_name.endswith('.pkl') or scene_name.endswith('.pickle'):
                    scene_name = scene_name.rsplit('.', 1)[0]
                split_scene_names.append(scene_name)
        
        return split_scene_names

    def _print_detailed_statistics(self):
        """Print detailed data statistics"""
        print("\n[INFO] Detailed data statistics:")
        
        packed_count = 0
        file_count = 0
        airport_stats = {}
        total_agents = 0
        
        for item in self.scenario_list:
            if isinstance(item, dict) and item.get('type') == 'packed':
                packed_count += 1
                airport = item.get('airport', 'unknown')
                airport_stats[airport] = airport_stats.get(airport, {'packed': 0, 'file': 0})
                airport_stats[airport]['packed'] += 1
                total_agents += item.get('agents', 0)
            else:
                file_count += 1
                if isinstance(item, str):
                    parts = item.split('/')
                    if len(parts) > 0:
                        airport = parts[0]
                        airport_stats[airport] = airport_stats.get(airport, {'packed': 0, 'file': 0})
                        airport_stats[airport]['file'] += 1
        
        print(f"[INFO] Total scenes: {len(self.scenario_list)}")
        print(f"[INFO] Packed data: {packed_count}")
        print(f"[INFO] Original files: {file_count}")
        
        if packed_count > 0:
            avg_agents = total_agents / packed_count
            print(f"[INFO] Average agents per scene: {avg_agents:.1f}")
        
        if airport_stats:
            print(f"\n[INFO] Airport distribution:")
            for airport, counts in airport_stats.items():
                total = counts['packed'] + counts['file']
                if counts['packed'] > 0 and counts['file'] > 0:
                    print(f"  {airport}: {total} scenes (packed:{counts['packed']}, original:{counts['file']})")
                elif counts['packed'] > 0:
                    print(f"  {airport}: {total} scenes (all packed)")
                else:
                    print(f"  {airport}: {total} scenes (all original)")
        
        # Print detailed information of first few samples
        print(f"\n[INFO] First 5 sample preview:")
        for i in range(min(5, len(self.scenario_list))):
            item = self.scenario_list[i]
            if isinstance(item, dict) and item.get('type') == 'packed':
                scene_name = item.get('scene_name', 'N/A')[:20]
                arcname = item.get('arcname', 'N/A')[:20]
                agents = item.get('agents', 0)
                print(f"  [{i}] Packed: {item['airport']}/{scene_name}.../{arcname}... (agents={agents})")
            else:
                filename = os.path.basename(str(item))[:40]
                print(f"  [{i}] File: {filename}...")

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
            elif key in ['mode_labels', 'unsupervised_encoding', 'encoding_info']:
                input_dict[key] = val_list
            elif key in ['rule_based_encoding']:
                val_list = [torch.from_numpy(x).type(torch.FloatTensor) for x in val_list]
                input_dict[key] = D.merge_seq2d_by_padding(
                    val_list, max_pad=self.k_agents
                )    
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
            shape=(num_agents, timesteps, 7))  # [x, y, z, heading, vx, vy, vz]
    
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
        rel_sequence[:, :, 3] = T.wrap_angle(headings - ego_heading)
        
        # knots/s  km/s
        # 1 knot = 1.852 km/h
        # 1 knot/s = 1.852 km/h/s = 1.852/3600 km/s = 0.000514444 km/s
        KNOTS_TO_KMS = 0.000514444
        
        headings_rad = np.radians(
            sequences[:, :, G.SEQ_IDX.Heading])
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
        MODE_MAP = {
            "TurnLeft_Accel": 0,
            "TurnLeft_Decel": 1,
            "TurnLeft_Normal": 2,
            "TurnLeft_Hold": 3,
        
            "TurnRight_Accel": 4,
            "TurnRight_Decel": 5,
            "TurnRight_Normal": 6,
            "TurnRight_Hold": 7,
        
            "Straight_Accel": 8,
            "Straight_Decel": 9,
            "Straight_Normal": 10,
            "Straight_Hold": 11,
        
            "Hold_Accel": 12,
            "Hold_Decel": 13,
            "Hold_Normal": 14,
            "Hold_Hold": 15,
        }


        sequences = scene_data['agent_sequences']
        agent_masks = scene_data['agent_masks']
        airport_id = scene_data['airport_id']
        # airport_id = 'kbos' # for fixing the error when generating scene dataset

        # TODO: fix this k-agent selection. This is assuming that the k selected agents are related
        #       or relevant to each other, but this is not a guarantee. They could be complete
        #       unrelated, and thus the scene representation may not be valid.
        # Define ego agent and agents in scene based on the sampling scheme
        # if self.sampling_strategy == 'random':
        #     # For k_random, slice k agents out of previously shuffled agent array
        #     agents_in_scene = scene_data['random_order'][:self.k_agents]
        # elif self.sampling_strategy == 'safety':
        #     # For safety oriented, splice k agents out of ordered agent array
        #     agents_in_scene = scene_data['critical_order'][:self.k_agents]
        # else:
        #     raise ValueError(f"Sampling strategy: {self.sampling_strategy} not supported!")

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

        # Choose an ego-agent from the valid ones. NOTE: Valid ones are should appear first.
        # num_agents = min(self.k_agents, 2)#scene_data['random_valid'])

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

        # TODO: debug
        context_map, adjacency = None, None
        if self.add_context:
            context_map, adjacency = self.transform_context(
                self.semantic_maps[airport_id], sequences, rel_sequences, ego_agent,
                self.limits[airport_id]
            )
        agent_types = np.asarray(scene_data['agent_types'])
        agent_types = agent_types[agents_in_scene]
        
        # Extract additional labels and encodings from scene_data
        mode_labels = scene_data.get('mode_labels', None)
        
        # FIX: Check if mode_labels is a list and convert each element
        if mode_labels is not None:
            if isinstance(mode_labels, list):
                mode_labels = [MODE_MAP[m] for m in mode_labels]
                mode_labels = np.asarray(mode_labels)[agents_in_scene]
            else:
                mode_labels = np.array([MODE_MAP[mode_labels]])
                mode_labels = mode_labels[agents_in_scene]

        rule_based_encoding = scene_data.get('rule_based_encoding', None)
        unsupervised_encoding = scene_data.get('unsupervised_encoding', None)
        
        # If they exist, index them based on agents_in_scene
        if rule_based_encoding is not None:
            rule_based_encoding = np.asarray(rule_based_encoding)[agents_in_scene]
        
        if unsupervised_encoding is not None:
            unsupervised_encoding = np.asarray(unsupervised_encoding)[agents_in_scene]

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
            'mode_labels': mode_labels,
            'rule_based_encoding': rule_based_encoding,
            'unsupervised_encoding': unsupervised_encoding,
        }

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
        """ Loads scene from given index - high-performance version"""
        self.total_samples_loaded += 1
        
        item = self.scenario_list[index]
        # Original file
        file_path = str(item)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Transform data
        transformed_data = self.transform_scene_data(data)
        
        return transformed_data