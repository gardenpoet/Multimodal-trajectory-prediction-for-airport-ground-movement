import os
import json
import math
import pickle
import random
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
# from tslearn.clustering import TimeSeriesKMeans

import amelia_scenes.utils.common as C
import amelia_scenes.utils.dataset as D
import amelia_scenes.utils.global_masks as G

from tqdm import tqdm
from easydict import EasyDict
from typing import Tuple, List
from joblib import Parallel, delayed

from amelia_scenes.scoring.kinematic import compute_kinematic_scores
from amelia_scenes.scoring.critical import compute_simple_scene_critical
from amelia_scenes.scoring.interactive import compute_interactive_scores
from amelia_scenes.scoring.crowdedness import compute_simple_scene_crowdedness


class SceneProcessor:
    """ Dataset class for pre-processing airport surface movement data into scenes. """

    def __init__(self, config: EasyDict) -> None:
        """
        Inputs
        ------
            config[EasyDict]: dictionary containing configuration parameters needed to process the
            airport trajectory data.
        """
        super(SceneProcessor, self).__init__()

        # Trajectory configuration
        self.airport = config.airport
        self.in_data_dir = os.path.join(config.in_data_dir, self.airport)
        self.out_data_dir = os.path.join(config.out_data_dir, self.airport)
        os.makedirs(self.out_data_dir, exist_ok=True)
        self.blacklist_dir = os.path.join(config.out_data_dir, 'blacklist')
        os.makedirs(self.blacklist_dir, exist_ok=True)
        self.out_data_summary_dir = os.path.join(config.out_summary_dir, self.airport)
        os.makedirs(self.out_data_summary_dir, exist_ok=True)

        self.parallel = config.parallel
        self.overwrite = config.overwrite
        self.add_scores_meta = config.add_scores_meta

        self.seed = config.seed
        self.extend = True
        self.seq_extent = 1

        self.pred_lens = config.pred_lens
        self.pred_len = max(self.pred_lens)
        self.hist_len = config.hist_len
        self.seq_len = self.hist_len + self.pred_len
        self.skip = config.skip
        self.min_agents = config.min_agents
        self.max_agents = config.max_agents
        self.min_valid_points = config.min_valid_points

        self.n_jobs = config.jobs

        # add configurations for unsupervised learning
        self.use_unsupervised_encoding = getattr(config, 'use_unsupervised_encoding', False)
        self.predict_faster = getattr(config, 'use_unsupervised_encoding', False)
        self.n_accel_clusters = getattr(config, 'n_accel_clusters', 5)
        self.n_turn_clusters = getattr(config, 'n_turn_clusters', 5)
        self.unsupervised_sample_size = getattr(config, 'unsupervised_sample_size', 200)
        self.encoding_method = getattr(config, 'encoding_method', 'separate')

        self.unsupervised_models = None
        self.cluster_centers = None

        limits_file = os.path.join(config.assets_dir, self.airport, 'limits.json')
        with open(limits_file, 'r') as fp:
            self.ref_data = EasyDict(json.load(fp))

        graph_data_dir = os.path.join(config.graph_data_dir, self.airport)
        print(f"Loading graph data from: {graph_data_dir}")
        pickle_map_filepath = os.path.join(graph_data_dir, "semantic_graph.pkl")
        with open(pickle_map_filepath, 'rb') as f:
            graph_pickle = pickle.load(f)
            self.hold_lines = graph_pickle['hold_lines'][:, 2:4]

        self.blacklist = []
        blacklist_file = os.path.join(self.blacklist_dir, f"{self.airport}.txt")
        if os.path.exists(blacklist_file) and not self.overwrite:
            with open(blacklist_file, 'r') as f:
                self.blacklist = f.read().splitlines()

        file_list = os.listdir(self.in_data_dir)
        for duplicate in list(set(self.blacklist) & set(file_list)):
            file_list.remove(duplicate)
        self.data_files = [os.path.join(self.in_data_dir, f) for f in file_list if f.endswith('.csv')]
        random.seed(self.seed)
        random.shuffle(self.data_files)
        self.data_files = self.data_files[:int(len(self.data_files) * config.perc_process)]
        self._compute_global_thresholds()

    def _compute_global_thresholds(self):
        """Compute turning and acceleration thresholds using percentiles, ignoring static sequences."""
        print("?? Computing global thresholds for rule-based labeling...")

        all_turn = []
        all_accel = []

        for f in tqdm(self.data_files, desc="Sampling taxiing statistics"):
            try:
                data = pd.read_csv(f)
                frames = data.Frame.unique().tolist()
                frame_data = [data[data.Frame == fr] for fr in frames]

                num_sequences = max(1, int(math.ceil((len(frames) - self.seq_len + 1) / self.skip)))

                for i in range(0, num_sequences * self.skip + 1, self.skip):
                    seq = np.concatenate(frame_data[i:i + self.seq_len], axis=0)

                    unique_agents = np.unique(seq[:, G.RAW_IDX.ID])
                    for agent_id in unique_agents:
                        agent_seq = seq[seq[:, G.RAW_IDX.ID] == agent_id]
                        if len(agent_seq) < self.seq_len:
                            continue

                        speed = agent_seq[:, G.RAW_IDX.Speed].astype(float)

                        # ?????? 0 ???
                        if np.allclose(speed, 0):
                            continue

                        heading = agent_seq[:, G.RAW_IDX.Heading].astype(float)
                        hdg_smooth = uniform_filter1d(heading, size=5)
                        turn = np.gradient(np.unwrap(np.deg2rad(hdg_smooth))) * 180 / np.pi
                        accel = np.gradient(speed)

                        turn_sum = np.sum(turn)
                        accel_mean = np.mean(accel)

                        all_turn.append(turn_sum)
                        all_accel.append(accel_mean)

            except Exception as e:
                print(f"Warning file skipped: {e}")

        all_turn = np.array(all_turn)
        all_accel = np.array(all_accel)

        # === Turning thresholds ===
        self.turn_th_net = float(np.percentile(all_turn[all_turn > 0], 75))
        self.turn_th_net_neg = float(np.percentile(all_turn[all_turn < 0], 25))

        # === Acceleration thresholds ===
        self.accel_th = float(np.percentile(all_accel, 75))  # strong acceleration
        self.decel_th = float(np.percentile(all_accel, 25))  # strong deceleration (negative)

        print("?? Learned thresholds:")
        print(f"  turn_th_net_pos = {self.turn_th_net:.3f}")
        print(f"  turn_th_net_neg = {self.turn_th_net_neg:.3f}")
        print(f"  accel_th        = {self.accel_th:.3f}")
        print(f"  decel_th        = {self.decel_th:.3f}")

        ddd = 1

    def process_data(self) -> None:
        """ Processes the CSV data files containing airport trajectory information, creating shards
        containing scenario-level pickle data. If self.parallel is True, it  will process the data
        in parallel, otherwise it will do it sequentially. Once the sharding is done, it will return
        a list containing all generated scenarios and will blacklist any CSV files that failed to
        generate scenarios for the specified conditions.
        """
        print(f"Processing data for airport {self.airport.upper()}.")

        if self.use_unsupervised_encoding:
            print("\n?? Training unsupervised clustering models on all data...")
            self._train_unsupervised_models()

        if self.parallel:
            scenes = Parallel(n_jobs=self.n_jobs)(
                delayed(self.process_file)(f) for f in tqdm(self.data_files))
            # Unpacking results
            for i in range(len(scenes)):
                res = scenes.pop()
                if res is None:
                    continue
                self.blacklist += res
            del scenes
        else:
            for f in tqdm(self.data_files):
                res = self.process_file(f)
                if res is None:
                    continue
                self.blacklist += res

        # Once all of the data has been processed and the blacklists collected, save them.
        blacklist_file = os.path.join(self.blacklist_dir, f'{self.airport}.txt')
        with open(blacklist_file, 'w') as fp:
            fp.write('\n'.join(self.blacklist))

        # get data percentiles
        # self.data_percentiles()

    # def _train_unsupervised_models(self):
    #     """train unsupervised models using sequence data (20 frames per sequence)"""
    #     print("?? Collecting features for unsupervised clustering...")
    #
    #     all_accels = []
    #     all_turn_rates = []
    #
    #     # collect sequences from all data files
    #     for f in tqdm(self.data_files, desc="Collecting sequence features"):  # memory size warning
    #         try:
    #             data = pd.read_csv(f)
    #
    #             # Get the number of unique frames
    #             frames = data.Frame.unique().tolist()
    #             frame_data = []
    #             for frame_num in frames:
    #                 frame = data[data.Frame == frame_num]
    #                 frame_data.append(frame)
    #
    #             num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))
    #             if num_sequences < 1:
    #                 continue
    #
    #             # Process each sequence
    #             for i in range(0, min(num_sequences * self.skip + 1, len(frames) - self.seq_len + 1), self.skip):
    #                 # Extract sequence data (same as in process_seq)
    #                 seq_data = np.concatenate(frame_data[i:i + self.seq_len], axis=0)
    #
    #                 # Filter sequences with valid aircraft
    #                 if math.isclose(seq_data[:, G.RAW_IDX.Speed].sum(), 0):
    #                     continue
    #                 if not np.isin(seq_data[:, G.RAW_IDX.Type].astype(int), C.AIRCRAFT).sum():
    #                     continue
    #
    #                 # Process each agent in the sequence
    #                 unique_agents = np.unique(seq_data[:, G.RAW_IDX.ID])
    #                 for agent_id in unique_agents:
    #                     agent_seq = seq_data[seq_data[:, G.RAW_IDX.ID] == agent_id]
    #
    #                     if len(agent_seq) < self.seq_len:  # Skip incomplete sequences
    #                         continue
    #
    #                     # Extract speed and heading for the 20-frame sequence
    #                     speeds = agent_seq[:, G.RAW_IDX.Speed].astype(float)
    #                     headings = agent_seq[:, G.RAW_IDX.Heading].astype(float)
    #
    #                     # Calculate acceleration and turning rate for the sequence
    #                     accel = np.gradient(speeds)
    #                     heading_rad = np.deg2rad(headings)
    #                     turn_rate = np.gradient(np.unwrap(heading_rad)) * 180 / np.pi
    #
    #                     # Only keep sequences with meaningful movement
    #                     if np.std(accel) < 0.01 and np.std(turn_rate) < 0.1:  # Filter static sequences
    #                         continue
    #
    #                     # Add to training dataset
    #                     all_accels.append(accel)
    #                     all_turn_rates.append(turn_rate)
    #
    #         except Exception as e:
    #             print(f"Error processing file {f} for clustering: {e}")
    #             continue
    #
    #     if len(all_accels) < 100:  # at least 100 sequences
    #         print("?? Not enough sequence data for unsupervised clustering, will use rule-based encoding only")
    #         self.use_unsupervised_encoding = False
    #         return
    #
    #     # Convert to numpy arrays
    #     all_accels = np.array(all_accels)
    #     all_turn_rates = np.array(all_turn_rates)
    #
    #     print(f"?? Collected {len(all_accels)} sequences of length {self.seq_len}")
    #
    #     # Sample to avoid memory issues
    #     sample_size = min(self.unsupervised_sample_size, len(all_accels))
    #     sample_indices = np.random.choice(len(all_accels), sample_size, replace=False)
    #     accel_sample = all_accels[sample_indices]
    #     turn_sample = all_turn_rates[sample_indices]
    #
    #     print(f"??? Training with {sample_size} sequences...")
    #
    #     # Train clustering models
    #     print("??? Training acceleration clustering model...")
    #     km_accel = TimeSeriesKMeans(
    #         n_clusters=self.n_accel_clusters,
    #         metric="dtw",
    #         verbose=True,
    #         random_state=self.seed,
    #         n_init=3
    #     )
    #     km_accel.fit(accel_sample)
    #
    #     print("??? Training turn rate clustering model...")
    #     km_turn = TimeSeriesKMeans(
    #         n_clusters=self.n_turn_clusters,
    #         metric="dtw",
    #         verbose=True,
    #         random_state=self.seed,
    #         n_init=3
    #     )
    #     km_turn.fit(turn_sample)
    #
    #     self.unsupervised_models = {
    #         'accel_model': km_accel,
    #         'turn_model': km_turn
    #     }
    #
    #     self.cluster_centers = {
    #         'accel_centers': km_accel.cluster_centers_,
    #         'turn_centers': km_turn.cluster_centers_
    #     }
    #
    #     print("? Unsupervised models trained successfully on sequence data!")


    def _compute_unsupervised_encoding(self, agent_seqs):
        if self.unsupervised_models is None:
            return None

        try:
            # extract features
            speeds = np.array([seq[:, G.RAW_IDX.Speed].astype(float) for seq in agent_seqs])
            headings = np.array([seq[:, G.RAW_IDX.Heading].astype(float) for seq in agent_seqs])

            # calculate accelerations and turning rate
            accels = np.array([np.gradient(speed) for speed in speeds])
            turn_rates = np.array([np.gradient(np.unwrap(np.deg2rad(heading))) * 180 / np.pi
                                   for heading in headings])

            # filter near-zero turn_rate
            # turn_rates = np.where(np.abs(turn_rates) < 2, 0, turn_rates)

            if self.predict_faster:
                # predict labels faster
                accel_centers = self.cluster_centers['accel_centers'].reshape(self.n_accel_clusters, -1)
                turn_centers = self.cluster_centers['turn_centers'].reshape(self.n_turn_clusters, -1)

                # for acceleration
                accels_flat = accels.reshape(len(accels), -1)
                dists_accel = np.linalg.norm(
                    accels_flat[:, None, :] - accel_centers[None, :, :],
                    axis=2
                )
                accel_labels = np.argmin(dists_accel, axis=1)

                # for turning
                turn_flat = turn_rates.reshape(len(turn_rates), -1)
                dists_turn = np.linalg.norm(
                    turn_flat[:, None, :] - turn_centers[None, :, :],
                    axis=2
                )
                turn_labels = np.argmin(dists_turn, axis=1)
            else:
                # predict clustering labels
                accel_labels = self.unsupervised_models['accel_model'].predict(accels)
                turn_labels = self.unsupervised_models['turn_model'].predict(turn_rates)

            if self.encoding_method == 'combined':
                encoding_df = self._create_combined_encoding(turn_labels, accel_labels)
            elif self.encoding_method == 'separate':
                encoding_df = self._create_separate_unsupervised_encoding(turn_labels, accel_labels)
            elif self.encoding_method == 'soft':
                encoding_df = self._create_soft_encoding(turn_rates, accels)
            elif self.encoding_method == 'hierarchical':
                encoding_df = self._create_hierarchical_encoding(turn_labels, accel_labels)
            else:
                encoding_df = self._create_separate_unsupervised_encoding(turn_labels, accel_labels)

            encoding_df['unsupervised_accel_label'] = accel_labels
            encoding_df['unsupervised_turn_label'] = turn_labels

            return encoding_df

        except Exception as e:
            print(f"Error in unsupervised encoding: {e}")
            return None

    def _create_combined_encoding(self, turn_labels, accel_labels):
        """
        Create combined one-hot encoding for turn + acceleration clusters.
        Each mode is represented as a string "T{t}_A{a}".
        """

        # Combine cluster labels into a single mode identifier
        combined_modes = [f"T{t}_A{a}" for t, a in zip(turn_labels, accel_labels)]

        # Get all unique combined modes (cluster pairs)
        unique_modes = sorted(set(combined_modes))
        mode_to_idx = {mode: idx for idx, mode in enumerate(unique_modes)}

        # One-hot encode the combined modes
        one_hot = np.zeros((len(combined_modes), len(unique_modes)))
        for i, mode in enumerate(combined_modes):
            one_hot[i, mode_to_idx[mode]] = 1

        return pd.DataFrame(
            one_hot,
            columns=[f"unsupervised_mode_{m}" for m in unique_modes]
        )

    def _create_separate_unsupervised_encoding(self, turn_labels, accel_labels):
        """
        Create separate one-hot encodings for turning clusters and acceleration clusters,
        and concatenate them into a single feature vector.
        """

        # One-hot encode turning modes
        turn_one_hot = np.zeros((len(turn_labels), self.n_turn_clusters))
        turn_one_hot[np.arange(len(turn_labels)), turn_labels] = 1

        # One-hot encode acceleration modes
        accel_one_hot = np.zeros((len(accel_labels), self.n_accel_clusters))
        accel_one_hot[np.arange(len(accel_labels)), accel_labels] = 1

        # Concatenate into a single encoding
        separate_one_hot = np.concatenate([turn_one_hot, accel_one_hot], axis=1)

        # Create column names
        turn_columns = [f"unsupervised_turn_{i}" for i in range(self.n_turn_clusters)]
        accel_columns = [f"unsupervised_accel_{i}" for i in range(self.n_accel_clusters)]
        columns = turn_columns + accel_columns

        return pd.DataFrame(separate_one_hot, columns=columns)

    def _create_soft_encoding(self, turn_features, accel_features):
        """
        Create soft (probabilistic) encoding using distances to cluster centers.
        Distances ? Softmax ? Probabilities.
        """

        # Compute distances to turning cluster centers
        turn_dists = np.array([
            [np.linalg.norm(turn_feature - center)
             for center in self.cluster_centers['turn_centers']]
            for turn_feature in turn_features
        ])

        # Compute distances to acceleration cluster centers
        accel_dists = np.array([
            [np.linalg.norm(accel_feature - center)
             for center in self.cluster_centers['accel_centers']]
            for accel_feature in accel_features
        ])

        # Softmax helper (convert distances into probabilities)
        def softmax(x, axis=1):
            e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e_x / np.sum(e_x, axis=axis, keepdims=True)

        # Negative distances ? higher probability for closer centers
        turn_probs = softmax(-turn_dists, axis=1)
        accel_probs = softmax(-accel_dists, axis=1)

        # Concatenate turning + acceleration probabilities
        soft_encoding = np.concatenate([turn_probs, accel_probs], axis=1)

        # Build column names
        turn_columns = [f"unsupervised_turn_prob_{i}" for i in range(self.n_turn_clusters)]
        accel_columns = [f"unsupervised_accel_prob_{i}" for i in range(self.n_accel_clusters)]
        columns = turn_columns + accel_columns

        return pd.DataFrame(soft_encoding, columns=columns)

    def _create_hierarchical_encoding(self, turn_labels, accel_labels):
        """
        Create hierarchical encoding:
        Each (turn_cluster, accel_cluster) pair corresponds to one unique one-hot index.
        Example: index = turn_label * n_accel_clusters + accel_label
        """

        num_modes = self.n_turn_clusters * self.n_accel_clusters
        hierarchical = np.zeros((len(turn_labels), num_modes))

        for i, (t_label, a_label) in enumerate(zip(turn_labels, accel_labels)):
            idx = t_label * self.n_accel_clusters + a_label
            hierarchical[i, idx] = 1

        columns = [f"unsupervised_hierarchical_{i}" for i in range(num_modes)]
        return pd.DataFrame(hierarchical, columns=columns)

    def process_file(self, f: str) -> Tuple[List, List, List, List, List, List]:
        """ Processes a single data file. It first obtains the number of possible sequences (given
        the parameters in the configuration file) and then generates scene-level pickle files with
        the corresponding scene's information.

        Inputs
        ------
            f[str]: name of the file to shard.
        """
        print(f"Processing file: {f}")
        base_name = f.split('/')[-1]
        shard_name = base_name.split('.')[0]
        airport_id = base_name.split('_')[0].lower()
        file_time = D._get_file_timestamp(base_name)
        data_dir = os.path.join(self.out_data_dir, shard_name)
        # Check if the file has been sharded already. If so, add sharded files to the scenario list.
        if not self.overwrite and (os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0):
            return None

        # Otherwise, shard the file and add it to the scenario list.
        data = pd.read_csv(f)

        # Get the number of unique frames
        frames = data.Frame.unique().tolist()
        frame_data = []
        for frame_num in frames:
            frame = data[:][data.Frame == frame_num]
            frame_data.append(frame)

        blacklist = []
        num_sequences = int(math.ceil((len(frames) - (self.seq_len) + 1) / self.skip))
        if num_sequences < 1:
            blacklist.append(f.removeprefix(self.in_data_dir+'/'))
            return blacklist

        sharded_files = []
        os.makedirs(data_dir, exist_ok=True)

        valid_seq = 0
        for i in range(0, num_sequences * self.skip + 1, self.skip):
            scenario_id = str(valid_seq).zfill(6)
            seq, agent_id, agent_type, agent_valid, agent_mask, mode_labels, rule_based_encoding, unsupervised_encoding = self.process_seq(
                frame_data=frame_data, frames=frames, seq_idx=i, airport_id=airport_id)
            if seq is None:
                continue
            # Get agent array based on random and safety criteria
            num_agents, _, _ = seq.shape
            time_meta = D._process_timestamp(
                scene_ts=file_time,
                frame_idx=i,
                airport_code=airport_id
            )
            scene = {
                'scenario_id': scenario_id,
                'num_agents': num_agents,
                'airport_id': airport_id,
                'agent_sequences': seq,
                'agent_ids': agent_id,
                'agent_types': agent_type,
                'agent_masks': agent_mask,
                'agent_valid': agent_valid,
                'time_meta': time_meta,
                'mode_labels': mode_labels,
                'rule_based_encoding': rule_based_encoding,
                'unsupervised_encoding': unsupervised_encoding,
                'encoding_info': {
                    'rule_based_columns': list(rule_based_encoding.columns) if rule_based_encoding is not None else [],
                    'unsupervised_columns': list(unsupervised_encoding.columns) if unsupervised_encoding is not None else [],
                    'encoding_method': self.encoding_method if unsupervised_encoding is not None else 'rule_based_only'
            }
            }
            scene['meta'] = None
            if self.add_scores_meta:
                scene['meta'] = self.process_scores(scene)
                # score = scene['meta']['scene_scores']['critical']
                # self.data_summary.scores += [score]
                # self.data_summary.num_scenes += 1
                # self.data_summary.files_scores += [(score, os.path.join(
                #     shard_name, f"{scenario_id}_n-{num_agents}.pkl"))]

            scene_filepath = os.path.join(data_dir, f"{scenario_id}_n-{num_agents}.pkl")
            with open(scene_filepath, 'wb') as f:
                pickle.dump(scene, f, protocol=pickle.HIGHEST_PROTOCOL)

            valid_seq += 1
            sharded_files.append(scene_filepath)

        # If directory is empty, remove it.
        if len(os.listdir(data_dir)) == 0:
            blacklist.append(f.removeprefix(self.in_data_dir+'/'))
            os.rmdir(data_dir)
        return blacklist

    def process_seq(
            self, frame_data: pd.DataFrame, frames: list, seq_idx: int, airport_id: str
    ) -> np.array:
        """ Processes all valid agent sequences with taxiing mode labels.

        Inputs:
        -------
            frame_data[pd.DataFrame]: dataframe containing the scene's trajectory information
            frames[list]: list of frames to process.
            seq_idx[int]: current sequence index to process.

        Outputs:
        --------
            seq[np.array]: numpy array containing all processed scene's sequences
            agent_id_list[list]: list with the agent IDs that were processed.
            agent_type_list[list]: list containing the type of agent
            valid_agent_list[list]: list of valid agents
            agent_masks[np.array]: agent masks
            mode_labels[list]: list of taxiing mode labels for each agent
            encoded_modes[pd.DataFrame]: one-hot coding for taxiing modes
        """
        none_outs = (None, None, None, None, None, None, None, None)
        # All data for the current sequence: from the curr index i to i + sequence length
        seq_data = np.concatenate(frame_data[seq_idx:seq_idx + self.seq_len], axis=0)

        # If the speed of all agents is zero or close, return None
        if math.isclose(seq_data[:, G.RAW_IDX.Speed].sum(), 0):
            return none_outs

        # If there are no aircraft in the sequence, return None
        if not np.isin(seq_data[:, G.RAW_IDX.Type].astype(int), C.AIRCRAFT).sum():
            return none_outs

        # IDs of agents in the current sequence
        unique_agents = np.unique(seq_data[:, G.RAW_IDX.ID])
        num_agents = len(unique_agents)
        if num_agents < self.min_agents or num_agents > self.max_agents:
            return none_outs

        num_agents_considered = 0
        seq = np.zeros((num_agents, self.seq_len, G.DIM))
        agent_masks = np.zeros((num_agents, self.seq_len)).astype(bool)
        agent_id_list, agent_type_list, valid_agent_list = [], [], []
        mode_labels = []  # storing taxiing mode labels
        agent_seqs = []  # storing original agent sequence for unsupervised learning

        # parameters for judging taxiing modes
        turn_th_net = 10.0
        decel_th = -0.5
        accel_th = 0.5
        smooth_window = 5

        alt_idx = G.RAW_IDX.Altitude

        for _, agent_id in enumerate(unique_agents):
            # Current sequence of agent with agent_id
            agent_seq = seq_data[seq_data[:, 1] == agent_id]
            agent_seqs.append(agent_seq)

            # Start frame for the current sequence of the current agent reported to 0
            pad_front = frames.index(agent_seq[0, 0]) - seq_idx

            # End frame for the current sequence of the current agent
            pad_end = frames.index(agent_seq[-1, 0]) - seq_idx + 1

            if pad_end - pad_front != self.seq_len:
                continue

            # Scale altitude
            mx = self.ref_data.limits.Altitude.max
            mn = self.ref_data.limits.Altitude.min
            agent_seq[:, alt_idx] = (agent_seq[:, alt_idx] - mn) / (mx - mn)

            agent_id_list.append(int(agent_id))
            agent_type_list.append(int(agent_seq[0, G.RAW_IDX.Type]))

            # Interpolated mask
            mask = agent_seq[:, G.RAW_IDX.Interp] == '[ORG]'
            # Not interpolated --> Valid
            agent_seq[mask, G.RAW_IDX.Interp] = 1.0
            # Interpolated --> Not valid
            agent_seq[~mask, G.RAW_IDX.Interp] = 0.0

            # Check if there's at least two valid points in the history segment, two valid points in
            # partial segment and two valid points in the future segment
            valid = mask[:self.hist_len].sum() >= self.min_valid_points
            if valid:
                for t in self.pred_lens:
                    if mask[self.hist_len:self.hist_len + t].sum() < self.min_valid_points:
                        valid = False
                        break
            valid_agent_list.append(valid)

            # TODO: Impute needs to be debugged. Imputing should not happen since it was done in SWIM.
            agent_seq = C.impute(agent_seq, self.seq_len)  # self.seq_len)
            # computing taxiing mode labels
            mode_label = self._compute_taxiing_mode(agent_seq, pad_front, pad_end,
                                                    turn_th_net, decel_th, smooth_window)
            valid_mask = agent_seq[:, G.RAW_IDX.Interp].astype(bool)
            agent_masks[num_agents_considered, pad_front:pad_end] = valid_mask

            agent_seq = agent_seq[:, G.RAW_SEQ_MASK]
            seq[num_agents_considered, pad_front:pad_end] = agent_seq[:, G.SEQ_ORDER]
            num_agents_considered += 1

            mode_labels.append(mode_label)

        # Return Nones if there aren't any valid agents
        valid_agent_list = np.asarray(valid_agent_list)
        if valid_agent_list.sum() == 0:
            return none_outs

        # Return Nones if the number of considered agents is less than the required
        if num_agents_considered < self.min_agents:
            return none_outs

        # rule based encoding
        rule_based_encoding = self._compute_separate_encoding(mode_labels)

        # unsupervised learning
        unsupervised_encoding = None
        if self.use_unsupervised_encoding and len(agent_seqs) > 0:
            unsupervised_encoding = self._compute_unsupervised_encoding(agent_seqs)
            if unsupervised_encoding is not None:
                print("?? Generated both rule-based and unsupervised encodings")
            else:
                print("?? Unsupervised encoding failed, using rule-based only")
        else:
            print("?? Using rule-based encoding only")

        return (seq[:num_agents_considered], agent_id_list, agent_type_list,
                valid_agent_list, agent_masks[:num_agents_considered], mode_labels,
                rule_based_encoding, unsupervised_encoding)

    def _compute_taxiing_mode(self, agent_seq, pad_front, pad_end, turn_th_net, decel_th, smooth_window):
        """Compute taxiing mode labels for a single agent."""
        try:
            spd_valid = agent_seq[pad_front:pad_end, G.RAW_IDX.Speed].astype(float)

            # ?????? 0,???? Hold_Hold
            if np.allclose(spd_valid, 0):
                return "Hold_Hold"

            hdg_valid = agent_seq[pad_front:pad_end, G.RAW_IDX.Heading].astype(float)

            # ?????
            hdg_smooth = uniform_filter1d(hdg_valid, size=smooth_window)
            hdg_rad = np.unwrap(np.deg2rad(hdg_smooth))
            turn_rate = np.gradient(hdg_rad) * 180 / np.pi
            net_turn = np.sum(turn_rate)

            accel = np.gradient(spd_valid)
            mean_accel = np.mean(accel)

            # ????
            if net_turn > self.turn_th_net:
                turn_label = "TurnLeft"
            elif net_turn < self.turn_th_net_neg:
                turn_label = "TurnRight"
            else:
                turn_label = "Straight"

            # ????
            if mean_accel < self.decel_th:
                speed_label = "Decel"
            elif mean_accel > self.accel_th:
                speed_label = "Accel"
            else:
                speed_label = "Normal"

            return f"{turn_label}_{speed_label}"

        except Exception as e:
            print(f"Error computing taxiing mode: {e}")
            return "Unknown_Unknown"

    def _compute_separate_encoding(self, taxiing_modes):
        """One-hot encoding for turning and speed modes separately, always include 4 predefined modes."""

        turn_modes_list = ['TurnLeft', 'TurnRight', 'Straight', 'Hold']
        speed_modes_list = ['Accel', 'Decel', 'Normal', 'Hold']

        turn_modes = []
        speed_modes = []

        for mode in taxiing_modes:
            if '_' in mode:
                turn, speed = mode.split('_')
                turn_modes.append(turn)
                speed_modes.append(speed)
            else:
                # ??????? Unknown(??????????)
                turn_modes.append('Unknown')
                speed_modes.append('Unknown')

        # one-hot encoding separately,????????
        df_turn = pd.DataFrame({'turn_mode': pd.Categorical(turn_modes, categories=turn_modes_list)})
        df_speed = pd.DataFrame({'speed_mode': pd.Categorical(speed_modes, categories=speed_modes_list)})

        turn_encoded = pd.get_dummies(df_turn['turn_mode'], prefix='turn')
        speed_encoded = pd.get_dummies(df_speed['speed_mode'], prefix='speed')

        # combining encoding results
        separate_encoded = pd.concat([turn_encoded, speed_encoded], axis=1)

        # ??????,?? Hold_Hold
        if len(taxiing_modes) > 0:
            print(f"Processed {len(taxiing_modes)} taxiing modes")
            print("Distribution of turning:", df_turn['turn_mode'].value_counts(dropna=False).to_dict())
            print("Distribution of speed:", df_speed['speed_mode'].value_counts(dropna=False).to_dict())

        return separate_encoded

    def process_scores(self, scene):
        """ Computes kinematic and interactive scores for all valid agent sequences.

        Inputs:
        -------
            scene[dict]: dictionary containing agent sequences and meta information.

        Outputs:
        --------
            scores[dict]: dictionary containing individual and interactive scores for each agent,
            and scene scores.
        """
        crowd_scene_score = compute_simple_scene_crowdedness(scene, self.max_agents)
        kin_agents_scores, kin_scene_score = compute_kinematic_scores(scene, self.hold_lines)
        int_agents_scores, int_scene_score = compute_interactive_scores(scene, self.hold_lines)
        crit_agent_scores, crit_scene_score = compute_simple_scene_critical(
            agent_scores_list=[kin_agents_scores.copy(), int_agents_scores].copy(),
            scene_score_list=[crowd_scene_score.copy(), kin_scene_score.copy(), int_scene_score.copy()]
        )
        return {
            'agent_scores': {
                'kinematic': kin_agents_scores,
                'interactive': int_agents_scores,
                'critical': crit_agent_scores
            },
            'agent_order': {
                'random': C.get_random_order(scene['num_agents'], scene['agent_valid'], self.seed),
                # 'kinematic': C.get_sorted_order(kin_agents_scores),
                'interactive': C.get_sorted_order(int_agents_scores),
                'critical': C.get_sorted_order(crit_agent_scores)
            },
            'scene_scores': {
                'crowdedness': crowd_scene_score,
                'kinematic': kin_scene_score,
                'interactive': int_scene_score,
                'critical': crit_scene_score
            },
        }

    # def data_percentiles(self):

    #     scores = np.array(self.data_summary.scores)
    #     percentiles = [50, 60, 70, 80, 90, 95, 99, 99.5]
    #     percentile_values = np.percentile(scores, percentiles)

    #     scenes_data = self.data_summary.files_scores
    #     data_summary = {"num_scenes": self.data_summary.num_scenes,
    #                     "percentile_scores": {},
    #                     }
    #     for i, p in enumerate(percentiles):
    #         threshold = percentile_values[i]
    #         filtered = [{scene_file: score} for (score, scene_file) in scenes_data if score >= threshold]
    #         data_summary["percentile_scores"][str(p)] = {
    #             "threshold": float(threshold),
    #             "num_scenes": len(filtered),
    #             "scenes": filtered
    #         }

    #     summary_file = os.path.join(self.out_data_summary_dir, f"{self.airport}_summary.json")
    #     with open(summary_file, 'w') as f:
    #         json.dump(data_summary, f, indent=4)
    #     print(f"Dataset summary saved to {summary_file}")
