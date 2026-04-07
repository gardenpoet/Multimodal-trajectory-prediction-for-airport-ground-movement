import os
import math
import json
import pickle
import random
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict
from shapely.geometry import LineString, Point


class TrajectoryExtractor:
    """
    Extract aircraft ground movement trajectories for mode recognition (e.g. HMM).
    Each trajectory sequence is of shape (seq_len, 6):
    [x, y, speed, heading, id, type]
    """

    def __init__(self, config: EasyDict) -> None:
        super(TrajectoryExtractor, self).__init__()

        # --- Configuration ---
        self.airport = config.airport
        self.in_data_dir = os.path.join(config.in_data_dir, self.airport)
        self.out_data_dir = os.path.join(config.out_data_dir, self.airport)
        self.graph_dir = os.path.join(config.graph_data_dir, self.airport)
        os.makedirs(self.out_data_dir, exist_ok=True)

        # --- Processing parameters ---
        self.seq_len = config.hist_len + max(config.pred_lens)
        self.skip = config.skip
        self.min_valid_points = config.min_valid_points
        self.parallel = config.parallel
        self.n_jobs = config.jobs
        self.seed = config.seed
        random.seed(self.seed)

        self.all_trajs_by_type = {}

    # -------------------------------------------------
    def load_all_data(self) -> pd.DataFrame:
        """Load and concatenate all CSV files for the airport."""
        csv_files = sorted([
            os.path.join(self.in_data_dir, f)
            for f in os.listdir(self.in_data_dir)
            if f.endswith(".csv")
        ])

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.in_data_dir}")

        print(f"📂 Loading {len(csv_files)} CSV files from {self.in_data_dir} ...")

        df_list = []
        for fpath in tqdm(csv_files, desc="Reading CSVs"):
            df = pd.read_csv(fpath)
            if not df.empty:
                df_list.append(df)

        combined = pd.concat(df_list, ignore_index=True)
        combined = combined.sort_values(["ID", "Frame"]).reset_index(drop=True)
        print(f"✅ Combined shape: {combined.shape}")

        # Convert agent type to integer categories to save memory
        combined["Type"] = combined["Type"].astype("category").cat.codes.astype(np.int16)

        return combined

    # -------------------------------------------------
    def extract_trajectories(self, df: pd.DataFrame) -> dict:
        """
        Extract valid sequences grouped by agent Type.
        Returns a dict: {type_id: [array(seq_len, 6), ...]}
        """

        file_path = os.path.join(self.graph_dir, "semantic_graph.pkl")

        with open(file_path, "rb") as f:
            mapdata = pickle.load(f)

        df = df[df["Type"].isin([0, 1, 2])]
        trajs_by_type = {t: [] for t in df["Type"].unique()}

        for agent_type, type_df in df.groupby("Type"):
            print(f"\n🚀 Extracting trajectories for Type={agent_type} ...")
            type_trajs = []

            for agent_id, agent_df in tqdm(type_df.groupby("ID"), desc=f"Type {agent_type} agents"):
                agent_df = agent_df.sort_values("Frame")
                if len(agent_df) < self.seq_len:
                    continue

                # convert to smaller dtypes to save RAM
                x = agent_df["x"].astype(np.float32).values
                y = agent_df["y"].astype(np.float32).values
                spd = agent_df["Speed"].astype(np.float32).values
                hdg = agent_df["Heading"].astype(np.float32).values

                for i in range(0, len(agent_df) - self.seq_len + 1, self.skip):
                    if np.sum(spd[i:i + self.seq_len] > 0.5) < self.min_valid_points:
                        continue
                    if np.any(spd[i:i + self.seq_len] > 50):  # e.g. >50 knot
                        continue
                    seq = np.stack([
                        x[i:i + self.seq_len],
                        y[i:i + self.seq_len],
                        spd[i:i + self.seq_len],
                        hdg[i:i + self.seq_len],
                        np.full(self.seq_len, agent_id, dtype=np.float32),
                        np.full(self.seq_len, agent_type, dtype=np.float32),
                    ], axis=1)

                    type_trajs.append(seq)
            # type_trajs_filter = self.separate_runway_trajectories(type_trajs, mapdata)
            trajs_by_type[agent_type] = type_trajs
            print(f"✅ Extracted {len(type_trajs)} valid sequences for Type {agent_type}.")
            gc.collect()

        return trajs_by_type

    def separate_runway_trajectories(self, trajs_by_type, map_data, distance_threshold=0.005):
        """
        Separate trajectories into taxiing and runway ones.

        Args:
            trajs_by_type: dict {type_id: [np.ndarray(seq_len, 6), ...]}
            map_data: dict including 'map_infos' with 'all_polylines' and 'thr_id'
            distance_threshold: float, max distance to consider "on runway"

        Returns:
            taxi_trajs_by_type: dict
            runway_trajs_by_type: dict
        """
        print("\n✈️ Separating runway and taxi trajectories ...")

        all_polylines = map_data["map_infos"]["all_polylines"]
        thr_ids = map_data["map_infos"]["thr_id"]

        # --- Build runway LineStrings ---
        runway_segments = []
        for info in thr_ids:
            poly_idx = info["polyline_index"]
            pl = all_polylines[poly_idx]
            start = (pl[2], pl[3])
            end = (pl[6], pl[7])
            runway_segments.append(LineString([start, end]))

        print(f"🛬 Loaded {len(runway_segments)} runway segments.")

        taxi_trajs_by_type = []
        runway_trajs_by_type = []

        # --- Check each trajectory ---
        for traj in tqdm(trajs_by_type):
            xy = traj[:, :2]
            traj_line = LineString(xy)

            # compute min distance to any runway segment
            min_dist = min(traj_line.distance(seg) for seg in runway_segments)

            if min_dist <= distance_threshold:
                runway_trajs_by_type.append(traj)
            else:
                taxi_trajs_by_type.append(traj)

        print(f"🚕 Taxi trajectories   : {len(taxi_trajs_by_type)}")
        print(f"🛫 Runway trajectories : {len(runway_trajs_by_type)}")

        return taxi_trajs_by_type

    # -------------------------------------------------
    def process_all(self):
        """Main pipeline: load all CSVs, extract trajectories (by type), and save."""
        combined_df = self.load_all_data()
        self.all_trajs_by_type = self.extract_trajectories(combined_df)
        self.save_all()
        return self.all_trajs_by_type

    # -------------------------------------------------
    def save_all(self):
        """Save each agent type's trajectories to separate pickle files."""
        for t, trajs in self.all_trajs_by_type.items():
            filename = f"aircraft_type_{t}_trajectories.pkl"
            out_file = os.path.join(self.out_data_dir, filename)
            with open(out_file, "wb") as f:
                pickle.dump(trajs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"💾 Saved {len(trajs)} trajectories to {out_file}")
