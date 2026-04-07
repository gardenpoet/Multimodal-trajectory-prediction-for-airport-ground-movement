import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns  # for KDE smoothing


class TrajectoryAnalyzer:
    """
    Analyze pre-extracted aircraft ground trajectories.
    Each trajectory is shape (seq_len, 6): [x, y, speed, heading, id, type]
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.trajs_by_type = self.load_all_pickles()

    # ---------------------------------------------
    def load_all_pickles(self):
        """Load all pickle files (grouped by type)."""
        trajs_by_type = {}
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".pkl"):
                t = int(fname.split("_")[2])  # assumes aircraft_type_{t}_trajectories.pkl
                path = os.path.join(self.data_dir, fname)
                with open(path, "rb") as f:
                    trajs = pickle.load(f)
                trajs_by_type[t] = trajs
                print(f"✅ Loaded {len(trajs)} trajectories for Type {t}")
        return trajs_by_type

    # ---------------------------------------------
    def compute_statistics(self):
        """Compute speed/turning/acceleration statistics for each agent type."""
        stats = {}

        for t, trajs in self.trajs_by_type.items():
            print(f"/n📊 Analyzing agent type {t} ...")
            all_speeds, all_turn_rates, all_accels = [], [], []

            for seq in tqdm(trajs):
                spd = seq[:, 2]
                hdg = np.unwrap(np.deg2rad(seq[:, 3]))  # unwrap heading (radians)
                turn_rate = np.gradient(hdg) * 180 / np.pi  # deg/frame
                accel = np.gradient(spd)

                all_speeds.extend(spd)
                all_turn_rates.extend(turn_rate)
                all_accels.extend(accel)

            stats[t] = {
                "speed_mean": np.mean(all_speeds),
                "speed_std": np.std(all_speeds),
                "accel_mean": np.mean(all_accels),
                "accel_std": np.std(all_accels),
                "turn_mean": np.mean(all_turn_rates),
                "turn_std": np.std(all_turn_rates),
                "num_trajs": len(trajs),
            }

            print(f"  🚀 Avg speed: {stats[t]['speed_mean']:.2f} ± {stats[t]['speed_std']:.2f}")
            print(f"  🔄 Avg turn rate: {stats[t]['turn_mean']:.2f} ± {stats[t]['turn_std']:.2f}")
            print(f"  ⚙️  Avg accel: {stats[t]['accel_mean']:.4f} ± {stats[t]['accel_std']:.4f}")

        self.stats = stats
        return stats

    # ---------------------------------------------
    def visualize_distributions(self):
        """Plot distributions of speed and turning rate for each agent type."""
        for t, trajs in self.trajs_by_type.items():
            all_speeds, all_turns = [], []

            for seq in trajs:
                spd = seq[:, 2]
                hdg = np.unwrap(np.deg2rad(seq[:, 3]))
                turn_rate = np.gradient(hdg) * 180 / np.pi

                all_speeds.extend(spd)
                all_turns.extend(turn_rate)

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(all_speeds, bins=40, alpha=0.7)
            plt.title(f"Speed Distribution (Type {t})")
            plt.xlabel("Speed (knots or m/s)")
            plt.ylabel("Count")

            plt.subplot(1, 2, 2)
            plt.hist(all_turns, bins=40, alpha=0.7, color='orange')
            plt.title(f"Turning Rate Distribution (Type {t})")
            plt.xlabel("Turning rate (deg/frame)")
            plt.ylabel("Count")

            plt.tight_layout()
            plt.show()

    # ---------------------------------------------
    def visualize_full_distributions(self, bins=60, kde=False):
  

        for t, trajs in self.trajs_by_type.items():
            all_speeds, all_turns, all_accels = [], [], []
    
            # collect all values with filtering
            for seq in trajs:
                spd = seq[:, 2]
                hdg = np.unwrap(np.deg2rad(seq[:, 3]))
                turn_rate = np.gradient(hdg) * 180 / np.pi
                accel = np.gradient(spd)
    
                # Filter values within specified ranges
                all_speeds.extend(spd)
                
                # Filter turn rate to [-20, 20]
                turn_rate_filtered = turn_rate[(turn_rate >= -20) & (turn_rate <= 20)]
                all_turns.extend(turn_rate_filtered)
                
                # Filter acceleration to [-10, 10]
                accel_filtered = accel[(accel >= -10) & (accel <= 10)]
                all_accels.extend(accel_filtered)
    
            print(f"Type {t}: {len(all_speeds)} speed points, {len(all_turns)} turn rate points, {len(all_accels)} acceleration points")
    
            # --------------- Plotting section --------------------
            plt.figure(figsize=(18, 5))
    
            # ---- Speed ----
            plt.subplot(1, 3, 1)
            sns.histplot(all_speeds, bins=bins, kde=kde)
            plt.title(f"Speed Distribution (Type {t})")
            plt.xlabel("Speed")
            plt.ylabel("Frequency")
    
            # ---- Acceleration (filtered) ----
            plt.subplot(1, 3, 2)
            sns.histplot(all_accels, bins=bins, kde=kde, color="green")
            plt.title(f"Acceleration Distribution (Type {t})\nFiltered: [-10, 10]")
            plt.xlabel("Acceleration")
            plt.ylabel("Frequency")
            plt.xlim(-10, 10)  # Ensure x-axis shows the filtered range
    
            # ---- Turn Rate (filtered) ----
            plt.subplot(1, 3, 3)
            sns.histplot(all_turns, bins=bins, kde=kde, color="orange")
            plt.title(f"Turn Rate Distribution (Type {t})\nFiltered: [-20, 20]")
            plt.xlabel("Turn Rate (deg/frame)")
            plt.ylabel("Frequency")
            plt.xlim(-20, 20)  # Ensure x-axis shows the filtered range
    
            plt.suptitle(f"Trajectory Dynamics Distributions: Type {t} (with filtering)", fontsize=15)
            plt.tight_layout()
            plt.savefig(f'trajectory_distributions_type_{t}.png', dpi=300, bbox_inches='tight')
            plt.close()  # Free memory


data_dir = "/data/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main2/datasets/amelia/traj_data_a10v08/proc_full_scenes/kbos/"
analyzer = TrajectoryAnalyzer(data_dir)

# stats = analyzer.compute_statistics()
analyzer.visualize_full_distributions()