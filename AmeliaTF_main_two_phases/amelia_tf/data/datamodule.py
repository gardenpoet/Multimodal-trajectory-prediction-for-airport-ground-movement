from collections import Counter
from torch.utils.data import DataLoader, Dataset, Subset
import os
from copy import deepcopy
from easydict import EasyDict
from lightning import LightningDataModule
from typing import Optional
import random
import torch
import numpy as np

from amelia_tf.utils import pylogger
from amelia_tf.utils import data_utils as D
from amelia_tf.utils.modes import VALID_TURN_MODES

log = pylogger.get_pylogger(__name__)


# =========================================================
# Agent-level mode statistics (EGO-ONLY)
# =========================================================
def summarize_agent_level_modes(dataset, name, max_scenes=10000):
    """
    Summarize ego-agent level mode distribution.
    
    This is more representative of what the model actually learns.
    """
    counter = Counter()
    n = len(dataset)
    indices = random.sample(range(n), min(max_scenes, n))
    
    for idx in indices:
        item = dataset[idx]
        modes = item["mode_labels"]  # (A,) values: 0-15
        
        # Get ego agent
        ego_agent_id = item.get("ego_agent_id", None)
        
        if ego_agent_id is not None and ego_agent_id < len(modes):
            ego_mode = modes[ego_agent_id].item() if torch.is_tensor(modes[ego_agent_id]) else modes[ego_agent_id]
            turn_m = int(ego_mode) // 4
            counter[turn_m] += 1
        else:
            # Fallback: use first agent
            log.warning(f"Sample {idx} missing ego_agent_id in {name}, using first agent")
            ego_mode = modes[0].item() if torch.is_tensor(modes[0]) else modes[0]
            turn_m = int(ego_mode) // 4
            counter[turn_m] += 1
    
    total = sum(counter.values())
    log.info(f"[EGO-ONLY] Agent-level turn-mode distribution ({name})")
    for m in sorted(counter):
        mode_name = ["TurnLeft", "TurnRight", "Straight", "Hold"][m]
        log.info(f"  {mode_name}: {counter[m]} ({counter[m]/total:.4f})")
        

def fast_agent_level_balance(dataset, keep_ratio=0.3):
    """
    Balance agent-level mode distribution by removing
    scenes dominated by frequent modes.

    Args:
        dataset: original dataset
        keep_ratio: fraction of scenes to keep (e.g. 0.7)

    Returns:
        Subset(dataset, selected_indices)
    """

    log.info("Agent-level balancing via dominant-scene pruning...")

    # --------------------------------------------------
    # 1. Collect agent-level mode statistics
    # --------------------------------------------------
    scene_modes_list = []
    global_counter = Counter()

    for item in dataset:
        modes = item["mode_labels"].tolist()
        scene_modes_list.append(modes)
        for m in modes:
            global_counter[int(m)] += 1

    total_agents = sum(global_counter.values())
    mode_freq = {
        m: global_counter[m] / total_agents
        for m in global_counter
    }

    log.info("Global agent-level mode frequency:")
    for m in sorted(mode_freq):
        log.info(f"  Mode {m}: {mode_freq[m]:.4f}")

    # --------------------------------------------------
    # 2. Compute penalty score per scene
    # --------------------------------------------------
    scene_scores = []
    for idx, modes in enumerate(scene_modes_list):
        counter = Counter(modes)
        score = 0.0
        for m, c in counter.items():
            score += c * mode_freq[m]   # dominant modes penalized more
        scene_scores.append((idx, score))

    # --------------------------------------------------
    # 3. Keep scenes with lowest penalty
    # --------------------------------------------------
    scene_scores.sort(key=lambda x: x[1])  # ascending
    keep_n = int(len(scene_scores) * keep_ratio)
    selected_indices = [idx for idx, _ in scene_scores[:keep_n]]

    log.info(
        f"Selected {keep_n}/{len(dataset)} scenes "
        f"({keep_ratio:.0%}) after balancing"
    )

    return Subset(dataset, selected_indices)


def create_balanced_test_set(dataset, samples_per_class=None, random_seed=42):
    """
    Create a balanced test set by sampling equal number of EGO agents from each mode.
    
    Args:
        dataset: Original test dataset
        samples_per_class: Number of ego-agent samples per class (if None, use min class count)
        random_seed: Random seed for reproducibility
    
    Returns:
        Subset(dataset, selected_indices) with balanced ego-agent distribution
    """
    log.info("Creating balanced test set (ego-agent only)...")
    
    # Collect indices for each mode (ego-agent level)
    mode_indices = {0: [], 1: [], 2: [], 3: []}  # TurnLeft, TurnRight, Straight, Hold
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        modes = item["mode_labels"].tolist()
        
        # Get ego agent
        ego_agent_id = item.get("ego_agent_id", None)
        
        if ego_agent_id is not None and ego_agent_id < len(modes):
            ego_mode = modes[ego_agent_id]
            turn_m = int(ego_mode) // 4
            if turn_m in mode_indices:
                mode_indices[turn_m].append(idx)
        else:
            # Fallback: use first agent
            log.warning(f"Sample {idx} missing ego_agent_id, using first agent")
            ego_mode = modes[0]
            turn_m = int(ego_mode) // 4
            if turn_m in mode_indices:
                mode_indices[turn_m].append(idx)
    
    # Log original distribution
    log.info("Original test set ego-agent distribution:")
    for mode in range(4):
        mode_name = ["TurnLeft", "TurnRight", "Straight", "Hold"][mode]
        log.info(f"  {mode_name}: {len(mode_indices[mode])} ego agents")
    
    # Determine samples per class
    if samples_per_class is None:
        min_count = min(len(indices) for indices in mode_indices.values())
        samples_per_class = min_count
        log.info(f"Using minimum class count: {samples_per_class}")
    else:
        log.info(f"Using specified samples per class: {samples_per_class}")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Sample indices for each class
    balanced_indices = set()
    for mode in range(4):
        if len(mode_indices[mode]) >= samples_per_class:
            sampled = random.sample(mode_indices[mode], samples_per_class)
        else:
            # If not enough samples, use all available and sample with replacement
            log.warning(f"Not enough ego agents for mode {mode}, using all {len(mode_indices[mode])} samples")
            sampled = mode_indices[mode]
            if len(sampled) < samples_per_class:
                sampled.extend(random.choices(mode_indices[mode], k=samples_per_class - len(sampled)))
                log.info(f"  Sampled with replacement to reach {samples_per_class}")
        
        balanced_indices.update(sampled)
    
    # Convert to list and shuffle
    balanced_indices = list(balanced_indices)
    random.shuffle(balanced_indices)
    
    # Verify new distribution
    new_mode_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for idx in balanced_indices:
        item = dataset[idx]
        modes = item["mode_labels"].tolist()
        ego_agent_id = item.get("ego_agent_id", 0)
        if ego_agent_id < len(modes):
            ego_mode = modes[ego_agent_id]
            turn_m = int(ego_mode) // 4
            new_mode_counts[turn_m] += 1
    
    log.info("Balanced test set ego-agent distribution:")
    for mode in range(4):
        mode_name = ["TurnLeft", "TurnRight", "Straight", "Hold"][mode]
        log.info(f"  {mode_name}: {new_mode_counts[mode]} ego agents")
    
    log.info(f"Created balanced test set with {len(balanced_indices)} scenes")
    
    return Subset(dataset, balanced_indices)


# =========================================================
# DataModule
# =========================================================
class DataModule(LightningDataModule):

    def __init__(self, dataset: Dataset, extra_params: EasyDict):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset = dataset
        self.eparams = extra_params
        self.data_prep = self.eparams.data_prep
        self.supported_airports = self.eparams.supported_airports
        self.task_name = self.eparams.task_name

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_test_balanced: Optional[Dataset] = None  # New balanced test set

        self.splits_suffix = f"{self.data_prep.split_type}_{self.data_prep.exp_suffix}"
        self.split_path = {
            "train": f"{self.data_prep.split_dir}/train_{self.splits_suffix}.txt",
            "val": f"{self.data_prep.split_dir}/val_{self.splits_suffix}.txt",
            "test": f"{self.data_prep.split_dir}/test_{self.splits_suffix}.txt"
        }

        assert self.task_name in self.eparams.task_names

        self.use_fraction = getattr(self.eparams.data_prep, "fraction", 1.0)
        self.fraction_seed = 42
        
        # New parameters for balanced test set
        self.use_balanced_test = getattr(self.eparams.data_prep, "use_balanced_test", True)
        self.test_balance_samples = getattr(self.eparams.data_prep, "test_balance_samples", None)
        self.balance_random_seed = getattr(self.eparams.data_prep, "balance_random_seed", 42)

    # -----------------------------------------------------
    # prepare_data
    # -----------------------------------------------------
    def prepare_data(self):
        log.info("Preparing dataset splits.")

        seen_airports = self.data_prep.seen_airports
        train_list, val_list, test_list = [], [], []

        for airport in seen_airports:
            filename = f"{airport}_{self.data_prep.split_type}"

            for split, container in zip(
                ["train", "val", "test"],
                [train_list, val_list, test_list]
            ):
                path = f"{self.data_prep.traj_data_dir}/splits/{split}_splits/{filename}.txt"
                with open(path, "r") as fp:
                    lines = [l.strip().replace("\\", "/") for l in fp]
                    container += lines[:int(len(lines) * self.data_prep.to_process)]

        self.blacklist = D.load_blacklist(self.data_prep, self.supported_airports)
        flat_blacklist = D.flatten_blacklist(self.blacklist)

        os.makedirs(self.data_prep.split_dir, exist_ok=True)

        self.train_list = D.remove_blacklisted(flat_blacklist, train_list)
        self.val_list = D.remove_blacklisted(flat_blacklist, val_list)
        self.test_list = D.remove_blacklisted(flat_blacklist, test_list)

        for split, lst in zip(["train", "val", "test"],
                              [self.train_list, self.val_list, self.test_list]):
            with open(self.split_path[split], "w") as fp:
                fp.write("\n".join(lst))

    def mode_frequency(self, dataset, debug_mode=True, debug_size=10000):
        """
        Prepare mode frequencies for loss weight using ONLY ego agent modes.
        
        This ensures loss weights reflect the actual distribution the model will
        encounter during training (only the ego agent's modes matter).
        """
        log.info("Prepare mode frequencies for loss weight (ego-agent only)...")
    
        NUM_MODES = len(VALID_TURN_MODES)
        global_counter = Counter()
    
        if debug_mode:
            indices = range(min(debug_size, len(dataset)))
            log.info(f"Debug mode: using first {debug_size} samples only.")
        else:
            indices = range(len(dataset))
    
        # Count mode occurrences for ego agent only
        for idx in indices:
            item = dataset[idx]
            modes = item["mode_labels"].tolist()  # (A,) mode values 0-15
            
            # Get ego agent ID from the scene
            ego_agent_id = item.get("ego_agent_id", None)
            
            if ego_agent_id is not None and ego_agent_id < len(modes):
                # Only count the mode for the ego agent
                ego_mode = modes[ego_agent_id]
                turn_m = int(ego_mode) // NUM_MODES  # mapping 0-3
                global_counter[turn_m] += 1
            else:
                # Fallback: if no ego_agent_id, assume first agent is ego
                log.warning(f"Sample {idx} missing ego_agent_id, using first agent as ego")
                if len(modes) > 0:
                    ego_mode = modes[0]
                    turn_m = int(ego_mode) // NUM_MODES
                    global_counter[turn_m] += 1
    
        total_agents = sum(global_counter.values())
    
        # Build full mode frequency dict (0..3)
        if total_agents == 0:
            log.warning("No modes found in dataset; using zero frequencies.")
            mode_freq = {m: 0.0 for m in range(NUM_MODES)}
        else:
            mode_freq = {
                m: global_counter.get(m, 0) / total_agents
                for m in range(NUM_MODES)
            }
    
        log.info("Global ego-agent mode frequency (for loss weights):")
        for m in range(NUM_MODES):
            mode_name = ["TurnLeft", "TurnRight", "Straight", "Hold"][m]
            log.info(f"  {mode_name}: {mode_freq[m]:.6f} ({global_counter.get(m, 0)}/{total_agents})")
    
        # Compute class-balanced weights
        alpha = 1  # You can adjust this hyperparameter
        eps = 1e-6
        mode_freq_tensor = torch.tensor(
            [mode_freq[m] for m in range(NUM_MODES)],
            dtype=torch.float32
        )
    
        mode_weights = (1.0 / (mode_freq_tensor + eps)) ** alpha
    
        # Never-seen modes -> zero weight
        never_seen = mode_freq_tensor == 0
        mode_weights[never_seen] = 0.0
    
        # Normalize
        mean_w = mode_weights.mean()
        if mean_w > 0:
            mode_weights = mode_weights / mean_w
        else:
            log.warning("All mode weights are zero; using uniform weights.")
            mode_weights = torch.ones(NUM_MODES)
    
        self.mode_weights = mode_weights
        log.info(f"Mode weights computed (alpha={alpha}):")
        for m in range(NUM_MODES):
            mode_name = ["TurnLeft", "TurnRight", "Straight", "Hold"][m]
            log.info(f"  {mode_name}: {mode_weights[m].item():.4f}")

    # -----------------------------------------------------
    # setup
    # -----------------------------------------------------
    def setup(self, stage: Optional[str] = None):

        if not self.data_train and not self.data_val and not self.data_test:

            # -------------------------------
            # Load datasets
            # -------------------------------
            if self.task_name == "train":
                self.data_train = deepcopy(self.dataset)
                self.data_train.set_split_list(self.split_path["train"])
                self.data_train.prepare_data()

                self.data_val = deepcopy(self.dataset)
                self.data_val.set_split_list(self.split_path["val"])
                self.data_val.prepare_data()

            self.data_test = deepcopy(self.dataset)
            self.data_test.set_split_list(self.split_path["test"])
            self.data_test.prepare_data()

            # -------------------------------
            # Create balanced test set
            # -------------------------------
            if self.use_balanced_test:
                log.info("Creating balanced version of test set...")
                self.data_test_balanced = create_balanced_test_set(
                    self.data_test,
                    samples_per_class=self.test_balance_samples,
                    random_seed=self.balance_random_seed
                )
                # Also summarize the balanced test set
                summarize_agent_level_modes(self.data_test_balanced, "test (balanced)")
            else:
                self.data_test_balanced = None

            # -------------------------------
            # Agent-level statistics (EGO-ONLY)
            # -------------------------------
            if self.task_name == "train":
                summarize_agent_level_modes(self.data_train, "train (ego-only)")
                summarize_agent_level_modes(self.data_val, "val (ego-only)")
                self.mode_frequency(self.data_train)

            # Optional: Balance training set (commented out by default)
            # if self.task_name == "train":
            #     self.data_train = fast_agent_level_balance(self.data_train)
            #     summarize_agent_level_modes(self.data_train, "train (balanced)")

            log.info("Dataset setup complete!")

            def _subsample_dataset(dataset, fraction, seed):
                n = len(dataset)
                keep_n = int(n * fraction)
                g = torch.Generator().manual_seed(seed)
                indices = torch.randperm(n, generator=g)[:keep_n].tolist()
                return Subset(dataset, indices)

            if self.task_name == "train" and self.use_fraction <= 1.0:
                log.info(f"Using only {self.use_fraction:.0%} of each split for experiment")

                self.data_train = _subsample_dataset(
                    self.data_train, self.use_fraction, self.fraction_seed
                )
                self.data_val = _subsample_dataset(
                    self.data_val, self.use_fraction, self.fraction_seed
                )
                self.data_test = _subsample_dataset(
                    self.data_test, self.use_fraction, self.fraction_seed
                )
                
                if self.data_test_balanced is not None:
                    self.data_test_balanced = _subsample_dataset(
                        self.data_test_balanced, self.use_fraction, self.fraction_seed
                    )

                log.info(
                    f"Train/Val/Test sizes after subsample: "
                    f"{len(self.data_train)}, "
                    f"{len(self.data_val)}, "
                    f"{len(self.data_test)}"
                )
                
                if self.data_test_balanced is not None:
                    log.info(f"Balanced test size after subsample: {len(self.data_test_balanced)}")

    # -----------------------------------------------------
    # dataloaders
    # -----------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.eparams.batch_size,
            shuffle=True,
            num_workers=self.eparams.num_workers,
            pin_memory=self.eparams.pin_memory,
            collate_fn=self.dataset.collate_batch,
            persistent_workers=self.eparams.persistent_workers,
            prefetch_factor=4
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.eparams.batch_size,
            shuffle=False,
            num_workers=self.eparams.num_workers,
            pin_memory=self.eparams.pin_memory,
            collate_fn=self.dataset.collate_batch,
            persistent_workers=self.eparams.persistent_workers
        )

    def test_dataloader(self, balanced=True):
        """
        Get test dataloader.
        
        Args:
            balanced: If True, return balanced test set; if False, return original test set
        """
        if balanced and self.data_test_balanced is not None:
            dataset = self.data_test_balanced
            log.info("Using BALANCED test set")
        else:
            dataset = self.data_test
            log.info("Using ORIGINAL test set")
            
        return DataLoader(
            dataset=dataset,
            batch_size=self.eparams.batch_size,
            shuffle=False,
            num_workers=self.eparams.num_workers,
            pin_memory=self.eparams.pin_memory,
            collate_fn=self.dataset.collate_batch,
            persistent_workers=self.eparams.persistent_workers
        )


if __name__ == "__main__":
    _ = DataModule