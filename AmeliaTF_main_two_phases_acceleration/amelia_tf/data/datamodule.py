from collections import Counter
from torch.utils.data import DataLoader, Dataset, Subset
import os
from copy import deepcopy
from easydict import EasyDict
from lightning import LightningDataModule
from typing import Optional
import random
import torch

from amelia_tf.utils import pylogger
from amelia_tf.utils import data_utils as D
from amelia_tf.utils.modes import VALID_TURN_MODES

log = pylogger.get_pylogger(__name__)


# =========================================================
# Agent-level mode statistics
# =========================================================
def summarize_agent_level_modes(dataset, name, max_scenes=10000):
    counter = Counter()
    n = len(dataset)
    indices = random.sample(range(n), min(max_scenes, n))
    for idx in indices:
        item = dataset[idx]
        modes = item["mode_labels"]  # (A,)  values: 0-15
        for m in modes.tolist():
            turn_m = int(m) // 4    # ? mapping 0-3
            counter[turn_m] += 1
    total = sum(counter.values())
    log.info(f"[FAST] Agent-level turn-mode distribution ({name})")
    for m in sorted(counter):
        log.info(f"  Turn mode {m}: {counter[m]} ({counter[m]/total:.4f})")
        

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

        self.splits_suffix = f"{self.data_prep.split_type}_{self.data_prep.exp_suffix}"
        self.split_path = {
            "train": f"{self.data_prep.split_dir}/train_{self.splits_suffix}.txt",
            "val": f"{self.data_prep.split_dir}/val_{self.splits_suffix}.txt",
            "test": f"{self.data_prep.split_dir}/test_{self.splits_suffix}.txt"
        }

        assert self.task_name in self.eparams.task_names

        self.use_fraction = getattr(self.eparams.data_prep, "fraction", 1.0)
        self.fraction_seed = 42

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
        log.info("Prepare mode frequencies for loss weight...")
    
        NUM_MODES = len(VALID_TURN_MODES)
        global_counter = Counter()
    
        if debug_mode:
            indices = range(min(debug_size, len(dataset)))
            log.info(f"Debug mode: using first {debug_size} samples only.")
        else:
            indices = range(len(dataset))
    
        # Count mode occurrences across all agents
        for idx in indices:
            item = dataset[idx]
            modes = item["mode_labels"].tolist()
            for m in modes:
                turn_m = int(m) // NUM_MODES  # mapping 0-3
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
    
        log.info("Global agent-level mode frequency:")
        for m in range(NUM_MODES):
            log.info(f"  Mode {m}: {mode_freq[m]:.6f}")
    
        # Compute class-balanced weights
        alpha = 1
        eps = 1e-6
        mode_freq_tensor = torch.tensor(
            [mode_freq[m] for m in range(NUM_MODES)],
            dtype=torch.float32
        )
    
        mode_weights = (1.0 / (mode_freq_tensor + eps)) ** alpha
    
        # Never-seen modes ? zero weight
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
        log.info(f"Mode weights computed (alpha={alpha}): {mode_weights.tolist()}")


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
            # Agent-level statistics
            # -------------------------------
            if self.task_name == "train":
                summarize_agent_level_modes(self.data_train, "train")
                summarize_agent_level_modes(self.data_val, "val")
                self.mode_frequency(self.data_train)

            # -------------------------------
            # Agent-level balancing: reduce dominant mode
            # -------------------------------
            # if self.task_name == "train":
            #     log.info("Optimizing agent-level balancing...")

                # scene_modes = [item["mode_labels"] for item in self.data_train]
                # mode_counter = Counter()
                # for modes in scene_modes:
                #     for m in modes.tolist():
                #         mode_counter[int(m)] += 1
                # 
                # min_count = min(mode_counter.values())
                # log.info(f"Minimum agent count across modes: {min_count}")
                # 
                # # target agent count per mode
                # target_count = {m: min_count for m in mode_counter}
                # 
                # selected_scenes = []
                # mode_accum = Counter()
                # 
                # indices = list(range(len(self.data_train)))
                # random.shuffle(indices)
                # 
                # for idx in indices:
                #     item = self.data_train[idx]
                #     modes = item["mode_labels"].tolist()
                # 
                #     add_scene = False
                #     for m in modes:
                #         if mode_accum[m] < target_count[m]:
                #             add_scene = True
                #             break
                # 
                #     if add_scene:
                #         selected_scenes.append(idx)
                #         for m in modes:
                #             mode_accum[m] += 1
                # 
                #     if all(mode_accum[m] >= target_count[m] for m in target_count):
                #         break
                # 
                # log.info(f"Selected {len(selected_scenes)}/{len(self.data_train)} scenes after balancing")
                # self.data_train = Subset(self.data_train, selected_scenes)
                # self.data_train = fast_agent_level_balance(self.data_train)
                #
                # summarize_agent_level_modes(self.data_train, "train (balanced)")

            log.info("Dataset setup complete!")

            def _subsample_dataset(dataset, fraction, seed):
                n = len(dataset)
                keep_n = int(n * fraction)
                g = torch.Generator().manual_seed(seed)
                indices = torch.randperm(n, generator=g)[:keep_n].tolist()
                return Subset(dataset, indices)

            if self.task_name == "train" and self.use_fraction < 1.0:
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

                log.info(
                    f"Train/Val/Test sizes after subsample: "
                    f"{len(self.data_train)}, "
                    f"{len(self.data_val)}, "
                    f"{len(self.data_test)}"
                )

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

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.eparams.batch_size,
            shuffle=False,
            num_workers=self.eparams.num_workers,
            pin_memory=self.eparams.pin_memory,
            collate_fn=self.dataset.collate_batch,
            persistent_workers=self.eparams.persistent_workers
        )


if __name__ == "__main__":
    _ = DataModule
