import os
import torch
from copy import deepcopy
from re import split
from easydict import EasyDict
from lightning import LightningDataModule
from math import floor
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Optional
import random
import numpy as np
from collections import Counter

from amelia_tf.utils import pylogger
from amelia_tf.utils import data_utils as D

log = pylogger.get_pylogger(__name__)


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
        ego_agent_id = item.get("ego_agent_id_test", None)
        
        if ego_agent_id is not None and ego_agent_id < len(modes):
            ego_mode = modes[ego_agent_id]
            turn_m = int(ego_mode) // 4
            if turn_m in mode_indices:
                mode_indices[turn_m].append(idx)
        else:
            # Fallback: use first agent
            log.warning(f"Sample {idx} missing ego_agent_id, using first agent")
            if len(modes) > 0:
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
    
    # ========== Ego Agent ID Distribution Statistics ==========
    log.info("=" * 50)
    log.info("Ego Agent ID Distribution Statistics:")
    log.info("=" * 50)
    
    # Collect ego_agent_id from original dataset
    original_ego_ids = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        ego_agent_id = item.get("ego_agent_id_test", None)
        if ego_agent_id is not None:
            original_ego_ids.append(ego_agent_id)
        else:
            original_ego_ids.append(-1)  # Mark missing as -1
    
    # Count distribution of ego_agent_id in original dataset
    from collections import Counter
    original_ego_counts = Counter(original_ego_ids)
    
    log.info("Original test set ego_agent_id distribution:")
    for agent_id, count in sorted(original_ego_counts.items()):
        if agent_id == -1:
            log.info(f"  Missing ego_agent_id: {count}")
        else:
            log.info(f"  Agent {agent_id}: {count}")
    
    # Collect ego_agent_id from balanced dataset
    balanced_ego_ids = []
    for idx in balanced_indices:
        item = dataset[idx]
        ego_agent_id = item.get("ego_agent_id_test", None)
        if ego_agent_id is not None:
            balanced_ego_ids.append(ego_agent_id)
        else:
            balanced_ego_ids.append(-1)
    
    # Count distribution of ego_agent_id in balanced dataset
    balanced_ego_counts = Counter(balanced_ego_ids)
    
    log.info("\nBalanced test set ego_agent_id distribution:")
    for agent_id, count in sorted(balanced_ego_counts.items()):
        if agent_id == -1:
            log.info(f"  Missing ego_agent_id: {count}")
        else:
            log.info(f"  Agent {agent_id}: {count}")
    
    # Summary
    log.info(f"\nTotal unique agent IDs in original: {len([k for k in original_ego_counts.keys() if k >= 0])}")
    log.info(f"Total unique agent IDs in balanced: {len([k for k in balanced_ego_counts.keys() if k >= 0])}")
    log.info("=" * 50)
    
    return Subset(dataset, balanced_indices)


class DataModule(LightningDataModule):
    """ DataModule wrapper based on: lightning.ai/docs/pytorch/stable/data/datamodule.html """

    def __init__(self, dataset: Dataset, extra_params: EasyDict):
        """
        Inputs
        ------
            dataset[Dataset]: pytorch dataset object.
            extra_params[EasyDict]: dictionary containing any additional parameters needed by the
                class. NOTE: This is used in order to avoid adding more init parameters.
        """
        super().__init__()

        # This line allows to access init parameters with 'self.hparams' attribute also ensures init
        # parameterss will be stored in ckpt.
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_test_balanced: Optional[Dataset] = None  # Balanced test set

        self.dataset = dataset

        self.eparams = extra_params
        self.data_prep = self.eparams.data_prep
        self.supported_airports = self.eparams.supported_airports

        self.task_name = self.eparams.task_name

        self.splits_suffix = f"{self.data_prep.split_type}_{self.data_prep.exp_suffix}"

        self.split_path = {
            "train": f"{self.data_prep.split_dir}/train_{self.splits_suffix}.txt",
            "val": f"{self.data_prep.split_dir}/val_{self.splits_suffix}.txt",
            "test": f"{self.data_prep.split_dir}/test_{self.splits_suffix}.txt"
        }

        assert self.task_name in self.eparams.task_names
        
        self.use_fraction = getattr(self.eparams.data_prep, "fraction", 1.0)
        self.fraction_seed = 42
        self.test_size = getattr(self.eparams.data_prep, "test_size", None)
        
        # New parameters for balanced test set
        self.use_balanced_test = getattr(self.eparams.data_prep, "use_balanced_test", False)
        self.test_balance_samples = getattr(self.eparams.data_prep, "test_balance_samples", None)
        self.balance_random_seed = getattr(self.eparams.data_prep, "balance_random_seed", 42)

    def prepare_data(self):
        """ Creates the data splits for training, validation and testing. """
        # NOTE: All inside this function SHOULD be done outside this repository.
        # TODO: Need to create 'amelia_dataset' repo to do data-preparation stuff. The data module
        # should only load the splits without having to deal with any of this.

        # ------------------------------------------------------------------------------------------
        # log.info("Creating dataset splits.")
        assert self.data_prep.split_type in self.eparams.supported_splits, \
            f"Data split type {self.data_prep.split_type} not in supported splits: {self.eparams.supported_splits}."

        # ------------------------------------------------------------------------------------------
        # Process 'seen' airports into train/val/test splits.
        log.info("Preparing dataset splits.")
        seen_airports = self.data_prep.seen_airports
        assert len(seen_airports) > 0, f"Train airport list is empty: {seen_airports}"
        assert len(seen_airports) == len(set(seen_airports)), f"Duplicate airports {seen_airports}"
        assert all(airport in self.supported_airports for airport in seen_airports), \
            f"Unsupported airport. Supported ones are {self.supported_airports}"

        train_list, val_list, test_list = [], [], []
        for airport in seen_airports:
            filename = f"{airport}_{self.data_prep.split_type}"
            with open(f"{self.data_prep.split_dir_}/train_splits/{filename}.txt", 'r') as fp:
                airport_list = [line.rstrip() for line in fp]
                train_list += airport_list[:int(len(airport_list) * self.data_prep.to_process)]

            with open(f"{self.data_prep.split_dir_}/val_splits/{filename}.txt", 'r') as fp:
                airport_list = [line.rstrip() for line in fp]
                val_list += airport_list[:int(len(airport_list) * self.data_prep.to_process)]

            with open(f"{self.data_prep.split_dir_}/test_splits/{filename}.txt", 'r') as fp:
                airport_list = [line.rstrip() for line in fp]
                test_list += airport_list[:int(len(airport_list) * self.data_prep.to_process)]

        # ------------------------------------------------------------------------------------------
        # If 'unseen' airports are specified, then it will first iterate over unseen_airports and
        # create a random test split for each and add it to the test list.
        unseen_airports = self.data_prep.unseen_airports
        if len(unseen_airports) > 0:
            assert all(not airport in seen_airports for airport in unseen_airports), \
                f"'Unseen' airport {airport} is in 'Seen' list {seen_airports}"
            assert all(airport in self.supported_airports for airport in unseen_airports), \
                f"Unsupported airport {airport}. Supported ones are {self.supported_airports}"

            for airport in unseen_airports:
                filename = f"{airport}_{self.data_prep.split_type}"
                with open(f"{self.data_prep.traj_data_dir}/splits/test_splits/{filename}.txt", 'r') as fp:
                    airport_list = [line.rstrip() for line in fp]
                    test_list += airport_list[:int(len(airport_list) * self.data_prep.to_process)]

        # ------------------------------------------------------------------------------------------
        # Load blacklist and remove files in blacklist from split files
        self.blacklist = D.load_blacklist(self.data_prep, self.supported_airports)
        flat_blacklist = D.flatten_blacklist(self.blacklist)

        # ------------------------------------------------------------------------------------------
        # Save 'temporary' train/val/test splits.
        # TODO: verify that split lists do not share information
        os.makedirs(self.data_prep.split_dir, exist_ok=True)
        self.train_list = D.remove_blacklisted(flat_blacklist, train_list)
        with open(self.split_path["train"], 'w') as fp:
            fp.write('\n'.join(self.train_list))

        self.val_list = D.remove_blacklisted(flat_blacklist, val_list)
        with open(self.split_path["val"], 'w') as fp:
            fp.write('\n'.join(self.val_list))

        self.test_list = D.remove_blacklisted(flat_blacklist, test_list)
        with open(self.split_path["test"], 'w') as fp:
            fp.write('\n'.join(self.test_list))

    def setup(self, stage: Optional[str] = None):
        """
        Processes the input data within the dataset object and randomly splits it.

        NOTE: This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so
        be careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            if self.task_name == "train":
                log.info(f"Processing train set")
                self.data_train = deepcopy(self.dataset)
                self.data_train.set_split_list(self.split_path["train"])
                # self.data_train.set_blacklist(self.blacklist)
                self.data_train.prepare_data()
                log.info(f"...done!")

                log.info(f"Processing validation set")
                self.data_val = deepcopy(self.dataset)
                self.data_val.set_split_list(self.split_path["val"])
                # self.data_val.set_blacklist(self.data_train.get_blacklist())
                self.data_val.prepare_data()
                log.info(f"...done!")

            log.info(f"Processing test set")
            self.data_test = deepcopy(self.dataset)
            self.data_test.set_split_list(self.split_path["test"])
            # self.data_test.set_blacklist(self.data_val.get_blacklist())
            self.data_test.prepare_data()
            if self.test_size is not None:
                n = len(self.data_test)
                g = torch.Generator().manual_seed(self.fraction_seed)
                indices = torch.randperm(n, generator=g)[:self.test_size].tolist()
                self.data_test = Subset(self.data_test, indices)
                log.info(f"Capped test set to {len(self.data_test)} samples (test_size={self.test_size})")
            log.info(f"...done!")
            
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
            else:
                self.data_test_balanced = None
            
            log.info(f"Dataset setup complete!")
            
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

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.eparams.batch_size,
            num_workers=self.eparams.num_workers,
            pin_memory=self.eparams.pin_memory,
            shuffle=True,
            collate_fn=self.dataset.collate_batch,
            persistent_workers=self.eparams.persistent_workers,
            prefetch_factor=4
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.eparams.batch_size,
            num_workers=self.eparams.num_workers,
            pin_memory=self.eparams.pin_memory,
            shuffle=False,
            collate_fn=self.dataset.collate_batch,
            persistent_workers=self.eparams.persistent_workers
        )

    def test_dataloader(self):
        """
        Return test dataloaders for comprehensive evaluation.

        Returns:
            List of dataloaders: [original_test_set, balanced_test_set] (if available)
        """
        dataloaders = []

        # 1. Original test set
        if hasattr(self, 'data_test') and self.data_test is not None:
            dataloaders.append(
                DataLoader(
                    dataset=self.data_test,
                    batch_size=self.eparams.batch_size,
                    shuffle=False,
                    num_workers=self.eparams.num_workers,
                    pin_memory=self.eparams.pin_memory,
                    collate_fn=self.dataset.collate_batch,
                    persistent_workers=self.eparams.persistent_workers
                )
            )
            log.info("Added ORIGINAL test set to test dataloader list")

        # 2. Balanced test set
        if self.use_balanced_test and self.data_test_balanced is not None:
            dataloaders.append(
                DataLoader(
                    dataset=self.data_test_balanced,
                    batch_size=self.eparams.batch_size,
                    shuffle=False,
                    num_workers=self.eparams.num_workers,
                    pin_memory=self.eparams.pin_memory,
                    collate_fn=self.dataset.collate_batch,
                    persistent_workers=self.eparams.persistent_workers
                )
            )
            log.info("Added BALANCED test set to test dataloader list")

        # For backward compatibility: if no balanced set, return single dataloader
        if len(dataloaders) == 1:
            return dataloaders[0]

        return dataloaders

    # Optional: Keep original method for backward compatibility
    def get_test_dataloader(self, balanced=True):
        """Legacy method for single test set."""
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
    _ = DataModule()