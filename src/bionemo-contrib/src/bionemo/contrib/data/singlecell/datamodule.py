# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import os
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from bionemo.contrib.data.mapped_dataset import ResamplingMappedDataset
from bionemo.contrib.data.singlecell.dataset import SingleCellDataset


class SingleCellDataModule(pl.LightningDataModule):
    """LightningDataModule wrapper of `SingleCellDataset`

    Args:
        data_path (Union[str, PosixPath]): Path to preprocessed single-cell data files
        tokenizer (Tokenizer): Maps gene names to ids and vice-versa
        collator: Used to batch samples
        process_item: Function defining how each item should be processed
        num_workers (int): Number of workers to use
        num_mask_per_sample (int): Number of masked versions of a single sample to be returned by each worker
        train_batch_size (int): Batch size for training
        val_batch_size (int): Batch size for validation

    Attributes:
        cfg (Config): Configuration object
        data_path (Union[str, PosixPath]): Path to preprocessed single-cell data files
        median_dict (dict): Dictionary containing median values
        tokenizer (Tokenizer): Tokenizer object
        setup_called (bool): Flag indicating if the setup method has been called
        dataset (SingleCellDataset): Single-cell dataset object

    """

    # Nothing says we cant pass in the dataset...
    def __init__(
        self,
        tokenizer: Tokenizer,
        train_dataset_path: str,
        val_dataset_path: str,
        test_dataset_path: str,
        median_dict: dict[str, float],
        mask_prob: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.5,  # 50/50 split between mask and random token
        seq_length: int = 2048,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        index_mapping_dir: Optional[str] = None,
        rampup_batch_size: Optional[List[int]] = None,
        seed: int = 42,
        num_workers: int = 10,  # TODO can this be automatically set?
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_path_train = train_dataset_path
        self.data_path_val = val_dataset_path
        self.data_path_test = test_dataset_path
        self.tokenizer = tokenizer
        self.median_dict = median_dict
        self.max_len = seq_length
        self.mask_prob = mask_prob
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.seed = seed
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.index_mapping_dir = index_mapping_dir or str(Path(self.data_path_train).parent)
        self._train_dataset_ori = SingleCellDataset(
            self.data_path_train, self.tokenizer, self.median_dict, self.max_len
        )
        self._val_dataset_ori = SingleCellDataset(self.data_path_val, self.tokenizer, self.median_dict, self.max_len)
        self._test_dataset_ori = SingleCellDataset(self.data_path_test, self.tokenizer, self.median_dict, self.max_len)

        # This is needed here, or you need to specify it in the megatron adapter thing TODO name?
        self.data_sampler = MegatronDataSampler(
            seq_len=self.max_len,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        assert (
            hasattr(self, "trainer") and self.trainer is not None
        ), "Setup should be completed when trainer and config are attached."

        # Trainer API
        max_train_steps = self.trainer.max_steps
        if self.trainer.max_epochs > 1:
            logging.warning(
                "Trainer is set to run for multiple epochs. This is not recommended due to the same shuffle being used in each. Instead set max_epochs to 1 and increase the number of max_steps."
            )
        assert max_train_steps > 0, "Please specify trainer.max_steps"
        eval_iters = int((max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches)
        test_iters = self.trainer.limit_test_batches
        num_train_samples = int(max_train_steps * self.data_sampler.global_batch_size)
        num_val_samples = int(eval_iters * self.data_sampler.global_batch_size)
        num_test_samples = int(test_iters * self.data_sampler.global_batch_size)

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            # This is to make sure we only have one epoch on every validation iteration
            num_val_samples = 1

        # This happens exactly once during setup.
        self._train_ds = self._sample_and_shuffle_dataset(self._train_dataset_ori, num_train_samples, 'train')
        self._validation_ds = self._sample_and_shuffle_dataset(self._val_dataset_ori, num_val_samples, 'val')
        self._test_ds = self._sample_and_shuffle_dataset(self._test_dataset_ori, num_test_samples, 'test')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            # collate_fn=dataset.collate_fn,  No special work happens in this dataloader outside of getitem
            **kwargs,
        )

    def _sample_and_shuffle_dataset(self, dataset: SingleCellDataset, num_samples: int, stage: str):
        """Sample the training dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from

        Returns:
            ResamplingMappedDataset: Resampled dataset

        """
        # This is where re-sampling occurs.
        os.makedirs(self.index_mapping_dir, exist_ok=True)
        return ResamplingMappedDataset(
            dataset,
            data_prefix=self.index_mapping_dir,
            num_samples=num_samples,
            max_seq_length=self.max_len,
            cfg=None,
            seed=self.seed + len(stage),
            name=f'{stage}_{num_samples}',
            index_mapping_dir=self.index_mapping_dir,
        )