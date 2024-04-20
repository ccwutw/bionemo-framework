# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from functools import lru_cache
from typing import Any, Sequence, Union

import torch
from nemo.utils.model_utils import import_class_by_path
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import bionemo.utils
from bionemo.data.singlecell.datamodule import AdamsonDataModule
from bionemo.model.core import MLPModel
from bionemo.model.core.encoder_finetuning import EncoderFineTuning


class FineTuneGeneformerModel(EncoderFineTuning):
    """
    Fine-tunes the Geneformer model for downstream tasks.

    Args:
        cfg (object): Configuration object.
        dataset_class (object): Dataset class.
        trainer (object): Trainer object.
        tokenizer (object): Tokenizer object.
        median_dict (dict): Dictionary containing median values.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, cfg, trainer, median_dict, **kwargs):
        # This changed too.
        self.encoder_frozen = cfg.encoder_frozen
        self.use_peft = cfg.get('peft.enabled', False)
        self.task_type = cfg.data.task_type
        super().__init__(cfg, trainer=trainer)
        self.tokenizer = self.encoder_model.tokenizer
        self.median_dict = median_dict
        # self.median_dict = self.encoder_model.median_dict

    def configure_optimizers(self) -> Union[Optimizer, tuple[list[Optimizer], list[_LRScheduler]]]:
        super().setup_optimization(optim_config=self.cfg.finetuning_optim)

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def build_loss_fn(self) -> Any:
        loss = bionemo.utils.lookup_or_use(torch.nn, self.cfg.loss_func)
        return loss

    def build_task_head(self) -> MLPModel:
        if self.task_type in ['regression', 'classification']:
            task_head = MLPModel(
                layer_sizes=[
                    self.encoder_model.cfg.model.input_size,
                    self.cfg.hidden_size,
                    self.cfg.data.target_size,
                ],
                dropout=0.0,
            )
        else:
            raise ValueError(f"Task type {self.task_type} not supported. Use one of ['regression', 'classification'].")

        return task_head

    def setup_encoder_model(self, cfg, trainer) -> torch.nn.Module:  # Or some decendant
        # Uses PEFT flag to determine whether to load enconder +/- adapters
        infer_class = import_class_by_path(self.cfg.encoder_cfg.infer_target)
        pretrained_model = infer_class(
            self.cfg.encoder_cfg,
            freeze=self.encoder_frozen,
            restore_path=self.cfg.restore_encoder_path,
            training=not self.cfg.encoder_frozen,
            adjust_config=False,
            strict_restore_from_path=False,
        )
        return pretrained_model

    @lru_cache
    def data_setup(self):
        self.data_module = AdamsonDataModule(
            self.cfg, self.trainer, self.tokenizer, self.median_dict, max_len=self.cfg.seq_length
        )

    def on_fit_start(self):
        self.build_train_valid_test_datasets()
        return super().on_fit_start()

    def build_train_valid_test_datasets(self):
        self._train_ds = self.data_module.get_sampled_train_dataset()
        self._validation_ds = self.data_module.get_sampled_val_dataset()
        self._test_ds = self.data_module.get_sampled_test_dataset()

    def encoder_forward(self, model: torch.nn.Module, batch: dict) -> Sequence:
        """
        Forward pass of the encoder module.

        Args:
            model (torch.nn.Module): The encoder model. Probably should be refined to some NeMo type, but we dont know what the limit is.
            batch (dict): The input batch containing 'input_ids', 'types', and 'padding_mask'.

        Returns:
            Sequence: The embeddings extracted from the model.

        """
        input_ids = batch.get('input_ids')
        token_types = batch.get("types", None)
        padding_mask = batch.get("padding_mask")
        embeddings = model.extract_embeddings(input_ids, padding_mask, token_type_ids=token_types)
        return embeddings

    def extract_for_task_head(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # NOTE investigate using mixed precision to remove need for float casting; maybe use setup_trainer method
        return input_tensor.float()

    def get_target_from_batch(self, batch: dict) -> torch.Tensor:
        """
        Extracts the target tensor from the given batch dictionary.

        Args:
            batch (dict): A dictionary containing the batch data with atleast the key 'target'

        Returns:
            torch.Tensor: The target tensor extracted from the batch.
        """
        return batch["target"]

    def _calc_step(self, batch: dict, batch_idx: int) -> tuple:
        output_tensor = self.forward(batch)
        target = batch['target']
        loss = self.loss_fn(output_tensor, target)
        return loss, output_tensor, target

    def state_dict(self, destination=None, prefix=None, keep_vars=False) -> dict:
        custom_state_dict = super(FineTuneGeneformerModel, self).state_dict()

        if self.use_peft:
            custom_state_dict.update(self.encoder_model.model.state_dict())

        return custom_state_dict

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        self.predict_dataset = self.data_module.get_test_dataset()
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            shuffle=False,
            batch_size=self.cfg.micro_batch_size,
        )
