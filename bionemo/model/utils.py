# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Optional, Type, TypeVar

import pytorch_lightning as pl
import torch
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.exp_manager import StatelessTimer, exp_manager
from nemo.utils.model_utils import import_class_by_path
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import (
    CheckpointConnector,
)

from bionemo.callbacks.utils import add_test_callbacks


M = TypeVar('M')
"""Supposed to be `bionemo.model.core.infer.M`, but cannot import that here due to circular imports.
"""

try:
    import apex
    from apex.transformer.pipeline_parallel.utils import (
        _reconfigure_microbatch_calculator,
    )
    from megatron.core import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


def set_cfg_key(key, value, cfg_section, msg=""):
    """Adds a value to a config if it is missing

    Args:
        key (str): Key in `cfg_section` to be set
        value (Any): value to set in config
        cfg_section (OmegaConf): Config object to set the key/value for

    Raises:
        ValueError: If `key` already exists in `cfg_section` and is different
            than `value`.
    """
    if key not in cfg_section or cfg_section.get(key) is None:
        with open_dict(cfg_section):
            cfg_section[key] = value
    elif cfg_section[key] != value:
        raise ValueError(f"{key}={cfg_section[key]}, which conflicts with target value: {value}.{msg}")


def infer_global_batch_size(
    micro_batch_size,
    n_devices=1,
    n_nodes=1,
    accumulate_grad_batches=1,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
):
    n_devices = get_num_devices(n_devices)
    world_size = n_devices * n_nodes
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size

    if (model_parallel_size > world_size) or world_size % model_parallel_size != 0:
        raise ValueError(
            f'Model parallel size: {model_parallel_size} '
            f'(tensor_model_paralle_size={tensor_model_parallel_size} x '
            f'pipeline_model_parallel_size={pipeline_model_parallel_size}) '
            f'must be <= and a divisor of world size: {world_size} ('
            f'n_devices={n_devices} x n_nodes={n_nodes}).'
        )

    data_parallel_size = world_size // model_parallel_size

    global_batch_size = micro_batch_size * data_parallel_size * accumulate_grad_batches

    return global_batch_size


# Use this fucntion to retreive number of devices
# Handles cases where n_devices is not integer (example: in multirun mode)
def get_num_devices(n_devices):
    if not isinstance(n_devices, int):
        n_devices = len(n_devices)

    return n_devices


class TrainerBuilder:
    @staticmethod
    def adjust_config(cfg):
        micro_batch_size = cfg.model.micro_batch_size
        tensor_model_parallel_size = cfg.model.tensor_model_parallel_size
        pipeline_model_parallel_size = cfg.model.pipeline_model_parallel_size
        n_devices = cfg.trainer.devices
        n_nodes = cfg.trainer.num_nodes
        acc_grad_batches = cfg.trainer.get("accumulate_grad_batches", 1)

        global_batch_size = infer_global_batch_size(
            micro_batch_size=micro_batch_size,
            n_devices=n_devices,
            n_nodes=n_nodes,
            accumulate_grad_batches=acc_grad_batches,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
        )

        # we always inferr global batch size based on micro batch size
        if cfg.get('do_training', False):
            # when training we validate
            set_cfg_key(
                'global_batch_size',
                global_batch_size,
                cfg.model,
                msg=' Please set global_batch_size in the config file to be null or to match global_batch_size = micro_batch_size * data_parallel_size * accumulate_grad_batches',
            )
        else:
            # otherwise we override at inference
            with open_dict(cfg):
                cfg.model['global_batch_size'] = global_batch_size

        # accumulate_grad_batches must always be 1 for NeMo Megatron
        with open_dict(cfg):
            cfg.trainer['accumulate_grad_batches'] = 1

    @staticmethod
    def configure_plugins(cfg):
        plugins = []
        if cfg.trainer.precision in [16, "bf16"]:
            scaler = None
            if cfg.trainer.precision == 16:
                scaler = GradScaler(
                    init_scale=cfg.model.get("native_amp_init_scale", 2**32),
                    growth_interval=cfg.model.get("native_amp_growth_interval", 1000),
                    hysteresis=cfg.model.get('hysteresis', 2),
                )
            if cfg.model.get('megatron_amp_O2', False):
                plugins.append(
                    MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler)
                )
            else:
                plugins.append(
                    PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler)
                )

        if cfg.get("cluster_type", None) == "BCP":
            plugins.append(TorchElasticEnvironment())

        return plugins

    @staticmethod
    def configure_callbacks(cfg, extra_callbacks=[]):
        callbacks = [ModelSummary(max_depth=3)]
        callbacks.extend(extra_callbacks)
        logging.info(f'Selected Callbacks: {[type(c) for c in callbacks]}')
        return callbacks

    @staticmethod
    def configure_strategy(cfg):
        # DDP communication hooks cause errors when used with megatron pipeline parallel
        return NLPDDPStrategy(
            no_ddp_communication_hook=True,
            # Allocate gradients in a contiguous bucket to save memory (less fragmentation and buffer memory)
            gradient_as_bucket_view=cfg.model.get('gradient_as_bucket_view', True),
            find_unused_parameters=cfg.model.get('find_unused_parameters', False),
        )

    @staticmethod
    def resume_checkpoint(cfg, trainer):
        # update resume from checkpoint found by exp_manager
        if cfg.model.resume_from_checkpoint is not None:
            resume_from_checkpoint = cfg.model.resume_from_checkpoint
        else:
            resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path
        logging.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")

        trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
        # Override timer callback to a stateless one
        for idx, callback in enumerate(trainer.callbacks):
            if isinstance(callback, Timer):
                trainer.callbacks[idx] = StatelessTimer(
                    cfg.trainer.max_time,
                )

        # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
        with open_dict(cfg):
            cfg.model.precision = cfg.trainer.precision


class InferenceTrainerBuilder(TrainerBuilder):
    @staticmethod
    def configure_callbacks(cfg, extra_callbacks=[]):
        return []

    @staticmethod
    def resume_checkpoint(cfg, trainer):
        pass


def setup_trainer(cfg, builder=None, callbacks=[], adjust_config=True, verbose=True, interactive: bool = False):
    """NeMo Trainer setup functions"""
    if builder is None:
        builder = TrainerBuilder

    if adjust_config:
        builder.adjust_config(cfg)
    plugins = builder.configure_plugins(cfg)
    mode = "train" if cfg.get("do_training", False) else "test"
    callbacks = builder.configure_callbacks(cfg, callbacks)
    add_test_callbacks(cfg, callbacks=callbacks, mode="infer" if builder == InferenceTrainerBuilder else mode)

    if interactive:
        strategy = 'auto'
        # strategy = 'ddp_notebook'
        print(f"Interactive mode selected, using {strategy=}")
    else:
        strategy = builder.configure_strategy(cfg)

    trainer = Trainer(plugins=plugins, strategy=strategy, callbacks=callbacks, **cfg.trainer)

    exp_manager(trainer, cfg.get("exp_manager", None))

    builder.resume_checkpoint(cfg, trainer)

    # log trainer configuration (which might be different from input cfg)
    if verbose:
        logging.info("\n\n************** Trainer configuration ***********")
        logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    return trainer


def setup_inference_trainer(cfg, builder=None, adjust_config=True, interactive: bool = False):
    """NeMo Trainer setup functions for inference"""
    if builder is None:
        builder = InferenceTrainerBuilder

    return setup_trainer(cfg, builder, adjust_config=adjust_config, interactive=interactive)


def restore_model(
    restore_path: str,
    trainer: Optional[pl.Trainer] = None,
    cfg: Optional[DictConfig] = None,
    model_cls: Optional[Type[M]] = None,
    adjust_config: bool = True,
    strict: bool = True,
    interactive: bool = False,
) -> M:
    """Restore model from checkpoint.

    The supplied `model_cls`
    """
    logging.info(f"Restoring model from {restore_path}")

    # infer model_cls from cfg if missing
    if model_cls is None:
        logging.info(f"Loading model class: {cfg.target}")
        model_cls = import_class_by_path(cfg.target)

    # merge the config from restore_path with the provided config
    restore_cfg = model_cls.restore_from(
        restore_path=restore_path,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        return_config=True,
    )
    with open_dict(cfg):
        cfg.model = OmegaConf.merge(restore_cfg, cfg.model) if hasattr(cfg, 'model') else restore_cfg

    # build trainer if not provided
    if trainer is None:
        trainer = setup_inference_trainer(cfg=cfg, adjust_config=adjust_config, interactive=interactive)

    # enforce trainer precition
    with open_dict(cfg):
        if "EquiDock" in cfg.model.get("name", ""):
            # overwrite model cfg from checkpoint (to get the input_edge_feats_dim)
            cfg.model = OmegaConf.merge(cfg.model, restore_cfg)
        cfg.model.precision = trainer.precision

    model = model_cls.restore_from(
        restore_path=restore_path,
        trainer=trainer,
        override_config_path=cfg,
        save_restore_connector=NLPSaveRestoreConnector(),
        strict=strict,
    )
    if cfg.get('load_from_checkpoint') is not None:
        model.load_from_checkpoint(checkpoint_path=cfg.load_from_checkpoint, strict=strict)

    return model


def load_model_for_inference(cfg: DictConfig, strict: bool = True):
    """load model with config for model inference. Freeze model and reconfigure DDP and batch size.

    Args:
        cfg (DictConfig): Omega config
        strict (bool, optional): Whether to ignore non-matching keys when loading model, ignore if set to False. Defaults to True.

    Returns:
        Loaded model
    """

    # load model class from config which is required to load the .nemo file
    model = restore_model(
        restore_path=cfg.restore_from_path,
        cfg=cfg,
        strict=strict,
    )

    # check whether the DDP is initialized
    if parallel_state.is_unitialized():
        logging.info("DDP is not initialized. Initializing...")

        def dummy():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()
    # Reconfigure microbatch sizes here because on model restore, this will contain the micro/global batch configuration used while training.
    _reconfigure_microbatch_calculator(
        rank=0,  # This doesn't matter since it is only used for logging
        rampup_batch_size=None,
        global_batch_size=1,
        micro_batch_size=1,  # Make sure that there is no "grad acc" while decoding.
        data_parallel_size=1,  # We check above to make sure that dataparallel size is always 1 at inference.
    )
    model.freeze()
    return model


# TODO this is taken and from NeMo and we should try to make sure we get this
# back upstream into NeMo
def extract_consumed_samples_from_ckpt(ckpt_path):
    try:
        init_consumed_samples = int(float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", ckpt_path)[0]))
    except (ValueError, TypeError, IndexError):
        logging.warning("Cannot parse the checkpoint file to get the consumed samples. assume it is zero.")
        init_consumed_samples = 0

    return init_consumed_samples


# TODO this is taken and from NeMo and we should try to make sure we get this
# back upstream into NeMo
def compute_consumed_samples(model, steps_since_resume=0):
    app_state = AppState()
    consumed_samples = (
        model.init_consumed_samples
        + steps_since_resume
        * app_state.data_parallel_size
        * model.cfg.micro_batch_size
        * model.trainer.accumulate_grad_batches
    )
    return int(consumed_samples)


def _reconfigure_inference_batch(global_batch_per_gpu, global_batch_size=None):
    """Reconfigure microbatch sizes for inference."""

    # This should happen only on the last batch of the validation/test dataset with drop_last=False.
    # apex.transformer.pipeline_parallel.utils.get_current_global_batch_size()
    cur_global_batch = apex.transformer.pipeline_parallel.utils.get_current_global_batch_size()
    cur_data_parallel_world_size = parallel_state.get_data_parallel_world_size()
    if global_batch_size is None:
        global_batch_size = global_batch_per_gpu * parallel_state.get_data_parallel_world_size()
    if global_batch_per_gpu != (cur_global_batch // cur_data_parallel_world_size):
        _reconfigure_microbatch_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=global_batch_size,
            micro_batch_size=global_batch_per_gpu,
            data_parallel_size=cur_data_parallel_world_size,
        )


def _dummy() -> None:
    return


def initialize_model_parallel(model, interactive: bool = False) -> None:
    # check whether the DDP is initialized
    if not interactive and parallel_state.is_unitialized():
        logging.info("DDP is not initialized. Initializing...")
        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(_dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()


def initialize_distributed_parallel_state(
    local_rank: int = 0,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_model_parallel_split_rank: int = 0,
    interactive: bool = False,
) -> None:
    # initialize pytorch DDP
    # if not interactive and not torch.distributed.is_initialized():
    if not torch.distributed.is_initialized():
        logging.info("pytorch DDP is not initialized. Initializing with pytorch-lightening...")
        trainer = pl.Trainer(gpus=1, strategy='ddp' if not interactive else "auto", num_nodes=1)

        if trainer.strategy.launcher is not None:
            trainer.strategy.launcher.launch(_dummy, trainer=trainer)
        trainer.strategy.setup_environment()

    if not interactive and parallel_state.is_unitialized():
        logging.info("Megatron DDP is not initialized. Initializing...")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        )
