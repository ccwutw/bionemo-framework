# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
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


from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.timer import Timer
from lightning_lite.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from bionemo.model.protein.prott5nv import ProtT5nvModel, T5SaveRestoreConnector
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager
from bionemo.data import UniRef50Preprocess
from bionemo.utils.callbacks.callback_utils import setup_callbacks


@hydra_runner(config_path="../../../examples/protein/prott5nv/conf", config_name="pretrain_small")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    # TODO: move this to model utils
    callbacks = [ModelSummary(max_depth=3)]
    callbacks.extend(setup_callbacks(cfg))
    logging.info(f'Selected Callbacks: {[type(c) for c in callbacks]}')

    # TODO: Should use setup_inference from model uitls
    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer, callbacks=callbacks)
    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path
    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        if cfg.restore_from_path is not None:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            model = ProtT5nvModel.restore_from(
                cfg.restore_from_path, cfg.model, trainer=trainer,
                save_restore_connector=T5SaveRestoreConnector()
            )
        else:
            model = ProtT5nvModel(cfg.model, trainer=trainer)

        trainer.fit(model)
        logging.info("************** Finish Training ***********")
    else:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = UniRef50Preprocess()
        preprocessor.prepare_dataset(url=cfg.model.data.data_url,
                                 output_dir=cfg.model.data.dataset_path)


if __name__ == '__main__':
    main()
