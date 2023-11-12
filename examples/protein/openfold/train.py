# Copyright (c) 2023, NVIDIA CORPORATION.

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

import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from nemo.core.config import hydra_runner
from nemo.core.optim.lr_scheduler import register_scheduler
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

from bionemo.data import OpenProteinSetPreprocess, PDBMMCIFPreprocess
from bionemo.model.protein.openfold.checkpoint_utils import load_pt_checkpoint
from bionemo.model.protein.openfold.lr_scheduler import AlphaFoldLRScheduler
from bionemo.model.protein.openfold.openfold_model import AlphaFold
from bionemo.model.utils import setup_trainer


@hydra_runner(config_path="conf", config_name="openfold_initial_training")
def main(cfg) -> None:
    cfg = instantiate(cfg)
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing required keys in config:\n{missing_keys}")

    register_scheduler(name='AlphaFoldLRScheduler', scheduler=AlphaFoldLRScheduler, scheduler_params=None)

    np.random.seed(cfg.model.seed)
    pl.seed_everything(cfg.model.seed)

    if cfg.get('do_preprocess', False):
        ops_preprocessor = OpenProteinSetPreprocess(
            dataset_root_path=cfg.model.data.dataset_path, **cfg.model.data.prepare.open_protein_set
        )
        pdb_mmcif_preprocessor = PDBMMCIFPreprocess(
            dataset_root_path=cfg.model.data.dataset_path, **cfg.model.data.prepare.pdb_mmcif
        )

        ops_preprocessor.prepare(**cfg.model.data.prepare.open_protein_set_actions)
        pdb_mmcif_preprocessor.prepare(**cfg.model.data.prepare.pdb_mmcif_actions)

    if cfg.get('do_training', False) or cfg.get('do_validation', False):
        trainer = setup_trainer(cfg, callbacks=[])
        if cfg.get('restore_from_path', None):
            # TODO: consider blocking restore if stage is not 'fine-tune'
            alphafold = AlphaFold.restore_from(
                restore_path=cfg.restore_from_path, override_config_path=cfg, trainer=trainer
            )
            alphafold.setup_training_data(cfg.model.train_ds)
            alphafold.setup_validation_data(cfg.model.validation_ds)
        else:
            alphafold = AlphaFold(cfg=cfg.model, trainer=trainer)
            if cfg.get('torch_restore', None):
                load_pt_checkpoint(model=alphafold, checkpoint_path=cfg.torch_restore)

        if cfg.get('do_validation', False):
            trainer.validate(alphafold)
        if cfg.get('do_training', False):
            trainer.fit(alphafold)


if __name__ == '__main__':
    main()