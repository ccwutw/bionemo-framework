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

from omegaconf.omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging

from bionemo.data import UniRef50Preprocess
from bionemo.model.protein.esm1nv import ESM1nvModel
from bionemo.model.utils import setup_trainer

from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from bionemo.utils.callbacks.callback_utils import setup_callbacks

@hydra_runner(config_path="../../../examples/protein/esm1nv/conf", config_name="pretrain_small")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    callbacks = setup_callbacks(cfg)

    trainer = setup_trainer(cfg, callbacks=callbacks)

    if cfg.do_training:
        logging.info("************** Starting Training ***********")
        if cfg.restore_from_path:
            logging.info("\nRestoring model from .nemo file " + cfg.restore_from_path)
            model = ESM1nvModel.restore_from(
                cfg.restore_from_path, cfg.model, trainer=trainer,
                save_restore_connector=NLPSaveRestoreConnector()
            )
        else:
            model = ESM1nvModel(cfg.model, trainer)
        trainer.fit(model)
        logging.info("************** Finished Training ***********")
    else:
        logging.info("************** Starting Preprocessing ***********")
        preprocessor = UniRef50Preprocess()
        preprocessor.prepare_dataset(url=cfg.model.data.data_url,
                                 output_dir=cfg.model.data.dataset_path)



if __name__ == '__main__':
    main()