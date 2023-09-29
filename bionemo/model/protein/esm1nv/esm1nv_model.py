
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

import os
import torch
from typing import Dict, Optional
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from bionemo.data.dataloader.protein_collate import ESM2BertCollate
from bionemo.data.mapped_dataset import NeMoUpsampling, Uniref90ClusterMappingDataset

from nemo.core.neural_types import NeuralType
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
)
from nemo.utils import logging
from bionemo.data.dataset_builder_utils import build_typed_dataset

from bionemo.data.molecule import megamolbart_build_train_valid_test_datasets
from bionemo.data.dataloader import ProteinBertCollate
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

try:
    from apex.transformer import tensor_parallel


    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ["ESM1nvModel"]

class ESM1nvModel(MegatronBertModel):
    """
    ESM1nv pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        """
        model_name = self._cfg.tokenizer.get('model_name')
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=model_name,
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.vocab_file),
            legacy=False,
        )
        # patch tokenizer for use with HF esm tokenizer
        if self._cfg.tokenizer.library == 'huggingface' and \
                str(model_name).startswith('facebook/esm2'):
            self.tokenizer.tokenizer.vocab = self.tokenizer.tokenizer.get_vocab()

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        assert self._cfg.data.dataloader_type == 'single', AssertionError(
            f'Only the Megatron sequential ("single") sampler is currently supported. {self._cfg.data.dataloader_type} was chosen.'
            )

        dataloader = super().build_pretraining_data_loader(dataset=dataset, consumed_samples=consumed_samples)

        # Add collate function and unpin memory to avoid crash with CUDA misaligned address
        dataloader.pin_memory = False # must be False with CSV dataset TODO check with binary
        pad_size_divisible_by_8 = True if self._cfg.masked_softmax_fusion else False

        dataloader.collate_fn = ProteinBertCollate(tokenizer=self.tokenizer,
                                                seq_length=self._cfg.seq_length,
                                                pad_size_divisible_by_8=pad_size_divisible_by_8,
                                                modify_percent=self._cfg.data.modify_percent,
                                                perturb_percent=self._cfg.data.perturb_percent,
                                                ).collate_fn
        return dataloader

    def setup_training_data(self, cfg):
        super().setup_training_data(cfg)

    def setup_validation_data(self, cfg):
        super().setup_validation_data(cfg)

    def setup_test_data(self, cfg):
        super().setup_test_data(cfg)

    @staticmethod
    def _build_train_valid_test_datasets(trainer, model_cfg):
        logging.info('Building Bert datasets.')
        global_batch_size = trainer.world_size * model_cfg.micro_batch_size / model_cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = trainer.max_steps * trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // trainer.val_check_interval + 1) * trainer.limit_val_batches
        test_iters = trainer.limit_test_batches

        train_valid_test_num_samples = {
            'train': int(max_train_steps * global_batch_size),
            'val': int(eval_iters * global_batch_size),
            'test': int(test_iters * global_batch_size),
            }

        _train_ds, _validation_ds, _test_ds = megamolbart_build_train_valid_test_datasets(
            cfg=model_cfg.data,
            train_valid_test_num_samples=train_valid_test_num_samples
        )

        logging.info(f'Length of train dataset: {len(_train_ds)}')
        logging.info(f'Length of val dataset: {len(_validation_ds)}')
        logging.info(f'Length of test dataset: {len(_test_ds)}')
        logging.info(f'Finished building Bert datasets.')
        return _train_ds, _validation_ds, _test_ds


    def build_train_valid_test_datasets(self):
        train, val, test = self._build_train_valid_test_datasets(self.trainer, self._cfg)
        self._train_ds = train
        self._validation_ds = val
        self._test_ds = test
        return self._train_ds, self._validation_ds, self._test_ds
        
    def validation_epoch_end(self, outputs):
        if not outputs:
            return
        averaged_loss = torch.stack(outputs).mean()
        self.log('val_loss', averaged_loss, prog_bar=True)
        self.log('val_loss_ECE', pow(2, averaged_loss)) #calculate exponential cross entropy loss for logs
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step - self.init_global_step))

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')
        logging.info(f'test_loss_ECE: {pow(2, averaged_loss[0])}')

    @property
    def input_names(self):
        return ['input_ids', 'attention_mask', ]

    @property
    def output_names(self):
        return ['output']

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'input_ids': {
                0: 'batch',
                1: 'time'
                },
            'attention_mask': {
                0: 'batch',
                1: 'time'
                }
            }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'output': {
                0: 'batch',
                1: 'time',
                2: 'size'
            }
        }

class ESM2nvModel(ESM1nvModel):
    """
    ESM2nv pretraining. Same as ESM1nv but with custom dataset construction.

    This model expects the following scheme for model.data in addition to those used by esm1nv

    (no file suffix, assumes csv, expandable with ..)
    model:
        data:
            # Fields from esm1nv
            dataset_path: /data/uniref2022_05 # parent directory for data, contains train / val / test folders. Needs to be writeable for index creation.
            dataset: # inclusive range of data files to load x[000..049] or can a single file, e.g. x000
                train: x[000..049]
                test: x[000..049]
                val: x[000..049]

            # New fields
            uf90_dataset: x[000..049]
            uf90_data_impl_kwargs:
            csv_mmap:
                data_col: 3 # 0-based
    """
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
    

    @staticmethod 
    def _build_train_valid_test_datasets(trainer, model_cfg):
        _train_ds, _validation_ds, _test_ds = ESM1nvModel._build_train_valid_test_datasets(trainer, model_cfg)

        # TODO: ADDME to config preconditions
        dataset_path = model_cfg.data.uf90.uniref90_path
        # TODO: ADDME to config preconditions
        split = 'uf90_csvs' # NOTE do we want this hardcoded? this is where they end up when preprocessed.
        ds_name = model_cfg.data.uf90.dataset.get(split, None)
        filepath: str = os.path.join(dataset_path, split, ds_name)

        # TODO: ADDME to config preconditions
        data_impl = model_cfg.data.uf90.get('data_impl', None)
        assert data_impl is not None, 'Config "cfg" should contain field "cfg.data_impl"'
        # NOTE train/val/test will all each into uniref90, since we are split on clusters, we now they are independent.
        #   hence, we only need one dataset object for uf90
        uniref90_dataset = build_typed_dataset(dataset_paths=filepath,
                                                data_impl=data_impl,
                                                cfg=model_cfg.data.uf90, 
                                                use_upsampling=False, 
                                                num_samples=None)

        results = []
        for ds in [_train_ds, _validation_ds, _test_ds]:
            # TypeHint for intellisense
            ds: NeMoUpsampling = ds

            data_prefix= ds.data_prefix
            index_mapping_dir = ds.index_mapping_dir
            # TODO: how do we set private seeds?
            seed = 1337

            # Setup the resampling
            ds = Uniref90ClusterMappingDataset(
                uniref50_dataset=ds, 
                uniref90_dataset=uniref90_dataset, 
                data_prefix=data_prefix,  # used for index creation
                seed=seed, # used for rng, although awkward because global statehood
                index_mapping_dir=index_mapping_dir, # stores index
                cluster_map_json_path=model_cfg.data.cluster_map_json_path
            )

            results.append(ds)

        [_train_ds, _validation_ds, _test_ds] = results
        logging.info(f'Length of train dataset: {len(_train_ds)}')
        logging.info(f'Length of val dataset: {len(_validation_ds)}')
        logging.info(f'Length of test dataset: {len(_test_ds)}')
        logging.info(f'Finished building Bert datasets.')
        return _train_ds, _validation_ds, _test_ds


    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        assert self._cfg.data.dataloader_type == 'single', AssertionError(
            f'Only the Megatron sequential ("single") sampler is currently supported. {self._cfg.data.dataloader_type} was chosen.'
            )

        # NOTE: this doesnt actually call the right super. might be okay since we are duplicating the work of the super anyway
        dataloader = super().build_pretraining_data_loader(dataset=dataset, consumed_samples=consumed_samples)

        # Add collate function and unpin memory to avoid crash with CUDA misaligned address
        dataloader.pin_memory = False # must be False with CSV dataset TODO check with binary
        pad_size_divisible_by_8 = True if self._cfg.masked_softmax_fusion else False

        dataloader.collate_fn = ESM2BertCollate(tokenizer=self.tokenizer,
                                                seq_length=self._cfg.seq_length,
                                                pad_size_divisible_by_8=pad_size_divisible_by_8,
                                                modify_percent=self._cfg.data.modify_percent,
                                                perturb_percent=self._cfg.data.perturb_percent,
                                                ).collate_fn
        return dataloader