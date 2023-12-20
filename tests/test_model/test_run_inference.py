# Copyright (c) 2023, NVIDIA CORPORATION.
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

import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import pytest

from bionemo.model import run_inference
from bionemo.triton.utils import load_model_config


_TEST_DATA: List[Tuple[str, int]] = [
    ('prott5nv_infer.yaml', 768),
    ('esm1nv_infer.yaml', 768),
    ('megamolbart_infer.yaml', 512),
]
"""(config filename, hidden embedding dimensionality)"""


@pytest.mark.needs_gpu
@pytest.mark.needs_checkpoint
@pytest.mark.parametrize('config_name, hidden_size', _TEST_DATA)
def test_model_inference(bionemo_home: Path, config_name: str, hidden_size: int):
    """Verify that inference with each model produces the correct size for hidden weights."""

    config_path = bionemo_home / "examples" / "tests" / "conf"

    cfg = load_model_config(config_path, config_name)

    with NamedTemporaryFile() as tempfi:
        run_inference.main(config_path, config_name, output_override=tempfi.name, overwrite=True)

        with open(tempfi.name, 'rb') as rb:
            embeddings = pickle.load(rb)

    dict_keys = cfg.model.downstream_task.outputs + ["sequence", "id"]
    for emb in embeddings:
        for key in dict_keys:
            if key not in emb.keys():
                assert False, f'Missing key {key} in embeddings file {cfg.model.data.output_fname}'
        if "hiddens" in dict_keys:
            assert emb["hiddens"].shape == (len(emb["sequence"]), hidden_size)
        if "embeddings" in dict_keys:
            assert emb["embeddings"].shape == (hidden_size,)