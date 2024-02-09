# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List, Type

import hydra
import numpy as np
import pytest
from guided_molecule_gen.optimizer import MoleculeGenerationOptimizer
from guided_molecule_gen.oracles import molmim_qed_with_similarity, qed

from bionemo.model.core.controlled_generation import ControlledGenerationPerceiverEncoderInferenceWrapper
from bionemo.model.core.infer import BaseEncoderDecoderInference
from bionemo.model.molecule.megamolbart import MegaMolBARTInference


def scoring_function(smis: List[str], reference: str, **kwargs) -> np.ndarray:
    scores = molmim_qed_with_similarity(smis, reference)
    return -1 * scores


# Inside of examples/tests/conf
INFERENCE_CONFIGS: List[str] = [
    "megamolbart_infer.yaml",
]

MODEL_CLASSES: List[Type[BaseEncoderDecoderInference]] = [
    MegaMolBARTInference,
]

# This test follows the example in
# https://gitlab-master.nvidia.com/bionemo/service/controlled-generation/-/blob/6dad5965469275e263fe0d0e3b2485a341629e11/guided_molecule_gen/optimizer_integration_test.py#L17
example_smis = [
    "C[C@@H](C(=O)C1=c2ccccc2=[NH+]C1)[NH+]1CCC[C@@H]1[C@@H]1CC=CS1",
    "CCN(C[C@@H]1CCOC1)C(=O)c1ccnc(Cl)c1",
    "CSCC(=O)NNC(=O)c1c(O)cc(Cl)cc1Cl",
]


@pytest.mark.slow
@pytest.mark.needs_gpu
@pytest.mark.parametrize("model_infer_config_path,model_cls", list(zip(INFERENCE_CONFIGS, MODEL_CLASSES)))
def test_property_guided_optimization_of_inference_model(
    model_infer_config_path: str, model_cls: Type[BaseEncoderDecoderInference], pop_size: int = 2
):
    # config_path relative to this script, so examples/test
    with hydra.initialize(config_path="conf"):
        cfg = hydra.compose(config_name=model_infer_config_path)
    inf_model = model_cls(cfg=cfg)
    controlled_gen_kwargs = {
        "additional_decode_kwargs": {"override_generate_num_tokens": 128},  # speed up sampling for this test
        "sampling_method": "beam-search",
        "sampling_kwarg_overrides": {
            "beam_size": 3,
            "keep_only_best_tokens": True,  # only return the best sequence from beam search. MoleculeGenerationOptimizer needs only one sample.
            "return_scores": False,  # Do not return extra things that will confuse the MoleculeGenerationOptimizer
        },
    }
    if model_cls == MegaMolBARTInference:
        token_ids, _ = inf_model.tokenize(example_smis)  # get the padded sequence length for this batch.
        model = ControlledGenerationPerceiverEncoderInferenceWrapper(
            inf_model, enforce_perceiver=False, hidden_steps=token_ids.shape[1], **controlled_gen_kwargs
        )  # just flatten the position for this.
    else:
        model = ControlledGenerationPerceiverEncoderInferenceWrapper(
            inf_model, **controlled_gen_kwargs
        )  # everything is inferred from the perciever config
    optimizer = MoleculeGenerationOptimizer(
        model,
        scoring_function,
        example_smis,
        popsize=pop_size,
        optimizer_args={"sigma": 1.0},
    )
    starting_qeds = qed(example_smis)
    optimizer.step()  # one step of optimization
    opt_generated_smiles = optimizer.generated_smis
    assert len(opt_generated_smiles) == len(example_smis)
    assert all(len(pops) == pop_size for pops in opt_generated_smiles)
    opt_qeds = np.array([qed(molecule_smis) for molecule_smis in opt_generated_smiles])
    # For all starting molecules, test that the QED improved (max over the population axis) after optimization
    assert np.all(np.max(opt_qeds, axis=1) >= starting_qeds)