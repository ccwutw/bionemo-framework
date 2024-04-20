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
from unittest.mock import MagicMock

import numpy as np
import scanpy

from bionemo.data.singlecell.dataset import SingleCellDataset


def test_dataset_init_lookup():
    data_path = "examples/tests/test_data/singlecell"
    tokenizer = MagicMock()

    dataset = SingleCellDataset(data_path, tokenizer)
    first = dataset.lookup_cell_by_idx(0)  # I think we can open these up directly to test
    print(first[2].keys())
    scanpy.read_h5ad(os.path.join(data_path, first[2]['file_path']))

    last = dataset.lookup_cell_by_idx(len(dataset) - 1)
    scanpy.read_h5ad(os.path.join(data_path, last[2]['file_path']))

    random = dataset.lookup_cell_by_idx(150)
    scanpy.read_h5ad(os.path.join(data_path, random[2]['file_path']))

    # Data was generated by taking 3x100 cell slices
    assert len(dataset) == 300


def test_dataset_ccum():
    data_path = "examples/tests/test_data/singlecell"
    tokenizer = MagicMock()

    dataset = SingleCellDataset(data_path, tokenizer)
    # should sum to the total length
    assert len(dataset) == sum(dataset.dataset_ccum) + 1  # Because there is a -1 in the first element
    assert len(dataset.dataset_ccum) == 3  # Three datasets.

    # we expect all three of our test files to end up in dataset_map
    assert all(
        val in dataset.dataset_map.values()
        for val in ('test_data/test3.h5ad', 'test_data/test2.h5ad', 'test_data/test1.h5ad')
    )

    # Exhaustive search over did lookup, 100 elements for each, should map to the appropriate dataset
    assert all(dataset.metadata_lookup(i) == dataset.metadata_lookup(0) for i in range(100))
    assert all(dataset.metadata_lookup(i) == dataset.metadata_lookup(100) for i in range(100, 200))
    assert all(dataset.metadata_lookup(i) == dataset.metadata_lookup(200) for i in range(200, 300))

    # Tests that the boundary of each metadata matches the metadata filenames
    #   boundaries are defined by the order in dataset_map, we designed this test set to contain 100 elements per file.
    for i, filename in enumerate(dataset.dataset_map.values()):
        assert dataset.metadata_lookup(i * 100) == dataset.metadata[filename]


def test_dataset_process_item():
    tokenizer = MagicMock()

    tokenizer.pad_token = 4
    tokenizer.cls_token = 5
    # Not using ensembl ids, this is a very specific usecase.
    tokenizer.gene_tok_to_ens = lambda x: x
    tokenizer.vocab = {'GENE0': 1, 'GENE1': 2, 'GENE2': 3}

    def tok_to_id(tok):
        if tok == tokenizer.pad_token:
            return 4
        if tok == tokenizer.cls_token:
            return 5
        if tok == tokenizer.mask_token:
            return 6
        if tok == 'GENE0':
            return 1
        if tok == 'GENE1':
            return 2
        if tok == 'GENE2':
            return 3

    tokenizer.token_to_id = tok_to_id
    # Create a sample input item
    input_item = {
        "expression": np.array([1, 2, 3]),
        "indices": np.array([0, 1, 2]),
        "metadata": {"feature_names": [f'GENE{i}' for i in range(3)]},
    }

    # Process the input item
    from bionemo.data.singlecell.dataset import process_item

    # no padding, no masking, no medianrank norm. checks that CLS and correct tokenization is applied
    processed_item = process_item(
        input_item['expression'],
        input_item['indices'],
        input_item['metadata'],
        tokenizer,
        gene_median=None,
        max_len=4,
        mask_prob=0,
    )
    assert all(processed_item['text'] == [5, 1, 2, 3])  # CLS, 1, 2, 3

    ###### Check median rank norm, sorts in ascending order. ######

    # 1/6/1=1/6 , 2/3/6 =2/18=1/9, 3/6/6 =3/36=1/12 => 3, 2, 1
    processed_item = process_item(
        input_item['expression'],
        input_item['indices'],
        input_item['metadata'],
        tokenizer,
        gene_median={'GENE0': 1, 'GENE1': 3, 'GENE2': 6},
        max_len=4,
        mask_prob=0,
        target_sum=1,
    )
    assert all(processed_item['text'] == [5, 3, 2, 1])  # CLS, 1, 2, 3

    # Checks median norm, should change the order due to medians.
    # 1/6/.5=1/3, 2/6/1=2/6=1/3, 3/6/2=3/12=1/4 => 3,1,2
    processed_item = process_item(
        input_item['expression'],
        input_item['indices'],
        input_item['metadata'],
        tokenizer,
        gene_median={'GENE0': 0.5, 'GENE1': 1, 'GENE2': 2},
        max_len=4,
        mask_prob=0,
        target_sum=1,
    )
    assert all(processed_item['text'] == [5, 3, 1, 2])  # CLS, 1, 2, 3

    # checks padding is added for a short sequence
    processed_item = process_item(
        input_item['expression'],
        input_item['indices'],
        input_item['metadata'],
        tokenizer,
        gene_median=None,
        max_len=5,
        mask_prob=0,
        target_sum=1,
    )
    assert all(processed_item['text'] == [5, 1, 2, 3, 4])  # CLS, 1, 2, 3, PAD

    #    Masking - test that no special tokens are masked, all when 100, none when 0
    processed_item = process_item(
        input_item['expression'],
        input_item['indices'],
        input_item['metadata'],
        tokenizer,
        gene_median=None,
        max_len=5,
        mask_prob=1.0,
        target_sum=1,
    )
    # NOTE: we need to set masked tokens to MASK so that they are decoded.
    assert all(processed_item['text'] == [5, 6, 6, 6, 4])  # CLS, MASK, MASK, MASK, PAD
    # NOTE: MASKed tokens are the only ones used by loss
    assert all(processed_item['loss_mask'] == [0, 1, 1, 1, 0])  # NO, MASK, MASK, MASK, NO
    # the ARBITRARY labels should be ignored due to loss mask.
    assert all(processed_item['labels'] == [-1, 1, 2, 3, -1])  # ARBITRARY, 1, 2, 3, ARBITRARY
    assert all(processed_item['is_random'] == 0)  # For now we don't support random masking.

    # checks sequence is truncated for a long sequence
    processed_item = process_item(
        input_item['expression'],
        input_item['indices'],
        input_item['metadata'],
        tokenizer,
        gene_median=None,
        max_len=3,
        mask_prob=0,
        target_sum=1,
    )
    # Randomly permutes the other values, no fixed order
    assert all(processed_item['text'][0] == [5])
    # Truncate to exactly three items
    assert len(processed_item['text']) == 3
    assert all(processed_item['loss_mask'] == [False, False, False])  # No mask applied
