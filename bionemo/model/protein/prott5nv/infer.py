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


import torch

from typing import List

from bionemo.model.core.infer import BaseEncoderDecoderInference

class ProtT5nvInference(BaseEncoderDecoderInference):
    '''
    All inference functions
    '''

    def __init__(self, cfg):
        super().__init__(cfg)

    def _tokenize(self, sequences: List[str]):
        """
        ProtT5 expects input/output format:
        
        encoder input ids - [tokens] (without <BOS> and <EOS>)
        decoder input ids - <BOS> + [tokens]
        decoder output ids - [tokens] + <EOS>
        """
        # Tokenize sequences
        token_ids = [self.tokenizer.text_to_ids(s) for s in sequences]

        return token_ids

    def seq_to_hiddens(self, sequences):
        '''
        Transforms Sequences into hidden state.
        This class should be implemented in a child class, since it is model specific.
        This class should return only the hidden states, without the special tokens such as
         <BOS> and <EOS> tokens, for example.

        Args:
            sequences (list[str]): list of sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''
        token_ids, enc_mask = self.tokenize(sequences)
        embedding = self.model.encode(tokens_enc=token_ids, enc_mask=enc_mask)

        return embedding, enc_mask

# TODO: switch to a new inference interface
class ProtT5nvValidationInference():
    def __init__(self, model, cfg):

        self.model = model
        self.cfg = cfg
        self.tokenizer = self.model.tokenizer

    def _transform(self, sequences):
        '''
        Transforms Protein Sequences into hidden state.

        Args:
            sequences (list[str]): list of protein sequences

        Returns:
            hidden_states (torch.Tensor, float):
            enc_mask (torch.Tensor, long): boolean mask for padded sections
        '''
        token_ids, mask = self._tokenize(sequences)
        embedding = self.model.encode(tokens_enc=token_ids, enc_mask=mask)

        return embedding, mask

    def _tokenize(self, sequences: List[str]):
        # Validate input seqs
        valids = [len(s) > self.model.cfg.seq_length - 2 for s in sequences]
        if True in valids:
            raise Exception(f'One or more sequence exceeds max length({self.model.cfg.seq_length - 2}).')

        token_ids = [self.tokenizer.text_to_ids(s) for s in sequences]

        # +2 for the terminal tokens
        pad_length = max([len(seq) for seq in token_ids])
        mask = [([1] * (len(seq) + 2)) + ([0] * (pad_length - len(seq))) for seq in token_ids]

        token_ids = [torch.tensor([self.tokenizer.bos_id] + s + [self.tokenizer.eos_id]).cuda() for s in token_ids]
        token_ids = torch.nn.utils.rnn.pad_sequence(token_ids,
                                                    batch_first=True,
                                                    padding_value=0.0)

        mask = torch.tensor(mask).half().cuda()
        return token_ids, mask
