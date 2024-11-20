import pytest
from bionemo.noodles.nvfaidx import NvFaidx
import os
import tempfile
import random
import pyfaidx
import torch


def test_getitem_bounds():
    # NOTE make this the correct path, check this file in since we are checking exactness of queries.
    index = NvFaidx('sub-packages/bionemo-noodles/tests/bionemo/noodles/data/sample.fasta')
    # first element
    assert index['chr1'][0] == 'A'
    # Slice up to the last element
    assert index['chr1'][0:-1] == 'ACTGACTGACT'
    # equivalent to above
    assert index['chr1'][:-1] == 'ACTGACTGACT'
    # -1 should get the last element
    assert index['chr1'][-1:] == 'G'
    # normal, in range, query
    assert index['chr1'][1:4] == 'CTG'
    # Going beyond the max bound in a slice should truncate at the end of the sequence
    assert index['chr1'][1:10000] == 'CTGACTGACTG'

def test_pyfaidx_nvfaidx_equivalence():
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)
    pyfaidx_fasta = pyfaidx.Fasta(fasta)
    nvfaidx_fasta = NvFaidx(fasta)


    for i in range(100):
        # Deterministically generate regions to generate
        seqid = f"contig{i % 2 + 1}"
        start = i * 1000
        end = start + 1000

        assert pyfaidx_fasta[seqid][start:end] == nvfaidx_fasta[seqid][start:end]

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, fasta_path, fasta_cls):
        self.fasta = fasta_cls(fasta_path)
        self.keys = list(self.fasta.keys())

    def __len__(self):
        # Gigantic, we dont care.
        return 99999999999

    def __getitem__(self, idx):
        # Always return the same thing to keep it easy, we assume the fasta_created is doing the right thing.
        return str(self.fasta['contig1'][150000:160000])


@pytest.mark.xfail(reason="This is a known failure mode for pyfaidx that we are trying to prevent with nvfaidx.")
def test_parallel_index_creation_pyfaidx():
    ''' 
    PyFaidx is a python replacement for faidx that provides a dictionary-like interface to reference genomes. Pyfaidx 
    is not process safe, and therefore does not play nice with pytorch dataloaders.

    Ref: https://github.com/mdshw5/pyfaidx/issues/211

    Naively, this problem can be fixed by keeping index objects private to each process. However, instantiating this object can be quite slow. 
        In the case of hg38, this can take between 15-30 seconds.

    For a good solution we need three things:
        1) Safe index creation, in multi-process or multi-node scenarios, this should be restricted to a single node where all workers block until it is complete (not implemented above)
        2) Index object instantion must be fast.
        3) Read-only use of the index object must be both thread safe and process safe with python.
    '''
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)
    dl = torch.utils.data.DataLoader(TestDataset(fasta, fasta_cls = pyfaidx.Fasta), batch_size=16, num_workers=16)
    max_i = 1000
    for i, batch in enumerate(dl):
        # assert length of all elements in batch is 10000
        if i > max_i: break
        lens = [len(x) for x in batch]
        lens_equal = [x == 10000 for x in lens]
        assert all(lens_equal), (set(lens), sum(lens_equal))

def test_parallel_index_creation_nvfaidx():
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)

    # NOTE: worker_init_fn could also be a way to handle this, where we let it instantiate its own reader.

    dl = torch.utils.data.DataLoader(TestDataset(fasta, fasta_cls = NvFaidx), batch_size=32, num_workers=16)
    max_i = 1000
    # NOTE this shouldnt be failing uh oh
    for i, batch in enumerate(dl):
        if i > max_i: break
        lens = [len(x) for x in batch]
        lens_equal = [x == 10000 for x in lens]
        assert all(lens_equal), (set(lens), sum(lens_equal))

def demo_failure_mode():
    ''' 
    PyFaidx is a python replacement for faidx that provides a dictionary-like interface to reference genomes. Pyfaidx 
    is not process safe, and therefore does not play nice with pytorch dataloaders.

    Ref: https://github.com/mdshw5/pyfaidx/issues/211

    Naively, this problem can be fixed by keeping index objects private to each process. However, instantiating this object can be quite slow. 
        In the case of hg38, this can take between 20-30 seconds.

    For a good solution we need three things:
        1) Safe index creation, in multi-process or multi-node scenarios, this should be restricted to a single node where all workers block until it is complete (not implemented above)
        2) Index object instantion must be fast.
        3) Read-only use of the index object must be both thread safe and process safe with python.
    '''
    fasta = create_test_fasta(num_seqs=2, seq_length=200000)
    dl = torch.utils.data.DataLoader(TestDataset(fasta, fasta_cls = pyfaidx.Fasta), batch_size=16, num_workers=16)
    max_i = 1000
    passed=True
    failure_set = set()
    for i, batch in enumerate(dl):
        # assert length of all elements in batch is 10000
        if i > max_i: break
        lens = [len(x) for x in batch]
        lens_equal = [x == 10000 for x in lens]
        if not all(lens_equal):
            passed = False
            failure_set = set(lens)
            break
    print(f"pyfaidx {passed=}, {failure_set=}")

    passed=True
    failure_set = set()
    dl = torch.utils.data.DataLoader(TestDataset(fasta, fasta_cls = NvFaidx), batch_size=16, num_workers=16)
    for i, batch in enumerate(dl):
        # assert length of all elements in batch is 10000
        if i > max_i: break
        lens = [len(x) for x in batch]
        lens_equal = [x == 10000 for x in lens]
    print(f"nvfaidx {passed=}, {failure_set=}")


## Benchmarks
def measure_index_creation_time():
    '''Observed performance.

    13.8x speedup for NvFaidx when using --release.
    '''
    import time
    # Too slow gen a big genome
    fasta = create_test_fasta(num_seqs=1, seq_length=200_000)
    # Remove the .fai file to prevent cheating.
    if os.path.exists(fasta + ".fai"):
        os.remove(fasta + ".fai")
    start = time.time()
    _ = pyfaidx.Fasta(fasta)
    end = time.time()
    elapsed_pyfaidx = end - start

    # Remove the .fai file to prevent cheating.
    if os.path.exists(fasta + ".fai"):
        os.remove(fasta + ".fai")
    start = time.time()
    _ = NvFaidx(fasta)
    end = time.time()
    elapsed_nvfaidx = end - start

    print(f"pyfaidx: {elapsed_pyfaidx=}")
    print(f"nvfaidx: {elapsed_nvfaidx=}")
    print(f"nvfaidx faster by: {elapsed_pyfaidx/elapsed_nvfaidx=}")

def measure_query_time():
    '''Observed perf:

    1.5x faster nvfaidx with --release when doing queries directly.
    1.3x faster nvfaidx with cargo --release when doing queries through our SequenceAccessor implementation in python land.

    NOTE: perf is slower for NvFaidx when not built with --release
    '''
    import time
    num_iters = 1000
    fasta = create_test_fasta(num_seqs=10, seq_length=200000)

    # So we are a little slower
    fasta_idx = NvFaidx(fasta)
    start = time.time()
    for i in range(num_iters):
        query_res = fasta_idx['contig1'][150000:160000]
    end= time.time()
    elapsed_nvfaidx = end - start


    fasta_idx = pyfaidx.Fasta(fasta)
    start = time.time()
    for i in range(num_iters):
        query_res = fasta_idx['contig1'][150000:160000]
    end= time.time()
    elapsed_pyfaidx = end - start

    print(f"pyfaidx query/s: {elapsed_pyfaidx/num_iters=}")
    print(f"nvfaidx query/s: {elapsed_nvfaidx/num_iters=}")
    print(f"nvfaidx faster by: {elapsed_pyfaidx/elapsed_nvfaidx=}")

# Utility function
def create_test_fasta(num_seqs=2, seq_length=1000):
    """
    Creates a FASTA file with random sequences.
    
    Args:
        num_seqs (int): Number of sequences to include in the FASTA file.
        seq_length (int): Length of each sequence.
    
    Returns:
        str: File path to the generated FASTA file.
    """
    temp_dir = tempfile.mkdtemp()
    fasta_path = os.path.join(temp_dir, "test.fasta")
    
    with open(fasta_path, "w") as fasta_file:
        for i in range(1, num_seqs + 1):
            # Write the header
            fasta_file.write(f">contig{i}\n")
            
            # Generate a random sequence of the specified length
            sequence = ''.join(random.choices("ACGT", k=seq_length))
            
            # Split the sequence into lines of 60 characters for FASTA formatting
            for j in range(0, len(sequence), 80):
                fasta_file.write(sequence[j:j+80] + "\n")
    
    return fasta_path

test_parallel_index_creation_nvfaidx()