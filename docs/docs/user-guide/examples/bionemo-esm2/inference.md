# ESM-2 Inference

This tutorial serves as a demo for [ESM2](https://www.science.org/doi/abs/10.1126/science.ade2574) Inference using a CSV file with `sequences` column. To pre-train the ESM2 model please refer to [ESM-2 Pretraining](./pretrain.md) tutorial.

# Setup and Assumptions

In this tutorial, we will demonstrate how to download ESM2 checkpoint, create a CSV file with protein sequences, and infer a ESM-2 model.

All commands should be executed inside the BioNeMo docker container, which has all ESM-2 dependencies pre-installed. This tutorial assumes that a copy of the BioNeMo framework repo exists on workstation or server and has been mounted inside the container at `/workspace/bionemo2`. For more information on how to build or pull the BioNeMo2 container, refer to the [Initialization Guide](../../getting-started/initialization-guide.md).

!!! note

    This `WORKDIR` may be `/workspaces/bionemo-framework` if you are using the VSCode Dev Container.

Similar to PyTorch Lightning, we have to define some key classes:

1. `MegatronStrategy` - To launch and setup parallelism for [NeMo](https://github.com/NVIDIA/NeMo/tree/main) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
2. `Trainer` - To configure training configurations and logging.
3. `ESMFineTuneDataModule` - To load sequence data for both fine-tuning and inference.
4. `ESM2Config` - To configure the ESM-2 model as `BionemoLightningModule`.

Please refer to [ESM-2 Pretraining](./pretrain.md) and [ESM-2 Fine-Tuning](./finetune.md) tutorials for detailed description of these classes.

# Create a CSV data file containing your protein sequences

We use the `InMemoryCSVDataset` class to load the protein sequence data from a `.csv` file. This data file should at least have a `sequences` column and can optionally have a `labels` column used for fine-tuning applications. Here is an example of how to create your own inference input data using a list of sequences in python:

```python
import pandas as pd

artificial_sequence_data = [
    "TLILGWSDKLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
    "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
    "GRFNVWLGGNESKIRQVLKAVKEIGVSPTLFAVYEKN",
    "DELTALGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
    "KLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI",
    "LFGAIGNAISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
    "LGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
    "LYSGDHSTQGARFLRDLAENTGRAEYELLSLF",
    "ISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP",
    "SGSKASSDSQDANQCCTSCEDNAPATSYCVECSEPLCETCVEAHQRVKYTKDHTVRSTGPAKT",
]

csv_file = "/home/bionemo/sequences.csv"
# Create a DataFrame
df = pd.DataFrame(dummy_protein_sequences, columns=["sequences"])
# Save the DataFrame to a CSV file
df.to_csv(csv_file, index=False)
```

For the purpose of this tutorial, we have already provided an example `.csv` file as a downloadable resource in Bionemo Framework:

```bash
download_bionemo_data esm2/testdata_esm2_inference:2.0 --source ngc
```

To run inference on this data using an ESM2 checkpoint you can use the `infer_esm2` executable which calls `$WORKDIR/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/infer_esm2.py`:

```bash
DATA_PATH=$(download_bionemo_data esm2/testdata_esm2_inference:2.0 --source ngc)
CHECKPOINT_PATH=$(download_bionemo_data esm2/650m:2.0 --source ngc)

infer_esm2
    --data-path ${DATA_PATH} \
    --checkpoint-path ${CHECKPOINT_PATH} \
    --results-path ${RESULTS_MOUNT}/esm2_inference_tutorial.pt \
    --micro-batch-size 2 \
    --include-hiddens \
    --include-embeddings \
    --include-logits
```

This script will create the `esm2_inference_tutorial.pt` file under the results mount of your container to stores the results. The `.pt` file containes a dictionary of `{'result_key': torch.Tensor}` that be loaded with PyTorch:

```python
import torch
data = torch.load(f'${RESULTS_MOUNT}/esm2_inference_tutorial.pt')
```
In this example `data` a python dict with the following keys `['token_logits', 'binary_logits', 'hidden_states', 'embeddings']`
