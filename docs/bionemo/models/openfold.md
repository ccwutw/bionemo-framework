# OpenFold 
# Model Overview

## User guide
We provide example training and inference scripts under `examples/protein/openfold` and their configuration files under the subdirectory `conf`. Users can download 4 different checkpoints: a pair of initial-training and fintuining from the public OpenFold repository and converted to .nemo format, and another pair of in-house trained checkpoints, with the following command.

```bash
python download_models.py openfold_finetuning_inhouse --download_dir models  # NGC setup is required
```

If users are interested in diving into BioNeMo in details, users can refer to `bionemo/model/protein/openfold` for the model class and `bionemo/data/protein/openfold` for data utilities. Configuration parsing is handled by [hydra config](https://hydra.cc/docs/intro/) and default parameters are stored in yaml format.

### Inference
Users can initiate inference with `examples/protein/openfold/infer.py` and provide the input sequence(s) and optional multiple sequence alignment(s) (MSA) for inference. Users can download example inference data through `scripts/download_sample_data.sh`, and follow a step-by-step guidance in jupyter notebook in `nbs/inference.ipynb` if desired.

For additional template-based inference, users would provide template hhr or generate templates on-the-fly after installing the necessary alignment software under `scripts/install_third_party.sh`. Users might also have to download pdb70 database. More details are available in `conf/infer.yaml`.

### Training
OpenFold training is a two-stage process, broken down into (1) initial training and (2) fine-tuning. Dataset download and preprocessing for both training stages is handled by the command

```bash
# Will take up to 2-3 days and ~2TB storage.
python examples/protein/openfold/train.py ++do_preprocess=true ++do_training=false
```
 To create smaller training data variant for accelerated testing, refer to option `sample` in `conf/openfold_initial_training.yaml`.

Users can initiate model training with `examples/protein/openfold/train.py`. To override initial training (default behavior) to fine-tuning, redirect the configuration with the following command
```bash
python examples/protein/openfold/train.py --config-name openfold_finetuning.yaml
```

## Description:
<!-- This document gives a brief summary of the BioNeMo framework's implementation of OpenFold. This implementation is derived from public OpenFold and DeepMind AlphaFold-2.  Below we summarize 
- user guide on inference and training
- dataset and preprocessing
- accuracy and runtime results -->

OpenFold predicts protein structures from protein sequence inputs and optional multiple sequence alignments (MSAs) and template(s). This implementation supports initial training, fine-tuning and inference under BioNeMo framework.

Users are advised to read the licensing terms under [public OpenFold](https://github.com/aqlaboratory/openfold) and [DeepMind AlphaFold-2](https://github.com/google-deepmind/alphafold) repositories as well as our copyright text.

This model is ready for commercial use. <br>

## References:
To cite OpenFold:
```bibtex
@article {Ahdritz2022.11.20.517210,
   author = {Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
   title = {{O}pen{F}old: {R}etraining {A}lpha{F}old2 yields new insights into its learning mechanisms and capacity for generalization},
   elocation-id = {2022.11.20.517210},
   year = {2022},
   doi = {10.1101/2022.11.20.517210},
   publisher = {Cold Spring Harbor Laboratory},
   URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
   eprint = {https://www.biorxiv.org/content/early/2022/11/22/2022.11.20.517210.full.pdf},
   journal = {bioRxiv}
}
```

To cite AlphaFold-2:
```bibtex
@Article{AlphaFold2021,
 author  = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and {\v{Z}}{\'\i}dek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A A and Ballard, Andrew J and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
 journal = {Nature},
 title   = {Highly accurate protein structure prediction with {AlphaFold}},
 year    = {2021},
 volume  = {596},
 number  = {7873},
 pages   = {583--589},
 doi     = {10.1038/s41586-021-03819-2}
}
```

If you use OpenProteinSet in initial training and fine-tuning, please also cite:
```bibtex
@misc{ahdritz2023openproteinset,
     title={{O}pen{P}rotein{S}et: {T}raining data for structural biology at scale},
     author={Gustaf Ahdritz and Nazim Bouatta and Sachin Kadyan and Lukas Jarosch and Daniel Berenberg and Ian Fisk and Andrew M. Watkins and Stephen Ra and Richard Bonneau and Mohammed AlQuraishi},
     year={2023},
     eprint={2308.05326},
     archivePrefix={arXiv},
     primaryClass={q-bio.BM}
}
```

## Model Architecture:
**Architecture Type:** Pose Estimation  <br>
**Network Architecture:** AlphaFold-2 <br>

## Input:
**Input Type(s):** Protein Sequence, (optional) Multiple Sequence Alignment(s) and (optional) Strutural Template(s) <br>
**Input Format(s):** None, a3m (text file), hhr (text file) <br>
**Input Parameters:** 1D <br>
**Other Properties Related to Input:** None <br>

## Output:
**Output Type(s):** Protrin Structure Pose(s), (optional) Confidence Metrics, (optional) Embeddings <br>
**Output Format:** PDB (text file), Pickle file, Pickle file <br>
**Output Parameters:** 3D <br>
**Other Properties Related to Output:** Pose (num_atm_ x 3), (optional) Confidence Metric: pLDDT (num_res_) and PAE (num_res_ x num_res_), (optional) Embeddings (num_res_ x emb_dims, or num_res_ x num_res_ x emb_dims) <br>

## Software Integration:
**Runtime Engine(s):**
* NeMo, BioNeMo <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* [Ampere] <br>
* [Hopper] <br>

**[Preferred/Supported] Operating System(s):** <br>
* [Linux]

## Model Version(s):
OpenFold under BioNeMo framework  <br>

# Training & Evaluation:

## Training Dataset:
**Link:**  [PDB-mmCIF dataset](https://www.rcsb.org), [OpenProteinSet](https://arxiv.org/abs/2308.05326)  <br>
**Data Collection Method by dataset** <br>
* PDB-mmCIF dataset: [Automatic] and [Human] <br>
* OpenProteinSet: [Automatic] <br>

**Labeling Method by dataset** <br>
* [Not Applicable] <br>

**Properties:** PDB-mmCIF dataset: 200k samples of experimental protein structures. OpenProteinSet: 269k samples on sequence alignments. <br>
**Dataset License(s):** PDB-mmCIF dataset: [CC0 1.0 Universal](https://www.rcsb.org/pages/usage-policy). OpenProteinSet: [CC BY 4.0](https://registry.opendata.aws/openfold/).

## Evaluation Dataset:
**Link:** [CAMEO](https://cameo3d.org/)   <br>
**Data Collection Method by dataset** <br>
* CAMEO dataset: [Automatic] and [Human] <br>

**Labeling Method by dataset** <br>
* [Not Applicable] <br>

**Properties:** Our during-training validation dataset is the CAMEO dataset on the date range 2021-09-17 to 2021-12-11.  There are ~200 sequence-structure pairs.  This validation dataset is automatically created by setting the cameo start and end dates in `examples/protein/openfold/conf/openfold_initial_training.yaml`.
 <br>
**Dataset License(s):** CAMEO dataset: [Terms and Conditions](https://cameo3d.org/cameong_terms/).

## Inference:
**Engine:** NeMo, BioNeMo, Triton <br>
**Test Hardware:** <br>
* [Ampere] <br>
* [Hopper] <br>

## Accuracy Benchmarks

Accuracy and performance benchmarks were obtain with the following machine configurations:


### Cluster specs

| Label  | GPU type             | Driver Version | memory per GPU  | number of gpu  | number of nodes | cpu type      | cores per cpu | num cpu per node |
| :----: | :------------------: | :--: | :-------------: | :------------: | :------------:  | :-----------: | :-----------: | :----------:|
| I      | NVIDA A100           | unknown |80 GB           |  128           | 16              | unknown            |     unknown        |  unknown |
| O      | NVIDIA A100-SXM4-80GB| 535.129.03 |80 GB           |  128           | 16              | [AMD EPYC 7J13](https://www.amd.com/en/products/cpu/amd-epyc-7713) |     64        | 256|

### Parameter settings
The following parameters are described and configured in the current version of `examples/proteint/openfold/conf/openfold_initial_training.yaml`
- model.num_steps_per_epoch
- model.optimisations
- trainer.precision


### Initial Training

| model.optimisations                             | trainer.precision | model.num_steps_per_epoch | machine spec | LDDT-CA [%] | job-completion date |
| :------------------------------------------------: | :------: | :----: | :----: | :--:| :----: |
| no optimisations implemented with this version | 32 | 80,000 | I | 89.82 | 2023q4* |
| []                                                |   32   | 80,000 | O | 90.030 | 2024-04-29 |
| [layernorm_inductor,layernorm_triton,mha_triton]  |   bf16-mixed   | 80,000 | O | 90.025 | 2024-04-19 |

These runs were carried out with
1. trainer.max_epochs=1

### Fine tuning

| model.optimisations                              | trainer.precision | model.num_steps_per_epoch | machine spec | LDDT-CA [%] | job-completion date |
| :----------------------------------------------: | :------: | :----: | :----: | :--: | :------: |
|  no optimisations implemented with this version                 |   32   | 12,000 | I | 91.0 | 2023q4* |


## Performance Benchmarks

### Initial Training

| model.optimisations                                | trainer.precision | model.num_steps_per_epoch | machine spec | time per training step (sec) | speed-up | job-completion date |
| :------------------------------------------------: | :------: | :----: | :----: | :---: | :---: | :---: |
| []                                                |   32   | 1000 | O | 6.71 | 1.0 | 2024-04-02 |
| [layernorm_inductor,layernorm_triton,mha_triton]  |   bf16-mixed   | 1000 | O | 6.00 | 1.12 | 2024-04-03 |

These runs were carried out with
1. trainer.max_epochs=1
2. The first 20 training steps are ommited from the 'time per training step', since they are consider 'warm-up'.


### Fine-tuning

| model.optimisations                              | trainer.precision | model.num_steps_per_epoch | machine spec | time per training step (sec) | job-completion date |
| :----------------------------------------------: | :---------------: | :----------------------:  | :----------: | :--------:  | :-----------------: |
|  no optimisations implemeention in this version  |           32      |  12,000                 | I             | 24.91         | 2023q4*     |


*Best guess.

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their supporting model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards.  Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).