# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import pickle

import numpy as np
import torch
import torch.nn.functional as F
from lightning import pytorch as pl
from nemo.core.config import hydra_runner
from omegaconf import DictConfig
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean
from torch_sparse import coalesce
from tqdm import tqdm

from bionemo.model.molecule.moco.data.molecule_datamodule import MoleculeDataModule
from bionemo.model.molecule.moco.data.molecule_dataset import full_atom_decoder
from bionemo.model.molecule.moco.metrics.metrics import (
    BasicMolecularMetrics,
    get_molecules,
)

#!TODO later on can create a Inference Class like Model Builder to handle all specfic benchmarking
from bionemo.model.molecule.moco.models.denoising_models import ModelBuilder
from bionemo.model.molecule.moco.models.interpolant import build_interpolant
from bionemo.model.molecule.moco.models.utils import InterpolantLossFunction


class Graph3DInterpolantModel(pl.LightningModule):
    def __init__(
        self,
        loss_params: DictConfig,
        optimizer_params: DictConfig,
        lr_scheduler_params: DictConfig,
        dynamics_params: DictConfig,
        interpolant_params: DictConfig,
        sampling_params: DictConfig,
    ):
        super(Graph3DInterpolantModel, self).__init__()
        self.save_hyperparameters()
        self.optimizer_params = optimizer_params
        self.lr_scheduler_params = lr_scheduler_params
        self.dynamics_params = dynamics_params
        self.interpolant_params = interpolant_params
        self.global_variable = interpolant_params.global_variable_name
        self.loss_params = loss_params
        self.loss_functions = self.initialize_loss_functions()
        self.interpolants = self.initialize_interpolants()
        self.sampling_params = sampling_params
        self.node_distribution = self.initialize_inference()
        self.dynamics = ModelBuilder().create_model(
            dynamics_params.model_name, dynamics_params.model_args, dynamics_params.wrapper_args
        )
        self.mol_metrics = BasicMolecularMetrics({"atom_decoder": full_atom_decoder})

    def initialize_loss_functions(self):
        loss_functions = {}
        for loss_params in self.loss_params.variables:
            index = loss_params.variable_name
            loss_functions[index] = InterpolantLossFunction(
                loss_scale=loss_params.loss_scale, aggregation=loss_params.aggregate, continuous=loss_params.continuous
            )
        return loss_functions

    def load_prior(self, fpath):
        if fpath[-3:] == "npy":
            array = np.load(fpath)
            tensor = torch.tensor(array)  # .to(self.device)
        else:
            raise ValueError("Currently only supports numpy prior arrays")
        return tensor

    def initialize_interpolants(self):
        interpolants = torch.nn.ModuleDict()
        for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
            index = interp_param.variable_name
            if interp_param.prior_type in ["mask", "absorb"]:
                interp_param.num_classes += 1
            elif interp_param.prior_type in ["custom", "data"]:
                interp_param = dict(interp_param)
                interp_param["custom_prior"] = self.load_prior(interp_param["custom_prior"]).float()
            interpolants[index] = build_interpolant(**interp_param)
        return interpolants

    def configure_optimizers(self):
        if self.optimizer_params.type == "adamw":
            optimizer = torch.optim.AdamW(
                self.dynamics.parameters(),
                lr=self.optimizer_params.lr,
                amsgrad=self.optimizer_params.amsgrad,
                weight_decay=self.optimizer_params.weight_decay,
            )
        else:
            raise NotImplementedError('Optimizer not supported: %s' % self.optimizer_params.type)

        if self.lr_scheduler_params:
            if self.lr_scheduler_params.type == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=self.lr_scheduler_params.factor,
                    patience=self.lr_scheduler_params.patience,
                    min_lr=self.lr_scheduler_params.min_lr,
                    cooldown=self.lr_scheduler_params.cooldown,
                )
            else:
                raise NotImplementedError('LR Scheduler not supported: %s' % self.lr_scheduler_params.type)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.lr_scheduler_params.interval,
                    "monitor": self.lr_scheduler_params.monitor,
                    "frequency": self.lr_scheduler_params.frequency,
                    "strict": False,
                },
            }
        else:
            return {
                "optimizer": optimizer,
            }

    def sample_time(self, batch):
        batch_size = int(batch.batch.max()) + 1
        time = self.interpolants['x'].sample_time(
            num_samples=batch_size,
            device=batch.x.device,
            method=self.interpolant_params.sample_time_method,
            mean=self.interpolant_params.sample_time_mean,
            scale=self.interpolant_params.sample_time_scale,
        )
        return time

    def pre_format_molecules(self, batch, batch_size):
        batch['x'] = batch['x'] - scatter_mean(batch['x'], index=batch.batch, dim=0, dim_size=batch_size)[batch.batch]

        if self.interpolants['h'].prior_type in ["mask", "absorb"]:
            # batch["h"] = torch.cat((batch["h"], torch.zeros((batch["h"].size(0), 1)).to(batch["h"].device)), dim=1).to(
            #     batch["h"].device
            # )
            batch["h"] = self.add_adsorbtion_state(batch["h"])

        # batch['h'] = F.one_hot(batch["h"], self.interpolants['h'].num_classes).float()

        # Load bond information from the dataloader
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=batch.edge_index, edge_attr=batch.edge_attr, sort_by_row=False
        )
        # Create Fully Connected Graph instead
        edge_index_global = torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1)).int().fill_diagonal_(0)
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        edge_attr_tmp = torch.full(
            size=(edge_index_global.size(-1),),
            fill_value=0,
            device=edge_index_global.device,
            dtype=torch.long,
        )
        edge_index_global = torch.cat([edge_index_global, bond_edge_index], dim=-1)
        edge_attr_tmp = torch.cat([edge_attr_tmp, bond_edge_attr], dim=0)
        edge_index_global, edge_attr_global = coalesce(
            index=edge_index_global, value=edge_attr_tmp, m=batch['x'].size(0), n=batch['x'].size(0), op="max"
        )
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
        )

        if self.interpolants['edge_attr'].prior_type in ["mask", "absorb"]:
            # TODO: should we use the no bond state as the mask? or create an extra dim
            edge_attr_global = self.add_absorbption_state(edge_attr_global)

        batch['edge_attr'] = edge_attr_global
        batch['edge_index'] = edge_index_global

        # batch['edge_attr'] = F.one_hot(batch["edge_attr"], self.interpolants['edge_attr'].num_classes).float()

        # TODO: anymore specfic shifting of molecule only data
        return batch

    def add_adsorbtion_state(self, h):
        # h is (N, C) and we want to add a column of all zeros at the end
        N, C = h.shape
        zeros_column = torch.zeros(N, 1, device=h.device)
        return torch.cat([h, zeros_column], dim=1)

    def interpolate(self, batch, time):
        for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
            interpolant = self.interpolants[f"{interp_param.variable_name}"]
            if interpolant is None:
                batch[f"{interp_param.variable_name}_t"] = batch[f"{interp_param.variable_name}"]
            if interp_param.variable_name == "edge_attr":
                _, batch[f"{interp_param.variable_name}_t"], _ = interpolant.interpolate_edges(
                    batch.batch, batch[f"{interp_param.variable_name}"], batch["edge_index"], time
                )
            else:
                _, batch[f"{interp_param.variable_name}_t"], _ = interpolant.interpolate(
                    batch.batch, batch[f"{interp_param.variable_name}"], time
                )
            if "discrete" in interp_param.interpolant_type:
                batch[f"{interp_param.variable_name}_t"] = F.one_hot(
                    batch[f"{interp_param.variable_name}_t"], interp_param.num_classes
                ).float()
                # batch[f"{interp_param.variable_name}"] = F.one_hot(
                #     batch[f"{interp_param.variable_name}"], interp_param.num_classes
                # ).float() #! I don't think we need to onehot this since for CE loss this can be classes

        return batch

    def aggregate_discrete_variables(self, batch):
        """
        Concatenate the flagged variable on its target and save the original in the batch via _og.
        """
        for interp_param in self.interpolant_params:
            if 'concat' in interp_param and interp_param['concat'] is not None:
                batch[f"{interp_param.concat}_og"] = batch[f"{interp_param.concat}_t"]
                batch[f"{interp_param.concat}_t"] = torch.concat(
                    [batch[f"{interp_param.concat}_t"], batch[f"{interp_param.variable_name}_t"]], dim=-1
                )
        return batch

    def separate_discrete_variables(self, out, batch):
        """
        Iterates throgh all outputs and restores interpolation for any aggregated variables.
        Converts output logits to the input necessary for Interpolant.step()
            - Discrete Diffusion Models assume class probabilities are given
            - Discrete Flow Models operate on the raw logits.
        Produces [Variable]_hat
        """
        for interp_param in self.interpolant_params.variables:
            if "discrete" in interp_param.interpolant_type:
                key = interp_param.variable_name
                if f"{key}_og" in batch:
                    batch[f"{key}_t"] = batch[f"{key}_og"]
                if self.interpolants[key].prior_type in ["absorb", "mask"]:
                    logits = out[f"{key}_logits"].clone()
                    logits[:, -1] = -1e9
                else:
                    logits = out[f"{key}_logits"]
                if "diffusion" in interp_param.interpolant_type:  #! Diffusion operates over the selected option
                    out[f"{key}_hat"] = logits.softmax(dim=-1)  # logits.argmax(dim=-1)
                else:  #! DFM assumes that you operate over the logits
                    out[f"{key}_hat"] = out[f"{key}_logits"]

        return out, batch

    def validation_step(self, batch, batch_idx):
        batch.h = batch.x
        batch.x = batch.pos
        batch.pos = None
        #! Swapping names for now
        time = self.sample_time(batch)
        out, batch, time = self(batch, time)
        loss, predictions = self.calculate_loss(batch, out, time, "val")
        # self.sample(100)
        # import ipdb; ipdb.set_trace()
        return loss

    def on_validation_epoch_end(self) -> None:
        try:
            mol_dict = self.sample(100)
            # TODO: put this into the wrapper since its assumes many things but can leave for now
            mols = get_molecules(mol_dict, {"atom_decoder": full_atom_decoder})
            stab_dict, valid_dict, stat_dict, valid_smi, stable_mols, valid_mols = self.mol_metrics(mols)
            res = {**stab_dict, **valid_dict, **stat_dict}
            print(res)
            self.log_dict(res)
        except Exception as e:
            print(f"The sampling has been failed with the error {e}")
        return super().on_validation_epoch_end()

    def training_step(self, batch, batch_idx):
        batch.h = batch.x
        batch.x = batch.pos
        batch.pos = None
        #! Swapping names for now
        time = self.sample_time(batch)
        out, batch, time = self(batch, time)
        loss, predictions = self.calculate_loss(batch, out, time, "train")
        return loss

    def calculate_loss(self, batch, out, time, stage="train"):
        batch_geo = batch.batch
        batch_size = int(batch.batch.max()) + 1
        ws_t = self.interpolants[self.global_variable].snr_loss_weight(time)
        loss = 0
        predictions = {}
        for key, loss_fn in self.loss_functions.items():
            if "edge" in key:
                sub_loss, sub_pred = loss_fn.edge_loss(
                    batch_geo,
                    out['edge_attr_logits'],
                    batch['edge_attr'],
                    index=batch['edge_index'][1],
                    num_atoms=batch_geo.size(0),
                    batch_weight=ws_t,
                )
            else:
                if loss_fn.continuous:
                    sub_loss, sub_pred = loss_fn(batch_geo, out[f'{key}_hat'], batch[f'{key}'], batch_weight=ws_t)
                else:
                    true_data = batch[f'{key}']
                    if len(true_data.shape) > 1:
                        if true_data.size(1) == 1:
                            true_data = true_data.unsqueeze(1)
                        else:
                            true_data = true_data.argmax(dim=-1)
                    sub_loss, sub_pred = loss_fn(batch_geo, out[f'{key}_logits'], true_data, batch_weight=ws_t)
            # print(key, sub_loss)
            self.log(f"{stage}/{key}_loss", sub_loss, batch_size=batch_size, prog_bar=True)
            loss = loss + sub_loss
            predictions[f'{key}'] = sub_pred

        if self.loss_params.use_distance:
            if "Z_hat" in out.keys() and self.loss_function.use_distance == "triple":
                z_hat = out["Z_hat"]
            else:
                z_hat = None
            distance_loss_tp, distance_loss_tz, distance_loss_pz = self.loss_function.distance_loss(
                batch_geo, batch['x'], predictions['x'], z_hat
            )
            distance_loss = distance_loss_tp + distance_loss_tz + distance_loss_pz
            self.log(f"{stage}/distance_loss", distance_loss, batch_size=batch_size)
            self.log(f"{stage}/distance_loss_tp", distance_loss_tp, batch_size=batch_size)
            self.log(f"{stage}/distance_loss_tz", distance_loss_tz, batch_size=batch_size)
            self.log(f"{stage}/distance_loss_pz", distance_loss_pz, batch_size=batch_size)
            loss = loss + distance_loss
        self.log(f"{stage}/loss", loss, batch_size=batch_size)
        return loss, predictions

    def forward(self, batch, time):
        """
        This forward function assumes we are doing some form (including none) of interpolation on positions X, node features H, and edge attributes edge_attr.
        1. Sample time from the distribution that is defined via the X interpolant params
        2. Shift X to 0 CoM, add absorbing state for H, create fully connected graph and edge features for edge_attr
        3. Interpolate all needed variables which are defined by "string" args in the config.
        4. Aggregate all the discrete non edge features in the H variable for modeling.
        5. Dynamics forward pass to predict clean data given noisy data.
        6. Seperate the aggregated discrete predictions for easier loss calculation.
        """
        batch_size = int(batch.batch.max()) + 1
        batch = self.pre_format_molecules(batch, batch_size)
        batch = self.interpolate(batch, time)  #! discrete variables are one hot after this point
        #    batch = self.initialize_pair_embedding(batch) #! Do in the dynamics
        batch = self.aggregate_discrete_variables(batch)
        out = self.dynamics(batch, time)
        out, batch = self.separate_discrete_variables(out, batch)
        return out, batch, time

    def one_hot(self, batch):
        """
        Convert class indices to one hot vectors.
        """
        for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
            if "discrete" in interp_param.interpolant_type:
                batch[f"{interp_param.variable_name}_t"] = F.one_hot(
                    batch[f"{interp_param.variable_name}_t"], interp_param.num_classes
                ).float()
        return batch

    def initialize_inference(self):
        if self.sampling_params.node_distribution:
            with open(self.sampling_params.node_distribution, "rb") as f:
                node_dict = pickle.load(f)
            max_n_nodes = max(node_dict.keys())
            n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
            for key, value in node_dict.items():
                n_nodes[key] += value

            n_nodes = n_nodes / n_nodes.sum()
        else:
            n_nodes = None
        return n_nodes

    def sample(self, num_samples, timesteps=500, time_discretization="linear"):
        time_type = self.interpolants[self.global_variable].time_type
        if time_type == "continuous":
            if time_discretization == "linear":
                timeline = torch.linspace(0, 1, timesteps + 1).tolist()  # [0, 1.0] timestpes + 1
            elif time_discretization == "log":
                timeline = (
                    (1 - torch.logspace(-2, 0, timesteps + 1)).flip(dims=[0]).tolist()
                )  # [0, 0.99] #timestpes + 1
            # timeline = torch.logspace(-2, 0, timesteps + 1) #[0.01, 1.0]
            DT = [t1 - t0 for t0, t1 in zip(timeline[:-1], timeline[1:])]  # timesteps
        else:
            timeline = torch.arange(timesteps + 1)
            DT = [1 / timesteps] * timesteps

        if self.node_distribution is not None:
            num_atoms = torch.multinomial(
                input=self.node_distribution,
                num_samples=num_samples,
                replacement=True,
            )
        else:
            num_atoms = torch.randint(20, 55, (num_samples,)).to(torch.int64)
        batch_index = torch.repeat_interleave(torch.arange(num_samples), num_atoms).to(self.device)
        edge_index = (
            torch.eq(batch_index.unsqueeze(0), batch_index.unsqueeze(-1)).int().fill_diagonal_(0).to(self.device)
        )  # N x N
        edge_index, _ = dense_to_sparse(edge_index)  # 2 x E
        edge_index = sort_edge_index(edge_index, sort_by_row=False)

        data, prior = {}, {}
        total_num_atoms = num_atoms.sum().item()

        for key, interpolant in self.interpolants.items():
            if "edge" in key:
                shape = (edge_index.size(1), interpolant.num_classes)
                prior[key], edge_index = interpolant.prior_edges(batch_index, shape, edge_index, self.device)
                data[f"{key}_t"] = prior[key]
            else:
                shape = (total_num_atoms, interpolant.num_classes)
                data[f"{key}_t"] = prior[key] = interpolant.prior(batch_index, shape, self.device)
        for idx in tqdm(list(range(len(DT))), total=len(DT)):
            t = timeline[idx]
            dt = DT[idx]
            time = torch.tensor([t] * num_samples).to(self.device)
            data = self.one_hot(data)
            data = self.aggregate_discrete_variables(data)
            data["batch"] = batch_index
            data["edge_index"] = edge_index
            out = self.dynamics(data, time)
            out, data = self.separate_discrete_variables(out, data)
            for key, interpolant in self.interpolants.items():
                if "edge" in key:
                    edge_index, data[f"{key}_t"] = interpolant.step_edges(
                        batch_index,
                        edge_index=edge_index,
                        edge_attr_t=data[f"{key}_t"],
                        edge_attr_hat=out[f"{key}_hat"],
                        time=time,
                    )
                else:
                    data[f"{key}_t"] = interpolant.step(
                        xt=data[f"{key}_t"],
                        x_hat=out[f"{key}_hat"],
                        x0=prior[key],
                        batch=batch_index,
                        time=time,
                        dt=dt,
                    )
        samples = {key: data[f"{key}_t"] for key in self.interpolants.keys()}
        samples["batch"] = batch_index
        samples["edge_index"] = edge_index
        return samples


@hydra_runner(config_path="conf", config_name="train")
def main(cfg: DictConfig):
    datamodule = MoleculeDataModule(cfg.data)
    train_dataloader = datamodule.test_dataloader()
    device = 'cuda:0'
    model = Graph3DInterpolantModel(
        loss_params=cfg.loss,
        optimizer_params=cfg.optimizer,
        lr_scheduler_params=cfg.lr_scheduler,
        dynamics_params=cfg.dynamics,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
    ).to(device)
    model.eval()
    with torch.no_grad():
        model.sample(10)
    for batch in train_dataloader:
        batch = batch.to(device)
        print(batch)
        model.training_step(batch, 0)


if __name__ == "__main__":
    main()
