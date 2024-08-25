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
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from lightning import pytorch as pl
from omegaconf import DictConfig, OmegaConf
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean
from torch_sparse import coalesce
from tqdm import tqdm

# TODO create general data module class that can do 3dmg, sbdd, docking etc
# TODO later on can create a Inference Class like Model Builder to handle all specfic benchmarking
from bionemo.model.molecule.moco.models.denoising_models import ModelBuilder
from bionemo.model.molecule.moco.models.interpolant import build_interpolant
from bionemo.model.molecule.moco.models.self_conditioning import SelfConditioningBuilder
from bionemo.model.molecule.moco.models.utils import InterpolantLossFunction


def compute_angles(pos, bond_matrix, edge_index):
    """
    Compute the angles between connected nodes in a graph based on their positions and the bond matrix.

    Args:
        positions (Tensor): Node positions [N, 3].
        bond_matrix (Tensor): Adjacency matrix [N, N].
        edge_index (Tensor): Edge index [2, E].

    Returns:
        Tensor: Vector of angles for existing bonds [E].
    """
    num_edges = edge_index.size(1)
    angle_list = []

    for edge_idx in range(num_edges):
        src, dst = edge_index[:, edge_idx]
        if bond_matrix[src, dst] > 0:
            # Find all other neighbors to compute the angle
            # src_neighbors = torch.nonzero(bond_matrix[src]).squeeze(1)
            dst_neighbors = torch.nonzero(bond_matrix[dst]).squeeze(1)

            # Remove self from neighbors
            # src_neighbors = src_neighbors[src_neighbors != dst]
            dst_neighbors = dst_neighbors[dst_neighbors != src]

            for k in dst_neighbors:
                if k != src:  # Avoid self-loop
                    # Compute the angle (i, j, k)
                    p_a = pos[src]
                    p_b = pos[dst]
                    p_c = pos[k]

                    v1 = p_b - p_a
                    v2 = p_c - p_a

                    # Normalize the vectors
                    v1_norm = v1 / (torch.norm(v1) + 1e-6)
                    v2_norm = v2 / (torch.norm(v2) + 1e-6)

                    # Compute cosine similarity
                    cosine_similarity = torch.dot(v1_norm, v2_norm)

                    # Clamp to avoid numerical errors
                    cosine_similarity = cosine_similarity.clamp(-1, 1)

                    # Compute angle
                    angle = torch.acos(cosine_similarity)
                    angle_list.append(angle)

    return torch.stack(angle_list).to(pos.device)  # Convert list to tensor and move to device


def cosine_distance_loss(true_angles, pred_angles):
    """
    Compute the cosine distance loss between true and predicted angles.

    Args:
        true_angles (Tensor): True angles [E].
        pred_angles (Tensor): Predicted angles [E].

    Returns:
        Tensor: Loss value.
    """
    true_cos = torch.cos(true_angles)
    pred_cos = torch.cos(pred_angles)
    loss = 1 - torch.mean(true_cos * pred_cos)
    return loss


def angle_cosine_loss(X, X_true, edge_index, edge_attr, batch):
    total_loss = 0.0

    for graph_idx in torch.unique(batch):
        # Mask to select the current graph
        mask = batch == graph_idx
        pos_pred = X[mask]  # Predicted positions for the current graph
        pos_true = X_true[mask]  # True positions for the current graph

        # Create bond matrix
        edge_mask = (edge_index[0] >= mask.nonzero(as_tuple=True)[0].min()) & (
            edge_index[0] <= mask.nonzero(as_tuple=True)[0].max()
        )
        bonds = edge_attr[edge_mask]
        bond_indices = edge_index[:, edge_mask]

        # Adjust bond indices to local molecule
        local_bond_indices = bond_indices - bond_indices[0].min()
        bond_matrix = torch.zeros((mask.sum().item(), mask.sum().item()), dtype=torch.long, device=X.device)
        for src, dst, bond in zip(local_bond_indices[0], local_bond_indices[1], bonds):
            bond_matrix[src, dst] = bond
            bond_matrix[dst, src] = bond

        # Compute angles
        # import ipdb; ipdb.set_trace()
        true_angles = compute_angles(pos_true, bond_matrix, local_bond_indices)
        pred_angles = compute_angles(pos_pred, bond_matrix, local_bond_indices)

        # Calculate loss
        loss = cosine_distance_loss(true_angles, pred_angles)
        total_loss += loss

    return total_loss / len(torch.unique(batch))


class Graph3DInterpolantModel(pl.LightningModule):
    def __init__(
        self,
        loss_params: DictConfig,
        optimizer_params: DictConfig,
        lr_scheduler_params: DictConfig,
        dynamics_params: DictConfig,
        interpolant_params: DictConfig,
        sampling_params: DictConfig,
        self_cond_params: DictConfig = None,
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
        self.self_conditioning_module = None
        if self_cond_params is not None:
            self.self_conditioning_module = self.configure_self_cond(self_cond_params)

    def configure_self_cond(self, self_cond_params):
        self_cond_params = OmegaConf.to_container(self_cond_params, resolve=True)
        for var in self_cond_params["variables"]:
            var["inp_dim"] = self.interpolants[var["variable_name"]].num_classes
        return SelfConditioningBuilder().create_self_cond(self_cond_params)

    # def setup(self, stage = None):
    #     self.loss_functions = self.initialize_loss_functions()

    def initialize_loss_functions(self):
        loss_functions = {}
        self.loss_clamps = {}
        for loss_params in self.loss_params.variables:
            index = loss_params.variable_name
            self.loss_clamps[index] = 1e6
            if "use_distance" in loss_params:
                loss_functions[index] = InterpolantLossFunction(
                    loss_scale=loss_params.loss_scale,
                    aggregation=loss_params.aggregate,
                    continuous=loss_params.continuous,
                    use_distance=loss_params.use_distance,
                    distance_scale=loss_params.distance_scale,
                )
            else:
                # if "edge" in index:
                #     import ipdb; ipdb.set_trace()
                #     weight = torch.tensor([1.0000e+00, 3.1123e+01, 4.7676e+02, 2.1425e+04, 7.9271e+01]).to(self.device)
                # else:
                #     weight = None
                loss_functions[index] = InterpolantLossFunction(
                    loss_scale=loss_params.loss_scale,
                    aggregation=loss_params.aggregate,
                    continuous=loss_params.continuous,
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
            if not interp_param.interpolant_type:
                interpolants[index] = None
                continue
            if interp_param.prior_type in ["mask", "absorb"]:
                interp_param.num_classes += 1
            elif interp_param.prior_type in ["custom", "data"]:
                interp_param = dict(interp_param)
                interp_param["custom_prior"] = self.load_prior(interp_param["custom_prior"]).float()
            interpolants[index] = build_interpolant(**interp_param)
        self.interpolant_param_variables = {
            interp_param.variable_name: interp_param for interp_param in self.interpolant_params.variables
        }
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
            elif self.lr_scheduler_params.type == "linear_warmup":
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.lr_scheduler_params.initial_lr
                    / self.lr_scheduler_params.final_lr,  # Start factor (initial learning rate / final learning rate)
                    end_factor=1.0,  # End factor (final learning rate / final learning rate)
                    total_iters=self.lr_scheduler_params.num_warmup_steps,  # Number of iterations to go from start_factor to end_factor
                )
            elif self.lr_scheduler_params.type == "linear_warmup_decay":
                # Warm-up phase using LinearLR
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.lr_scheduler_params.initial_lr / self.lr_scheduler_params.final_lr,
                    end_factor=1.0,
                    total_iters=self.lr_scheduler_params.num_warmup_steps,  # Steps
                )

                # Decay phase using LinearLR (kicks in after milestone)
                decay_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=self.lr_scheduler_params.min_lr_decay / self.lr_scheduler_params.final_lr,
                    total_iters=self.lr_scheduler_params.num_decay_steps,  # Steps
                )

                # SequentialLR to combine both schedulers
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, decay_scheduler],
                    milestones=[self.lr_scheduler_params.milestone_steps],  # Milestone in steps
                )
            else:
                raise NotImplementedError('LR Scheduler not supported: %s' % self.lr_scheduler_params.type)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.lr_scheduler_params.interval,
                    # "monitor": self.lr_scheduler_params.monitor,
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
        # for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
        for index, interp_param in self.interpolant_param_variables.items():
            # index = interp_param.variable_name
            if index == "x":
                batch['x'] = (
                    batch['x'] - scatter_mean(batch['x'], index=batch.batch, dim=0, dim_size=batch_size)[batch.batch]
                )
            elif index == "h":
                if interp_param.prior_type in ["mask", "absorb"]:
                    batch["h"] = self.add_adsorbtion_state(batch["h"])
            elif index == "edge_attr":
                # Load bond information from the dataloader
                bond_edge_index, bond_edge_attr = sort_edge_index(
                    edge_index=batch.edge_index, edge_attr=batch.edge_attr, sort_by_row=False
                )
                # Create Fully Connected Graph instead
                edge_index_global = (
                    torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1)).int().fill_diagonal_(0)
                )
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

                if interp_param.prior_type in ["mask", "absorb"]:
                    # TODO: should we use the no bond state as the mask? or create an extra dim
                    edge_attr_global = self.add_absorbption_state(edge_attr_global)

                batch['edge_attr'] = edge_attr_global
                batch['edge_index'] = edge_index_global

            elif index == "charges":
                batch['charges'] = batch['charges'] + interp_param.offset
        return batch

    def add_adsorbtion_state(self, h):
        # h is (N, C) and we want to add a column of all zeros at the end
        N, C = h.shape
        zeros_column = torch.zeros(N, 1, device=h.device)
        return torch.cat([h, zeros_column], dim=1)

    def interpolate(self, batch, time):
        # for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
        for index, interp_param in self.interpolant_param_variables.items():
            interpolant = self.interpolants[f"{interp_param.variable_name}"]
            if interpolant is None:
                batch[f"{interp_param.variable_name}_t"] = batch[f"{interp_param.variable_name}"]
            else:
                if interp_param.variable_name == "edge_attr":
                    _, batch[f"{interp_param.variable_name}_t"], _ = interpolant.interpolate_edges(
                        batch.batch, batch[f"{interp_param.variable_name}"], batch["edge_index"], time
                    )
                else:
                    _, batch[f"{interp_param.variable_name}_t"], _ = interpolant.interpolate(
                        batch.batch, batch[f"{interp_param.variable_name}"], time
                    )

            if "concat" in interp_param or "discrete" in interp_param.interpolant_type:
                batch[f"{interp_param.variable_name}_t"] = F.one_hot(
                    batch[f"{interp_param.variable_name}_t"], interp_param.num_classes
                ).float()

        return batch

    def aggregate_discrete_variables(self, batch):
        """
        Concatenate the flagged variable on its target and save the original in the batch via _og.
        """
        for interp_param in self.interpolant_params.variables:
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
            if "concat" in interp_param or "discrete" in interp_param.interpolant_type:
                key = interp_param.variable_name
                combined_keys = [key]
                interpolant_type = interp_param.interpolant_type
                if "concat" in interp_param:
                    target = interp_param.concat
                    combined_keys.append(target)
                    K = interp_param.num_classes
                    N = out[f"{target}_logits"].shape[-1]
                    out[f"{target}_logits"], out[f"{key}_logits"] = torch.split(
                        out[f"{target}_logits"], [N - K, K], dim=-1
                    )
                    interpolant_type = self.interpolant_param_variables[target].interpolant_type

                for _key in combined_keys:
                    if f"{_key}_og" in batch:
                        batch[f"{_key}_t"] = batch[f"{_key}_og"]
                    if self.interpolants[_key] and self.interpolants[_key].prior_type in ["absorb", "mask"]:
                        logits = out[f"{_key}_logits"].clone()
                        logits[:, -1] = -1e9
                    else:
                        logits = out[f"{_key}_logits"]
                    if "diffusion" in interpolant_type:  #! Diffusion operates over the probability
                        out[f"{_key}_hat"] = logits.softmax(dim=-1)
                    else:  #! DFM assumes that you operate over the logits
                        out[f"{_key}_hat"] = out[f"{_key}_logits"]

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
        return loss

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
                    level=self.loss_clamps[key],
                )
            else:
                if loss_fn.continuous:
                    # import ipdb; ipdb.set_trace()
                    sub_loss, sub_pred = loss_fn(
                        batch_geo, out[f'{key}_hat'], batch[f'{key}'], batch_weight=ws_t, level=self.loss_clamps[key]
                    )
                else:
                    true_data = batch[f'{key}']
                    if len(true_data.shape) > 1:
                        if true_data.size(1) == 1:
                            true_data = true_data.unsqueeze(1)
                        else:
                            true_data = true_data.argmax(dim=-1)
                    sub_loss, sub_pred = loss_fn(
                        batch_geo, out[f'{key}_logits'], true_data, batch_weight=ws_t, level=self.loss_clamps[key]
                    )

            self.loss_clamps[key] = min(self.loss_clamps[key], sub_loss.item() / loss_fn.scale * 5)
            # print(key, sub_loss)
            self.log(f"{stage}/{key}_loss", sub_loss, batch_size=batch_size, prog_bar=True)
            loss = loss + sub_loss
            # predictions[f'{key}'] = sub_pred
            # bbloss = self.loss_functions['x'].backbone_loss(
            #     batch_geo, out['x_hat'], batch['x'], batch_weight=(time > 250).int()
            # )
            # self.log(f"{stage}/backbone_loss", bbloss, batch_size=batch_size, prog_bar=True)
            # loss = loss + bbloss

            if loss_fn.use_distance in ["single", "triple"]:
                if "Z_hat" in out.keys() and loss_fn.use_distance == "triple":
                    z_hat = out["Z_hat"]
                else:
                    z_hat = None
                distance_loss_tp, distance_loss_tz, distance_loss_pz = loss_fn.distance_loss(
                    batch_geo, out[f'{key}_hat'], batch[f'{key}'], z_hat
                )
                distance_loss = distance_loss_tp + distance_loss_tz + distance_loss_pz
                self.log(f"{stage}/distance_loss", distance_loss, batch_size=batch_size)
                self.log(f"{stage}/distance_loss_tp", distance_loss_tp, batch_size=batch_size)
                self.log(f"{stage}/distance_loss_tz", distance_loss_tz, batch_size=batch_size)
                self.log(f"{stage}/distance_loss_pz", distance_loss_pz, batch_size=batch_size)
                loss = loss + loss_fn.distance_scale * distance_loss
            # if loss_fn.use_distance in ["angle"]:
            #     # import ipdb; ipdb.set_trace()
            #     angle_loss = angle_cosine_loss(X= out[f'x_hat'], X_true = batch[f'x'], edge_index = batch[f'edge_index'], edge_attr = batch[f'edge_attr'], batch=batch["batch"])
            #     self.log(f"{stage}/bond_angle_loss", angle_loss, batch_size=batch_size, prog_bar=True)
            #     loss = loss + angle_loss
        self.log(f"{stage}/loss", loss, batch_size=batch_size)
        print(self.loss_clamps)
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
        if self.self_conditioning_module is not None:
            batch, _ = self.self_conditioning(batch, time)
        batch = self.aggregate_discrete_variables(batch)
        out = self.dynamics(batch, time)
        out, batch = self.separate_discrete_variables(out, batch)
        return out, batch, time

    def one_hot(self, batch):
        """
        Convert class indices to one hot vectors.
        """
        for interpolant_idx, interp_param in enumerate(self.interpolant_params.variables):
            if interp_param.interpolant_type is not None and "discrete" in interp_param.interpolant_type:
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

    # def on_after_backward(self):
    #     # Compute and log the maximum gradient norm before clipping
    #     max_grad_norm = 0
    #     for p in self.dynamics.parameters():
    #         if p.grad is not None:
    #             param_norm = p.grad.data.norm(2)
    #             if param_norm > max_grad_norm:
    #                 max_grad_norm = param_norm

    #     # Log the maximum gradient norm before clipping
    #     print("TEST", max_grad_norm)
    #     results = {'train/max_grad_norm_before_clipping': max_grad_norm}
    #     self.log_dict(results, sync_dist=True)

    @torch.no_grad()
    def sample(self, num_samples, timesteps=500, time_discretization="linear", batch=None, num_atoms=None):
        """
        Generates num_samples. Can supply a batch for inital starting points for conditional sampling for any interpolants set to None.
        """
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

        if num_atoms is None:
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

        # Sample from all Priors
        for key, interpolant in self.interpolants.items():
            if interpolant is None:
                if batch is not None:
                    prior[key] = batch[key]
                    data[f"{key}_t"] = prior[key]
                else:
                    # If no batch is supplied just give zeros
                    data[f"{key}_t"] = torch.zeros(
                        (total_num_atoms, self.interpolant_param_variables[key].num_classes)
                    ).to(self.device)
                    if "offset" in self.interpolant_param_variables[key]:
                        data[f"{key}_t"] += self.interpolant_param_variables[key].offset
                continue
            if "edge" in key:
                shape = (edge_index.size(1), interpolant.num_classes)
                prior[key], edge_index = interpolant.prior_edges(batch_index, shape, edge_index, self.device)
                data[f"{key}_t"] = prior[key]
            else:
                shape = (total_num_atoms, interpolant.num_classes)
                data[f"{key}_t"] = prior[key] = interpolant.prior(batch_index, shape, self.device)

        # Iterate through time, query the dynamics, apply interpolant step update
        out = {}
        # print("DT", len(DT))
        for idx in tqdm(list(range(len(DT))), total=len(DT)):
            t = timeline[idx]
            dt = DT[idx]
            time = torch.tensor([t] * num_samples).to(self.device)
            data = self.one_hot(data)
            # Apply Self Conditioning
            pre_conditioning_variables = {}
            if self.self_conditioning_module is not None:
                data, pre_conditioning_variables = self.self_conditioning(data, time, conditional_batch=out)
            data = self.aggregate_discrete_variables(data)
            data["batch"] = batch_index
            data["edge_index"] = edge_index
            out = self.dynamics(data, time, conditional_batch=out, timesteps=timesteps)
            out, data = self.separate_discrete_variables(out, data)
            for key in pre_conditioning_variables:
                data[key] = pre_conditioning_variables[key]
            for key, interpolant in self.interpolants.items():
                if interpolant is None:
                    continue
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
        for interp_params in self.interpolant_params.variables:
            if "offset" in interp_params:
                samples[interp_params.variable_name] -= interp_params.offset
        return samples

    def self_conditioning(self, batch, time, conditional_batch=None):
        pre_conditioning_variables = {}
        if self.training:
            with torch.no_grad():
                batch = self.aggregate_discrete_variables(batch)
                out = self.dynamics(batch, time)
                conditional_batch, batch = self.separate_discrete_variables(out, batch)
                for key in conditional_batch:
                    conditional_batch[key].detach()
            batch, pre_conditioning_variables = self.self_conditioning_module(batch, conditional_batch)
            if torch.rand(1).item() <= 0.5:
                for key in pre_conditioning_variables:
                    # hack to avoid unused parameters error
                    batch[key] = pre_conditioning_variables[key] + 0 * batch[key]
        else:
            if conditional_batch is not None and len(conditional_batch) > 0:
                batch, pre_conditioning_variables = self.self_conditioning_module(batch, conditional_batch)
        return batch, pre_conditioning_variables

    @torch.no_grad()
    def conditional_sample(
        self,
        batch,
        conditional_variables=['h', 'edge_attr', 'charges'],
        timesteps=500,
        time_discretization="linear",
        early_stop=None,
        save_all=False,
    ):
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

        batch_size = int(batch.batch.max()) + 1
        num_samples = batch_size
        batch = self.pre_format_molecules(batch, batch_size)
        batch_index = batch.batch
        edge_index = batch.edge_index
        total_num_atoms = batch["h"].shape[0]
        data, prior = {}, {}
        if save_all:
            data_all = defaultdict(list)

        for key, interpolant in self.interpolants.items():
            if key in conditional_variables:
                prior[key] = batch[key]
                data[f"{key}_t"] = prior[key]
            elif interpolant is None:
                data[f"{key}_t"] = torch.zeros(
                    (total_num_atoms, self.interpolant_param_variables[key].num_classes)
                ).to(self.device)
                if "offset" in self.interpolant_param_variables[key]:
                    data[f"{key}_t"] += self.interpolant_param_variables[key].offset
            elif "edge" in key:
                shape = (edge_index.size(1), interpolant.num_classes)
                prior[key], edge_index = interpolant.prior_edges(batch_index, shape, edge_index, self.device)
                data[f"{key}_t"] = prior[key]
            else:
                shape = (total_num_atoms, interpolant.num_classes)
                data[f"{key}_t"] = prior[key] = interpolant.prior(batch_index, shape, self.device)

        if save_all:
            for key in self.interpolants.keys():
                data_all[key].append(data[f"{key}_t"])

        for idx in tqdm(list(range(len(DT))), total=len(DT)):
            if early_stop and idx >= early_stop:
                break
            t = timeline[idx]
            dt = DT[idx]
            time = torch.tensor([t] * num_samples).to(self.device)
            data = self.one_hot(data)
            data = self.aggregate_discrete_variables(data)
            data["batch"] = batch_index
            data["edge_index"] = edge_index
            out = self.dynamics(data, time, timesteps=timesteps)
            out, data = self.separate_discrete_variables(out, data)
            for key, interpolant in self.interpolants.items():
                if t / timesteps > 0.5 and key in conditional_variables:
                    data[f"{key}_t"] = prior[key]
                    continue
                if interpolant is None:
                    continue
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
            if save_all:
                for key in self.interpolants.keys():
                    data_all[key].append(data[f"{key}_t"])

        samples = {key: data[f"{key}_t"] for key in self.interpolants.keys()}
        samples["batch"] = batch_index
        samples["edge_index"] = edge_index
        for interp_params in self.interpolant_params.variables:
            if "offset" in interp_params:
                samples[interp_params.variable_name] -= interp_params.offset
        if save_all:
            data_all["batch"].append(batch_index)
            data_all["edge_index"].append(edge_index)
            return samples, data_all
        return samples
