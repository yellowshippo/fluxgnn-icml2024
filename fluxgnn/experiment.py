
import copy
import datetime
import os
import pathlib
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import siml
import torch
from tqdm import tqdm
import yaml

from fluxgnn.dataset import TorchFVDataset, collate_function
from fluxgnn.fluxgnn import FluxGNN
from fluxgnn.mixture import MixtureSolver
from fluxgnn.navier_stokes import NavierStokesSolver
from fluxgnn.setting import SimulationSetting, MLSetting, ModelSetting
from fluxgnn.torch_fv import TorchFVSolver
from fluxgnn import util
from fluxgnn.util import torch_tensor


class Experiment():

    DICT_DATASET_TO_MODE = {
        'cavity': 'navier_stokes',
        'convection_diffusion': 'convection_diffusion',
        'cavity_mixture': 'mixture',
    }
    RANK0_NAMES = [
        'phi', 'p', 'nu', 'diffusion', 'alpha', 'nu_solute', 'nu_solvent',
        'rho_solute', 'rho_solvent', 'diffusion_alpha',
        'space_scale_factor', 'time_scale_factor', 'mass_scale_factor',
    ]
    RANK1_NAMES = ['u', 'gravity']
    CONSERVATIVE_NAMES = ['phi', 'alpha']
    EPSILON = 1e-8

    DICT_MODE2DIM = {
        'navier_stokes': {
            'u': {'length': 1, 'time': -1, 'mass': 0},
            'p': {'length': 2, 'time': -2, 'mass': 0},
            'nu': {'length': 2, 'time': -1, 'mass': 0},
            'space_scale_factor': {'length': 0, 'time': 0, 'mass': 0},
            'time_scale_factor': {'length': 0, 'time': 0, 'mass': 0},
            'mass_scale_factor': {'length': 0, 'time': 0, 'mass': 0},
        },
        'mixture': {
            'u': {'length': 1, 'time': -1, 'mass': 0},
            'p': {'length': -1, 'time': -2, 'mass': 1},
            'alpha': {'length': 0, 'time': 0, 'mass': 0},
            'nu_solute': {'length': 2, 'time': -1, 'mass': 0},
            'nu_solvent': {'length': 2, 'time': -1, 'mass': 0},
            'rho_solute': {'length': -3, 'time': 0, 'mass': 1},
            'rho_solvent': {'length': -3, 'time': 0, 'mass': 1},
            'diffusion_alpha': {'length': 2, 'time': -1, 'mass': 1},
            'gravity': {'length': 1, 'time': -2, 'mass': 0},
            'space_scale_factor': {'length': 0, 'time': 0, 'mass': 0},
            'time_scale_factor': {'length': 0, 'time': 0, 'mass': 0},
            'mass_scale_factor': {'length': 0, 'time': 0, 'mass': 0},
        },
        'convection_diffusion': {
            'phi': {'length': 0, 'time': 0, 'mass': 0},
            'u': {'length': 1, 'time': -1, 'mass': 0},
            'diffusion': {'length': 2, 'time': -1, 'mass': 0},
            'space_scale_factor': {'length': 0, 'time': 0, 'mass': 0},
            'time_scale_factor': {'length': 0, 'time': 0, 'mass': 0},
            'mass_scale_factor': {'length': 0, 'time': 0, 'mass': 0},
        },
    }

    def __init__(
            self,
            model_name,
            dataset_name,
            train_data_directories,
            validation_data_directories,
            simulation_setting_dict,
            ml_setting_dict,
            model_setting_dict,
            fv_solver_setting_dict,
            output_directory_base,
            *,
            model_directory=None,
            output_directory=None,
            output_directory_suffix=None,
            output_directory_additional_suffix=None,
            device_id=-1,
            load_dataset=True,
            initialize=True,
            symmetry_decimal=3,
            conservative_decimal=3,
            write_initial=True,
            dataset_file_prefix=None,
            force_load=False,
            save_dataset=True,
            ml=True,
            plot=False,
            stats_mode='nodal',
            **kwargs,
    ):
        self.set_device(device_id)

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.mode = self.DICT_DATASET_TO_MODE[self.dataset_name]
        self.train_data_directories = train_data_directories
        self.validation_data_directories = validation_data_directories
        self.simulation_setting = SimulationSetting(**simulation_setting_dict)
        self.ml_setting = MLSetting(**ml_setting_dict)
        self.model_setting = ModelSetting(**model_setting_dict)
        self.fv_solver_setting_dict = fv_solver_setting_dict
        self.output_directory = None
        self.output_directory_base = output_directory_base
        self.output_directory_suffix = output_directory_suffix
        if output_directory_additional_suffix is None:
            self.output_directory_additional_suffix = ''
        else:
            self.output_directory_additional_suffix \
                = output_directory_additional_suffix
        self.symmetry_decimal = symmetry_decimal
        self.conservative_decimal = conservative_decimal

        self.write_initial = write_initial
        self.plot = plot
        self.ml = ml
        self.stats_mode = stats_mode

        if not self.ml:
            self.model_setting.property_for_ml = False
        if self.model_setting.property_for_ml:
            print('Use scaled properties for ML')

        if dataset_file_prefix is None or len(dataset_file_prefix) == 0:
            self.dataset_file_prefix = ''
        else:
            self.dataset_file_prefix = dataset_file_prefix + '_'
        self.force_load = force_load
        self.save_dataset = save_dataset
        self.scale_answer = kwargs.get('scale_answer', False)
        self.power_scale_answer = kwargs.get('power_scale_answer', 1.)
        self.diffusion_alpha = kwargs.get('diffusion_alpha', 1.e-4)
        self.prepost_scale = kwargs.get('prepost_scale', False)

        self.continue_count = 0
        self.best_state = None
        self.best_epoch = 0
        self.best_continue = 0
        self.min_loss = torch.tensor(float('inf'))

        # Load dataset
        if self.mode == 'convection_diffusion':
            self.torch_fv_constructor = TorchFVSolver
        elif self.mode == 'navier_stokes':
            self.torch_fv_constructor = NavierStokesSolver
        elif self.mode == 'mixture':
            self.torch_fv_constructor = MixtureSolver
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        self.set_indices()

        if load_dataset:
            train_dataset = TorchFVDataset(
                self.dataset_name, self.train_data_directories,
                self.torch_fv_constructor, self.fv_solver_setting_dict,
                start_index=self.timeseries_start_index,
                end_index=self.timeseries_start_index
                + self.timeseries_end_index,
                dataset_file_name=self.dataset_file_prefix + 'train',
                force_load=self.force_load,
                save=self.save_dataset,
                diffusion_alpha=self.diffusion_alpha,
                compute_variance=self.scale_answer)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, collate_fn=collate_function,
                batch_size=1, shuffle=True, num_workers=0)

            validation_dataset = TorchFVDataset(
                self.dataset_name, self.validation_data_directories,
                self.torch_fv_constructor, self.fv_solver_setting_dict,
                start_index=self.timeseries_start_index,
                end_index=self.timeseries_start_index
                + self.timeseries_end_index,
                dataset_file_name=self.dataset_file_prefix + 'validation',
                force_load=self.force_load,
                diffusion_alpha=self.diffusion_alpha,
                save=self.save_dataset)
            self.validation_loader = torch.utils.data.DataLoader(
                validation_dataset, collate_fn=collate_function,
                batch_size=1, shuffle=False, num_workers=0)

            if self.scale_answer:
                total_scale = sum([
                    1 / v**self.power_scale_answer
                    for v in train_dataset.dict_variance.values()])
                self.dict_answer_scales = {
                    k: 1 / v**self.power_scale_answer / total_scale
                    for k, v in train_dataset.dict_variance.items()}
                print(f"Scale answer: {self.dict_answer_scales}")
            else:
                self.dict_answer_scales = {
                    k: 1 for k in train_dataset[0][-1].keys()}

            if self.prepost_scale:
                self.scale_for_prepost = {
                    k: v**.5
                    for k, v in train_dataset.dict_variance.items()}

        else:
            self.dict_answer_scales = None

        if initialize:
            self.initialize(model_directory=model_directory)
        return

    def set_indices(self):
        self.factor_delta_t = round(
            self.simulation_setting.delta_t
            / self.simulation_setting.original_delta_t)
        float_factor_delta_t = self.simulation_setting.delta_t \
            / self.simulation_setting.original_delta_t
        if abs(self.factor_delta_t - float_factor_delta_t) > 1e-5 \
                or self.factor_delta_t < 1:
            raise ValueError(
                'Invalid delta t setting: '
                f"{self.simulation_setting.delta_t} is given but original is "
                f"{self.simulation_setting.original_delta_t}")
        self.answer_start_index = (round(
            self.simulation_setting.t_max
            / self.simulation_setting.delta_t
            / self.model_setting.n_forward
            * (self.model_setting.n_forward - 1)) + 1) * self.factor_delta_t
        self.answer_end_index = round(
            self.simulation_setting.t_max
            / self.simulation_setting.original_delta_t) + 1  # Include last
        factor_shift = round(
            self.simulation_setting.shift_t / self.simulation_setting.delta_t)
        float_factor_shift = \
            self.simulation_setting.shift_t / self.simulation_setting.delta_t
        if abs(factor_shift - float_factor_shift) > 1e-5 \
                or self.simulation_setting.shift_t \
                < self.simulation_setting.original_delta_t:
            raise ValueError(
                'Invalid shift_t setting: '
                f"{self.simulation_setting.shift_t}")
        self.index_shift = self.factor_delta_t * factor_shift
        self.timeseries_start_index = round(
            self.simulation_setting.start_t
            / self.simulation_setting.original_delta_t)
        self.timeseries_end_index = round(
            self.simulation_setting.max_timeseries_t
            / self.simulation_setting.original_delta_t) + 1  # Include last
        self.eval_answer_end_index = round(
            self.simulation_setting.eval_t_max
            / self.simulation_setting.original_delta_t) + 1  # Include last
        self.test_answer_end_index = round(
            self.simulation_setting.test_t_max
            / self.simulation_setting.original_delta_t) + 1  # Include last
        return

    def initialize(self, model_directory=None):
        self.set_seed(self.ml_setting.seed)

        # Initialize model
        block_setting = siml.setting.BlockSetting(
            type='fluxgnn', bias=False,
            nodes=self.model_setting.nodes,
            activations=self.model_setting.activations,
            optional={
                'propagations': self.model_setting.mode,
                'dimension': {
                    'length': 0,
                    'time': 0,
                    'mass': 0,
                }
            })
        block_setting.optional.update(vars(self.model_setting))
        if self.mode == 'convection_diffusion':
            self.predict_sample_function \
                = self.predict_sample_convection_diffusion
            self.compute_single_loss_function \
                = self.compute_single_loss_dict
        elif self.mode == 'navier_stokes':
            self.predict_sample_function = self.predict_sample_navier_stokes
            self.compute_single_loss_function \
                = self.compute_single_loss_dict
        elif self.mode == 'mixture':
            self.predict_sample_function = self.predict_sample_mixture
            self.compute_single_loss_function \
                = self.compute_single_loss_dict
        else:
            raise ValueError(f"Unexpected mode: {self.mode}")

        # Prepare for training
        self.dict_loss_function = {
            k: self._select_loss_function(v)
            for k, v in self.ml_setting.dict_loss_function_name.items()}
        print('Loss function:')
        for k, v in self.ml_setting.dict_loss_function_name.items():
            print(f"- {k}: {v}")
        self.early_stopping_count = 0

        if self.output_directory is None:
            if self.output_directory_suffix is None:
                suffix = f"{self.model_name}_{siml.util.date_string()}" \
                    + self.output_directory_additional_suffix
            else:
                suffix = self.output_directory_suffix \
                    + self.output_directory_additional_suffix
            self.original_output_directory = self.output_directory_base \
                / suffix
            self.output_directory = self.output_directory_base / suffix

        if not self.ml:
            return

        self.fluxgnn = FluxGNN(block_setting).to(self.device_id)
        if model_directory is not None:
            self.load_model_from_directory(model_directory)

        self.optimizer = torch.optim.Adam(
            self.fluxgnn.parameters(), lr=self.ml_setting.lr)

        return

    def _select_loss_function(self, name):
        if name in ['mse', 'l2']:
            return self.mse
        if name == 'rmse':
            return self.rmse
        elif name == 'relative_l2':
            return self.relative_l2
        elif name == 'cosine_distance':
            return self.cosine_distance
        else:
            raise ValueError(f"Unexpected loss function name: {name}")

    def train(self):
        self.output_directory.mkdir(parents=True, exist_ok=True)
        print('--')
        print(f"Using device: {self.device_id}")
        print(f"Output directory: {self.output_directory}")
        print('--')

        self.load_model_if_needed()

        n_epoch = self.ml_setting.n_epoch

        self.print_write_log(
            'epoch, train_loss, validation_loss, elapsed_time')
        start_time = datetime.datetime.now()
        finish = False
        for i_epoch in range(n_epoch):
            if finish:
                break

            with tqdm(
                    leave=False, total=len(self.train_loader),
                    ncols=80, ascii=True) as pbar:
                pbar.set_description(f"loss: {0.:.5e}")

                loss = 0.
                cumurative_n = 0
                self.optimizer.zero_grad()
                for i_loader, (list_input, list_dict_answer) in enumerate(
                        self.train_loader):
                    if finish:
                        break

                    list_part_input, list_part_dict_answer = self.part_data(
                        list_input, list_dict_answer, shuffle=True)
                    len_part = len(list_part_input)
                    max_iter_per_epoch = len(self.train_loader) * len_part
                    # print(f"{len_part = }")
                    # print(f"{max_iter_per_epoch = }")

                    for i_part in range(len_part):
                        list_dict_train_prediction = self.predict_batch(
                            [list_part_input[i_part]])

                        l, n = self.compute_loss(
                            [list_part_input[i_part]],
                            list_dict_train_prediction,
                            [list_part_dict_answer[i_part]])
                        if torch.isnan(l):
                            print(f"NaN detected. Break at: {i_epoch}")
                            finish = True
                            break
                        pbar.set_description(f"loss: {l:.5e}")
                        loss += l * n
                        cumurative_n += n
                        iteration = i_loader * len_part + i_part + 1
                        # print(f"{iteration = }")
                        if iteration % self.ml_setting.update_period == 0 \
                                or iteration == max_iter_per_epoch:
                            # print('update')
                            self.step(loss / cumurative_n)
                            loss = 0.
                            cumurative_n = 0
                            self.optimizer.zero_grad()
                    pbar.update(1)

            if finish:
                break
            train_loss = self.eval(self.train_loader, 'train')
            validation_loss = self.eval(self.validation_loader, 'validation')
            current_time = datetime.datetime.now()
            elapsed_time = (current_time - start_time).total_seconds()
            self.print_write_log(
                f"{i_epoch:5d}, {train_loss:.5e}, {validation_loss:.5e}, "
                f"{elapsed_time:8.1f}"
            )

            if validation_loss > self.ml_setting.factor_loss_upper_limit \
                    * self.min_loss:
                limit = self.ml_setting.factor_loss_upper_limit * self.min_loss
                print(
                    'Validation loss exceeded the limit: '
                    f"{validation_loss:.5e} > {limit:.5e}. "
                    f"Break at: {i_epoch}")
                finish = True
                break

            if torch.isnan(train_loss) or torch.isnan(validation_loss):
                print(f"NaN detected. Break at: {i_epoch}")
                finish = True
                break

            self.save_model_if_needed(i_epoch, validation_loss)

            # Early stopping
            if (i_epoch + 1) % self.ml_setting.stop_trigger_epoch == 0:
                finish = self._test_early_stopping(i_epoch)
                if finish:
                    print(f"Eearly stopping at: {i_epoch}")
                    break

        self.continue_training_if_needed()
        return

    def _test_early_stopping(self, i_epoch):
        if self.continue_count == self.best_continue:
            if i_epoch - self.best_epoch >= self.ml_setting.stop_trigger_epoch:
                self.early_stopping_count += 1
            else:
                self.early_stopping_count = 0
        else:
            self.early_stopping_count += 1

        return self.early_stopping_count >= self.ml_setting.patience

    def part_data(self, list_input, list_dict_answer, shuffle=False):
        ret_list_input = []
        ret_list_dict_answer = []

        for tuple_input, dict_answer in zip(list_input, list_dict_answer):

            torch_fv = tuple_input[0]
            dirichlet = tuple_input[1]['dirichlet']
            neumann = tuple_input[1]['neumann']
            periodic = tuple_input[1]['periodic']
            property_ = tuple_input[1]['property']

            start_indices = np.arange(
                0,
                self.timeseries_end_index - self.answer_end_index + 1,
                self.index_shift)
            if len(start_indices) == 0:
                start_indices = np.array([0])

            if shuffle:
                np.random.shuffle(start_indices)

            for start_index in start_indices:
                timeseries_indices = list(range(
                    start_index + self.answer_start_index,
                    start_index + self.answer_end_index,
                    self.factor_delta_t))
                initial = {
                    k: v[start_index]
                    for k, v in dict_answer.items()}
                ret_dict_answer = {
                    k: v[timeseries_indices]
                    for k, v in dict_answer.items()}
                ret_list_input.append(
                    (torch_fv, {
                        'initial': initial,
                        'dirichlet': dirichlet,
                        'neumann': neumann,
                        'periodic': periodic,
                        'property': property_,
                    }))
                ret_list_dict_answer.append(ret_dict_answer)

        # return [(list_input, list_dict_answer)]
        return ret_list_input, ret_list_dict_answer

    def print_write_log(self, string):
        print(string)
        with open(self.output_directory / 'log.csv', 'a') as f:
            f.write(string + '\n')
        return

    def continue_training_if_needed(self):
        if self.continue_count >= self.ml_setting.n_continue:
            return

        self.ml_setting.lr = self.ml_setting.lr * .5
        self.ml_setting.stop_trigger_epoch \
            = self.ml_setting.stop_trigger_epoch * 2
        self.continue_count += 1
        self.output_directory = pathlib.Path(
            str(self.original_output_directory)
            + f"_cont{self.continue_count}")
        print('===========')
        print(f"Continue: {self.continue_count}")

        self.initialize()
        self.train()
        return

    def eval(self, loader, name=''):
        nloss = 0.
        n_cell = 0
        with torch.no_grad():
            with tqdm(
                    leave=False, total=len(loader),
                    ncols=80, ascii=True) as pbar:
                for list_input, list_dict_answer in loader:
                    list_dict_prediction = self.predict_batch(
                        list_input, t_max=self.simulation_setting.eval_t_max,
                        evaluate=True)

                    # NOTE: Compute loss from the first step.
                    # Thus, loss value may differ from training
                    # when pushforward trick is applied.
                    l, n = self.compute_loss(
                        list_input, list_dict_prediction, list_dict_answer,
                        answer_start_index=self.factor_delta_t,
                        answer_end_index=self.eval_answer_end_index,
                        factor_delta_t=self.factor_delta_t,
                        only_last=True,  # Evaluate at the last step only
                    )

                    pbar.set_description(f"{name} eval: {l:.5e}")
                    pbar.update(1)
                    nloss += l * n
                    n_cell += n
        return nloss / n_cell

    def save_model_if_needed(self, i_epoch, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.best_state = copy.deepcopy(
                self.fluxgnn.to('cpu').state_dict())
            self.fluxgnn.to(self.device_id)
            torch.save(
                self.best_state, self.output_directory
                / f"model_{i_epoch}.pth")
            self.best_continue = self.continue_count
            self.best_epoch = i_epoch
        return

    def step(self, loss):
        loss.backward()
        if self.ml_setting.show_weights:
            print('--\nweights')
            for k, v in self.fluxgnn.dict_mlp.items():
                if hasattr(v, 'linears'):
                    print(f"{k}:")
                    for lin in v.linears:
                        print(f"{lin.weight[:4, :4]}")
                    print('--')

        if self.ml_setting.clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.fluxgnn.parameters(), self.clip)

        if self.ml_setting.show_grads:
            print('--\ngrads')
            for k, v in self.fluxgnn.dict_mlp.items():
                if hasattr(v, 'linears'):
                    if v.linears[0].weight.grad is not None:
                        print(f"{k}:")
                        for lin in v.linears:
                            print(f"{lin.weight.grad[:3, :3]}")
                        print('--')

        self.optimizer.step()
        return

    def compute_loss(
            self, list_input, list_dict_prediction, list_dict_answer,
            answer_start_index=None,
            answer_end_index=None,
            factor_delta_t=None,
            only_last=False):
        loss = 0.
        n_cell = 0
        for elem_input, dict_prediction, dict_answer in zip(
                list_input, list_dict_prediction, list_dict_answer):
            loss += self.compute_single_loss(
                elem_input[0], dict_prediction, dict_answer,
                answer_start_index=answer_start_index,
                answer_end_index=answer_end_index,
                factor_delta_t=factor_delta_t,
                only_last=only_last,
            )
            n_cell += len(elem_input[0].cell_pos)
        return loss, n_cell

    def compute_single_loss(
            self, torch_fv, dict_prediction, dict_answer,
            answer_start_index=None,
            answer_end_index=None,
            factor_delta_t=None,
            only_last=False):
        return self.compute_single_loss_function(
            torch_fv, dict_prediction, self.convert(dict_answer),
            answer_start_index=answer_start_index,
            answer_end_index=answer_end_index,
            factor_delta_t=factor_delta_t,
            only_last=only_last,
        )

    def compute_single_loss_dict(
            self, torch_fv, dict_prediction, dict_answer,
            answer_start_index=None,
            answer_end_index=None,
            factor_delta_t=None,
            only_last=False):
        if answer_start_index is None:
            answer_start_index = self.answer_start_index
        if answer_end_index is None:
            answer_end_index = self.answer_end_index
        if factor_delta_t is None:
            factor_delta_t = self.factor_delta_t

        dict_loss = {
            k: self.compute_loss_variable(
                self.dict_loss_function[k],
                torch_fv, dict_prediction[k], dict_answer[k],
                answer_start_index=answer_start_index,
                answer_end_index=answer_end_index,
                factor_delta_t=factor_delta_t,
                only_last=only_last)
            for k in dict_prediction.keys()}
        # for k, v in dict_loss.items():
        #     print(f"{k}: {v}")
        return torch.sum(torch.stack([
            dict_loss[k] * self.ml_setting.dict_loss_weight[k]
            * self.dict_answer_scales[k]
            for k in dict_prediction.keys()
        ]))

    def compute_loss_variable(
            self, loss_function, torch_fv, pred, ans, *,
            answer_start_index=None, answer_end_index=None,
            factor_delta_t=None, only_last=False):
        if len(pred) == len(ans):
            extracted_answer = ans.to(pred.device)
        else:
            extracted_answer = ans[
                answer_start_index:answer_end_index:factor_delta_t].to(
                    pred.device)
        if only_last:
            n_bundle = self.model_setting.n_bundle
            return loss_function(
                torch_fv, pred[-n_bundle:], extracted_answer[-n_bundle:])
        else:
            return loss_function(torch_fv, pred, extracted_answer)

    def predict_batch(self, batch, t_max=None, evaluate=False):
        return [
            self.predict_sample(b, t_max=t_max, evaluate=evaluate)[0]
            for b in batch]

    def predict_sample(
            self, sample, write=False, output_directory=None, evaluate=False,
            t_max=None, delta_t=None):
        if t_max is None:
            t_max = self.simulation_setting.t_max

        if delta_t is None:
            delta_t = self.simulation_setting.delta_t

        t0 = datetime.datetime.now()

        res = self.predict_sample_function(
            self.convert(sample), t_max=t_max, delta_t=delta_t,
            write=write, output_directory=output_directory, evaluate=evaluate)

        t1 = datetime.datetime.now()
        time = (t1 - t0).total_seconds()

        return res, time

    def predict_sample_convection_diffusion(
            self, sample, t_max, delta_t,
            write=False, output_directory=None, evaluate=False):
        torch_fv = sample[0]
        cell_initial_phi = sample[1]['initial']['phi']
        facet_dirichlet_phi = sample[1]['dirichlet']['phi']
        facet_neumann_phi = sample[1]['neumann']['phi']
        periodic = sample[1]['periodic']

        facet_u = sample[1]['property']['u']
        facet_diffusion = sample[1]['property']['diffusion']

        if self.ml:
            return self.fluxgnn.solve_convection_diffusion(
                t_max, delta_t,
                torch_fv,
                cell_initial_phi=cell_initial_phi,
                facet_velocity=facet_u, facet_diffusion=facet_diffusion,
                facet_dirichlet=facet_dirichlet_phi,
                facet_neumann=facet_neumann_phi,
                periodic=periodic,
                write=write, output_directory=output_directory,
                evaluate=evaluate)
        else:
            torch_fv.trainable = False
            return torch_fv.solve_convection_diffusion(
                t_max, delta_t,
                cell_initial_phi=cell_initial_phi,
                facet_velocity=facet_u, facet_diffusion=facet_diffusion,
                facet_dirichlet=facet_dirichlet_phi,
                facet_neumann=facet_neumann_phi,
                periodic=periodic,
                write=write, output_directory=output_directory)

    def predict_sample_navier_stokes(
            self, sample, t_max, delta_t,
            write=False, output_directory=None, evaluate=False):
        torch_fv = sample[0]
        cell_initial_u = sample[1]['initial']['u']
        cell_initial_p = sample[1]['initial']['p']
        facet_dirichlet_u = sample[1]['dirichlet']['u']
        facet_dirichlet_p = sample[1]['dirichlet']['p']
        facet_neumann_u = sample[1]['neumann']['u']
        facet_neumann_p = sample[1]['neumann']['p']
        global_nu = sample[1]['property']['nu']

        if self.ml:
            return self.fluxgnn.solve_navier_stokes(
                t_max, delta_t,
                torch_fv,
                cell_initial_u=cell_initial_u,
                cell_initial_p=cell_initial_p,
                global_nu=global_nu,
                facet_dirichlet_u=facet_dirichlet_u,
                facet_neumann_u=facet_neumann_u,
                facet_dirichlet_p=facet_dirichlet_p,
                facet_neumann_p=facet_neumann_p,
                write=write, output_directory=output_directory,
                evaluate=evaluate)
        else:
            torch_fv.trainable = False
            return torch_fv.solve_navier_stokes(
                t_max, delta_t,
                cell_initial_u=cell_initial_u,
                cell_initial_p=cell_initial_p,
                global_nu=global_nu,
                facet_dirichlet_u=facet_dirichlet_u,
                facet_neumann_u=facet_neumann_u,
                facet_dirichlet_p=facet_dirichlet_p,
                facet_neumann_p=facet_neumann_p,
                write=write, output_directory=output_directory)

    def predict_sample_mixture(
            self, sample, t_max, delta_t,
            write=False, output_directory=None, evaluate=False):
        torch_fv = sample[0]
        if self.prepost_scale:
            u_s = self.scale_for_prepost['u']
            p_s = self.scale_for_prepost['p']
            alpha_s = self.scale_for_prepost['alpha']

            cell_initial_u = sample[1]['initial']['u'] / u_s
            cell_initial_p = sample[1]['initial']['p'] / p_s
            cell_initial_alpha = sample[1]['initial']['alpha'] / alpha_s
            facet_dirichlet_u = sample[1]['dirichlet']['u'] / u_s
            facet_dirichlet_p = sample[1]['dirichlet']['p'] / p_s
            facet_dirichlet_alpha = sample[1]['dirichlet']['alpha'] / alpha_s
            facet_neumann_u = sample[1]['neumann']['u'] / u_s
            facet_neumann_p = sample[1]['neumann']['p'] / p_s
            facet_neumann_alpha = sample[1]['neumann']['alpha'] / alpha_s
        else:
            cell_initial_u = sample[1]['initial']['u']
            cell_initial_p = sample[1]['initial']['p']
            cell_initial_alpha = sample[1]['initial']['alpha']
            facet_dirichlet_u = sample[1]['dirichlet']['u']
            facet_dirichlet_p = sample[1]['dirichlet']['p']
            facet_dirichlet_alpha = sample[1]['dirichlet']['alpha']
            facet_neumann_u = sample[1]['neumann']['u']
            facet_neumann_p = sample[1]['neumann']['p']
            facet_neumann_alpha = sample[1]['neumann']['alpha']

        global_rho_solute = sample[1]['property']['rho_solute']
        global_rho_solvent = sample[1]['property']['rho_solvent']
        global_gravity = sample[1]['property']['gravity']
        time_scale_factor = float(np.squeeze(
            sample[1]['property']['time_scale_factor'].cpu().numpy()))

        if self.model_setting.property_for_ml:
            nu = .1
            global_diffusion_alpha = .1
            global_nu_solute = util.torch_tensor(np.array([nu])[None, :]).to(
                self.device_id)
            global_nu_solvent = util.torch_tensor(np.array([nu])[None, :]).to(
                self.device_id)
            facet_diffusion_alpha = util.torch_tensor(
                global_diffusion_alpha
                * np.ones((len(torch_fv.fv_data.facet_pos), 1))).to(
                    self.device_id)
        else:
            global_nu_solute = sample[1]['property']['nu_solute']
            global_nu_solvent = sample[1]['property']['nu_solvent']
            facet_diffusion_alpha = sample[1]['property']['diffusion_alpha']

        if self.ml:
            dict_results = self.fluxgnn.solve_mixture(
                t_max * time_scale_factor, delta_t * time_scale_factor,
                torch_fv,
                cell_initial_u=cell_initial_u,
                cell_initial_p=cell_initial_p,
                cell_initial_alpha=cell_initial_alpha,
                global_nu_solute=global_nu_solute,
                global_nu_solvent=global_nu_solvent,
                global_rho_solute=global_rho_solute,
                global_rho_solvent=global_rho_solvent,
                facet_diffusion_alpha=facet_diffusion_alpha,
                global_gravity=global_gravity,
                facet_dirichlet_u=facet_dirichlet_u,
                facet_neumann_u=facet_neumann_u,
                facet_dirichlet_p=facet_dirichlet_p,
                facet_neumann_p=facet_neumann_p,
                facet_dirichlet_alpha=facet_dirichlet_alpha,
                facet_neumann_alpha=facet_neumann_alpha,
                write=write, output_directory=output_directory,
                evaluate=evaluate)
            if self.prepost_scale:
                dict_results['u'] = dict_results['u'] * u_s
                dict_results['p'] = dict_results['p'] * p_s
                dict_results['alpha'] = dict_results['alpha'] * alpha_s
        else:
            torch_fv.trainable = False
            dict_results = torch_fv.solve_mixture(
                t_max * time_scale_factor, delta_t * time_scale_factor,
                cell_initial_u=cell_initial_u,
                cell_initial_p=cell_initial_p,
                cell_initial_alpha=cell_initial_alpha,
                global_nu_solute=global_nu_solute,
                global_nu_solvent=global_nu_solvent,
                global_rho_solute=global_rho_solute,
                global_rho_solvent=global_rho_solvent,
                facet_diffusion_alpha=facet_diffusion_alpha,
                global_gravity=global_gravity,
                facet_dirichlet_u=facet_dirichlet_u,
                facet_neumann_u=facet_neumann_u,
                facet_dirichlet_p=facet_dirichlet_p,
                facet_neumann_p=facet_neumann_p,
                facet_dirichlet_alpha=facet_dirichlet_alpha,
                facet_neumann_alpha=facet_neumann_alpha,
                write=write, output_directory=output_directory)

        return dict_results

    def convert(self, x):
        if isinstance(x, dict):
            return {k: self.convert(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return [self.convert(_) for _ in x]
        else:
            return x.to(self.device_id)

    def validate_physics(
            self, loader=None, raise_error=True, output_directory_base=None):
        self.load_model_if_needed()

        if loader is None:
            loader = self.validation_loader
        if loader.batch_size != 1:
            raise ValueError(
                'Batch size for validate_physics should be 1 '
                f"(given: {loader.batch_size}")

        list_n_cell = []
        list_dict_rotational_symmetry_error = []
        list_dict_scaling_symmetry_error = []
        list_dict_conservative_error = []
        for (list_input, list_dict_answer), data_directory in zip(
                loader, loader.dataset.data_directories):
            if output_directory_base is not None:
                name = '/'.join(data_directory.parts[-3:])
                output_directory = output_directory_base / name
                rotational_output_directory = output_directory / 'rotation'
                scaling_output_directory = output_directory / 'scaling'
            else:
                rotational_output_directory = None
                scaling_output_directory = None

            tfv, dict_input = list_input[0]
            dict_answer = list_dict_answer[0]
            dict_prediction = self.predict_sample(
                (tfv, dict_input), evaluate=True)[0]

            dict_rotational_symmetry_error = self.validate_rotational_symmetry(
                tfv, self.convert(dict_input), self.convert(dict_prediction),
                dict_answer, raise_error=raise_error,
                output_directory=rotational_output_directory)
            dict_scaling_symmetry_error = self.validate_scaling_symmetry(
                tfv, self.convert(dict_input), self.convert(dict_prediction),
                dict_answer, raise_error=raise_error,
                output_directory=scaling_output_directory)
            dict_conservative_error, _ = self.validate_conservation(
                tfv, self.convert(dict_input), self.convert(dict_prediction),
                dict_answer, raise_error=raise_error)

            list_n_cell.append(tfv.n_cell)
            list_dict_rotational_symmetry_error.append(
                dict_rotational_symmetry_error)
            list_dict_scaling_symmetry_error.append(
                dict_scaling_symmetry_error)
            list_dict_conservative_error.append(dict_conservative_error)

        dict_summarized_rotational_symmetry_error \
            = self.summarize_list_dict_loss(
                list_n_cell, list_dict_rotational_symmetry_error)
        dict_summarized_scaling_symmetry_error \
            = self.summarize_list_dict_loss(
                list_n_cell, list_dict_scaling_symmetry_error)
        dict_summarized_conservative_error = self.summarize_list_dict_loss(
            list_n_cell, list_dict_conservative_error)

        self.print_dict_loss(
            'Rotation', dict_summarized_rotational_symmetry_error)
        self.print_dict_loss(
            'Scaling', dict_summarized_scaling_symmetry_error)
        self.print_dict_loss(
            'Conservation', dict_summarized_conservative_error)

        return

    def summarize_list_dict_loss(self, list_n_cell, list_dict_loss):
        keys = list(list_dict_loss[0].keys())
        dict_cummurated_loss = {k: 0. for k in keys}
        total_n_cell = 0
        for n_cell, dict_loss in zip(list_n_cell, list_dict_loss):
            for k, v in dict_loss.items():
                dict_cummurated_loss[k] += n_cell * v
            total_n_cell += n_cell

        return {k: v / total_n_cell for k, v in dict_cummurated_loss.items()}

    def print_dict_loss(self, name, dict_loss):
        if len(dict_loss) == 0:
            return

        print(f"--\n{name}:")
        for k, v in dict_loss.items():
            print(f"  {k}: {v:.8e}")
        return

    def validate_rotational_symmetry(
            self, tfv, dict_input, dict_prediction, dict_answer,
            raise_error=False, output_directory=None):
        rotated_fem_data, _rotation_matrix = util.rotate_fem_data(
            tfv.fv_data.fem_data)
        rotation_matrix = torch_tensor(_rotation_matrix, device=tfv.device)
        copied_tfv = copy.deepcopy(tfv).to('cpu')
        options = vars(copied_tfv)
        options.pop('fv_data')
        rotated_tfv = self.torch_fv_constructor(
            rotated_fem_data, **options).to(self.device_id)
        dict_rotated_input = self.rotate_dict_variable(
            rotation_matrix, dict_input)
        dict_rotated_prediction = self.rotate_dict_variable(
            rotation_matrix, dict_prediction)
        with torch.no_grad():
            dict_prediction_w_rotated_input = self.predict_sample(
                (rotated_tfv, dict_rotated_input), evaluate=True)[0]

        if output_directory is not None:
            self.write(
                output_directory, rotated_tfv,
                dict_rotated_input,
                dict_prediction_w_rotated_input, dict_rotated_prediction,
                extract_answer_time_series=False)
            np.savetxt(
                output_directory / 'rotation_matrix.txt', _rotation_matrix)

        return_dict = {}
        for k in dict_rotated_prediction.keys():
            loss = self.relative_l2(
                rotated_tfv,
                dict_prediction_w_rotated_input[k],
                dict_rotated_prediction[k]).cpu().detach().numpy()
            if raise_error:
                print(k)
                np.testing.assert_almost_equal(
                    loss, 0, decimal=self.symmetry_decimal)
            return_dict[k] = loss

        return return_dict

    def rotate_dict_variable(self, rotation_matrix, dict_variables):
        return_dict = {}
        for k, v in dict_variables.items():
            if k == 'neumann':
                return_dict[k] = self.rotate_neumann(rotation_matrix, v)
                continue
            if k == 'periodic':
                return_dict[k] = v
                continue
            if isinstance(v, dict):
                return_dict[k] = self.rotate_dict_variable(rotation_matrix, v)
                continue

            if k in self.RANK0_NAMES:
                return_dict[k] = v
            elif k in self.RANK1_NAMES:
                return_dict[k] = torch.einsum(
                    'pqa,...qb->...pb', rotation_matrix, v)
            else:
                raise ValueError(f"Unexpected variable name: {k}")

        return return_dict

    def rotate_neumann(self, rotation_matrix, dict_neumann_variables):
        return_dict = {}
        # Neumann tensor's rank = original variable's rank + 1
        for k, v in dict_neumann_variables.items():
            if k in self.RANK0_NAMES:
                return_dict[k] = torch.einsum(
                    'pqa,...qb->...pb', rotation_matrix, v)
            elif k in self.RANK1_NAMES:
                return_dict[k] = torch.einsum(
                    'pqa,rsa,...qsb->...prb',
                    rotation_matrix, rotation_matrix, v)
            else:
                raise ValueError(f"Unexpected variable name: {k}")

        return return_dict

    def validate_scaling_symmetry(
            self, tfv, dict_input, dict_prediction, dict_answer,
            raise_error=False, output_directory=None):
        # Scale factors in range [.5, 2.]
        space_scale_factor = np.random.rand() * 1.5 + .5
        time_scale_factor = np.random.rand() * 1.5 + .5
        mass_scale_factor = np.random.rand() * 1.5 + .5

        scaled_fem_data, _ = util.scale_fem_data(
            tfv.fv_data.fem_data, scale_factor=space_scale_factor)
        copied_tfv = copy.deepcopy(tfv).to('cpu')
        options = vars(copied_tfv)
        options.pop('fv_data')
        scaled_tfv = self.torch_fv_constructor(
            scaled_fem_data, **options).to(self.device_id)
        dict_scaled_input = self.scale_dict_variable(
            space_scale_factor, time_scale_factor, mass_scale_factor,
            dict_input)
        dict_scaled_prediction = self.scale_dict_variable(
            space_scale_factor, time_scale_factor, mass_scale_factor,
            dict_prediction)
        with torch.no_grad():
            dict_prediction_w_scaled_input = self.predict_sample(
                (scaled_tfv, dict_scaled_input), evaluate=True,
                t_max=self.simulation_setting.t_max * time_scale_factor,
                delta_t=self.simulation_setting.delta_t * time_scale_factor)[0]

        if output_directory is not None:
            self.write(
                output_directory, scaled_tfv,
                dict_scaled_input,
                dict_prediction_w_scaled_input, dict_scaled_prediction,
                extract_answer_time_series=False)

        return_dict = {}
        for k in dict_scaled_prediction.keys():
            loss = self.relative_l2(
                scaled_tfv,
                dict_prediction_w_scaled_input[k],
                dict_scaled_prediction[k]).cpu().detach().numpy()
            if raise_error:
                print(k)
                np.testing.assert_almost_equal(
                    loss, 0, decimal=self.symmetry_decimal)
            return_dict[k] = loss

        return return_dict

    def scale_dict_variable(
            self, space_scale_factor, time_scale_factor, mass_scale_factor,
            dict_variables, mode=None):
        if mode is None:
            mode = self.mode

        return_dict = {}
        for k, v in dict_variables.items():
            if k == 'periodic':
                return_dict[k] = v
                continue
            if isinstance(v, dict):
                return_dict[k] = self.scale_dict_variable(
                    space_scale_factor, time_scale_factor, mass_scale_factor,
                    v)
                continue

            if k in self.DICT_MODE2DIM[mode]:
                d = self.DICT_MODE2DIM[mode][k]
                return_dict[k] = v \
                    * space_scale_factor**d['length'] \
                    * time_scale_factor**d['time'] \
                    * mass_scale_factor**d['mass']
            else:
                raise ValueError(f"Unexpected variable name: {k}")

        return return_dict

    def validate_conservation(
            self, tfv, dict_input, dict_prediction, dict_answer,
            raise_error=False):
        mse_dict = {}
        ts_dict = {}
        for conservative_name in self.CONSERVATIVE_NAMES:
            if conservative_name not in dict_input['initial']:
                continue

            initial_total_integral = tfv.compute_total_mean(
                dict_input['initial'][conservative_name].to(self.device_id)
            ).cpu().detach().numpy()
            if abs(initial_total_integral) < 1e-5:
                scale = 1.
            else:
                scale = 1 / initial_total_integral
            resulting_total_integral = np.array([
                tfv.compute_total_mean(c).cpu().detach().numpy()
                for c in dict_prediction[conservative_name]])

            ts = (resulting_total_integral - initial_total_integral) \
                * scale

            loss = np.mean(ts**2)**.5

            if raise_error:
                np.testing.assert_almost_equal(
                    loss, 0., decimal=self.conservative_decimal)
            mse_dict[conservative_name] = loss
            ts_dict[conservative_name] = ts

        return mse_dict, ts_dict

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def set_device(self, device_id):
        if isinstance(device_id, (str, torch.device)):
            self.device_id = device_id
            return

        if not isinstance(device_id, int):
            raise ValueError(f"Unexpected device_id type: {device_id}")

        if not torch.cuda.is_available():
            if device_id >= 0:
                print(
                    f"Use CPU because GPU not available (given: {device_id})")
            device_id = -1

        if device_id < 0:
            self.device_id = torch.device('cpu')
        else:
            self.device_id = torch.device(device_id)

        print(f"Using device: {self.device_id}")
        return

    def send(self, dict_tensor, device):
        if dict_tensor is None:
            return {}
        return {k: v.to(device) for k, v in dict_tensor.items()}

    def mse(self, torch_fv, y, t):
        return torch_fv.compute_total_mean((y - t)**2, time_series=True)

    def simple_mse(self, torch_fv, y, t):
        # Simple mse without weight
        return torch.mean((y - t)**2)

    def simple_std_error(self, torch_fv, y, t):
        # Simple standard error without weight
        return torch.std((y - t)**2).cpu() \
            / np.sqrt(t.cpu().detach().numpy().size)

    def rmse(self, torch_fv, y, t):
        return self.mse(torch_fv, y, t)**.5

    def relative_l2(self, torch_fv, y, t):
        return torch_fv.compute_total_integral((y - t)**2, time_series=True) \
            / (1e-5 + torch_fv.compute_total_integral(
                (t - torch_fv.compute_total_mean(t, time_series=True))**2,
                time_series=True))

    def cosine_distance(self, torch_fv, y, t):
        t_mean = torch_fv.compute_total_mean(t, time_series=True)
        return 1 - torch_fv.compute_total_integral(
            (y - t_mean) * (t - t_mean), time_series=True) / (1e-5 + (
                torch_fv.compute_total_integral(
                    (y - t_mean)**2, time_series=True)
                * torch_fv.compute_total_integral(
                    (t - t_mean)**2, time_series=True))**.5)

    def squared_cosine_distance(self, torch_fv, y, t):
        raise ValueError('Not working when cos < 0')
        t_mean = torch_fv.compute_total_mean(t)
        return 1 - torch_fv.compute_total_integral(
            (y - t_mean) * (t - t_mean), time_series=True)**2 \
            / (
                1e-5
                + torch_fv.compute_total_integral(
                    (y - t_mean)**2, time_series=True)
                * torch_fv.compute_total_integral(
                    (t - t_mean)**2, time_series=True)
        )

    def load_model_if_needed(self):
        if self.best_state is not None:
            self.fluxgnn.load_state_dict(self.best_state)
            print(
                'Best state loaded at:\n'
                f"- continue: {self.best_continue}\n"
                f"- epoch: {self.best_epoch}")
        return

    def load_model_from_directory(self, model_directory):
        if (model_directory / 'model.pth').is_file():
            snapshot = model_directory / 'model.pth'
        else:
            df = pd.read_csv(
                model_directory / 'log.csv', header=0, index_col=None,
                skipinitialspace=True)
            self.best_epoch = df['epoch'].iloc[df['validation_loss'].idxmin()]
            snapshot = model_directory / f"model_{self.best_epoch}.pth"
        print(f"Load pretrained model: {snapshot}")

        self.best_state = torch.load(snapshot, map_location=self.device_id)
        self.fluxgnn.load_state_dict(self.best_state)
        return

    def predict_and_write(
            self, data_directories, output_directory_base,
            validate_physics=True, data_name=None):
        self.load_model_if_needed()
        if data_name is None:
            data_name = 'test'
        dataset = TorchFVDataset(
            self.dataset_name, data_directories,
            self.torch_fv_constructor, self.fv_solver_setting_dict,
            start_index=self.timeseries_start_index,
            end_index=self.timeseries_start_index+self.timeseries_end_index,
            dataset_file_name=self.dataset_file_prefix + data_name,
            force_load=self.force_load,
            diffusion_alpha=self.diffusion_alpha,
            save=self.save_dataset)
        if len(dataset) == 0:
            print('No test data given. Skip predict_and_write.')
            return

        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=collate_function,
            batch_size=1, shuffle=False, num_workers=0)

        if validate_physics:
            self.validate_physics(
                loader=data_loader, raise_error=False,
                output_directory_base=output_directory_base)

        dict_tmp_total_prediction = {}
        dict_tmp_total_answer = {}
        dict_tmp_total_nodal_prediction = {}
        dict_tmp_total_nodal_answer = {}
        dict_total_conservative_error = {}
        dict_tmp_ts_conservation = []
        dict_answer_scale = {}
        list_preprocess_time = []
        list_prediction_time = []
        for data_directory, (list_input, list_dict_answer) in zip(
                data_directories, data_loader):
            name = '/'.join(data_directory.parts[-3:])
            output_directory = output_directory_base / name

            torch_fv = list_input[0][0]
            dict_input = list_input[0][1]
            dict_answer = list_dict_answer[0]

            with torch.no_grad():
                dict_prediction, time = self.predict_sample(
                    list_input[0], write=False, output_directory=None,
                    evaluate=True, t_max=self.simulation_setting.test_t_max)
            list_preprocess_time.append(torch_fv.fv_data.preprocess_time)
            list_prediction_time.append(time)

            dict_nodal_prediction = self.to_nodal(
                torch_fv, dict_input, dict_prediction)
            dict_nodal_answer = self.to_nodal(
                torch_fv, dict_input, dict_answer)

            dict_stats = self.analyse_stats(
                torch_fv, dict_prediction, dict_answer, rescale=True,
                dict_nodal_prediction=dict_nodal_prediction,
                dict_nodal_answer=dict_nodal_answer)
            dict_conservative_error, dict_ts_conservation \
                = self.validate_conservation(
                    torch_fv, self.convert(dict_input),
                    self.convert(dict_prediction),
                    dict_answer, raise_error=False)

            if len(dict_tmp_total_prediction) == 0:
                dict_tmp_total_prediction = {
                    k: [v] for k, v in dict_prediction.items()}
                dict_tmp_total_answer = {
                    k: [v] for k, v in dict_answer.items()}

                dict_tmp_total_nodal_prediction = {
                    k: [v] for k, v in dict_nodal_prediction.items()}
                dict_tmp_total_nodal_answer = {
                    k: [v] for k, v in dict_nodal_answer.items()}

                dict_answer_scale = {
                    k: [v[2]] for k, v in dict_stats.items()}
                dict_total_conservative_error = {
                    k: [v] for k, v in dict_conservative_error.items()}
                dict_tmp_ts_conservation = {
                    k: [v] for k, v in dict_ts_conservation.items()}

            else:
                for k in dict_tmp_total_prediction.keys():
                    dict_tmp_total_prediction[k].append(dict_prediction[k])
                    dict_tmp_total_answer[k].append(dict_answer[k])

                    dict_tmp_total_nodal_prediction[k].append(
                        dict_nodal_prediction[k])
                    dict_tmp_total_nodal_answer[k].append(
                        dict_nodal_answer[k])

                    dict_answer_scale[k].append(dict_stats[k][2])

                for k in dict_total_conservative_error.keys():
                    dict_total_conservative_error[k].append(
                        dict_conservative_error[k])
                    dict_tmp_ts_conservation[k].append(
                        dict_ts_conservation[k])

            print('--')
            print(f" in: {data_directory}")
            print(f"out: {output_directory}")
            print('Scaled MSE:')
            for k, v in dict_stats.items():
                mse = v[0]
                std_error = v[1]
                print(f"  {k.rjust(10)}: {mse:.5e} +/- {std_error:.5e}")
            print('Conservation error:')
            for k, v in dict_conservative_error.items():
                print(f"  {k.rjust(10)}: {v:.5e}")
            print(f"Prediction time: {time}")

            self.write(
                output_directory, torch_fv,
                dict_input, dict_prediction, dict_answer,
                dict_nodal_prediction=dict_nodal_prediction,
                dict_nodal_answer=dict_nodal_answer)

        dict_total_prediction = {
            k: torch.cat(
                [_v / _s for _v, _s in zip(v, dict_answer_scale[k])],
                dim=1)
            for k, v in dict_tmp_total_prediction.items()}
        dict_total_answer = {
            k: torch.cat(
                [
                    _v.cpu() / _s.cpu()
                    for _v, _s in zip(v, dict_answer_scale[k])],
                dim=1)
            for k, v in dict_tmp_total_answer.items()}
        dict_total_nodal_prediction = {
            k: torch.cat(
                [
                    _v.cpu() / _s.cpu()
                    for _v, _s in zip(v, dict_answer_scale[k])],
                dim=1)
            for k, v in dict_tmp_total_nodal_prediction.items()}
        dict_total_nodal_answer = {
            k: torch.cat(
                [
                    _v.cpu() / _s.cpu()
                    for _v, _s in zip(v, dict_answer_scale[k])],
                dim=1)
            for k, v in dict_tmp_total_nodal_answer.items()}
        dict_total_stats = self.analyse_stats(
            torch_fv, dict_total_prediction, dict_total_answer,
            dict_nodal_prediction=dict_total_nodal_prediction,
            dict_nodal_answer=dict_total_nodal_answer,
            rescale=False)
        dict_time_stats = self.analyse_time(
            np.array(list_preprocess_time), np.array(list_prediction_time))

        stats_file = output_directory_base / 'global_stats.csv'
        with open(stats_file, 'w') as f:
            print('=========================================')
            print('Total scaled MSE:')
            for k, v in dict_total_stats.items():
                mse = v[0]
                std_error = v[1]
                print(f"  {k.rjust(10)}: {mse:.5e} +/- {std_error:.5e}")
                f.write(f"global_mse_{k},{mse}\n")
                f.write(f"global_std_error_{k},{std_error}\n")

            print('Total conservation error:')
            for k, v in dict_total_conservative_error.items():
                mean = np.mean(v)
                std_error = np.std(v) / len(v)**.5
                print(f"  {k.rjust(10)}: {mean:.5e} +/- {std_error:.5e}")
                f.write(f"global_mean_conservation_error_{k},{mean}\n")
                f.write(
                    f"global_std_error_conservation_error_{k},{std_error}\n")

            total_time_mean = dict_time_stats['global_mean_total_time']
            total_time_std_error = dict_time_stats[
                'global_std_error_total_time']
            print(
                f"Mean time: {total_time_mean:.5e} +/- "
                f"{total_time_std_error:.5e}")
            for k, v in dict_time_stats.items():
                f.write(f"{k},{v}\n")
        print(f"Global stats written in: {stats_file}")

        for k, v in dict_tmp_ts_conservation.items():
            conservation_file = output_directory_base / f"conservation_{k}.csv"
            ts_conservation_error = np.mean(np.stack(v, axis=0)**2, axis=0)**.5
            delta_t = self.simulation_setting.delta_t
            with open(conservation_file, 'w') as f:
                for i, e in enumerate(ts_conservation_error, 1):
                    f.write(f"{i * delta_t:.5e},{e:.5e}\n")
            print(f"Conservation log written in: {conservation_file}")

        return

    def to_nodal(self, torch_fv, dict_input, dict_variable, time_series=True):
        dict_nodal_data = {}
        for k, v in dict_variable.items():
            if time_series:
                dict_nodal_data[k] = torch.stack([
                    self.to_nodal_core(
                        torch_fv, _v,
                        dict_input['dirichlet'][k],
                        dict_input['neumann'][k],
                        dict_input['periodic'])
                    for _v in v], dim=0)
            else:
                dict_nodal_data[k] = self.to_nodal_core(
                    torch_fv, v,
                    dict_input['dirichlet'][k],
                    dict_input['neumann'][k],
                    dict_input['periodic'])

        return dict_nodal_data

    def to_nodal_core(self, torch_fv, v, dirichlet, neumann, periodic):
        facet_v = torch_fv.compute_facet_value_center(v.to(torch_fv.device))
        facet_v = torch_fv.apply_boundary_facet(
            facet_v.to(torch_fv.device),
            dirichlet.to(torch_fv.device), neumann.to(torch_fv.device),
            periodic=periodic)
        stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            nodal_v = torch_tensor(
                torch_fv.fv_data.facet_fem_data.convert_elemental2nodal(
                    self.to_2d(facet_v), mode='mean'),
                device=v.device, dtype=v.dtype)
        sys.stdout = stdout
        return nodal_v

    def analyse_stats(
            self, torch_fv, dict_prediction, dict_answer, rescale,
            dict_nodal_prediction=None, dict_nodal_answer=None):
        if self.stats_mode == 'nodal':
            if dict_nodal_prediction is None or dict_nodal_answer is None:
                raise ValueError(
                    'Feed nodal variables when stats_mode == "nodal"')
            dict_prediction_eval = dict_nodal_prediction
            dict_answer_eval = dict_nodal_answer
        elif self.stats_mode == 'elemental':
            dict_prediction_eval = dict_prediction
            dict_answer_eval = dict_answer
        else:
            raise ValueError(f"Unexpected stats_mode: {self.stats_mode}")
        dict_stats = {
            k: self.compute_stats_variable(
                torch_fv, dict_prediction_eval[k], dict_answer_eval[k],
                answer_start_index=self.factor_delta_t,
                answer_end_index=self.test_answer_end_index,
                factor_delta_t=self.factor_delta_t,
                rescale=rescale)
            for k in dict_prediction.keys()}
        return dict_stats

    def analyse_time(
            self, preprocess_times, prediction_times):
        n_sample = len(preprocess_times)

        preprocess_time_mean = np.mean(preprocess_times)
        preprocess_time_std_error = np.std(preprocess_times) \
            / np.sqrt(n_sample)

        prediction_time_mean = np.mean(prediction_times)
        prediction_time_std_error = np.std(prediction_times) \
            / np.sqrt(n_sample)

        total_times = preprocess_times + prediction_times
        total_time_mean = np.mean(total_times)
        total_time_std_error = np.std(total_times) / np.sqrt(n_sample)

        return {
            'global_mean_preprocess_time': preprocess_time_mean,
            'global_std_error_preprocess_time': preprocess_time_std_error,
            'global_mean_prediction_time': prediction_time_mean,
            'global_std_error_prediction_time': prediction_time_std_error,
            'global_mean_total_time': total_time_mean,
            'global_std_error_total_time': total_time_std_error,
        }

    def compute_stats_variable(
            self, torch_fv, pred, ans, *,
            answer_start_index=None, answer_end_index=None,
            factor_delta_t=None, rescale=True):
        if len(pred) == len(ans):
            extracted_answer = ans.to(pred.device)
        else:
            extracted_answer = ans[
                answer_start_index:answer_end_index:factor_delta_t].to(
                    pred.device)

        if rescale:
            answer_scale = torch.std(extracted_answer)
        else:
            answer_scale = 1.

        mse = self.simple_mse(torch_fv, pred, extracted_answer) \
            / answer_scale**2
        std_error = self.simple_std_error(torch_fv, pred, extracted_answer) \
            / answer_scale**2
        return mse, std_error, answer_scale

    def write(
            self, output_directory, torch_fv, dict_input,
            dict_prediction, dict_answer, extract_answer_time_series=True,
            dict_nodal_prediction=None,
            dict_nodal_answer=None):

        if dict_nodal_prediction is not None and dict_nodal_answer is not None:
            nodal = True
        else:
            nodal = False

        stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if extract_answer_time_series:
                dict_extracted_answer = {
                    k: v[self.factor_delta_t:self.test_answer_end_index:self
                         .factor_delta_t]
                    for k, v in dict_answer.items()}
            else:
                dict_extracted_answer = dict_answer

            if nodal:
                if extract_answer_time_series:
                    dict_extracted_nodal_answer = {
                        k: v[
                            self.factor_delta_t:self.test_answer_end_index:self
                            .factor_delta_t]
                        for k, v in dict_nodal_answer.items()}
                else:
                    dict_extracted_nodal_answer = dict_nodal_answer

            len_time = len(dict_prediction[list(dict_prediction.keys())[0]])
            fem_data = torch_fv.fv_data.fem_data

            if self.write_initial:
                if nodal:
                    nodal_initial = self.to_nodal(
                        torch_fv, dict_input, dict_input['initial'],
                        time_series=False)
                for k in dict_prediction.keys():
                    initial = dict_input['initial'][k].cpu().detach().numpy()
                    diff = initial - initial
                    fem_data.elemental_data.update_data(
                        fem_data.elements.ids, {
                            f"predicted_{k}": self.to_2d(initial),
                            f"answer_{k}": self.to_2d(initial),
                            f"diff_{k}": self.to_2d(diff),
                        }, allow_overwrite=True)

                    if nodal:
                        initial = nodal_initial[k].cpu().detach().numpy()
                        diff = initial - initial
                        fem_data.nodal_data.update_data(
                            fem_data.nodes.ids, {
                                f"predicted_{k}": self.to_2d(initial),
                                f"answer_{k}": self.to_2d(initial),
                                f"diff_{k}": self.to_2d(diff),
                            }, allow_overwrite=True)
                fem_data.write(
                    'vtu', output_directory / f"mesh.{0:08d}.vtu")

            for i_time in range(len_time):
                if self.plot:
                    plt.cla()
                for k in dict_prediction.keys():
                    pred = dict_prediction[k][i_time].cpu().detach().numpy()

                    ans = dict_extracted_answer[k][
                        i_time].cpu().detach().numpy()

                    diff = pred - ans
                    fem_data.elemental_data.update_data(
                        fem_data.elements.ids, {
                            f"predicted_{k}": self.to_2d(pred),
                            f"answer_{k}": self.to_2d(ans),
                            f"diff_{k}": self.to_2d(diff),
                        }, allow_overwrite=True)

                    if nodal:
                        pred = dict_nodal_prediction[k][
                            i_time].cpu().detach().numpy()

                        ans = dict_extracted_nodal_answer[k][
                            i_time].cpu().detach().numpy()

                        diff = pred - ans
                        fem_data.nodal_data.update_data(
                            fem_data.nodes.ids, {
                                f"predicted_{k}": self.to_2d(pred),
                                f"answer_{k}": self.to_2d(ans),
                                f"diff_{k}": self.to_2d(diff),
                            }, allow_overwrite=True)

                    if self.plot:
                        cell_x = torch_fv.fv_data.cell_pos[:, 0]
                        plt.plot(cell_x, ans, '-', label='answer')
                        plt.plot(cell_x, pred, '.', ms=3, label='pred')
                        plt.ylim(-1.5, 1.5)
                        plt.legend()
                        plt.pause(.1)
                fem_data.write(
                    'vtu', output_directory / f"mesh.{i_time+1:08d}.vtu")
        sys.stdout = stdout

        if self.plot:
            plt.show()
        return

    def to_2d(self, x):
        d = len(x.shape)
        if d == 1:
            return self.sanitize(x[:, None])
        elif d == 2:
            return self.sanitize(x)
        elif d == 3:
            if x.shape[-1] != 1:
                raise ValueError(f"Unexpected shape: {x.shape}")
            return self.sanitize(x[..., 0])
        else:
            raise ValueError(f"Unexpected shape: {x.shape}")

    def sanitize(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        x[np.abs(x) < self.EPSILON] = 0.
        return x


def run(
        settings_yaml, device_id=0, perform_train=False, model_directory=None,
        load_dataset=True, data_directories=None, output_directory=None,
        write_directory=None, **kwargs):
    dict_settings = siml.util.load_yaml(settings_yaml)

    # Overwrite setting
    first_time = True
    overwritten = False
    output_directory_additional_suffix = ''
    if 'test_t_max' in kwargs and kwargs['test_t_max'] is not None:
        if first_time:
            print('=============================')
        dict_settings['simulation']['test_t_max'] = kwargs['test_t_max']
        first_time = False
        overwritten = True
        print(f"Set test_t_max: {dict_settings['simulation']['test_t_max']}")

    if 'lr' in kwargs and kwargs['lr'] is not None:
        if first_time:
            print('=============================')
        dict_settings['ml']['lr'] = kwargs['lr']
        first_time = False
        overwritten = True
        print(f"Set lr: {dict_settings['ml']['lr']}")
        output_directory_additional_suffix \
            = output_directory_additional_suffix + f"_lr{kwargs['lr']:.5e}"

    if 'stop_trigger_epoch' in kwargs \
            and kwargs['stop_trigger_epoch'] is not None:
        if first_time:
            print('=============================')
        dict_settings['ml']['stop_trigger_epoch'] \
            = kwargs['stop_trigger_epoch']
        first_time = False
        overwritten = True
        print(f"Set stop trigger: {dict_settings['ml']['stop_trigger_epoch']}")
        output_directory_additional_suffix \
            = output_directory_additional_suffix \
            + f"_stop{kwargs['stop_trigger_epoch']}"

    if 'n_continue' in kwargs \
            and kwargs['n_continue'] is not None:
        if first_time:
            print('=============================')
        dict_settings['ml']['n_continue'] \
            = kwargs['n_continue']
        first_time = False
        overwritten = True
        print(f"Set # continue: {dict_settings['ml']['n_continue']}")
        output_directory_additional_suffix \
            = output_directory_additional_suffix \
            + f"_ncon{kwargs['n_continue']}"

    if 'threshold_p' in kwargs \
            and kwargs['threshold_p'] is not None:
        if first_time:
            print('=============================')
        dict_settings['solver']['threshold_p'] \
            = kwargs['threshold_p']
        first_time = False
        overwritten = True
        print(
            'Set convergence threshold for pressure: '
            f"{dict_settings['solver']['threshold_p']}")
        output_directory_additional_suffix \
            = output_directory_additional_suffix \
            + f"_threshold{kwargs['threshold_p']}"

    if 'print_period' in kwargs \
            and kwargs['print_period'] is not None:
        if first_time:
            print('=============================')
        dict_settings['solver']['print_period'] \
            = kwargs['print_period']
        first_time = False
        overwritten = True
        print(f"Set print period: {dict_settings['solver']['print_period']}")

    if 'print_period_p' in kwargs \
            and kwargs['print_period_p'] is not None:
        if first_time:
            print('=============================')
        dict_settings['solver']['print_period_p'] \
            = kwargs['print_period_p']
        first_time = False
        overwritten = True
        print(
            'Set pressure print period: '
            f"{dict_settings['solver']['print_period_p']}")

    if 'validate_physics' in kwargs \
            and kwargs['validate_physics'] is not None:
        if first_time:
            print('=============================')
        dict_settings['task']['validate_physics'] \
            = kwargs['validate_physics']
        first_time = False
        overwritten = True
        print(
            'Set validate physics: '
            f"{dict_settings['task']['validate_physics']}")

    if overwritten:
        print('=============================')

    output_directory_base = pathlib.Path(
        dict_settings['data']['output_directory_base'])
    dataset_file_prefix = dict_settings['data'].get('dataset_file_prefix', '')
    force_load = dict_settings['data'].get('force_load', False)
    save_dataset = dict_settings['data'].get('save_dataset', True)

    train_directories = [
        pathlib.Path(p) for p in dict_settings['data']['train']]
    validation_directories = [
        pathlib.Path(p) for p in dict_settings['data']['validation']]
    if data_directories is None:
        data_directories = [
            pathlib.Path(p) for p in dict_settings['data']['test']]
        data_name = None
    else:
        data_name = kwargs.get('data_name', None)
        if data_name is None:
            force_load = True
            save_dataset = False
        else:
            force_load = False
            save_dataset = True

    name = settings_yaml.stem
    if 'experiment' in dict_settings:
        dict_experiment = dict_settings['experiment']
    else:
        dict_experiment = {}
    experiment = Experiment(
        name, dict_settings['task']['dataset'],
        train_directories, validation_directories,
        simulation_setting_dict=dict_settings['simulation'],
        ml_setting_dict=dict_settings['ml'],
        model_setting_dict=dict_settings['model'],
        model_directory=model_directory,
        fv_solver_setting_dict=dict_settings['solver'],
        output_directory_base=output_directory_base,
        output_directory=output_directory,
        output_directory_additional_suffix=output_directory_additional_suffix,
        device_id=device_id,
        load_dataset=load_dataset,
        dataset_file_prefix=dataset_file_prefix, force_load=force_load,
        save_dataset=save_dataset,
        **dict_experiment,
        **kwargs,
    )

    if perform_train:
        experiment.output_directory.mkdir(parents=True, exist_ok=False)
        with open(experiment.output_directory / 'settings.yml', 'w') as f:
            yaml.dump(dict_settings, f)
        experiment.train()

    # if load_dataset:
    #     experiment.validate_physics(
    #         raise_error=dict_settings['task']['validate_physics'])

    if dict_settings['task']['predict_and_write']:
        if write_directory is None:
            write_directory = experiment.output_directory / 'prediction'
        write_directory.mkdir(parents=True, exist_ok=False)
        with open(write_directory / 'settings.yml', 'w') as f:
            yaml.dump(dict_settings, f)
        experiment.predict_and_write(
            data_directories, write_directory,
            validate_physics=dict_settings['task']['validate_physics'],
            data_name=data_name)

    return experiment


def train(settings_yaml, device_id=-1, model_directory=None, **kwargs):
    experiment = run(
        settings_yaml, device_id=device_id, perform_train=True,
        model_directory=model_directory, **kwargs)
    return experiment


def evaluate(
        model_directory, device_id=-1, data_directories=None, test_t_max=None,
        print_period=None, **kwargs):
    settings_yaml = pathlib.Path(
        re.sub(r'_cont\d+', '', str(model_directory))) / 'settings.yml'
    experiment = run(
        settings_yaml, device_id=device_id, perform_train=False,
        model_directory=model_directory, load_dataset=False,
        write_directory=model_directory
        / f"prediction/{siml.util.date_string()}",
        data_directories=data_directories,
        test_t_max=test_t_max, print_period=print_period, **kwargs)
    return experiment


def simulate(
        settings_yaml, device_id=-1, test_t_max=None, print_period=None,
        **kwargs):
    experiment = run(
        settings_yaml, device_id=device_id, perform_train=False,
        load_dataset=False, ml=False,
        test_t_max=test_t_max, print_period=print_period, **kwargs)
    return experiment
