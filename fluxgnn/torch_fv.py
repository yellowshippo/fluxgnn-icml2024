
from datetime import datetime
import os
import sys

import femio
import numpy as np
import siml
from siml.networks.sparse import mul
import torch

from fluxgnn.fv_data import FVData
from fluxgnn import util
from fluxgnn.util import torch_tensor, torch_sparse


BENCHMARK = False


class TorchFVSolver:

    EPSILON = 1e-8
    LIST_DENSE_NAMES = [
        'facet_pos',
        'facet_normal_vector',
        'facet_filter_boundary',
        'facet_area',
        'facet_relative_position',
        'facet_divergence_vector_minimum_correction',
        'facet_divergence_vector_over_relaxed',
        'facet_nonorthogonal_over_relaxed',
        'facet_normalized_dot',
        'facet_normalized_d',
        'facet_differential_d',
        'facet_inv_d_norm',
        'facet_length_scale',
        'cell_pos',
        'cell_volume',
        'cell_length_scale',
    ]
    LIST_SPARSE_NAMES = [
        'cf_inc',
        'cf_weighted_inc',
        'cf_signed_inc',
        'cf_normal_inc',
        'fc_positive_inc',
        'fc_negative_inc',
        'fc_inc',
        'fc_signed_inc',
        'fc_weighted_inc',
        'fc_weighted_positive_inc',
        'fc_weighted_negative_inc',
    ]

    @classmethod
    def from_vtu(cls, vtu_file_path, **kwargs):

        with open(os.devnull, 'w') as f:
            sys.stdout = f
            fem_data = femio.read_files('vtu', vtu_file_path)
        sys.stdout = sys.__stdout__

        return cls(fem_data, **kwargs)

    def __init__(
            self, fem_data, *,
            trainable=True,
            mode=None,
            diffusion_method='minimum_correction',
            time_evolution_method='explicit',
            n_time_repeat=None,
            n_time_factor=None,
            convergence_threshold=1e-3,
            skip_names=None,
            nonorthogonal_correction=False,
            upwind=True,
            ml_interpolation=None,
            clone_boundary=False,
            print_period=-1,
            write_mesh_period=-1,
            store_results_period=-1,
            center_data=False,
            debug=False,
            dtype=torch.float32,
            device=torch.device('cpu'),
            apply_setting=True,
            **kwargs):
        self.fv_data = FVData(fem_data)
        self.dtype = dtype
        self.device = device

        for dense_name in self.LIST_DENSE_NAMES:
            if 'filter' in dense_name:
                setattr(
                    self, dense_name,
                    torch_tensor(
                        getattr(self.fv_data, dense_name),
                        device=device, dtype=dtype)[..., 0])
            else:
                setattr(
                    self, dense_name,
                    torch_tensor(
                        getattr(self.fv_data, dense_name),
                        device=device, dtype=dtype))

        for sparse_name in self.LIST_SPARSE_NAMES:
            setattr(
                self, sparse_name,
                torch_sparse(
                    getattr(self.fv_data, sparse_name),
                    device=device, dtype=dtype))

        self.n_point = len(fem_data.nodes)
        self.n_facet = len(self.facet_pos)
        self.n_cell = len(self.cell_pos)

        if apply_setting:
            self.apply_setting(
                trainable=trainable,
                mode=mode,
                diffusion_method=diffusion_method,
                time_evolution_method=time_evolution_method,
                n_time_repeat=n_time_repeat,
                n_time_factor=n_time_factor,
                convergence_threshold=convergence_threshold,
                skip_names=skip_names,
                nonorthogonal_correction=nonorthogonal_correction,
                upwind=upwind,
                ml_interpolation=ml_interpolation,
                clone_boundary=clone_boundary,
                print_period=print_period,
                write_mesh_period=write_mesh_period,
                store_results_period=store_results_period,
                center_data=center_data,
                debug=debug,
                dtype=dtype,
                device=device,
                **kwargs)
        return

    def apply_setting(
            self,
            trainable=True,
            mode=None,
            diffusion_method='minimum_correction',
            time_evolution_method='explicit',
            n_time_repeat=None,
            n_time_factor=None,
            convergence_threshold=1e-3,
            skip_names=None,
            nonorthogonal_correction=False,
            upwind=True,
            ml_interpolation=None,
            clone_boundary=False,
            print_period=-1,
            write_mesh_period=-1,
            store_results_period=-1,
            center_data=False,
            debug=False,
            dtype=torch.float32,
            device=torch.device('cpu'),
            **kwargs):
        self.trainable = trainable
        if mode is None:
            self.mode = ''
        else:
            self.mode = mode

        if ml_interpolation is None:
            self.ml_interpolation = 'separate'  # or 'concatenate'
        else:
            self.ml_interpolation = ml_interpolation

        # Solver setting
        self.diffusion_method = diffusion_method
        self.nonorthogonal_correction = nonorthogonal_correction
        self.upwind = upwind
        self.rhie_chow = False
        self.time_evolution_method = time_evolution_method
        self.convergence_threshold = convergence_threshold
        self.debug = debug
        self.clone_boundary = clone_boundary
        if skip_names is None:
            self.skip_names = []
        else:
            self.skip_names = skip_names

        if n_time_factor is None:
            if n_time_repeat is None:
                self.n_time_repeat = 1
            else:
                self.n_time_repeat = n_time_repeat
        else:
            if n_time_repeat is not None:
                raise ValueError(
                    'n_time_repeat and n_time_factor cannot be fed '
                    'in the same time')
            self.n_time_repeat = max(int(n_time_factor * self.n_cell), 2)

        if self.time_evolution_method == 'explicit':
            self.time_evolution_function = self.step_explicit
        elif self.time_evolution_method == 'bb':
            self.time_evolution_function = self.step_bb
        else:
            raise ValueError(
                'Unexpected time_evolution_method: '
                f"{self.time_evolution_method}")

        # Simulation setting
        self.center_data = center_data

        # Utility setting
        self.print_period = print_period
        self.write_mesh_period = write_mesh_period
        self.store_results_period = store_results_period

        return

    def step(
            self, step_core_function, updatable_dict, *,
            n_loop=None, skip_names=None):
        """
        Compute step u^{t+1} <- F[u^{t}].

        Args:
            step_core_function: callable
                Function to compute dict of du / dt * delta_t.
            updatable_dict: dict[str, torch.Tensor]
                Input data.
            n_loop: int
                The number of loops.
            skip_names: list[str]
                List of variable names to be skipped for convergence threshold.

        Returns:
            updated_dict: dict[str, torch.Tensor]
        """
        if n_loop is None:
            n_loop = self.n_time_repeat
        if skip_names is None:
            skip_names = self.skip_names
        return self.time_evolution_function(
            step_core_function, updatable_dict,
            n_loop=n_loop, skip_names=skip_names)

    def step_explicit(
            self, step_core_function, updatable_dict, n_loop, **kwargs):
        for i_repeat in range(n_loop):
            dict_du = step_core_function(updatable_dict, i_repeat)
            updatable_dict = {
                k: updatable_dict[k] + dict_du[k] / n_loop
                for k in updatable_dict.keys()}
        return updatable_dict

    def step_bb(
            self, step_core_function, updatable_dict,
            n_loop, skip_names, **kwargs):
        initial_u = updatable_dict
        previous_u = initial_u
        initial_du = step_core_function(updatable_dict, 0)
        initial_residual = {k: -v for k, v in initial_du.items()}
        initial_residual_norm = self.dict_norm(initial_residual)

        # Initially, R[v] = u(t) - u(t) - D[v] dt = - D[v] dt
        previous_residual = initial_residual

        # First assume alpha = 1
        # v_{i+1} <- u(t) + D[v_{i}] dt
        u = self.operate_dict(initial_u, initial_du, torch.add)

        for i_repeat in range(n_loop - 1):
            du_dt_dt = step_core_function(u, i_repeat)

            # R[v] = v - u(t) - D[v] dt
            residual = self.operate_dict(
                self.operate_dict(u, initial_u, torch.sub),
                du_dt_dt, torch.sub)

            alpha = self.calculate_alpha(
                u, previous_u, residual, previous_residual)

            # - dv = alpha_i R[v_i]
            negative_du = self.operate_dict(alpha, residual, torch.mul)

            previous_u = u
            previous_residual = residual

            # v_{i+1} = v_i - (- dv)
            u = self.operate_dict(u, negative_du, torch.sub)

            residual_norm = self.dict_norm(residual)

            relative_residual = self.compute_relative_residual(
                residual_norm, initial_residual_norm, skip_names=skip_names)
            if self.debug:
                print('--')
                print(f"n_cell: {self.n_cell}")
                print(
                    f"{i_repeat + 2} / {n_loop}: "
                    f"{float(relative_residual.cpu().detach()):.5e}")
                for k in alpha.keys():
                    residual = residual_norm[k] / (
                        initial_residual_norm[k] + self.EPSILON)
                    print(f"- {k}:")
                    print(f"  - alpha: {alpha[k]:.5e}")
                    print(f"  - r: {residual:.5e}")

            if relative_residual < self.convergence_threshold:
                if self.debug:
                    print('break')
                break

        return u

    def dict_norm(self, dict_data):
        return {
            k: self.compute_total_integral(
                torch.einsum('c...,c...->c', v, v)[:, None])**.5
            for k, v in dict_data.items()}

    def compute_relative_residual(self, r, r_initial, skip_names):
        return torch.sum(torch.stack([
            r[k] / (r_initial[k] + self.EPSILON) for k in r.keys()
            if k not in skip_names])) / (len(r) - len(skip_names))

    def operate_dict(self, dict_a, dict_b, op):
        return {k: op(dict_a[k], dict_b[k]) for k in dict_a.keys()}

    def calculate_alpha(self, u, previous_u, residual, previous_residual):
        return {
            k: self._calculate_single_alpha(
                u[k], previous_u[k], residual[k], previous_residual[k])
            for k in u.keys()}

    def _calculate_single_alpha(
            self, u, previous_u, residual, previous_residual):
        delta_u = u - previous_u
        delta_r = residual - previous_residual

        denominator = self.compute_total_integral(
            torch.einsum('c...,c...->c', delta_r, delta_r)[:, None])
        if denominator < self.EPSILON:
            return 1.
        numerator = self.compute_total_integral(
            torch.einsum('c...,c...->c', delta_u, delta_r)[:, None])
        return numerator / denominator

    def to(self, device):
        self.device = device
        for dense_name in self.LIST_DENSE_NAMES:
            setattr(
                self, dense_name, getattr(self, dense_name).to(device))

        for sparse_name in self.LIST_SPARSE_NAMES:
            sparse_data = getattr(self, sparse_name)
            if isinstance(sparse_data, (list, tuple)):
                setattr(
                    self, sparse_name, [s.to(device) for s in sparse_data])
            else:
                setattr(
                    self, sparse_name, sparse_data.to(device))
        return self

    def facet_from_global(self, x):
        if isinstance(x, (int, float)):
            return (torch.ones((len(self.facet_pos), 1)) * x).to(
                device=self.device, dtype=self.dtype)
        elif isinstance(x, np.ndarray):
            converted = torch_tensor(x)
        elif isinstance(x, torch.Tensor):
            converted = x
        else:
            raise ValueError(f"Unexpedted input type: {x.__class__}")
        if len(converted) == len(self.facet_pos):
            return converted.to(device=self.device, dtype=self.dtype)
        else:
            return torch.einsum(
                'f,...->f...', torch.ones(len(self.facet_pos)),
                util.regularize_shape(converted)).to(
                    device=self.device, dtype=self.dtype)

    def solve_convection_diffusion(
            self, t_max, delta_t,
            cell_initial_phi, facet_velocity, facet_diffusion,
            facet_dirichlet, facet_neumann, periodic=None, dict_mlp=None,
            write=False, store=False, output_directory=None):
        facet_velocity = self.facet_from_global(facet_velocity)
        facet_diffusion = self.facet_from_global(facet_diffusion)

        ts = np.arange(0., t_max + delta_t / 2, delta_t)
        cell_phi = cell_initial_phi
        dict_results = self.write_store_if_needed(
            dict_results={}, dict_mlp=dict_mlp, dict_data={'phi': cell_phi},
            i_time=0, time=0., output_directory=output_directory,
            write=write, store=store)

        list_cell_return_phi = []

        def step_core(dict_input, i_repeat):
            cell_phi = self.step_convection_diffusion(
                delta_t=delta_t, cell_phi=dict_input['cell_phi'],
                facet_velocity=facet_velocity,
                facet_diffusion=facet_diffusion,
                facet_dirichlet=facet_dirichlet,
                facet_neumann=facet_neumann,
                periodic=periodic,
                dict_mlp=dict_mlp)
            return {'cell_phi': cell_phi}

        for i, t in enumerate(ts[1:], 1):

            dict_variable = self.step(
                step_core,
                updatable_dict={'cell_phi': cell_phi})
            cell_phi = dict_variable['cell_phi']

            if self.print_period > 0:
                if i % self.print_period == 0:
                    total_integral = self.compute_total_integral(cell_phi)
                    print(f"t = {t:9.5f}, Total integral: {total_integral:9f}")
            dict_results = self.write_store_if_needed(
                dict_results, dict_mlp, dict_data={'phi': cell_phi},
                i_time=i, time=t, output_directory=output_directory,
                write=write, store=store)
            list_cell_return_phi.append(cell_phi.clone())

        cell_return_phi = torch.stack(list_cell_return_phi, dim=0)
        dict_results.update({'phi': cell_return_phi})

        return dict_results

    def step_convection_diffusion(
            self, delta_t,
            cell_phi, facet_velocity, facet_diffusion,
            facet_dirichlet, facet_neumann,
            periodic=None, dict_mlp=None, mode=None):

        if self.center_data:
            mean_phi = self.compute_total_mean(
                cell_phi, dim=0, keepdim=True)
            cell_phi = cell_phi - mean_phi
            facet_dirichlet = facet_dirichlet - mean_phi

        cell_convection = self.compute_cc_scalar_convection(
            cell_phi, facet_velocity, facet_dirichlet, facet_neumann,
            periodic=periodic,
            dict_mlp=dict_mlp, mode=mode)
        cell_diffusion = self.compute_cc_diffusion_scalar(
            cell_phi, facet_diffusion, facet_dirichlet, facet_neumann,
            periodic=periodic,
            dict_mlp=dict_mlp, mode=mode)
        dphi_dt = - cell_convection + cell_diffusion

        if self.nonorthogonal_correction:
            cell_diffusion_correction \
                = self.compute_cc_diffusion_scalar_correction(
                    cell_phi, facet_diffusion,
                    facet_dirichlet, facet_neumann, periodic=periodic,
                    dict_mlp=dict_mlp, mode=mode)

            # Multiply with volume to have source term
            cell_diffusion_correction_source = torch.einsum(
                'c...a,cb->c...a',
                cell_diffusion_correction, self.cell_volume)
            dphi_dt = dphi_dt + cell_diffusion_correction_source
            # dphi_dt = dphi_dt + cell_diffusion_correction

        return dphi_dt * delta_t

    def apply_mlp(
            self, dict_mlp, key_, input_tensor,
            filter_for_scale=None, mode=None, reduce_op=sum,
            n_split=None, scale=False, l2=False):
        if mode is None:
            mode = self.mode
        key = f"{mode}/{key_}"
        if dict_mlp is None or key not in dict_mlp:
            if self.trainable:
                if dict_mlp is None:
                    return input_tensor
                    # raise ValueError('Feed dict_mlp when trainable is True')
                else:
                    raise ValueError(f"{key} not in: {dict_mlp.keys()}")
            else:
                return input_tensor

        # norm = torch.einsum('...,...->', input_tensor, input_tensor)**.5
        # print(f"{mode}/{key_}: {norm}")

        mlp = dict_mlp[key]
        # print(f"{key} is called")
        if isinstance(mlp, torch.nn.ModuleList):
            # Detach to keep grad valid
            if n_split is None:
                # Encoder
                if scale:
                    if l2:
                        s = torch.mean(
                            self._get_weight(mlp[0])**2).detach()**.5
                        return reduce_op([
                            self._apply_mlp(
                                m, input_tensor,
                                filter_for_scale=filter_for_scale)
                            * s
                            / torch.mean(self._get_weight(m).detach()**2)**.5
                            for m in mlp])
                    else:
                        s = torch.sum(self._get_weight(mlp[0])).detach()
                        return reduce_op([
                            self._apply_mlp(
                                m, input_tensor,
                                filter_for_scale=filter_for_scale)
                            * s / torch.sum(self._get_weight(m).detach())
                            for m in mlp])
                else:
                    return reduce_op([
                        self._apply_mlp(
                            m, input_tensor, filter_for_scale=filter_for_scale)
                        for m in mlp])
            else:
                # Decoder
                tensors = torch.tensor_split(input_tensor, n_split, dim=-1)
                if scale:
                    if l2:
                        s = torch.mean(
                            self._get_weight(mlp[0])**2).detach()**.5
                        return reduce_op([
                            self._apply_mlp(
                                m, t, filter_for_scale=filter_for_scale)
                            / s
                            * torch.mean(self._get_weight(m).detach()**2)**.5
                            for m, t in zip(mlp, tensors)])
                    else:
                        s = torch.sum(self._get_weight(mlp[0])).detach()
                        return reduce_op([
                            self._apply_mlp(
                                m, t, filter_for_scale=filter_for_scale)
                            / s * torch.sum(self._get_weight(m).detach())
                            for m, t in zip(mlp, tensors)])
                else:
                    return reduce_op([
                        self._apply_mlp(
                            m, t, filter_for_scale=filter_for_scale)
                        for m, t in zip(mlp, tensors)])
        else:
            return self._apply_mlp(
                mlp, input_tensor, filter_for_scale=filter_for_scale)

    def _get_weight(self, mlp):
        if isinstance(mlp, siml.networks.proportional.Proportional):
            return mlp.get_weight()
        elif hasattr(mlp, 'reference_block'):
            return self._get_weight(mlp.reference_block)
        else:
            raise ValueError(mlp)

    def _apply_mlp(self, mlp, input_tensor, filter_for_scale=None):
        if isinstance(
                mlp, siml.networks.tensor_operations.EnSEquivariantMLP):
            if filter_for_scale is None:
                h = mlp([
                    input_tensor, self.facet_length_scale, self.time_scale,
                    self.mass_scale])
            else:
                h = mlp([
                    input_tensor, self.facet_length_scale[filter_for_scale],
                    self.time_scale, self.mass_scale[filter_for_scale]])
        else:
            h = mlp(input_tensor)
        if mlp.block_setting.coeff is None:
            return h
        else:
            return h * mlp.block_setting.coeff

    def compute_cc_scalar_convection(
            self,
            cell_phi, facet_velocity,
            facet_dirichlet_scalar, facet_neumann_scalar, dict_mlp,
            mode=None, periodic=None):

        facet_value = self.compute_facet_value(
            cell_phi, facet_velocity,
            facet_dirichlet_scalar, facet_neumann_scalar,
            dict_mlp=dict_mlp, mode=mode,
            periodic=periodic)

        facet_flux = self.apply_mlp(
            dict_mlp, 'conv_phi_flux_mlp',
            torch.einsum('fpa,fa->fpa', facet_velocity, facet_value),
            mode=mode)
        cell_convection = self.compute_cf_divergence_default(facet_flux)

        return cell_convection

    def compute_cc_diffusion_scalar(
            self, cell_phi, facet_diffusion, facet_dirichlet, facet_neumann,
            dict_mlp, mode=None, periodic=None):
        if self.diffusion_method == 'default':
            return self.compute_cc_diffusion_default(
                cell_phi, facet_diffusion=facet_diffusion,
                facet_dirichlet=facet_dirichlet, facet_neumann=facet_neumann,
                dict_mlp=dict_mlp, mode=mode, periodic=periodic)
        elif self.diffusion_method == 'minimum_correction':
            return self.compute_cc_diffusion_minimum_correction(
                cell_phi, facet_diffusion=facet_diffusion,
                facet_dirichlet=facet_dirichlet, facet_neumann=facet_neumann,
                dict_mlp=dict_mlp, mode=mode, periodic=periodic)
        elif self.diffusion_method == 'over_relaxation':
            return self.compute_cc_diffusion_over_relaxation(
                cell_phi, facet_diffusion=facet_diffusion,
                facet_dirichlet=facet_dirichlet, facet_neumann=facet_neumann,
                dict_mlp=dict_mlp, mode=mode, periodic=periodic)
        else:
            raise ValueError(self.diffusion_method)

    def compute_cc_diffusion_scalar_correction(
            self, cell_phi, facet_diffusion, facet_dirichlet, facet_neumann,
            dict_mlp, mode=None, periodic=None):
        if self.diffusion_method == 'over_relaxation':
            return self.compute_cc_diffusion_over_relaxation_correction(
                cell_phi, facet_diffusion=facet_diffusion,
                facet_dirichlet=facet_dirichlet, facet_neumann=facet_neumann,
                dict_mlp=dict_mlp, mode=None, periodic=periodic)
        else:
            raise NotImplementedError(
                'Set nonorthogonal_correction False when '
                f"diffusion_method is {self.diffusion_method}")

    def compute_facet_value(
            self, cell_value, facet_velocity,
            facet_dirichlet, facet_neumann,
            cell_p=None, facet_rho=None, facet_diffusion=None,
            facet_dirichlet_p=None, facet_neumann_p=None,
            dict_mlp=None, mlp_base_name=None, mode=None, periodic=None):
        if mode is None:
            mode = self.mode
        if self.trainable and dict_mlp is not None \
                and self.ml_interpolation != 'non_ml':
            facet_upwind_value = self.compute_facet_value_upwind(
                cell_value, facet_velocity)

            # Use deepset-like method
            if self.ml_interpolation == 'upwind':
                facet_value = self.apply_mlp(
                    dict_mlp, input_tensor=facet_upwind_value,
                    key_='interpolation_upwind', mode=mode)
            else:
                if self.ml_interpolation == 'separate':
                    if dict_mlp[f"{mode}/interpolation_both"] is None:
                        return self.compute_facet_value(
                            cell_value, facet_velocity,
                            facet_dirichlet, facet_neumann,
                            dict_mlp=None, mlp_base_name=mlp_base_name,
                            mode=mode, periodic=periodic)

                    facet_positive_value = mul(
                        self.fc_weighted_positive_inc, cell_value)
                    facet_negative_value = - mul(
                        self.fc_weighted_negative_inc, cell_value)
                    facet_center_value = facet_positive_value \
                        + facet_negative_value
                    facet_value = (
                        + self.apply_mlp(
                            dict_mlp, input_tensor=facet_positive_value * 2,
                            key_='interpolation_both', mode=mode)
                        + self.apply_mlp(
                            dict_mlp, input_tensor=facet_negative_value * 2,
                            key_='interpolation_both', mode=mode)
                        + self.apply_mlp(
                            dict_mlp, input_tensor=facet_upwind_value,
                            key_='interpolation_upwind', mode=mode)
                        + self.apply_mlp(
                            dict_mlp, input_tensor=facet_center_value,
                            key_='interpolation_center', mode=mode)
                    ) / 4
                elif self.ml_interpolation == 'bounded':
                    if dict_mlp[f"{mode}/interpolation_both"] is None:
                        return self.compute_facet_value(
                            cell_value, facet_velocity,
                            facet_dirichlet, facet_neumann,
                            dict_mlp=None, mlp_base_name=mlp_base_name,
                            mode=mode, periodic=periodic)

                    facet_positive_value = mul(
                        self.fc_weighted_positive_inc, cell_value) * 2
                    facet_negative_value = - mul(
                        self.fc_weighted_negative_inc, cell_value) * 2
                    shape = facet_positive_value.shape
                    if len(shape) == 2:  # Scalar
                        facet_min = torch.min(
                            facet_positive_value, facet_negative_value)
                        facet_max = torch.min(
                            facet_positive_value, facet_negative_value)
                    else:  # Tensor
                        facet_positive_norm = torch.einsum(
                            'f...a,f...a->fa',
                            facet_positive_value, facet_positive_value)
                        facet_negative_norm = torch.einsum(
                            'f...a,f...a->fa',
                            facet_negative_value, facet_negative_value)
                        positive_smaller = facet_positive_norm \
                            < facet_negative_norm
                        positive_larger = ~positive_smaller
                        facet_min = torch.einsum(
                            'fa,f...a->f...a',
                            positive_smaller, facet_positive_value) \
                            + torch.einsum(
                                'fa,f...a->f...a',
                                positive_larger, facet_negative_value)
                        facet_max = torch.einsum(
                            'fa,f...a->f...a',
                            positive_smaller, facet_negative_value) \
                            + torch.einsum(
                                'fa,f...a->f...a',
                                positive_larger, facet_positive_value)

                    facet_limiter = (
                        + self.apply_mlp(
                            dict_mlp,
                            input_tensor=torch.cat([
                                facet_positive_value,
                                facet_negative_value], dim=-1),
                            key_='interpolation_both', mode=mode)
                        + self.apply_mlp(
                            dict_mlp,
                            input_tensor=torch.cat([
                                facet_negative_value,
                                facet_positive_value], dim=-1),
                            key_='interpolation_both', mode=mode)
                    ) / 2
                    facet_value = torch.einsum(
                        'fa,f...a->f...a',
                        facet_limiter, facet_max - facet_min) + facet_min

                    if self.debug:
                        if len(shape) == 2:  # Scalar
                            if not torch.all(
                                torch.logical_and(
                                    facet_min <= facet_value + 1e-5,
                                    facet_value <= facet_max + 1e-5)):
                                filter_not_good = ~torch.logical_and(
                                    facet_min <= facet_value + 1e-5,
                                    facet_value <= facet_max + 1e-5)
                                raise ValueError(
                                    'Interpolation failed for alpha\n'
                                    'Max:\n'
                                    f"{facet_max[filter_not_good]}\n"
                                    'Interp:\n'
                                    f"{facet_value[filter_not_good]}\n"
                                    'Min:\n'
                                    f"{facet_min[filter_not_good]}\n"
                                    'limiter:\n'
                                    f"{facet_limiter[filter_not_good]}\n"
                                )
                        else:
                            # NOTE: In equivariant setting, norm is not
                            # necessarily bounded for higher order tensors
                            pass

                elif self.ml_interpolation == 'upwind_center':
                    if dict_mlp[f"{mode}/interpolation_upwind"] is None:
                        return self.compute_facet_value(
                            cell_value, facet_velocity,
                            facet_dirichlet, facet_neumann,
                            dict_mlp=None, mlp_base_name=mlp_base_name,
                            mode=mode, periodic=periodic)

                    facet_center_value = self.compute_facet_value_center(
                        cell_value)
                    facet_value = (
                        + self.apply_mlp(
                            dict_mlp, input_tensor=facet_upwind_value,
                            key_='interpolation_upwind', mode=mode)
                        + self.apply_mlp(
                            dict_mlp, input_tensor=facet_center_value,
                            key_='interpolation_center', mode=mode)
                    ) / 2
                elif self.ml_interpolation == 'upwind_center_concatenate':
                    if dict_mlp[f"{mode}/interpolation_both"] is None:
                        return self.compute_facet_value(
                            cell_value, facet_velocity,
                            facet_dirichlet, facet_neumann,
                            dict_mlp=None, mlp_base_name=mlp_base_name,
                            mode=mode, periodic=periodic)

                    facet_center_value = self.compute_facet_value_center(
                        cell_value)
                    facet_value = self.apply_mlp(
                        dict_mlp, input_tensor=torch.cat(
                            [facet_upwind_value, facet_center_value],
                            dim=-1),
                        key_='interpolation_both', mode=mode)
                elif self.ml_interpolation == 'concatenate':
                    if dict_mlp[f"{mode}/interpolation_both"] is None:
                        return self.compute_facet_value(
                            cell_value, facet_velocity,
                            facet_dirichlet, facet_neumann,
                            dict_mlp=None, mlp_base_name=mlp_base_name,
                            mode=mode, periodic=periodic)

                    facet_positive_value = mul(
                        self.fc_weighted_positive_inc, cell_value)
                    facet_negative_value = - mul(
                        self.fc_weighted_negative_inc, cell_value)
                    facet_center_value = facet_positive_value \
                        + facet_negative_value
                    facet_concatenated_positive = torch.cat([
                        facet_positive_value * 2,
                        facet_upwind_value,
                        facet_center_value], dim=-1)
                    facet_concatenated_negative = torch.cat([
                        facet_negative_value * 2,
                        facet_upwind_value,
                        facet_center_value], dim=-1)
                    facet_value = (
                        + self.apply_mlp(
                            dict_mlp, input_tensor=facet_concatenated_positive,
                            key_='interpolation_both', mode=mode)
                        + self.apply_mlp(
                            dict_mlp, input_tensor=facet_concatenated_negative,
                            key_='interpolation_both', mode=mode)
                    ) / 2
                else:
                    raise ValueError(f"Unexpected {self.ml_interpolation = }")
        else:
            if self.rhie_chow and cell_p is not None:
                facet_value = self.compute_facet_value_center(cell_value)
            elif self.upwind:
                facet_value = self.compute_facet_value_upwind(
                    cell_value, facet_velocity)
            else:
                facet_value = self.compute_facet_value_center(cell_value)

        if self.rhie_chow and cell_p is not None:
            facet_value = facet_value \
                + self.compute_facet_rhie_chow(
                    cell_p, facet_rho, facet_diffusion, facet_value,
                    facet_dirichlet_p, facet_neumann_p)
        return self.apply_boundary_facet(
            facet_value,
            facet_dirichlet, facet_neumann, periodic=periodic)

    def compute_facet_rhie_chow(
            self, cell_p, facet_diffusion, facet_u,
            facet_dirichlet_p, facet_neumann_p):
        raise NotImplementedError('Should be overwritten')

    def compute_facet_value_upwind(self, cell_value, facet_velocity):
        facet_dot_normal_velocity = torch.einsum(
            'fpa,fpb->fb', self.facet_normal_vector, facet_velocity)

        facet_positive_value = mul(
            self.fc_positive_inc, cell_value)
        facet_negative_value = - mul(
            self.fc_negative_inc, cell_value)

        facet_filter_positive_dot = (
            facet_dot_normal_velocity > 0)
        facet_upwind_value = \
            + torch.einsum(
                'f...a,fa->f...a',
                facet_positive_value, facet_filter_positive_dot) \
            + torch.einsum(
                'f...a,fa->f...a',
                facet_negative_value, (~facet_filter_positive_dot))

        facet_filter_boundary_positive = torch.logical_and(
            self.facet_filter_boundary, facet_dot_normal_velocity[..., 0] <= 0)
        shape = facet_upwind_value.shape
        if len(shape) == 2:
            facet_upwind_value[
                facet_filter_boundary_positive] = facet_positive_value[
                    facet_filter_boundary_positive]
        elif len(shape) == 3:
            # print(facet_upwind_value)
            for i_dim in range(shape[-2]):
                facet_upwind_value[:, i_dim, :][
                    facet_filter_boundary_positive] = facet_positive_value[
                        :, i_dim, :][facet_filter_boundary_positive]
            # print(facet_upwind_value)
        else:
            raise ValueError(f"Unexpected {shape = }")

        return facet_upwind_value

    def compute_cf_divergence_default(self, facet_vector):
        cell_divergence = (
            + mul(
                self.cf_normal_inc[0],
                torch.einsum(
                    'fa,fb->fa', facet_vector[:, 0], self.facet_area))
            + mul(
                self.cf_normal_inc[1],
                torch.einsum(
                    'fa,fb->fa', facet_vector[:, 1], self.facet_area))
            + mul(
                self.cf_normal_inc[2],
                torch.einsum(
                    'fa,fb->fa', facet_vector[:, 2], self.facet_area))
        ) / self.cell_volume
        return cell_divergence

    def compute_cc_diffusion_default(
            self, cell_phi, facet_dirichlet, facet_neumann, dict_mlp,
            facet_diffusion=None, mode=None, periodic=None):

        facet_value = self.compute_facet_value_center(cell_phi)
        facet_value = self.apply_boundary_facet(
            facet_value, facet_dirichlet, facet_neumann, periodic=periodic)
        cell_gradient = self.compute_cf_grad(facet_value)
        facet_gradient = self.compute_facet_value_center(cell_gradient)

        facet_gradient_before_mlp = facet_gradient.clone()
        facet_gradient_after_mlp = self.apply_mlp(
            dict_mlp, 'diffusion_grad_mlp', facet_gradient, mode=mode)
        facet_gradient = self.apply_boundary_facet_gradient(
            facet_gradient_before_mlp, facet_gradient_after_mlp,
            facet_dirichlet, facet_neumann, periodic=periodic)

        if facet_diffusion is None:
            cell_diffusion = self.compute_cf_divergence_default(
                facet_gradient)
        else:
            cell_diffusion = self.compute_cf_divergence_default(
                torch.einsum('fa,fpa->fpa', facet_diffusion, facet_gradient))

        return cell_diffusion

    def compute_cc_diffusion_minimum_correction(
            self, cell_phi, facet_dirichlet, facet_neumann, dict_mlp,
            facet_diffusion=None, mode=None, periodic=None):
        facet_gradient = self.compute_fc_grad(
            cell_phi, facet_dirichlet, facet_neumann, dict_mlp=dict_mlp,
            mode=mode, periodic=periodic)

        if facet_diffusion is None:
            cell_diffusion = self.compute_cf_divergence_minimum_correction(
                facet_gradient)
        else:
            cell_diffusion = self.compute_cf_divergence_minimum_correction(
                torch.einsum(
                    'fa,f...a->f...a', facet_diffusion, facet_gradient))
        return cell_diffusion

    def compute_cc_diffusion_over_relaxation(
            self, cell_phi, facet_dirichlet, facet_neumann, dict_mlp,
            facet_diffusion=None, mode=None, periodic=None):
        facet_gradient = self.compute_fc_grad(
            cell_phi, facet_dirichlet, facet_neumann, periodic=periodic,
            dict_mlp=dict_mlp, mode=mode)

        if facet_diffusion is None:
            facet_flux = facet_gradient
        else:
            facet_flux = torch.einsum(
                'fa,f...a->f...a', facet_diffusion, facet_gradient)

        cell_diffusion = self.compute_cf_divergence_over_relaxed(facet_flux)

        return cell_diffusion

    def compute_cc_diffusion_over_relaxation_correction(
            self, cell_phi, facet_dirichlet, facet_neumann, dict_mlp,
            facet_diffusion=None, mode=None, periodic=None):
        facet_value = self.compute_facet_value_center(cell_phi)
        facet_value = self.apply_boundary_facet(
            facet_value, facet_dirichlet, facet_neumann, periodic=periodic)
        cell_gradient = self.compute_cf_grad(facet_value)
        facet_gradient = self.compute_facet_value_center(cell_gradient)

        facet_gradient_before_mlp = facet_gradient.clone()
        facet_gradient_after_mlp = self.apply_mlp(
            dict_mlp, 'diff_corr_mlp', facet_gradient, mode=mode)
        facet_gradient = self.apply_boundary_facet_gradient(
            facet_gradient_before_mlp, facet_gradient_after_mlp,
            facet_dirichlet, facet_neumann, periodic=periodic)

        if facet_diffusion is None:
            facet_flux = facet_gradient
        else:
            facet_flux = torch.einsum(
                'fa,fpa->fpa', facet_diffusion, facet_gradient)

        cell_correction = self.compute_cf_divergence_correction_over_relaxed(
            facet_flux)
        return cell_correction

    def compute_fc_grad(
            self, cell_phi, facet_dirichlet, facet_neumann, dict_mlp,
            mode=None, periodic=None):
        if BENCHMARK:
            t0 = datetime.now()

        if self.diffusion_method == 'default':
            facet_gradient = torch.stack([
                - mul(
                    self.cf_normal_inc[0].transpose(0, 1),
                    cell_phi / self.cell_volume)
                * self.facet_area,
                - mul(
                    self.cf_normal_inc[1].transpose(0, 1),
                    cell_phi / self.cell_volume)
                * self.facet_area,
                - mul(
                    self.cf_normal_inc[2].transpose(0, 1),
                    cell_phi / self.cell_volume)
                * self.facet_area,
            ], axis=1)
        else:
            facet_gradient = self._compute_fc_grad_core(cell_phi)

        if BENCHMARK:
            t1 = datetime.now()

        if self.trainable and dict_mlp is not None:
            facet_gradient_before_mlp = facet_gradient.clone()
            facet_gradient_after_mlp = self.apply_mlp(
                dict_mlp, 'diffusion_grad_mlp', facet_gradient, mode=mode)
        else:
            facet_gradient_before_mlp = facet_gradient
            facet_gradient_after_mlp = facet_gradient

        if BENCHMARK:
            t2 = datetime.now()

        facet_return_gradient = self.apply_boundary_facet_gradient(
            facet_gradient_before_mlp, facet_gradient_after_mlp,
            facet_dirichlet, facet_neumann, periodic=periodic)

        if BENCHMARK:
            t3 = datetime.now()
            print(f"Grad: {(t1 - t0).total_seconds()}")
            print(f"ML: {(t2 - t1).total_seconds()}")
            print(f"Boundary: {(t3 - t2).total_seconds()}")

        return facet_return_gradient

    def _compute_fc_grad_core(self, cell_phi):
        return - torch.einsum(
            'fpa,f...b->fp...b',
            self.facet_differential_d,
            mul(self.fc_signed_inc, cell_phi))

    def compute_cc_vector_laplacian(
            self, cell_u, facet_diffusion, facet_dirichlet, facet_neumann,
            dict_mlp, mode=None, periodic=None):
        if self.diffusion_method == 'minimum_correction':
            return self.compute_cc_diffusion_minimum_correction(
                cell_u, facet_diffusion=facet_diffusion,
                facet_dirichlet=facet_dirichlet, facet_neumann=facet_neumann,
                dict_mlp=dict_mlp, mode=mode, periodic=periodic)
        if self.diffusion_method == 'over_relaxation':
            return self.compute_cc_diffusion_over_relaxation(
                cell_u, facet_diffusion=facet_diffusion,
                facet_dirichlet=facet_dirichlet, facet_neumann=facet_neumann,
                dict_mlp=dict_mlp, mode=mode, periodic=periodic)
        else:
            raise ValueError(self.diffusion_method)

    def apply_boundary_facet_gradient(
            self, facet_original_gradient, facet_gradient,
            facet_dirichlet, facet_neumann, periodic=None):
        if BENCHMARK:
            t0 = datetime.now()

        # Dirichlet boundary condition
        n_shape = len(facet_dirichlet.shape)
        if n_shape == 2:
            facet_filter_dirichlet = ~ torch.isnan(facet_dirichlet[:, 0])
        elif n_shape == 3:
            facet_filter_dirichlet = ~ torch.isnan(facet_dirichlet[:, 0, 0])
        else:
            raise ValueError(
                f"Invalid dirichlet shape: {facet_dirichlet.shape}")

        if BENCHMARK:
            t1 = datetime.now()

        if torch.sum(facet_filter_dirichlet) > 1:
            dirichlet_gradient = torch.einsum(
                'fpa,f...b->fp...b',
                self.facet_differential_d[facet_filter_dirichlet],
                facet_dirichlet[facet_filter_dirichlet])
            boundary_gradient = dirichlet_gradient \
                + facet_original_gradient[facet_filter_dirichlet]
            facet_gradient[facet_filter_dirichlet] \
                = boundary_gradient

        if BENCHMARK:
            t2 = datetime.now()

        # Neumann boundary condition
        # facet_filter_neumann = ~ torch.isnan(facet_neumann)
        n_shape = len(facet_neumann.shape)
        if n_shape == 3:
            facet_filter_neumann = ~ torch.isnan(facet_neumann[:, 0, 0])
        elif n_shape == 4:
            facet_filter_neumann = ~ torch.isnan(facet_neumann[:, 0, 0, 0])
        else:
            raise ValueError(
                f"Invalid dirichlet shape: {facet_dirichlet.shape}")
        facet_gradient[facet_filter_neumann] = facet_neumann[
            facet_filter_neumann]

        if BENCHMARK:
            t3 = datetime.now()

        if periodic is not None and len(periodic) != 0:
            # print('boundary grad')
            mean_gradient = (
                facet_original_gradient[periodic['destination']]
                + facet_original_gradient[periodic['source']]) / 2
            # mean_gradient = (
            #     facet_gradient[periodic['destination']]
            #     + facet_gradient[periodic['source']]) / 2
            facet_gradient[periodic['destination']] = mean_gradient
            facet_gradient[periodic['source']] = mean_gradient

        if BENCHMARK:
            t4 = datetime.now()
            print(f"Filter: {(t1 - t0).total_seconds()}")
            print(f"Dirichlet: {(t2 - t1).total_seconds()}")
            print(f"Neumann: {(t3 - t2).total_seconds()}")
            print(f"Periodic: {(t4 - t3).total_seconds()}")

        return facet_gradient

    def compute_cf_grad(self, facet_value):
        cell_gradient = mul(
            self.cf_signed_inc, torch.einsum(
                'fpa,fa,fb->fpb',
                self.facet_normal_vector, self.facet_area, facet_value))
        return torch.einsum('cpb,ca->cpb', cell_gradient, 1 / self.cell_volume)

    def compute_cf_divergence_minimum_correction(self, facet_vector):
        cell_divergence = torch.einsum(
            'c...,ca->c...',
            mul(
                self.cf_signed_inc,
                torch.einsum(
                    'fpa,fp...b->f...b',
                    self.facet_divergence_vector_minimum_correction,
                    facet_vector)),
            1 / self.cell_volume)
        return cell_divergence

    def compute_cf_divergence_over_relaxed(self, facet_vector):
        cell_divergence = torch.einsum(
            'c...b,ca->c...b',
            mul(
                self.cf_signed_inc,
                torch.einsum(
                    'fpa,fp...b->f...b',
                    self.facet_divergence_vector_over_relaxed,
                    facet_vector)),
            1 / self.cell_volume)
        return cell_divergence

    def compute_cf_divergence_correction_over_relaxed(self, facet_vector):
        cell_divergence = mul(
            self.cf_signed_inc,
            torch.einsum(
                'fpa,fp...b->f...b',
                self.facet_nonorthogonal_over_relaxed,
                facet_vector)) / self.cell_volume
        return cell_divergence

    def apply_boundary_facet(
            self, facet_value,
            facet_dirichlet, facet_neumann, periodic=None):
        shape = facet_value.shape
        if len(shape) == 2:
            return self.apply_boundary_facet_scalar(
                facet_value, facet_dirichlet, facet_neumann, periodic=periodic)
        elif len(shape) == 3:
            return self.apply_boundary_facet_vector(
                facet_value, facet_dirichlet, facet_neumann, periodic=periodic)
        else:
            raise ValueError(f"Unexpected facet_value shape: {shape}")

    def apply_boundary_facet_scalar(
            self, facet_value,
            facet_dirichlet, facet_neumann, periodic=None):
        if self.clone_boundary:
            facet_return_value = facet_value.clone()
        else:
            facet_return_value = facet_value

        # Dirichlet boundary condition
        filter_not_nan = ~ torch.isnan(facet_dirichlet)
        facet_return_value[filter_not_nan] = facet_dirichlet[filter_not_nan]

        # Neumann boundary condition
        filter_not_nan = ~ torch.isnan(facet_neumann[:, 0])
        dot_neumann = torch.einsum(
            'fpa,fpb->fb',
            self.facet_relative_position, facet_neumann)
        facet_return_value[filter_not_nan] = (facet_value + dot_neumann)[
            filter_not_nan]

        if periodic is not None and len(periodic) != 0:
            mean_value = (
                facet_value[periodic['destination']]
                + facet_value[periodic['source']]) / 2
            facet_return_value[periodic['destination']] = mean_value
            facet_return_value[periodic['source']] = mean_value

        return facet_return_value

    def apply_boundary_facet_vector(
            self, facet_value, facet_dirichlet, facet_neumann, periodic=None):
        n_dim = facet_value.shape[-2]
        facet_value = torch.stack([
            self.apply_boundary_facet_scalar(
                facet_value[..., i_dim, :], facet_dirichlet[..., i_dim, :],
                facet_neumann[..., i_dim, :, :], periodic=periodic)
            for i_dim in range(n_dim)], dim=-2)
        return facet_value

    def apply_boundary_cell(
            self,
            cell_value, facet_dirichlet, facet_neumann, periodic=None):

        facet_value = self.compute_facet_value_center(cell_value)
        facet_value = self.apply_boundary_facet(
            facet_value, facet_dirichlet, facet_neumann, periodic=periodic)

        cell_weighted_sum = mul(
            self.cf_inc,
            torch.einsum('fa,fb->fa', facet_value, self.facet_inv_d_norm))
        cell_normalizer = mul(
            self.cf_inc,
            self.facet_inv_d_norm)

        return cell_weighted_sum / cell_normalizer

    def compute_facet_value_center(self, cell_phi, **kwargs):
        # NOTE: Assume fc_weighted_inc is normalized
        return mul(self.fc_weighted_inc, cell_phi)

    def compute_total_integral(
            self, x, time_series=False, return_time_series=False):
        if time_series:
            string = 't'
            if return_time_series:
                ret_string = 't'
            else:
                ret_string = ''
        else:
            string = ''
            ret_string = ''
        return torch.einsum(
            f"{string}c...f,cf->{ret_string}",
            torch_tensor(x), self.cell_volume)

    def compute_total_mean(
            self, x, time_series=False, return_time_series=False, **kwargs):
        if time_series:
            string = 't'
            if return_time_series:
                ret_string = 't'
                scale = 1
            else:
                ret_string = ''
                scale = 1 / len(x)
        else:
            string = ''
            ret_string = ''
            scale = 1
        weighted_x = torch.einsum(
            f"{string}c...a,cb->{ret_string}c...a",
            torch_tensor(x), self.cell_volume)
        return torch.sum(weighted_x, **kwargs) / torch.sum(self.cell_volume) \
            * scale

    def compute_l2_squared(self, x, time_series=False):
        return self.compute_total_integral(x**2, time_series=time_series)

    def write_store_if_needed(
            self, dict_results, dict_mlp, dict_data,
            i_time, time, write, store,
            *,
            output_directory=None,
            force_store=False, force_write=False, overwrite=False,
            prefix=None):
        if write and output_directory is None:
            raise ValueError('Feed output_directory when write is True')
        if write and self.write_mesh_period < 1:
            raise ValueError(
                'Set write_mesh_period > 0 when write is True '
                f"(given: {self.write_mesh_period})")
        if store and self.store_results_period < 1:
            raise ValueError(
                'Set store_results_period > 0 when store is True '
                f"(given: {self.store_results_period})")

        if write and i_time % self.write_mesh_period == 0:
            write_now = True
        else:
            write_now = False
        if force_write:
            write_now = True

        if store and i_time % self.store_results_period == 0:
            store_now = True
        else:
            store_now = False
        if force_store:
            store_now = True
        if store_now or write:
            decoded_data = self.decode(dict_mlp, dict_data)
        else:
            return dict_results

        if write_now:
            self.write(
                i_time=i_time, dict_data=decoded_data,
                output_directory=output_directory, overwrite=overwrite,
                prefix=prefix)

        if store_now:
            if 'stored_data' not in dict_results:
                dict_results['stored_data'] = {}
            if 'time' not in dict_results['stored_data']:
                dict_results['stored_data']['time'] = []

            dict_results['stored_data']['time'].append(time)
            for k, v in decoded_data.items():
                if k not in dict_results['stored_data']:
                    dict_results['stored_data'][k] = []
                dict_results['stored_data'][k].append(v)
        return dict_results

    def decode(self, dict_mlp, dict_data):
        return {
            k: self.apply_mlp(
                dict_mlp, f"inv_{k}_mlp", v).cpu().detach().numpy()
            for k, v in dict_data.items()}

    def write(
            self, i_time, dict_data, output_directory,
            prefix=None, overwrite=False):
        if output_directory is None:
            raise ValueError(
                'Feed output_directory when write_mesh_period > 0')

        if prefix is None:
            output_name = output_directory / f"mesh.{i_time:08d}.vtu"
        else:
            output_name = output_directory / prefix \
                / f"mesh.{i_time:08d}.vtu"

        write_dict_data = {}
        for k, v in dict_data.items():
            # Make very small number zero for visualization
            v[np.abs(v) < 1e-8] = 0.
            if len(v.shape) > 2:
                write_dict_data[k] = np.squeeze(v)
            else:
                write_dict_data[k] = v
        self.fv_data.fem_data.elemental_data.update_data(
            self.fv_data.fem_data.elements.ids, write_dict_data,
            allow_overwrite=True)

        with open(os.devnull, 'w') as f:
            sys.stdout = f
            self.fv_data.fem_data.write(
                'vtu', output_name, overwrite=overwrite)
            sys.stdout = sys.__stdout__

        return
