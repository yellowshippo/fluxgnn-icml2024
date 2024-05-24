
from datetime import datetime

import numpy as np
import siml
from siml.networks.sparse import mul
import torch

from .navier_stokes import NavierStokesSolver
from .util import cat_tail


BENCHMARK = False


class MixtureSolver(NavierStokesSolver):

    def __init__(self, fem_data, **kwargs):

        original_apply_setting = kwargs.pop('apply_setting', True)
        kwargs['apply_setting'] = False
        super().__init__(fem_data, **kwargs)

        if original_apply_setting:
            self.apply_setting(**kwargs)
        return

    def apply_setting(self, **kwargs):
        super().apply_setting(**kwargs)

        self.diffusion_divide_by_rho = kwargs.get(
            'diffusion_divide_by_rho', True)

        self.n_smoothing = kwargs.get('n_smoothing', 0)
        self.smoothing_coefficient = kwargs.get('smoothing_coefficient', 1.e-1)
        if self.n_smoothing > 0:
            print(
                f"Smoothing {self.n_smoothing} with coeff "
                f"{self.smoothing_coefficient}")

        self.n_source_smoothing = kwargs.get('n_source_smoothing', 0)
        self.source_smoothing_coefficient = kwargs.get(
            'source_smoothing_coefficient', 1.e-1)
        if self.n_source_smoothing > 0:
            print(
                f"Source smoothing {self.n_source_smoothing} with coeff "
                f"{self.source_smoothing_coefficient}")

        self.fix_u = kwargs.get('fix_u', False)
        self.alpha_interpolation_for_properties = kwargs.get(
            'alpha_interpolation_for_properties', 'center')

        n_alpha_repeat = kwargs.get('n_alpha_repeat', None)
        n_alpha_factor = kwargs.get('n_alpha_factor', None)
        if n_alpha_factor is None:
            if n_alpha_repeat is None:
                self.n_alpha_repeat = 1
            else:
                self.n_alpha_repeat = n_alpha_repeat
        else:
            if n_alpha_repeat is not None:
                raise ValueError(
                    'n_alpha_repeat and n_alpha_factor cannot be fed '
                    'in the same time')
            self.n_alpha_repeat = max(int(n_alpha_factor * self.n_cell), 2)
        print(f"n_alpha_repeat: {self.n_alpha_repeat}")

        # Could be overwritten by FluxGNN
        self.deep_processor = False
        self.encoded_pressure = True
        self.n_bundle = 1
        self.scale_encdec = False
        self.l2_for_scale_encdec = False

        return

    def solve_mixture(
            self, t_max, delta_t,
            cell_initial_u, cell_initial_p, cell_initial_alpha,
            facet_dirichlet_u, facet_neumann_u,
            facet_dirichlet_p, facet_neumann_p,
            facet_diffusion_alpha=None,
            facet_dirichlet_alpha=None,
            facet_neumann_alpha=None,
            global_gravity=None,
            global_nu_solvent=None, global_nu_solute=None,
            global_rho_solvent=None, global_rho_solute=None,
            facet_original_dirichlet_u=None,
            facet_original_neumann_u=None,
            facet_original_dirichlet_p=None,
            facet_original_neumann_p=None,
            dict_mlp=None, output_directory=None, write=False, store=False):

        ts = np.arange(0., t_max + delta_t / 2, delta_t)
        cell_u = cell_initial_u
        cell_p = cell_initial_p
        cell_alpha = cell_initial_alpha

        list_cell_return_u = []
        list_cell_return_p = []
        list_cell_return_alpha = []

        write_dict = {
            'u': cell_initial_u, 'p': cell_initial_p,
            'alpha': cell_initial_alpha}
        dict_results = self.write_store_if_needed(
            dict_results={}, dict_mlp=dict_mlp, dict_data=write_dict,
            i_time=0, time=0., output_directory=output_directory,
            write=write, store=store)

        cell_gh = torch.einsum('cpa,cpb->cb', self.cell_pos, global_gravity)
        cell_gh = cell_gh - torch.min(cell_gh)
        facet_gh = torch.einsum('fpa,mpb->fb', self.facet_pos, global_gravity)
        facet_gh = facet_gh - torch.min(facet_gh)

        for i, t in enumerate(ts[1:], 1):
            if self.fix_u:
                facet_tmp_u = mul(self.fc_weighted_inc, cell_u)
                facet_previous_u = self.compute_facet_value(
                    cell_u, facet_tmp_u,
                    facet_dirichlet_u, facet_neumann_u,
                    dict_mlp=dict_mlp, mode='u_for_alpha')
            else:
                facet_previous_u = None

            def step_core(dict_input, i_repeat):
                if BENCHMARK:
                    t0 = datetime.now()
                    print(f"\n========\nStart {i_repeat}: {t0}")

                cell_u, cell_p, cell_alpha = self.step_mixture(
                    delta_t=delta_t,
                    cell_u=dict_input['cell_u'], cell_p=dict_input['cell_p'],
                    cell_alpha=dict_input['cell_alpha'],
                    global_nu_solvent=global_nu_solvent,
                    global_nu_solute=global_nu_solute,
                    global_rho_solvent=global_rho_solvent,
                    global_rho_solute=global_rho_solute,
                    facet_diffusion_alpha=facet_diffusion_alpha,
                    cell_gh=cell_gh, facet_gh=facet_gh,
                    facet_dirichlet_u=facet_dirichlet_u,
                    facet_neumann_u=facet_neumann_u,
                    facet_dirichlet_p=facet_dirichlet_p,
                    facet_neumann_p=facet_neumann_p,
                    facet_dirichlet_alpha=facet_dirichlet_alpha,
                    facet_neumann_alpha=facet_neumann_alpha,
                    facet_previous_u=facet_previous_u,
                    facet_original_dirichlet_u=facet_original_dirichlet_u,
                    facet_original_neumann_u=facet_original_neumann_u,
                    facet_original_dirichlet_p=facet_original_dirichlet_p,
                    facet_original_neumann_p=facet_original_neumann_p,
                    i_step=i, i_repeat=i_repeat,
                    dict_mlp=dict_mlp, output_directory=output_directory)

                if BENCHMARK:
                    t1 = datetime.now()
                    print(f"Finish {i_repeat}: {t1}")
                    print(f"{(t1 - t0).total_seconds()}")

                return {
                    'cell_u': cell_u, 'cell_p': cell_p,
                    'cell_alpha': cell_alpha}

            dict_variable = self.step(
                step_core,
                updatable_dict={
                    'cell_u': cell_u, 'cell_p': cell_p,
                    'cell_alpha': cell_alpha})
            cell_u = dict_variable['cell_u']
            cell_p = dict_variable['cell_p']
            cell_alpha = dict_variable['cell_alpha']

            # Update variables
            if BENCHMARK:
                t_now = datetime.now()
                print(f"--\nStart append: {t_now}")
                t_prev = t_now

            list_cell_return_u.append(cell_u.clone())
            list_cell_return_p.append(cell_p.clone())
            list_cell_return_alpha.append(cell_alpha.clone())

            if self.print_period > 0 and i % self.print_period == 0:
                total_integral = self.compute_total_integral(cell_alpha)
                print(f"t = {t:9.5f}, Total integral: {total_integral:9f}")

            write_dict = {'u': cell_u, 'p': cell_p, 'alpha': cell_alpha}
            dict_results = self.write_store_if_needed(
                dict_results, dict_mlp, dict_data=write_dict,
                i_time=i, time=t, output_directory=output_directory,
                write=write, store=store)

            if BENCHMARK:
                t_now = datetime.now()
                print(f"Finish append: {t_now}")
                print(f"{(t_now - t_prev).total_seconds()}")
                t_prev = t_now

        if BENCHMARK:
            t_now = datetime.now()
            print(f"--\nStart stack: {t_now}")
            t_prev = t_now

        cell_return_u = torch.stack(list_cell_return_u, dim=0)
        cell_return_p = torch.stack(list_cell_return_p, dim=0)
        cell_return_alpha = torch.stack(list_cell_return_alpha, dim=0)
        dict_results.update({
            'u': cell_return_u, 'p': cell_return_p,
            'alpha': cell_return_alpha})

        if BENCHMARK:
            t_now = datetime.now()
            print(f"Finish stack: {t_now}")
            print(f"{(t_now - t_prev).total_seconds()}")
            t_prev = t_now

        return dict_results

    def step_mixture(
            self, delta_t,
            cell_u, cell_p, cell_alpha,
            global_nu_solvent, global_nu_solute,
            global_rho_solvent, global_rho_solute,
            facet_diffusion_alpha, cell_gh, facet_gh,
            facet_dirichlet_u, facet_neumann_u,
            facet_dirichlet_p, facet_neumann_p,
            facet_dirichlet_alpha, facet_neumann_alpha,
            i_step, i_repeat,
            facet_original_dirichlet_u=None,
            facet_original_neumann_u=None,
            facet_original_dirichlet_p=None,
            facet_original_neumann_p=None,
            facet_previous_u=None, dict_mlp=None, output_directory=None):

        if BENCHMARK:
            t_now = datetime.now()
            print(f"--\nStart preprocess: {t_now}")
            t_prev = t_now

        if self.deep_processor:
            label = i_repeat
        else:
            label = ''

        if self.center_data:
            mean_u = self.compute_total_mean(
                cell_u, dim=0, keepdim=True)
            mean_p = self.compute_total_mean(
                cell_p, dim=0, keepdim=True)
            cell_u = cell_u - mean_u
            facet_dirichlet_u = facet_dirichlet_u - mean_u
            cell_p = cell_p - mean_p
            facet_dirichlet_p = facet_dirichlet_p - mean_p

        if self.n_smoothing > 0:
            facet_smoothing_diffusion = self.facet_length_scale**2 / delta_t \
                * self.smoothing_coefficient
        if self.n_source_smoothing > 0:
            facet_source_smoothing_diffusion = self.facet_length_scale**2 \
                / delta_t * self.source_smoothing_coefficient
        else:
            facet_source_smoothing_diffusion = None

        if BENCHMARK:
            t_now = datetime.now()
            print(f"Finish preprocess: {t_now}")
            print(f"{(t_now - t_prev).total_seconds()}")
            t_prev = t_now

        if BENCHMARK:
            t_now = datetime.now()
            print(f"--\nStart interpolation u: {t_now}")
            t_prev = t_now

        # Solve alpha diffusion
        if self.fix_u:
            facet_u = facet_previous_u
        else:
            facet_tmp_u = mul(self.fc_weighted_inc, cell_u)
            facet_u = self.compute_facet_value(
                cell_u, facet_tmp_u,
                facet_dirichlet_u, facet_neumann_u,
                dict_mlp=dict_mlp, mode=f"u_for_alpha{label}")

        if BENCHMARK:
            t_now = datetime.now()
            print(f"Finish interpolation u: {t_now}")
            print(f"{(t_now - t_prev).total_seconds()}")
            t_prev = t_now

        if BENCHMARK:
            t_now = datetime.now()
            print(f"--\nStart prepare for alpha: {t_now}")
            t_prev = t_now

        # Facet alpha for rho and mu, so no ML
        if self.alpha_interpolation_for_properties == 'center':
            facet_alpha = self.compute_facet_value_center(cell_alpha)
        elif self.alpha_interpolation_for_properties == 'upwind':
            facet_alpha = self.compute_facet_value_upwind(cell_alpha, facet_u)
        else:
            raise ValueError(
                f"Unexpected {self.alpha_interpolation_for_properties = }")
        facet_alpha = self.apply_boundary_facet(
            facet_alpha, facet_dirichlet_alpha, facet_neumann_alpha)

        facet_mu = facet_alpha * global_nu_solute * global_rho_solute + (
            1 - facet_alpha) * global_nu_solvent * global_rho_solvent
        facet_original_rho = facet_alpha * global_rho_solute + (
            1 - facet_alpha) * global_rho_solvent
        cell_original_rho = cell_alpha * global_rho_solute + (
            1 - cell_alpha) * global_rho_solvent
        facet_dirichlet_rho = facet_dirichlet_alpha * global_rho_solute + (
            1 - facet_dirichlet_alpha) * global_rho_solvent
        facet_neumann_rho = facet_neumann_alpha * global_rho_solute \
            - facet_neumann_alpha * global_rho_solvent

        if not self.encoded_pressure:
            cell_original_decoded_rho = self.apply_mlp(
                dict_mlp, 'inv_alpha_mlp', cell_original_rho,
                mode=f"{self.mode}",
                reduce_op=cat_tail, n_split=self.n_bundle,
                scale=self.scale_encdec, l2=self.l2_for_scale_encdec).detach()
            facet_dirichlet_decoded_rho = self.apply_mlp(
                dict_mlp, 'inv_alpha_mlp', facet_dirichlet_rho,
                mode=f"{self.mode}",
                reduce_op=cat_tail, n_split=self.n_bundle,
                scale=self.scale_encdec, l2=self.l2_for_scale_encdec).detach()
            facet_neumann_decoded_rho = self.apply_mlp(
                dict_mlp, 'inv_alpha_mlp', facet_neumann_rho,
                mode=f"{self.mode}",
                reduce_op=cat_tail, n_split=self.n_bundle,
                scale=self.scale_encdec, l2=self.l2_for_scale_encdec).detach()

        if self.diffusion_divide_by_rho:
            divider = facet_original_rho
        else:
            divider = 1

        if BENCHMARK:
            t_now = datetime.now()
            print(f"Finish prepare for alpha: {t_now}")
            print(f"{(t_now - t_prev).total_seconds()}")
            t_prev = t_now

        if BENCHMARK:
            t_now = datetime.now()
            print(f"--\nStart solve alpha: {t_now}")
            t_prev = t_now

        if self.n_alpha_repeat > 1:
            # Subloop for alpha
            def step_core_alpha(dict_input, i_repeat):
                cell_alpha = self.step_convection_diffusion(
                    delta_t=delta_t / self.n_time_repeat,
                    cell_phi=dict_input['cell_alpha'],
                    facet_velocity=facet_u,
                    facet_diffusion=facet_diffusion_alpha / divider,
                    facet_dirichlet=facet_dirichlet_alpha,
                    facet_neumann=facet_neumann_alpha,
                    dict_mlp=dict_mlp, mode=f"alpha{label}")
                return {'cell_alpha': cell_alpha}
            dict_variable = self.step(
                step_core_alpha,
                updatable_dict={'cell_alpha': cell_alpha},
                n_loop=self.n_alpha_repeat,
                skip_names=[])
            cell_updated_alpha = dict_variable['cell_alpha']

        else:
            cell_updated_alpha = cell_alpha + self.step_convection_diffusion(
                delta_t=delta_t, cell_phi=cell_alpha,
                facet_velocity=facet_u,
                facet_diffusion=facet_diffusion_alpha / divider,
                facet_dirichlet=facet_dirichlet_alpha,
                facet_neumann=facet_neumann_alpha,
                dict_mlp=dict_mlp, mode=f"alpha{label}")

        if BENCHMARK:
            t_now = datetime.now()
            print(f"Finish solve alpha: {t_now}")
            print(f"{(t_now - t_prev).total_seconds()}")
            t_prev = t_now

        cell_updated_rho = cell_updated_alpha * global_rho_solute + (
            1 - cell_updated_alpha) * global_rho_solvent

        if self.encoded_pressure:
            cell_drho_dt = (cell_updated_rho - cell_original_rho) / delta_t
        else:
            cell_updated_decoded_rho = self.apply_mlp(
                dict_mlp, 'inv_alpha_mlp', cell_updated_rho,
                mode=f"{self.mode}",
                reduce_op=cat_tail, n_split=self.n_bundle,
                scale=self.scale_encdec, l2=self.l2_for_scale_encdec).detach()
            cell_drho_dt = (
                cell_updated_decoded_rho - cell_original_decoded_rho) \
                / delta_t

        if self.ns_solver == 'fractional_step':
            if BENCHMARK:
                t_now = datetime.now()
                print(f"--\nStart solve u: {t_now}")
                t_prev = t_now

            cell_rho_convection_diffusion = \
                self.compute_cc_ns_rho_convection_diffusion(
                    cell_u, cell_p, cell_original_rho,
                    facet_u,
                    facet_original_rho, facet_mu,
                    facet_dirichlet_u, facet_neumann_u,
                    facet_dirichlet_p, facet_neumann_p,
                    facet_previous_u=facet_previous_u,
                    dict_mlp=dict_mlp, label=label)

            if BENCHMARK:
                t_now = datetime.now()
                print(f"Finish solve u: {t_now}")
                print(f"{(t_now - t_prev).total_seconds()}")
                t_prev = t_now

            cell_intermediate_rho_u = torch.einsum(
                'ca,cpa->cpa', cell_original_rho, cell_u) \
                + delta_t * cell_rho_convection_diffusion
            cell_intermediate_u = torch.einsum(
                'ca,cpa->cpa',
                1 / cell_original_rho, cell_intermediate_rho_u)

            # Compute smoothing if needed
            for _ in range(self.n_smoothing):
                cell_intermediate_u = cell_intermediate_u \
                    + self.compute_cc_ns_diffusion(
                        cell_intermediate_u,
                        facet_dirichlet_u, facet_neumann_u,
                        facet_diffusion=facet_smoothing_diffusion,
                        dict_mlp=None) * delta_t

            facet_intermediate_u = self.compute_facet_value_center(
                cell_intermediate_u)
            facet_intermediate_u = self.apply_boundary_facet_vector(
                facet_intermediate_u, facet_dirichlet_u, facet_neumann_u)
            facet_intermediate_rho_u = torch.einsum(
                'fa,fpa->fpa', facet_original_rho, facet_intermediate_u)

            # Solve pressure
            if BENCHMARK:
                t_now = datetime.now()
                print(f"--\nStart solve p: {t_now}")
                t_prev = t_now

            if self.encoded_pressure:
                facet_grad_original_rho = self.compute_fc_grad(
                    cell_original_rho, facet_dirichlet_rho, facet_neumann_rho,
                    dict_mlp=dict_mlp, mode=f"rho{label}")
                facet_intermediate_rho_u_for_p = facet_intermediate_rho_u
                facet_dirichlet_u_for_p = facet_dirichlet_u
                facet_neumann_u_for_p = facet_neumann_u
            else:
                facet_grad_original_rho = self.compute_fc_grad(
                    cell_original_decoded_rho, facet_dirichlet_decoded_rho,
                    facet_neumann_decoded_rho, dict_mlp=None)
                facet_intermediate_rho_u_for_p = self.apply_mlp(
                    dict_mlp, 'inv_u_mlp', facet_intermediate_rho_u,
                    mode=f"{self.mode}",
                    reduce_op=cat_tail, n_split=self.n_bundle,
                    scale=self.scale_encdec, l2=self.l2_for_scale_encdec)
                facet_dirichlet_u_for_p = torch.cat([
                    facet_original_dirichlet_u] * self.n_bundle, dim=-1)
                facet_neumann_u_for_p = torch.cat([
                    facet_original_neumann_u] * self.n_bundle, dim=-1)

            facet_gh_grad_original_rho = torch.einsum(
                'fa,fpa->fpa', facet_gh, facet_grad_original_rho)

            if output_directory is None:
                poisson_output_directory = None
            else:
                poisson_output_directory = output_directory \
                    / f"p_iter_for_{i_step}"
            cell_updated_p, cell_source = self.solve_p(
                delta_t,
                facet_intermediate_rho_u_for_p, cell_p,
                facet_gh_grad_original_rho, cell_drho_dt,
                i_step,
                facet_dirichlet_u_for_p, facet_neumann_u_for_p,
                facet_dirichlet_p, facet_neumann_p, dict_mlp=dict_mlp,
                facet_smoothing_diffusion=facet_source_smoothing_diffusion,
                output_directory=poisson_output_directory)

            if BENCHMARK:
                t_now = datetime.now()
                print(f"Finish solve p: {t_now}")
                print(f"{(t_now - t_prev).total_seconds()}")
                t_prev = t_now

            if BENCHMARK:
                t_now = datetime.now()
                print(f"--\nStart update u: {t_now}")
                t_prev = t_now

            # Update velocity
            facet_updated_p = mul(self.fc_weighted_inc, cell_updated_p)
            facet_updated_p = self.apply_boundary_facet(
                facet_updated_p, facet_dirichlet_p, facet_neumann_p)
            cell_pressure_gradient = self.compute_cf_grad(
                facet_updated_p)
            if not self.encoded_pressure:
                cell_pressure_gradient = self.apply_mlp(
                    dict_mlp, 'encode_u_mlp', cell_pressure_gradient,
                    reduce_op=cat_tail, scale=self.scale_encdec,
                    l2=self.l2_for_scale_encdec)

            if dict_mlp is None or isinstance(
                    dict_mlp[f"rho{label}/rho"],
                    siml.networks.identity.Identity):
                facet_original_rho_buoyancy = facet_original_rho
            else:
                facet_original_rho_buoyancy = self.apply_mlp(
                    dict_mlp, 'rho', facet_original_rho, mode=f"rho{label}")
                facet_original_rho_buoyancy[self.facet_filter_boundary] = \
                    facet_original_rho[self.facet_filter_boundary]

            cell_gh_grad_original_rho = torch.einsum(
                'ca,cpa->cpa',
                cell_gh, self.compute_cf_grad(facet_original_rho_buoyancy))
            cell_updated_u = torch.einsum(
                'ca,cpa->cpa', 1 / cell_updated_rho,
                cell_intermediate_rho_u - delta_t * (
                    cell_pressure_gradient + cell_gh_grad_original_rho))

            if BENCHMARK:
                t_now = datetime.now()
                print(f"Finish update u: {t_now}")
                print(f"{(t_now - t_prev).total_seconds()}")
                t_prev = t_now

        else:
            raise ValueError(f"Unexpected ns_solver: {self.ns_solver}")

        if self.debug:
            print("gradp: {}".format(torch.einsum(
                '...,...->',
                cell_pressure_gradient, cell_pressure_gradient)))
            print("gh_gradrho: {}".format(torch.einsum(
                '...,...->',
                cell_gh_grad_original_rho, cell_gh_grad_original_rho)))

        # Return du/dt * delta_t, dp/dt * delta_t
        return cell_updated_u - cell_u, cell_updated_p - cell_p, \
            cell_updated_alpha - cell_alpha

    def compute_cc_ns_rho_convection_diffusion(
            self, cell_u, cell_p, cell_rho, facet_u, facet_rho, facet_mu,
            facet_dirichlet_u, facet_neumann_u,
            facet_dirichlet_p, facet_neumann_p,
            facet_previous_u=None, dict_mlp=None, label=None):
        if label is None:
            label = ''

        if BENCHMARK:
            t_now = datetime.now()
            print(f"--\nStart prepare u: {t_now}")
            t_prev = t_now

        if facet_previous_u is None:
            facet_u_dot_n = torch.einsum(
                'fpa,fpb->fa', facet_u, self.facet_normal_vector)
        else:
            facet_u_dot_n = torch.einsum(
                'fpa,fpb->fa', facet_previous_u, self.facet_normal_vector)
        facet_rho_u = torch.einsum('fa,fpa->fpa', facet_rho, facet_u)

        if BENCHMARK:
            t_now = datetime.now()
            print(f"Finish prepare u: {t_now}")
            print(f"{(t_now - t_prev).total_seconds()}")
            t_prev = t_now

        if BENCHMARK:
            t_now = datetime.now()
            print(f"--\nStart convection u: {t_now}")
            t_prev = t_now

        # Convection
        facet_rho_u_cross_normal_area = torch.einsum(
            'fpa,fa,fb->fpa',
            facet_rho_u, facet_u_dot_n, self.facet_area)
        if self.trainable:
            facet_rho_u_cross_normal_area_after_mlp = self.apply_mlp(
                dict_mlp, 'conv_u_mlp', facet_rho_u_cross_normal_area,
                mode=f"{self.mode}{label}")
            facet_rho_u_cross_normal_area_after_mlp[
                self.facet_filter_boundary] = \
                facet_rho_u_cross_normal_area[self.facet_filter_boundary]
        else:
            facet_rho_u_cross_normal_area_after_mlp = \
                facet_rho_u_cross_normal_area
        cell_rho_convection = torch.einsum(
            'cpa,cb->cpa',
            mul(self.cf_signed_inc, facet_rho_u_cross_normal_area_after_mlp),
            1 / self.cell_volume / 2
        )

        if BENCHMARK:
            t_now = datetime.now()
            print(f"Finish convection u: {t_now}")
            print(f"{(t_now - t_prev).total_seconds()}")
            t_prev = t_now

        # Diffusion
        cell_rho_diffusion = self.compute_cc_ns_diffusion(
            cell_u, facet_dirichlet_u, facet_neumann_u,
            facet_diffusion=facet_mu,
            dict_mlp=dict_mlp, mode=f"{self.mode}{label}")

        if BENCHMARK:
            t_now = datetime.now()
            print(f"Finish diffusion u: {t_now}")
            print(f"{(t_now - t_prev).total_seconds()}")
            t_prev = t_now

        if self.debug:
            print('--')
            print("rho: {}".format(torch.einsum(
                '...a,...a->a', facet_rho, facet_rho)))
            print("conv: {}".format(torch.einsum(
                '...,...->', cell_rho_convection, cell_rho_convection)))
            print("diff: {}".format(torch.einsum(
                '...,...->', cell_rho_diffusion, cell_rho_diffusion)))
        return - cell_rho_convection + cell_rho_diffusion

    def solve_p(
            self,
            delta_t,
            facet_intermediate_rho_u, cell_p,
            facet_gh_grad_rho, cell_drho_dt,
            i_step,
            facet_dirichlet_u, facet_neumann_u,
            facet_dirichlet_p, facet_neumann_p,
            dict_mlp,
            facet_smoothing_diffusion=None,
            output_directory=None):
        # [p] ~ [L^2] / [T^2]
        # nabla . nabla p = - nabla . (gh nabla rho) \
        #                   + (1 / t) * (nabla . (rho u) + (d rho / d t))

        if self.nonorthogonal_correction:
            correction = self.compute_cf_divergence_correction_over_relaxed(
                - facet_gh_grad_rho + 1 / delta_t * facet_intermediate_rho_u)
            cell_source = self.compute_cf_divergence_over_relaxed(
                - facet_gh_grad_rho + 1 / delta_t * facet_intermediate_rho_u) \
                + 1 / delta_t * cell_drho_dt \
                + correction
        else:
            cell_source = self.compute_cf_divergence_minimum_correction(
                - facet_gh_grad_rho + 1 / delta_t * facet_intermediate_rho_u) \
                + 1 / delta_t * cell_drho_dt

        # Compute source smoothing if needed
        for _ in range(self.n_source_smoothing):
            cell_source = cell_source \
                + self.compute_cc_diffusion_scalar(
                    cell_source,
                    facet_smoothing_diffusion,
                    facet_dirichlet_p, facet_neumann_p,
                    dict_mlp=None) * delta_t

        # NOTE: Do not pass MLPs because of too deep computation graph
        dict_results = self.poisson_solver.solve_poisson(
            cell_p, cell_source=cell_source,
            facet_dirichlet=facet_dirichlet_p,
            facet_neumann=facet_neumann_p,
            dict_mlp=None, output_directory=output_directory)
        cell_p = dict_results['phi']

        return cell_p, cell_source
