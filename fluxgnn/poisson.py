
import torch

from .torch_fv import TorchFVSolver


class PoissonSolver(TorchFVSolver):

    def __init__(self, fem_data, **kwargs):

        original_apply_setting = kwargs.pop('apply_setting', True)
        kwargs['apply_setting'] = False
        super().__init__(fem_data, **kwargs)

        if original_apply_setting:
            self.apply_setting(**kwargs)
        return

    def apply_setting(self, **kwargs):
        super().apply_setting(**kwargs)

        self.linear_solver = kwargs.get('linear_solver', 'cg')
        self.n_iter = kwargs.get('n_iter', len(self.fv_data.cell_volume))
        print(f"n_iter for Poisson: {self.n_iter}")
        self.n_correction = kwargs.get('n_correction', 0)
        self.threshold = kwargs.get('threshold', 1e-6)
        self.factor_initial_threshold = kwargs.get(
            'factor_initial_threshold', 1e-3)
        self.raise_non_convergence = kwargs.get(
            'raise_non_convergence', False)

        return

    def solve_poisson(
            self, cell_initial_phi, cell_source,
            facet_dirichlet, facet_neumann, dict_mlp=None,
            output_directory=None, write=False, store=False):

        initial_residual_phi = None
        cell_iter_phi = cell_initial_phi.clone()
        for i_correction in range(self.n_correction + 1):
            if self.print_period > 0 and self.n_correction > 0:
                print(f"Correction: {i_correction}")

            cell_iter_phi, initial_residual_phi, dict_results = self.solve_cg(
                cell_iter_phi, cell_source,
                initial_residual_phi=initial_residual_phi,
                facet_dirichlet=facet_dirichlet, facet_neumann=facet_neumann,
                dict_mlp=dict_mlp,
                output_directory=output_directory, write=write, store=store,
                prefix=f"correction{i_correction}")

        dict_results.update({'phi': cell_iter_phi})
        return dict_results

    def solve_cg(
            self, cell_initial_phi, cell_source, initial_residual_phi,
            facet_dirichlet, facet_neumann, dict_mlp=None,
            write=False, store=False, output_directory=None, prefix=None):
        residual_phi = torch.tensor(float('inf'))
        cell_iter_phi = cell_initial_phi.clone()
        v = self.cell_volume

        if self.nonorthogonal_correction:
            correction = self.compute_cc_diffusion_scalar_correction(
                cell_iter_phi, facet_diffusion=None,
                facet_dirichlet=facet_dirichlet,
                facet_neumann=facet_neumann, dict_mlp=dict_mlp)
            correction = torch.einsum(
                'c...a,cb->c...a', correction, self.cell_volume)
        else:
            correction = 0.

        # NOTE: Solve (-A) x = (-b) to obtain a positive definite matrix
        a_phi = self.compute_cc_diffusion_scalar(
            cell_iter_phi, facet_diffusion=None,
            facet_dirichlet=facet_dirichlet,
            facet_neumann=facet_neumann, dict_mlp=dict_mlp)
        cg_r = - cell_source - (- a_phi - correction)
        cg_p = cg_r
        cg_r_dot = torch.einsum('if,if,ia->', cg_r, cg_r, v)

        if initial_residual_phi is None:
            initial_residual_phi = cg_r_dot**.5
            if initial_residual_phi < self.threshold \
                    * self.factor_initial_threshold:
                if self.print_period > 0:
                    print(
                        f"Poisson solver converged without loop, "
                        f"residual: {initial_residual_phi:.5e}")
                return cell_iter_phi, initial_residual_phi, {}

        residual_phi = cg_r_dot**.5 / initial_residual_phi

        # NOTE:
        # - Do not apply original boundary conditions because this is
        #   residual computation
        # - Do not add non-orthodonal correction because it make
        #   the matrix ill-condiioned

        # Generate new boundary condition for CG loop since
        # the original b.c.s are already included
        facet_dirichlet_in_loop = torch.ones(
            facet_dirichlet.shape,
            dtype=cell_initial_phi.dtype,
            device=cell_initial_phi.device) * torch.tensor(float('nan'))
        facet_dirichlet_in_loop[~torch.isnan(facet_dirichlet)] = 0.
        facet_neumann_in_loop = torch.ones(
            facet_neumann.shape,
            dtype=cell_initial_phi.dtype,
            device=cell_initial_phi.device) * torch.tensor(float('nan'))
        facet_neumann_in_loop[
            torch.logical_and(
                torch.isnan(facet_dirichlet[:, 0]),
                self.facet_filter_boundary)] = 0.

        converged = False
        dict_results = self.write_store_if_needed(
            dict_results={}, dict_mlp=dict_mlp,
            dict_data={'phi': cell_iter_phi},
            i_time=0, time=0, output_directory=output_directory, write=write,
            store=store)
        for i in range(self.n_iter):
            cg_a_p = - self.compute_cc_diffusion_scalar(
                cg_p, facet_diffusion=None,
                facet_dirichlet=facet_dirichlet_in_loop,
                facet_neumann=facet_neumann_in_loop, dict_mlp=dict_mlp)
            cg_p_a_p = torch.einsum('if,if,ia->', cg_p, cg_a_p, v)
            if cg_p_a_p < 0:
                raise ValueError(
                    f"{cg_p_a_p}\n{cg_p[..., 0]}\n{cg_a_p[..., 0]}")

            cg_alpha = cg_r_dot / (cg_p_a_p + 1e-8)
            cell_new_phi = cell_iter_phi + cg_alpha * cg_p
            cg_r = cg_r - cg_alpha * cg_a_p

            new_cg_r_dot = torch.einsum('if,if,ia->', cg_r, cg_r, v)
            cg_beta = new_cg_r_dot / cg_r_dot
            cg_p = cg_r + cg_beta * cg_p

            cg_r_dot = new_cg_r_dot

            cell_iter_phi = cell_new_phi
            residual_phi = new_cg_r_dot**.5 / initial_residual_phi
            if self.print_period > 0 and i % self.print_period == 0:
                print(f"{i:10d}, residual: {residual_phi:.5e}")
            dict_results = self.write_store_if_needed(
                dict_results=dict_results, dict_mlp=dict_mlp,
                dict_data={'phi': cell_iter_phi},
                i_time=i+1, time=i+1, output_directory=output_directory,
                write=write, store=store)
            if residual_phi < self.threshold:
                if self.print_period > 0:
                    print(
                        f"Poisson solver converged at {i:10d}, "
                        f"residual: {residual_phi:.5e}")
                converged = True
                break

        if not converged:
            print(
                f"!!!Poisson solver NOT converged with {i:10d} loops, "
                f"residual: {residual_phi:.5e} vs {self.threshold:.5e}")
            if self.raise_non_convergence:
                raise ValueError(
                    f"!!!Poisson solver NOT converged with {i:10d} loops, "
                    f"residual: {residual_phi:.5e} vs {self.threshold:.5e}")

        dict_results = self.write_store_if_needed(
            dict_results=dict_results, dict_mlp=dict_mlp,
            dict_data={'phi': cell_iter_phi},
            i_time=i+1, time=i+1, output_directory=output_directory,
            force_store=True, overwrite=True, write=write, store=store)
        return cell_iter_phi, initial_residual_phi, dict_results
