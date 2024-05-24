
from .torch_fv import TorchFVSolver
from .poisson import PoissonSolver


class NavierStokesSolver(TorchFVSolver):

    def __init__(self, fem_data, **kwargs):

        original_apply_setting = kwargs.pop('apply_setting', True)
        kwargs['apply_setting'] = False
        super().__init__(fem_data, **kwargs)

        # Initialize Poisson solver with default setting to apply setting later
        self.poisson_solver = PoissonSolver(
            self.fv_data.fem_data, apply_setting=False)

        if original_apply_setting:
            self.apply_setting(**kwargs)
        return

    def apply_setting(self, **kwargs):
        super().apply_setting(**kwargs)

        self.ns_solver = kwargs.get('ns_solver', 'fractional_step')
        self.u_interpolation_scheme = kwargs.get(
            'u_interpolation_scheme', 'upwind')

        self.p_solver = kwargs.get('p_solver', 'cg')
        self.n_iter_p_factor = kwargs.get('n_iter_p_factor', None)
        if self.n_iter_p_factor is None:
            self.n_iter_p = kwargs.get(
                'n_iter_p', len(self.fv_data.cell_pos) * int(1e3))
        else:
            self.n_iter_p = int(
                len(self.fv_data.cell_pos) * self.n_iter_p_factor)
        self.diffusion_method_p = kwargs.get(
            'diffusion_method_p', 'minimum_correction')
        self.nonorthogonal_correction_p = kwargs.get(
            'nonorthogonal_correction_p', False)
        self.n_correction_p = kwargs.get('n_correction_p', 0)
        self.threshold_p = kwargs.get('threshold_p', 5.e-2)
        self.print_period_p = kwargs.get('print_period_p', -1)
        self.write_mesh_period_p = kwargs.get('write_mesh_period_p', -1)
        self.raise_non_convergence = kwargs.get('raise_non_convergence', False)

        if self.u_interpolation_scheme == 'upwind':
            self.compute_facet_u = self.compute_facet_value_upwind
        elif self.u_interpolation_scheme == 'center':
            self.compute_facet_u = self.compute_facet_value_center
        else:
            raise ValueError(
                f"Unexpected {self.u_interpolation_scheme = }")

        self.poisson_solver.apply_setting(
            linear_solver=self.p_solver,
            n_iter=self.n_iter_p,
            threshold=self.threshold_p,
            diffusion_method=self.diffusion_method_p,
            nonorthogonal_correction=self.nonorthogonal_correction_p,
            write_mesh_period=self.write_mesh_period_p,
            print_period=self.print_period_p, trainable=False,
            n_correction=self.n_correction_p,
            raise_non_convergence=self.raise_non_convergence,
            dtype=self.dtype)

        print(f"n_time_repeat: {self.n_time_repeat}")
        return

    def to(self, device):
        super().to(device)
        self.poisson_solver.to(device)
        return self

    def compute_cc_ns_diffusion(
            self, cell_u, facet_dirichlet_u, facet_neumann_u, dict_mlp,
            facet_diffusion=None, mode=None):
        return self.compute_cc_vector_laplacian(
            cell_u, facet_diffusion=facet_diffusion,
            facet_dirichlet=facet_dirichlet_u,
            facet_neumann=facet_neumann_u, dict_mlp=dict_mlp, mode=mode)
