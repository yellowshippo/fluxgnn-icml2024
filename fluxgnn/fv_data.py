
from datetime import datetime
import numpy as np


class FVData:

    def __init__(
            self, fem_data, *,
            diffusion_method='default',
            upwind=True,
            t_max=1.,
            delta_t=1e-3,
            print_period=100,
            write_mesh_period=100,
    ):
        self.fem_data = fem_data

        # Solver setting
        self.diffusion_method = diffusion_method
        self.upwind = upwind

        # Simulation setting
        self.t_max = t_max
        self.delta_t = delta_t

        # Utility setting
        self.print_period = print_period
        self.write_mesh_period = write_mesh_period

        self.init_fem_data()
        return

    def init_fem_data(self):

        # NOTE: This information can be obtained from data file
        self.facet_fem_data, self.cf_signed_inc, \
            self.facet_normal_vector \
            = self.fem_data.calculate_normal_incidence_matrix()

        t0 = datetime.now()
        self.fc_signed_inc = self.cf_signed_inc.transpose()
        s = self.cf_signed_inc

        self.cf_positive_inc = (s + np.abs(s)) / 2
        self.cf_negative_inc = - (- s + np.abs(s)) / 2
        self.fc_positive_inc = self.cf_positive_inc.transpose()
        self.fc_negative_inc = self.cf_negative_inc.transpose()

        self.facet_filter_boundary = np.ravel(
            self.cf_signed_inc.T.dot(
                np.ones((len(self.fem_data.elements), 1))) > .5)

        self.cf_normal_inc = [
            self.cf_signed_inc.multiply(
                self.facet_normal_vector[:, [0]].T),
            self.cf_signed_inc.multiply(
                self.facet_normal_vector[:, [1]].T),
            self.cf_signed_inc.multiply(
                self.facet_normal_vector[:, [2]].T),
        ]  # (dim, n_cell, n_facet)-shape, where n_facet does not double count.

        self.cell_volume = self.fem_data.calculate_element_volumes()
        self.facet_area = self.facet_fem_data.calculate_element_areas()
        self.fc_inc = self.facet_fem_data \
            .calculate_relative_incidence_metrix_element(
                self.fem_data, minimum_n_sharing=3).astype(int)
        self.cf_inc = self.fc_inc.transpose()

        self.cell_pos = self.fem_data.convert_nodal2elemental(
            self.fem_data.nodes.data, calc_average=True)
        self.facet_pos = self.facet_fem_data.convert_nodal2elemental(
            self.facet_fem_data.nodes.data, calc_average=True)

        facet_positive_node = self.cf_positive_inc.T.dot(
            self.cell_pos)
        facet_negative_node = self.cf_negative_inc.T.dot(
            self.cell_pos)
        self.facet_relative_position = - (
            facet_positive_node + facet_negative_node)

        # At boundary, d_n := (n . d) n
        self.facet_relative_position[self.facet_filter_boundary] = \
            self.facet_pos[self.facet_filter_boundary] \
            - self.fc_inc.dot(self.cell_pos)[
                self.facet_filter_boundary]
        # (d . n)
        self.facet_n_dot_d = np.einsum(
            'fq,fq->f',
            self.facet_normal_vector, self.facet_relative_position)[:, None]
        self.facet_relative_position[self.facet_filter_boundary] \
            = np.einsum(
                'fa,fp->fp',
                self.facet_n_dot_d,
                self.facet_normal_vector)[self.facet_filter_boundary]

        # (d . d)
        self.facet_d_square = np.einsum(
            'fp,fp->f',
            self.facet_relative_position,
            self.facet_relative_position)[:, None]

        # (d . n) / (d . d)
        self.facet_normalized_dot = self.facet_n_dot_d / self.facet_d_square

        # Minimum correction approach
        # S [(d . n) / (d . d)] d
        self.facet_divergence_vector_minimum_correction = np.einsum(
            'fp,fq,fq->fp',
            self.facet_relative_position,
            self.facet_normalized_dot,
            self.facet_area)

        # Over-relaxed approach
        # [S / (d . n)] d
        self.facet_divergence_vector_over_relaxed = np.einsum(
            'fa,fa,fp->fp',
            self.facet_area,
            1 / self.facet_n_dot_d, self.facet_relative_position)

        # At boundary, S n
        facet_area_normal = np.einsum(
            'fq,fp->fp', self.facet_area, self.facet_normal_vector)
        self.facet_divergence_vector_minimum_correction[
            self.facet_filter_boundary] = facet_area_normal[
                self.facet_filter_boundary]
        self.facet_divergence_vector_over_relaxed[
            self.facet_filter_boundary] = facet_area_normal[
                self.facet_filter_boundary]

        self.facet_nonorthogonal_minimum_correction = facet_area_normal \
            - self.facet_divergence_vector_minimum_correction
        self.facet_nonorthogonal_over_relaxed = facet_area_normal \
            - self.facet_divergence_vector_over_relaxed

        facet_d = self.facet_relative_position
        facet_d_sq_norm = np.einsum('fp,fp->f', facet_d, facet_d)[:, None]
        self.facet_d_norm = facet_d_sq_norm**.5
        min_d = np.min(self.facet_d_norm)
        self.facet_inv_d_norm = 1 / (self.facet_d_norm + 1e-5 * min_d)

        fc_inc_cell_position = [
            self.fc_inc.T.multiply(self.cell_pos[:, [0]]).T,
            self.fc_inc.T.multiply(self.cell_pos[:, [1]]).T,
            self.fc_inc.T.multiply(self.cell_pos[:, [2]]).T,
        ]
        fc_inc_facet_position = [
            self.fc_inc.multiply(self.facet_pos[:, [0]]),
            self.fc_inc.multiply(self.facet_pos[:, [1]]),
            self.fc_inc.multiply(self.facet_pos[:, [2]]),
        ]
        self.fc_inc_relative_position = [
            fc_inc_cell_position[0] - fc_inc_facet_position[0],
            fc_inc_cell_position[1] - fc_inc_facet_position[1],
            fc_inc_cell_position[2] - fc_inc_facet_position[2],
        ]
        self.fc_inc_distance = np.sqrt(
            self.fc_inc_relative_position[0].multiply(
                self.fc_inc_relative_position[0])
            + self.fc_inc_relative_position[1].multiply(
                self.fc_inc_relative_position[1])
            + self.fc_inc_relative_position[2].multiply(
                self.fc_inc_relative_position[2]))
        self.fc_weighted_inc = self.fc_inc_distance.multiply(
            1 / np.array(self.fc_inc_distance.sum(axis=1)))
        self.cf_weighted_inc = self.fc_weighted_inc.transpose()
        self.fc_weighted_positive_inc = self.cf_positive_inc.T.multiply(
            self.fc_weighted_inc)
        self.fc_weighted_negative_inc = self.cf_negative_inc.T.multiply(
            self.fc_weighted_inc)

        self.facet_normalized_d = facet_d / self.facet_d_norm
        # (1 / | d |) * (d / | d |)
        self.facet_differential_d = facet_d / facet_d_sq_norm

        # # For boundary conditions
        self.bc_signed_inc = self.cf_signed_inc.T[
            self.facet_filter_boundary]
        self.bc_grad_inc = [
            self.bc_signed_inc.T.multiply(
                self.facet_differential_d[
                    self.facet_filter_boundary, [0]]).T,
            self.bc_signed_inc.T.multiply(
                self.facet_differential_d[
                    self.facet_filter_boundary, [1]]).T,
            self.bc_signed_inc.T.multiply(
                self.facet_differential_d[
                    self.facet_filter_boundary, [2]]).T,
        ]

        self.facet_volume = self.compute_facet_value_center(
            self.cell_volume)
        self.cell_length_scale = self.cell_volume**(1/3)
        self.facet_length_scale = self.facet_volume / self.facet_area

        t1 = datetime.now()
        self.preprocess_time = (t1 - t0).total_seconds()
        print(f"Preprocess time: {self.preprocess_time}")

        # For backward compatibility - not used
        self.cf_div_inc_minimum_correction = [
            self.cf_signed_inc.multiply(
                self.facet_divergence_vector_minimum_correction[:, [0]].T),
            self.cf_signed_inc.multiply(
                self.facet_divergence_vector_minimum_correction[:, [1]].T),
            self.cf_signed_inc.multiply(
                self.facet_divergence_vector_minimum_correction[:, [2]].T),
        ]

        self.fc_inc_gradient = [
            - self.cf_signed_inc.multiply(
                self.facet_differential_d[:, [0]].T).T,
            - self.cf_signed_inc.multiply(
                self.facet_differential_d[:, [1]].T).T,
            - self.cf_signed_inc.multiply(
                self.facet_differential_d[:, [2]].T).T,
        ]

        return

    def compute_facet_value_center(self, phi):
        ave_facet_value = self.fc_weighted_inc.dot(phi) \
            / np.array(self.fc_weighted_inc.sum(axis=1))

        facet_value = ave_facet_value
        return facet_value
