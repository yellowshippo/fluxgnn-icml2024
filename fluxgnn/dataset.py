
import os
import pathlib
import re
import sys

import femio
import numpy as np
import torch

from fluxgnn import util


TORCH_NAN = torch.tensor(float('nan'))


class TorchFVDataset(torch.utils.data.Dataset):

    def __init__(
            self, dataset_name, data_directories, torch_fv_constructor,
            setting_dict, property_for_ml=False, start_index=0, end_index=None,
            save=True, dataset_file_name='train', force_load=False,
            compute_variance=False,
            **kwargs):
        self.diffusion_alpha = kwargs.pop('diffusion_alpha', 1.e-4)
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.data_directories = data_directories
        self.torch_fv_constructor = torch_fv_constructor
        self.setting_dict = setting_dict
        self.start_index = start_index
        self.end_index = end_index
        self.property_for_ml = property_for_ml

        common_path = pathlib.Path(os.path.commonpath(data_directories))
        dataset_file = common_path / f"{dataset_file_name}.pth"
        if dataset_file.is_file() and not force_load:
            print(f"Loading cached dataset in: {dataset_file}")
            dataset = torch.load(dataset_file)
            self.torch_fv_dataset = dataset.torch_fv_dataset
            for i in range(len(self.torch_fv_dataset)):
                self.torch_fv_dataset[i][0].apply_setting(**setting_dict)
        else:
            if not force_load:
                print(f"{dataset_file} not found. Load raw data.")
            self.torch_fv_dataset = self.load_dataset()
            if save:
                print(f"Saving {dataset_file}... ", end='')
                torch.save(self, dataset_file)
                print('Saved.')

        if compute_variance:
            self.compute_variance()
        return

    def __len__(self):
        return len(self.data_directories)

    def __getitem__(self, i):
        """
        Returns
            torch_fv_solver: TorchFVSolver
            dict_conditions: dict[dict[str, torch.Tensor]]
                - initial: dict[str, torch.Tensor]
                - dirichlet: dict[str, torch.Tensor]
                - neumann: dict[str, torch.Tensor]
                - property: dict[str, torch.Tensor]
            dict_answer: dict[str, torch.tensor]
        """
        return self.torch_fv_dataset[i]

    def compute_variance(self):
        dict_tmp_stats = {
            k: {'n': 0, 'sum': 0., 'squared_sum': 0.}
            for k in self.torch_fv_dataset[0][-1].keys()}
        for data in self.torch_fv_dataset:
            d = data[-1]
            for k, v in d.items():
                dict_tmp_stats[k]['n'] = dict_tmp_stats[k]['n'] \
                    + v.shape[0] * v.shape[1]  # n_time * n_cell
                dict_tmp_stats[k]['sum'] = dict_tmp_stats[k]['sum'] \
                    + torch.sum(v)
                dict_tmp_stats[k]['squared_sum'] = dict_tmp_stats[k][
                    'squared_sum'] + torch.sum(v**2)

        self.dict_variance = {}
        for k, v in dict_tmp_stats.items():
            n = v['n']
            if k == 'u':
                # Assume mean is zero for velocity
                self.dict_variance[k] = v['squared_sum'] / n
            else:
                self.dict_variance[k] = v['squared_sum'] / n - (
                    v['sum'] / n)**2
        return

    def load_dataset(self):
        return [
            self.load_single_data(d) for d in self.data_directories]

    def load_single_data(self, data_directory):
        sorted_vtu_paths = get_sorted_vtu_paths(data_directory)[
            self.start_index:self.end_index]
        vtu_path = sorted_vtu_paths[0]
        torch_fv = self.torch_fv_constructor.from_vtu(
            vtu_path, **self.setting_dict)
        torch_fv.trainable = True

        conditions, answers = self.generate_conditions(
            data_directory, torch_fv, sorted_vtu_paths)
        return torch_fv, conditions, answers

    def generate_conditions(self, data_directory, torch_fv, sorted_vtu_path):
        dict_transforms = self.load_transforms(data_directory)
        if self.dataset_name == 'cavity':
            return self.generate_conditions_cavity(
                torch_fv, sorted_vtu_path, dict_transforms=dict_transforms)
        elif self.dataset_name == 'convection_diffusion':
            return self.generate_conditions_convection_diffusion(
                torch_fv, sorted_vtu_path, dict_transforms=dict_transforms)
        elif self.dataset_name == 'cavity_mixture':
            return self.generate_conditions_cavity_mixture(
                torch_fv, sorted_vtu_path, dict_transforms=dict_transforms)
        else:
            raise ValueError(f"Unexpected dataset_name: {self.dataset_name}")

    def load_transforms(self, data_directory):
        rotation_matrix_path = data_directory / 'rotation_matrix.txt'
        if rotation_matrix_path.is_file():
            rotation_matrix = np.loadtxt(rotation_matrix_path)
            print(f"Loaded rotation matirx:\n{rotation_matrix}")
        else:
            rotation_matrix = np.eye(3)

        space_scale_factor_path = data_directory / 'space_scale_factor.txt'
        if space_scale_factor_path.is_file():
            space_scale_factor = float(np.squeeze(
                np.loadtxt(space_scale_factor_path)))
            time_scale_factor = float(np.squeeze(
                np.loadtxt(data_directory / 'time_scale_factor.txt')))
            mass_scale_factor = float(np.squeeze(
                np.loadtxt(data_directory / 'mass_scale_factor.txt')))
            print('Loaded scaling factors:')
            print(f"  - Space: {space_scale_factor}")
            print(f"  - Time: {time_scale_factor}")
            print(f"  - Mass: {mass_scale_factor}")
        else:
            space_scale_factor = 1.
            time_scale_factor = 1.
            mass_scale_factor = 1.

        return {
            'rotation_matrix': rotation_matrix,
            'space_scale_factor': space_scale_factor,
            'time_scale_factor': time_scale_factor,
            'mass_scale_factor': mass_scale_factor,
        }

    def generate_conditions_cavity(
            self, torch_fv, sorted_vtu_paths, dict_transforms):
        cell_initial_u = util.torch_tensor(
            torch_fv.fv_data.fem_data.elemental_data.get_attribute_data('U'))
        cell_initial_p = util.torch_tensor(
            torch_fv.fv_data.fem_data.elemental_data.get_attribute_data('p'))

        if '3d' in str(sorted_vtu_paths[0]):
            print(f"Apply 3D boundary condition: {sorted_vtu_paths[0]}")
            facet_dirichlet_u, facet_neumann_u, \
                facet_dirichlet_p, facet_neumann_p \
                = util.generate_facet_3d_cavity_boundary_tensors(
                    torch_fv.fv_data.facet_pos)
        else:
            facet_dirichlet_u, facet_neumann_u, \
                facet_dirichlet_p, facet_neumann_p \
                = util.generate_facet_cavity_boundary_tensors(
                    torch_fv.fv_data.facet_pos)

        dict_timeseries = self.load_timeseries(
            sorted_vtu_paths, ['U', 'p'])
        cell_answer_u = dict_timeseries['U']
        cell_answer_p = dict_timeseries['p']

        global_nu = .1

        return {
            'initial': {'u': cell_initial_u, 'p': cell_initial_p},
            'dirichlet': {'u': facet_dirichlet_u, 'p': facet_dirichlet_p},
            'neumann': {'u': facet_neumann_u, 'p': facet_neumann_p},
            'periodic': {},
            'property': {'nu': util.torch_tensor(global_nu)},
        }, {'u': cell_answer_u, 'p': cell_answer_p}

    def generate_conditions_cavity_mixture(
            self, torch_fv, sorted_vtu_paths, dict_transforms):
        cell_initial_u = util.torch_tensor(
            torch_fv.fv_data.fem_data.elemental_data.get_attribute_data('U'))
        cell_initial_p = util.torch_tensor(
            torch_fv.fv_data.fem_data.elemental_data.get_attribute_data(
                'p_rgh'))
        cell_initial_alpha = util.torch_tensor(
            torch_fv.fv_data.fem_data.elemental_data.get_attribute_data(
                'alpha.solute'))
        np_rotation_matrix = dict_transforms['rotation_matrix']

        space_scale_factor = dict_transforms['space_scale_factor']
        time_scale_factor = dict_transforms['time_scale_factor']
        mass_scale_factor = dict_transforms['mass_scale_factor']

        str_path = str(sorted_vtu_paths[0].parent)
        if '3d' in str(str_path):
            print(f"Apply 3D boundary condition: {sorted_vtu_paths[0]}")
            facet_dirichlet_u, facet_neumann_u, \
                facet_dirichlet_p, facet_neumann_p, \
                facet_dirichlet_alpha, facet_neumann_alpha \
                = util.generate_facet_3d_closed_mixture_boundary_tensors(
                    torch_fv.fv_data, rotation_matrix=np_rotation_matrix)
        else:
            facet_dirichlet_u, facet_neumann_u, \
                facet_dirichlet_p, facet_neumann_p, \
                facet_dirichlet_alpha, facet_neumann_alpha \
                = util.generate_facet_2d_closed_mixture_boundary_tensors(
                    torch_fv.fv_data, rotation_matrix=np_rotation_matrix)

        dict_timeseries = self.load_timeseries(
            sorted_vtu_paths, ['U', 'p_rgh', 'alpha.solute'])
        cell_answer_u = dict_timeseries['U']
        cell_answer_p = dict_timeseries['p_rgh']
        cell_answer_alpha = dict_timeseries['alpha.solute']

        global_diffusion_alpha = self.diffusion_alpha \
            * space_scale_factor**2 * time_scale_factor**(-1)

        nu = 1e-3 \
            * space_scale_factor**2 * time_scale_factor**(-1)
        global_nu_solute = util.torch_tensor(np.array([nu])[None, :])
        global_nu_solvent = util.torch_tensor(np.array([nu])[None, :])
        facet_diffusion_alpha = util.torch_tensor(
            global_diffusion_alpha
            * np.ones((len(torch_fv.fv_data.facet_pos), 1)))

        global_gravity = util.torch_tensor(
            np.einsum(
                'pq,q->p', np_rotation_matrix, np.array([0., -9.81, 0.])
            )[None, :, None]) \
            * space_scale_factor**1 * time_scale_factor**(-2)

        if 'rho' in str(str_path):
            captures = re.findall(r'rho(\d+)', str_path)
            rho = float(captures[-1]) \
                * space_scale_factor**(-3) * mass_scale_factor**1
        else:
            rho = 990. \
                * space_scale_factor**(-3) * mass_scale_factor**1
        global_rho_solute = util.torch_tensor(np.array([rho])[None, :])
        print(f"diffusion for alpha: {global_diffusion_alpha} for {str_path}")
        print(f"rho_solute: {rho} for {str_path}")
        global_rho_solvent = util.torch_tensor(np.array([1000.])[None, :]) \
            * space_scale_factor**(-3) * mass_scale_factor**1

        return {
            'initial': {
                'u': cell_initial_u, 'p': cell_initial_p,
                'alpha': cell_initial_alpha,
            },
            'dirichlet': {
                'u': facet_dirichlet_u, 'p': facet_dirichlet_p,
                'alpha': facet_dirichlet_alpha,
            },
            'neumann': {
                'u': facet_neumann_u, 'p': facet_neumann_p,
                'alpha': facet_neumann_alpha,
            },
            'periodic': {},
            'property': {
                'nu_solute': global_nu_solute,
                'nu_solvent': global_nu_solvent,
                'rho_solute': global_rho_solute,
                'rho_solvent': global_rho_solvent,
                'diffusion_alpha': facet_diffusion_alpha,
                'gravity': global_gravity,
                'space_scale_factor': util.torch_tensor(
                    np.array([space_scale_factor])[None, :]),
                'time_scale_factor': util.torch_tensor(
                    np.array([time_scale_factor])[None, :]),
                'mass_scale_factor': util.torch_tensor(
                    np.array([mass_scale_factor])[None, :]),
            },
        }, {'u': cell_answer_u, 'p': cell_answer_p, 'alpha': cell_answer_alpha}

    def generate_conditions_convection_diffusion(
            self, torch_fv, sorted_vtu_paths, dict_transforms):
        # Initial
        cell_initial_phi = util.torch_tensor(
            torch_fv.fv_data.fem_data.elemental_data.get_attribute_data(
                'phi'))

        # Boundary
        facet_x = torch_fv.fv_data.facet_pos[:, 0]
        min_x = np.min(facet_x)
        max_x = np.max(facet_x)
        facet_filter_source = torch.from_numpy(np.abs(facet_x - max_x) < 1e-5)
        facet_filter_destination = torch.from_numpy(
            np.abs(facet_x - min_x) < 1e-5)

        facet_dirichlet = torch.ones((torch_fv.n_facet, 1)) * TORCH_NAN

        facet_neumann = torch.ones((torch_fv.n_facet, 3, 1)) * TORCH_NAN
        facet_neumann[torch_fv.facet_filter_boundary] = 0.
        facet_neumann[facet_filter_source] = TORCH_NAN
        facet_neumann[facet_filter_destination] = TORCH_NAN

        periodic = {
            'source': facet_filter_source,
            'destination': facet_filter_destination}

        # Property
        ux = torch_fv.fv_data.fem_data.elemental_data.get_attribute_data(
            'u')[0, 0]
        d = torch_fv.fv_data.fem_data.elemental_data.get_attribute_data(
            'diffusion')[0, 0]
        facet_u = torch.zeros((torch_fv.n_facet, 3, 1))
        facet_u[:, 0, 0] = ux
        facet_diffusion = torch.ones((torch_fv.n_facet, 1)) * d

        facet_dirichlet_u, facet_neumann_u, \
            facet_dirichlet_p, facet_neumann_p \
            = util.generate_facet_cavity_boundary_tensors(
                torch_fv.fv_data.facet_pos)

        dict_timeseries = self.load_timeseries(
            sorted_vtu_paths, ['phi'])
        cell_answer_phi = dict_timeseries['phi']
        return {
            'initial': {'phi': cell_initial_phi},
            'dirichlet': {'phi': facet_dirichlet},
            'neumann': {'phi': facet_neumann},
            'periodic': periodic,
            'property': {'u': facet_u, 'diffusion': facet_diffusion},
        }, {'phi': cell_answer_phi}

    def load_timeseries(self, sorted_vtu_paths, keys):
        with open(os.devnull, 'w') as f:
            sys.stdout = f

            data_dict = {k: [] for k in keys}
            for sorted_vtu_path in sorted_vtu_paths:
                fem_data = femio.read_files('vtu', sorted_vtu_path)
                for k in keys:
                    data_dict[k].append(
                        util.torch_tensor(
                            fem_data.elemental_data.get_attribute_data(k)))

        sys.stdout = sys.__stdout__
        return {k: torch.stack(v, dim=0) for k, v in data_dict.items()}



def collate_function(batch):
    # ((TorchFVSolver, condition), answer)
    return [b[:2] for b in batch], [b[2] for b in batch]


def get_sorted_vtu_paths(data_directory, key=None):
    if key is None:
        key = '*.vtu'
    print(f"Loading: {data_directory}")
    vtu_paths = np.array(list(data_directory.glob('**/' + key)))
    steps = [
        extract_number(str(p))
        for p in vtu_paths]
    sorted_indices = np.argsort(steps)
    return vtu_paths[sorted_indices]


def extract_number(string):
    captures = re.findall(r'(\d+)', string)
    return int(captures[-1])
