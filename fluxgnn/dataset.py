
import os
import pathlib

import torch


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
            raise NotImplementedError
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


def collate_function(batch):
    # ((TorchFVSolver, condition), answer)
    return [b[:2] for b in batch], [b[2] for b in batch]
