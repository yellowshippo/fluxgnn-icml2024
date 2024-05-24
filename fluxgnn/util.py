
import femio
import numpy as np
import torch
from scipy.stats import special_ortho_group


def rmse(x, y, weight=None, relative=False):
    x_ = numpy_array(x)
    y_ = numpy_array(y)

    if weight is None:
        weight = np.ones((len(x_), 1))
    if relative:
        coeff = 1 / (
            np.mean(
                np.einsum('na,n...a->...', weight, y_**2)
                / np.sum(weight)))**.5
    else:
        coeff = 1.
    if len(x_) != len(weight) or len(y_) != len(weight):
        raise ValueError(
            f"Invalid length of data: {len(x_)}, {len(y_)}, {len(weight)}")
    return (np.mean(
        np.einsum('na,n...a->...', weight, (x_ - y_)**2)
        / np.sum(weight)))**.5 * coeff


def assert_close(x, y, decimal=7, weight=None):
    if isinstance(x, torch.Tensor):
        np_x = x.detach().numpy()
    else:
        np_x = x

    if isinstance(y, torch.Tensor):
        np_y = y.detach().numpy()
    else:
        np_y = y

    np.testing.assert_almost_equal(
        rmse(np_x, np_y, weight=weight), 0, decimal=decimal)
    return


def rotate_fem_data(fem_data):
    rotation_matrix = special_ortho_group.rvs(3)
    rotated_nodes = np.einsum(
        'pq,iq->ip', rotation_matrix, fem_data.nodes.data)
    rotated_fem_data = femio.FEMData(
        nodes=femio.FEMAttribute(
            'NODE', fem_data.nodes.ids, rotated_nodes),
        elements=fem_data.elements)
    return rotated_fem_data, rotation_matrix


def scale_fem_data(fem_data, scale_factor=2.):
    scaled_fem_data = femio.FEMData(
        nodes=femio.FEMAttribute(
            'NODE', fem_data.nodes.ids,
            fem_data.nodes.data * scale_factor),
        elements=fem_data.elements)
    return scaled_fem_data, scale_factor


def torch_sparse(
        scipy_sparse, device=torch.device('cpu'), dtype=torch.float32):
    if isinstance(scipy_sparse, list):
        return [torch_sparse(s, dtype=dtype) for s in scipy_sparse]

    coo = scipy_sparse.tocoo()
    row = torch.LongTensor(coo.row)
    col = torch.LongTensor(coo.col)
    data = torch_tensor(coo.data, device=device, dtype=dtype)[..., 0]
    return torch.sparse_coo_tensor(
        torch.stack([row, col]), data, scipy_sparse.shape)


def torch_tensor(numpy_array, device=None, dtype=torch.float32):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array.to(device=device, dtype=dtype)
    if not isinstance(numpy_array, np.ndarray):
        numpy_array = np.atleast_2d(numpy_array)

    if numpy_array.dtype == bool:
        # Ignore dtype when the original is bool
        return torch.from_numpy(regularize_shape(numpy_array)).to(
            device=device, dtype=bool)
    else:
        return torch.from_numpy(regularize_shape(numpy_array)).to(
            device=device, dtype=dtype)


def regularize_shape(x):
    if x.shape[-1] > 1:
        return x[..., None]
    else:
        return x


def numpy_array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().numpy()
    elif isinstance(x, np.ndarray):
        return x


def cat_tail(tensors):
    return torch.cat(tensors, dim=-1)


def stack_tail(tensors):
    return torch.stack(tensors, dim=-1)


def list_mean(tensors):
    return torch.mean(torch.stack(tensors, dim=0), dim=0)
