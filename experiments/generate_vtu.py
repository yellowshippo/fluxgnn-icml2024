import argparse
from distutils.util import strtobool
import pathlib
import re

import femio
import numpy as np
import pandas as pd
import siml

import stats


DT = .2
REQUIRED_FILE_NAME = 'predicted_nodal_u_step8.npy'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'predicted_base_directory',
        type=pathlib.Path,
        help='Directory containing predicted data')
    parser.add_argument(
        'interim_base_directory',
        type=pathlib.Path,
        help='The corresponding root interim directory')
    parser.add_argument(
        '-p', '--preprocessors-pkl',
        type=pathlib.Path,
        default=None,
        help='Pretrained directory name')
    parser.add_argument(
        '-f', '--force-renew',
        type=strtobool,
        default=0,
        help='If True, force renew data')
    args = parser.parse_args()

    if args.preprocessors_pkl is not None:
        converter = siml.prepost.Converter(args.preprocessors_pkl)
    df = pd.read_csv(
        args.predicted_base_directory / 'prediction.csv', header=0,
        index_col=0, skipinitialspace=True)

    data_directories = sorted([
        f.parent for f
        in args.predicted_base_directory.glob(f"**/{REQUIRED_FILE_NAME}")])

    list_dict = []
    stats_file = args.predicted_base_directory / 'stats.csv'
    with open(stats_file, 'w') as f:
        f.write(
            'directory,'
            'mse_u,stderror_u,'
            'mse_p,stderror_p,'
            'mse_alpha,stderror_alpha,'
            'conservation_error_alpha,'
            'prediction_time\n'
        )
    for data_directory in data_directories:
        fem_data, interim_path = process_one_directory(
            data_directory, args, converter)
        relative_directory = '/'.join(
            data_directory.relative_to(
                args.predicted_base_directory).parts[3:])
        if 'rotation' in str(data_directory):
            directory = \
                '../data/mixture/transformed/preprocessed/test/rotation/' \
                f"{relative_directory}"
        elif 'scaling' in str(data_directory):
            directory = \
                '../data/mixture/transformed/preprocessed/test/scaling/' \
                f"{relative_directory}"
        elif 'larger' in str(data_directory):
            directory = \
                '../data/mixture/larger/preprocessed/' \
                f"{relative_directory}"
        elif 'taller' in str(data_directory):
            directory = \
                '../data/mixture/taller/preprocessed/' \
                f"{relative_directory}"
        elif 'dim3' in str(data_directory):
            directory = \
                '../data/mixture/dim3/preprocessed/' \
                f"{relative_directory}"
        else:
            # Reference data
            directory = '../data/mixture/preprocessed/test/' \
                f"{relative_directory}"
        try:
            record = df.loc[directory]
        except KeyError as e:
            raise ValueError(
                f"{e}\n{df.index} vs {directory}\n"
                f"data_directory: {data_directory}")
        prediction_time = record['prediction_time']
        graph_creation_time = record['graph_creation_time']

        loss_dict = stats.calculate_single_loss_mixture(
            data_directory, interim_path, fem_data,
            force_renew=args.force_renew)
        loss_dict.update({
            'prediction_time': prediction_time,
            'graph_creation_time': graph_creation_time,
        })
        list_dict.append(loss_dict)

        with open(stats_file, 'a') as f:
            f.write(
                f"{directory},"
                f"{loss_dict['mse_u']},"
                f"{loss_dict['stderror_u']},"
                f"{loss_dict['mse_p']},"
                f"{loss_dict['stderror_p']},"
                f"{loss_dict['mse_alpha']},"
                f"{loss_dict['stderror_alpha']},"
                f"{loss_dict['conservation_error_alpha']},"
                f"{prediction_time}\n"
            )
    print(f"Stats written in: {stats_file}")

    global_mse_u, global_std_error_u = stats.calculate_global_stats(
        list_dict, 'u')
    global_mse_p, global_std_error_p = stats.calculate_global_stats(
        list_dict, 'p')
    global_mse_alpha, global_std_error_alpha = stats.calculate_global_stats(
        list_dict, 'alpha')

    errors = np.stack([d['conservation_error_alpha'] for d in list_dict])
    global_mean_conservation_error_alpha = np.mean(errors)
    global_std_error_conservation_error_alpha = np.std(
        errors) / len(errors)**.5

    mean_time, std_error_time = stats.calculate_time(list_dict)
    graph_mean_time, graph_std_error_time = stats.calculate_time(
        list_dict, key='graph_creation_time')

    print('--')
    global_stats_file = args.predicted_base_directory / 'global_stats.csv'
    with open(global_stats_file, 'w') as f:
        print(
            f"mse_u: {global_mse_u:.5e} +/- {global_std_error_u:5e}")
        print(
            f"mse_p: {global_mse_p:.5e} +/- {global_std_error_p:5e}")
        print(
            f"mse_alpha: {global_mse_alpha:.5e} +/- "
            f"{global_std_error_alpha:5e}")

        print(
            'mean_conservation_error_alpha: '
            f"{global_mean_conservation_error_alpha:.5e} +/- "
            f"{global_std_error_conservation_error_alpha:5e}")
        print(
            f"prediction_time: {mean_time:.5e} +/- "
            f"{std_error_time:.5e}")

        print(
            f"prediction_time: {mean_time:.5e} +/- "
            f"{std_error_time:.5e}")
        print(
            f"graph_creation_time: {graph_mean_time:.5e} +/- "
            f"{graph_std_error_time:.5e}")

        f.write(f"global_mse_u,{global_mse_u}\n")
        f.write(f"global_std_error_u,{global_std_error_u}\n")
        f.write(f"global_mse_p,{global_mse_p}\n")
        f.write(f"global_std_error_p,{global_std_error_p}\n")
        f.write(f"global_mse_alpha,{global_mse_alpha}\n")
        f.write(f"global_std_error_alpha,{global_std_error_alpha}\n")

        f.write(
            'global_mean_conservation_error_alpha,'
            f"{global_mean_conservation_error_alpha}\n")
        f.write(
            'global_std_error_conservation_error_alpha,'
            f"{global_std_error_conservation_error_alpha}\n")

        f.write(f"global_mean_prediction_time,{mean_time}\n")
        f.write(f"global_std_error_prediction_time,{std_error_time}\n")

        f.write(f"global_mean_graph_time,{graph_mean_time}\n")
        f.write(f"global_std_error_graph_time,{graph_std_error_time}\n")

    print(f"Global stats written in: {global_stats_file}")

    ts_errors = np.stack([
        d['timeseries_conservation_error_alpha'] for d in list_dict], axis=0)
    ts_mse = np.mean(ts_errors**2, axis=0)**.5
    conservation_file = args.predicted_base_directory \
        / 'conservation_alpha.csv'
    with open(conservation_file, 'w') as f:
        for i, e in enumerate(ts_mse, 1):
            f.write(f"{i * DT:.5e},{e:.5e}\n")
    print(f"Conservation log written in: {conservation_file}")

    return


def process_one_directory(data_directory, args, converter):
    print(f"--\nProcessing: {data_directory}")
    relative_path = data_directory.relative_to(args.predicted_base_directory)
    str_path = str(data_directory)
    if 'train' in str_path or 'validation' in str_path or 'test' in str_path:
        interim_path = args.interim_base_directory \
            / '/'.join(relative_path.parts[2:])
    else:
        interim_path = args.interim_base_directory \
            / '/'.join(relative_path.parts[3:])
    if not interim_path.is_dir():
        raise ValueError(
            f"Interim path {interim_path} does not exist\n"
            f"data_directory: {data_directory}\n"
            f"predicted_base_directory: {args.predicted_base_directory}\n"
        )

    fem_data = femio.read_files('vtu', interim_path / f"mesh.{160:08d}.vtu")
    fem_data.nodal_data.update_data(
        fem_data.nodes.ids, {
            'input_nodal_u_0':
            fem_data.nodal_data.get_attribute_data('nodal_u'),
            'input_nodal_p_0':
            fem_data.nodal_data.get_attribute_data('nodal_p'),
            'input_nodal_alpha_0':
            fem_data.nodal_data.get_attribute_data('nodal_alpha'),
        })
    npy_files = list(data_directory.glob('*step*.npy'))
    for npy_file in npy_files:
        data = np.load(npy_file)
        if args.preprocessors_pkl is not None:
            data = converter.converters[
                re.sub(
                    r'_step\d+',
                    '',
                    npy_file.stem
                    .replace('predicted_', '').replace('answer_', '')
                    .replace('input_', ''))
            ].inverse(data)
        fem_data.nodal_data.update_data(fem_data.nodes.ids, {
            f"{npy_file.stem}": data})

    return fem_data, interim_path


if __name__ == '__main__':
    main()
