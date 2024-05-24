import argparse
from distutils.util import strtobool
import pathlib

import femio
import numpy as np
import siml

from experiments import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_path',
        type=pathlib.Path,
        help='Pretrained model path')
    parser.add_argument(
        'data_directories',
        type=pathlib.Path,
        nargs='+',
        help='Data directory')
    parser.add_argument(
        '-o', '--out-dir',
        type=pathlib.Path,
        default=None,
        help='Output directory name')
    parser.add_argument(
        '-b', '--output-base',
        type=pathlib.Path,
        default=None,
        help='Output base directory name')
    parser.add_argument(
        '-p', '--preprocessors-pkl',
        type=pathlib.Path,
        default=None,
        help='Preprocessors.pkl file')
    parser.add_argument(
        '--perform-preprocess',
        type=strtobool,
        default=0,
        help='If true, perform preprocess')
    parser.add_argument(
        '-w', '--write-simulation-base',
        type=pathlib.Path,
        default=None,
        help='Simulation base directory to write inferred data')
    parser.add_argument(
        '-a', '--analyse-error-mode',
        type=str,
        default=None,
        help='If fed, analyse error stats [mixture, cd]')
    parser.add_argument(
        '-e', '--time-series-end',
        type=int,
        default=32,
        help='Time series slice''s end to predict')

    args = parser.parse_args()

    if args.data_directories[0].is_file():
        with open(args.data_directories[0]) as f:
            lines = f.readlines()
        data_directories = [
            pathlib.Path(line.strip()) for line in lines if line.strip() != '']
    else:
        data_directories = args.data_directories

    inferer = siml.inferer.Inferer.from_model_directory(
        args.model_path, save=True,
        load_function=load_function,
        converter_parameters_pkl=args.preprocessors_pkl)
    for g in inferer.setting.model.groups:
        if g.name == 'GROUP1':
            g.time_series_length = round(args.time_series_end / 2)
            print(f"time_series_length for {g.name}: {g.time_series_length}")
    inferer.setting.trainer.gpu_id = -1
    if args.output_base is not None:
        inferer.setting.inferer.output_directory_base = args.output_base
        print(inferer.setting.inferer.output_directory_base)
    else:
        inferer.setting.inferer.output_directory = args.out_dir
    inferer.setting.inferer.write_simulation = True
    if args.write_simulation_base:
        inferer.setting.inferer.write_simulation_base \
            = args.write_simulation_base
    inferer.setting.inferer.read_simulation_type = 'polyvtk'
    inferer.setting.inferer.write_simulation_type = 'polyvtk'
    inferer.setting.conversion.skip_femio = True
    inferer.setting.conversion.required_file_names = ['*.vtu']

    if args.time_series_end is not None:
        for key, value in inferer.setting.trainer.inputs.variables.items():
            if np.any(value.time_series):
                slice_ = value.time_slice
                start = slice_.start
                step = slice_.step
                for i in range(len(value.variables)):
                    value.variables[i].time_slice = slice(
                        start, start + args.time_series_end, step)
        for key, value in inferer.setting.trainer.outputs.variables.items():
            if np.any(value.time_series):
                slice_ = value.time_slice
                start = slice_.start
                step = slice_.step
                for i in range(len(value.variables)):
                    value.variables[i].time_slice = slice(
                        start, start + args.time_series_end, step)

    results = inferer.infer(
        data_directories=data_directories,
        perform_preprocess=args.perform_preprocess)

    if args.analyse_error_mode is None:
        pass
    else:
        output_base = stats.determine_output_base(results)
        if args.analyse_error_mode == 'mixture':
            stats.mixture_analyse_error(results, output_base)
        else:
            raise ValueError(
                f"Invalid --analyse-error-mode: {args.analyse_error_mode}")

    return


def load_function(data_files, write_simulation_base):
    fem_data = femio.read_files('vtu', data_files[0])
    fem_data.nodal_data.reset()
    return {}, fem_data


if __name__ == '__main__':
    main()
