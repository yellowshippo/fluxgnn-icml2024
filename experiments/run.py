
import argparse
import pathlib

from fluxgnn import experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        type=str,
        help='Mode of the experiment ([train], eval, fv)')
    parser.add_argument(
        'input_path',
        type=pathlib.Path,
        help='''Input path of the experiment.
        (train, fv: YAML file name of settings,
        eval: trained model directory)''')
    parser.add_argument(
        '-g',
        '--gpu-id',
        type=int,
        default=0,
        help='GPU ID [0]')
    parser.add_argument(
        '-d',
        '--data-directories',
        type=pathlib.Path,
        default=None,
        nargs='+',
        help='Data directories for eval')
    parser.add_argument(
        '-n',
        '--data-name',
        type=str,
        default=None,
        help='Name of the given data_directories')
    parser.add_argument(
        '-p',
        '--pretrained-directory',
        type=pathlib.Path,
        default=None,
        help='Pretrained model directory for train')
    parser.add_argument(
        '-t',
        '--test-t-max',
        type=float,
        default=None,
        help='Max time for test data evaluation')
    parser.add_argument(
        '-l',
        '--lr',
        type=float,
        default=None,
        help='Learning rate')
    parser.add_argument(
        '-s',
        '--stop-trigger-epoch',
        type=int,
        default=None,
        help='Stop trigger epoch')
    parser.add_argument(
        '-c',
        '--n-continue',
        type=int,
        default=None,
        help='Number of continues')
    parser.add_argument(
        '-w',
        '--print-period',
        type=int,
        default=None,
        help='Print period')
    parser.add_argument(
        '-q',
        '--print-period-p',
        type=int,
        default=None,
        help='Print period for pressure solver')
    parser.add_argument(
        '-v',
        '--validate-physics',
        action='store_true',
        help='If set, validate physical symmetry on evaluation')
    parser.add_argument(
        '-r',
        '--threshold_p',
        type=float,
        default=None,
        help='Convergence threshold for pressure solver')
    args = parser.parse_args()
    args.device_id = args.gpu_id

    if args.mode == 'train':
        args.model_directory = args.pretrained_directory
        experiment.train(args.input_path, **vars(args))
    elif args.mode == 'eval':
        experiment.evaluate(args.input_path, **vars(args))
    elif args.mode == 'fv':
        experiment.simulate(args.input_path, **vars(args))
    else:
        raise ValueError(f"Unexpected mode: {args.mode}")
    return


if __name__ == '__main__':
    main()
