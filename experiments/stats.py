import os
import pathlib
import re

import femio
import numpy as np


INITIAL_STEP = 160
INTERVAL = 2
DT = .2

MODE = 'nodal'


def mixture_analyse_error(
        results, prediction_root_directory, mode=MODE):
    list_dict = []
    stats_file = prediction_root_directory / 'stats.csv'
    with open(stats_file, 'w') as f:
        f.write(
            'directory,'
            'mse_u,stderror_u,'
            'mse_p,stderror_p,'
            'mse_alpha,stderror_alpha,'
            'conservation_error_alpha,'
            'prediction_time\n'
        )

    for result in results:
        directory = result['data_directory']
        interim_directory = pathlib.Path(
            str(directory).replace('preprocessed', 'interim'))
        prediction_time = result['inference_time']

        fem_data = result['fem_data']
        loss_dict = calculate_single_loss_mixture(
            result['output_directory'], interim_directory, fem_data, mode=mode)
        loss_dict.update({'prediction_time': prediction_time})
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

    global_mse_u, global_std_error_u = calculate_global_stats(
        list_dict, 'u')
    global_mse_p, global_std_error_p = calculate_global_stats(
        list_dict, 'p')
    global_mse_alpha, global_std_error_alpha = calculate_global_stats(
        list_dict, 'alpha')

    errors = np.stack([d['conservation_error_alpha'] for d in list_dict])
    global_mean_conservation_error_alpha = np.mean(errors)
    global_std_error_conservation_error_alpha = np.std(
        errors) / len(errors)**.5

    mean_time, std_error_time = calculate_time(list_dict)

    print('--')
    global_stats_file = prediction_root_directory / 'global_stats.csv'
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

    print(f"Global stats written in: {global_stats_file}")

    ts_errors = np.stack([
        d['timeseries_conservation_error_alpha'] for d in list_dict], axis=0)
    ts_mse = np.mean(ts_errors**2, axis=0)**.5
    conservation_file = prediction_root_directory / 'conservation_alpha.csv'
    with open(conservation_file, 'w') as f:
        for i, e in enumerate(ts_mse, 1):
            f.write(f"{i * DT:.5e},{e:.5e}\n")
    print(f"Conservation log written in: {conservation_file}")

    return


def calculate_single_loss_mixture(
        output_directory, interim_directory, fem_data, mode=MODE,
        force_renew=False):
    u_names = sorted(
        [
            k for k in fem_data.nodal_data.keys()
            if 'predicted' in k and '_u' in k],
        key=lambda s: int(re.findall(r'\d+', s)[0])
    )
    p_names = sorted(
        [
            k for k in fem_data.nodal_data.keys()
            if 'predicted' in k and '_p' in k],
        key=lambda s: int(re.findall(r'\d+', s)[0])
    )
    alpha_names = sorted(
        [
            k for k in fem_data.nodal_data.keys()
            if 'predicted' in k and '_alpha' in k],
        key=lambda s: int(re.findall(r'\d+', s)[0])
    )
    if len(u_names) == len(p_names) == len(alpha_names):
        pass
    else:
        raise ValueError(
            f"Names invalid: {u_names}, {p_names}, {alpha_names}")
    n_time = len(u_names)

    nodal_predicted_u = np.stack([
        fem_data.nodal_data.get_attribute_data(name)
        for name in u_names], axis=0)
    nodal_predicted_p = np.stack([
        fem_data.nodal_data.get_attribute_data(name)
        for name in p_names], axis=0)
    nodal_predicted_alpha = np.stack([
        fem_data.nodal_data.get_attribute_data(name)
        for name in alpha_names], axis=0)

    nodal_answer_u = np.stack([
        fem_data.nodal_data.get_attribute_data(
            name.replace('predicted', 'answer'))
        for name in u_names], axis=0)
    nodal_answer_scale_u = np.std(nodal_answer_u)

    nodal_answer_p = np.stack([
        fem_data.nodal_data.get_attribute_data(
            name.replace('predicted', 'answer'))
        for name in p_names], axis=0)
    nodal_answer_scale_p = np.std(nodal_answer_p)

    nodal_answer_alpha = np.stack([
        fem_data.nodal_data.get_attribute_data(
            name.replace('predicted', 'answer'))
        for name in alpha_names], axis=0)
    nodal_answer_scale_alpha = np.std(nodal_answer_alpha)

    elemental_predicted_u = np.stack([
        fem_data.convert_nodal2elemental(
            u, calc_average=True)
        for u in nodal_predicted_u], axis=0)
    elemental_predicted_p = np.stack([
        fem_data.convert_nodal2elemental(
            p, calc_average=True)
        for p in nodal_predicted_p], axis=0)
    elemental_predicted_alpha = np.stack([
        fem_data.convert_nodal2elemental(
            alpha, calc_average=True)
        for alpha in nodal_predicted_alpha], axis=0)

    # Loading elemental answers, skipping initial
    n_t = len(nodal_predicted_u)
    elemental_answer_u = np.load(interim_directory / 'elemental_u.npy')[
        INTERVAL:n_t+INTERVAL][..., 0]
    elemental_answer_scale_u = np.std(elemental_answer_u)
    elemental_answer_p = np.load(interim_directory / 'elemental_p.npy')[
        INTERVAL:n_t+INTERVAL]
    elemental_answer_scale_p = np.std(elemental_answer_p)
    elemental_answer_alpha = np.load(
        interim_directory / 'elemental_alpha.npy')[INTERVAL:n_t+INTERVAL]
    elemental_answer_scale_alpha = np.std(elemental_answer_alpha)

    # Write initial
    fem_data = femio.read_files(
        'vtu', interim_directory / f"mesh.{INITIAL_STEP:08d}.vtu")
    nodal_dict_initial = {
        'answer_u':
        fem_data.nodal_data.get_attribute_data('nodal_u'),
        'predicted_u':
        fem_data.nodal_data.get_attribute_data('nodal_u'),
        'answer_p':
        fem_data.nodal_data.get_attribute_data('nodal_p'),
        'predicted_p':
        fem_data.nodal_data.get_attribute_data('nodal_p'),
        'answer_alpha':
        fem_data.nodal_data.get_attribute_data('nodal_alpha'),
        'predicted_alpha':
        fem_data.nodal_data.get_attribute_data('nodal_alpha'),
    }
    elemental_dict_initial = {
        'answer_u':
        fem_data.elemental_data.get_attribute_data('elemental_u'),
        'predicted_u':
        fem_data.elemental_data.get_attribute_data('elemental_u'),
        'answer_p':
        fem_data.elemental_data.get_attribute_data('elemental_p'),
        'predicted_p':
        fem_data.elemental_data.get_attribute_data('elemental_p'),
        'answer_alpha':
        fem_data.elemental_data.get_attribute_data('elemental_alpha'),
        'predicted_alpha':
        fem_data.elemental_data.get_attribute_data('elemental_alpha'),
    }
    fem_data.nodal_data.reset()
    fem_data.nodal_data.update_data(
        fem_data.nodes.ids, nodal_dict_initial, allow_overwrite=True)
    fem_data.elemental_data.update_data(
        fem_data.elements.ids, elemental_dict_initial, allow_overwrite=True)
    fem_data.write(
        'vtu', output_directory / f"ts/mesh.{0:08d}.vtu",
        overwrite=force_renew)

    for i in range(n_time):
        fem_data.nodal_data.update_data(
            fem_data.nodes.ids, {
                'answer_u': nodal_answer_u[i],
                'predicted_u': nodal_predicted_u[i],
                'answer_p': nodal_answer_p[i],
                'predicted_p': nodal_predicted_p[i],
                'answer_alpha': nodal_answer_alpha[i],
                'predicted_alpha': nodal_predicted_alpha[i],
            }, allow_overwrite=True)
        fem_data.elemental_data.update_data(
            fem_data.elements.ids, {
                'answer_u': elemental_answer_u[i],
                'predicted_u': elemental_predicted_u[i],
                'answer_p': elemental_answer_p[i],
                'predicted_p': elemental_predicted_p[i],
                'answer_alpha': elemental_answer_alpha[i],
                'predicted_alpha': elemental_predicted_alpha[i],
            }, allow_overwrite=True)
        fem_data.write(
            'vtu', output_directory / f"ts/mesh.{i+1:08d}.vtu",
            overwrite=force_renew)

    if mode == 'nodal':
        error, ts_error = compute_conservation_stats(
            nodal_predicted_alpha, nodal_dict_initial['predicted_alpha'])
        return_dict = {
            'mse_u':
            mse(nodal_predicted_u, nodal_answer_u) / nodal_answer_scale_u**2,
            'stderror_u':
            std_error(nodal_predicted_u, nodal_answer_u)
            / nodal_answer_scale_u**2,
            'answer_scale_u': nodal_answer_scale_u,

            'mse_p':
            mse(nodal_predicted_p, nodal_answer_p) / nodal_answer_scale_p**2,
            'stderror_p':
            std_error(nodal_predicted_p, nodal_answer_p)
            / nodal_answer_scale_p**2,
            'answer_scale_p': nodal_answer_scale_p,

            'mse_alpha':
            mse(nodal_predicted_alpha, nodal_answer_alpha)
            / nodal_answer_scale_alpha**2,
            'stderror_alpha':
            std_error(nodal_predicted_alpha, nodal_answer_alpha)
            / nodal_answer_scale_alpha**2,
            'answer_scale_alpha': nodal_answer_scale_alpha,

            'conservation_error_alpha': error,
            'timeseries_conservation_error_alpha': ts_error,

            'answer_u': nodal_answer_u,
            'predicted_u': nodal_predicted_u,
            'answer_p': nodal_answer_p,
            'predicted_p': nodal_predicted_p,
            'answer_alpha': nodal_answer_alpha,
            'predicted_alpha': nodal_predicted_alpha,
        }
    elif mode == 'elemental':
        error, ts_error = compute_conservation_stats(
            nodal_predicted_alpha, nodal_dict_initial['predicted_alpha'])
        return_dict = {
            'mse_u':
            mse(elemental_predicted_u, elemental_answer_u)
            / elemental_answer_scale_u**2,
            'stderror_u':
            std_error(elemental_predicted_u, elemental_answer_u)
            / elemental_answer_scale_u**2,
            'answer_scale_u': elemental_answer_scale_u,

            'mse_p':
            mse(elemental_predicted_p, elemental_answer_p)
            / elemental_answer_scale_p**2,
            'stderror_p':
            std_error(elemental_predicted_p, elemental_answer_p)
            / elemental_answer_scale_p**2,
            'answer_scale_p': elemental_answer_scale_p,

            'mse_alpha':
            mse(elemental_predicted_alpha, elemental_answer_alpha)
            / elemental_answer_scale_alpha**2,
            'stderror_alpha':
            std_error(elemental_predicted_alpha, elemental_answer_alpha)
            / elemental_answer_scale_alpha**2,
            'answer_scale_alpha': elemental_answer_scale_alpha,

            'conservation_error_alpha': error,
            'timeseries_conservation_error_alpha': ts_error,

            'answer_u': elemental_answer_u,
            'predicted_u': elemental_predicted_u,
            'answer_p': elemental_answer_p,
            'predicted_p': elemental_predicted_p,
            'answer_alpha': elemental_answer_alpha,
            'predicted_alpha': elemental_predicted_alpha,
        }
    else:
        raise ValueError(f"Unexpected mode: {mode}")

    return return_dict


def compute_conservation_stats(v, initial):
    c_init = np.sum(initial)
    c = np.array([np.sum(_v) for _v in v])
    ts_rela_error = (c - c_init) / c_init
    mse_error = np.mean(ts_rela_error**2)**.5
    return mse_error, ts_rela_error


def determine_output_base(results):
    candidate = results[0]['output_directory']
    for result in results[1:]:
        candidate = os.path.commonpath([candidate, result['output_directory']])
    return pathlib.Path(candidate)


def calculate_global_stats(list_dict, key, concat_axis=1):
    answer = np.concatenate(
        [d[f"answer_{key}"] / d[f"answer_scale_{key}"] for d in list_dict],
        axis=concat_axis)
    pred = np.concatenate(
        [d[f"predicted_{key}"] / d[f"answer_scale_{key}"] for d in list_dict],
        axis=concat_axis)
    square_error = (pred - answer)**2
    mse = np.mean(square_error)
    std_error = np.std(square_error) / np.sqrt(answer.size)
    return mse, std_error


def calculate_time(list_dict, key='prediction_time'):
    times = np.array([d[key] for d in list_dict])
    n_sample = len(list_dict)
    return np.mean(times), np.std(times) / np.sqrt(n_sample)


def mse(a, b):
    return np.mean((a - b)**2)


def std_error(a, b):
    return np.std((a - b)**2) / np.sqrt(a.size)
