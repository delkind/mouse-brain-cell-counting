import itertools
import math

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression

from figures.figure1 import find_leaves
from figures.util import produce_figure, main_regions, get_subplots, plot_grid, \
    plot_annotations, plot_scatter

CORRELATION_THRESHOLD = 0.25
MIN_SAMPLE_COUNT = 100
SYMMETRY_SCORE_THRESHOLD = 0.155
CELL_COUNT_THRESHOLD = 20000


def prepare_data(data, plot=False):
    valid_data = data
    valid_data = filter_dark_brains(valid_data)
    valid_data = filter_small_regions(valid_data)
    valid_data = filter_brightness_correlated_data(valid_data, plot)
    valid_data = filter_asymmetric_regions(valid_data)
    valid_data = filter_count_outliers(valid_data)

    print(f"Brains discarded: {len(data.experiment_id.unique()) - len(valid_data.experiment_id.unique())} out "
          f"of {len(data.experiment_id.unique())}. {len(valid_data.experiment_id.unique())} good brains left.")
    print(f"Regions discarded: {len(data.region.unique()) - len(valid_data.region.unique())} out "
          f"of {len(data.region.unique())}. {len(valid_data.region.unique())} good brains left.")

    return valid_data


def filter_count_outliers(data):
    regions = data.region.unique()
    control_regions = regions[find_leaves(regions)]

    table = pd.pivot_table(data[data.region.isin(control_regions)],
                           columns=['experiment_id', 'region'])['count'].unstack()
    table = ((table - table.median()) / (table.mad() * 1.4826))

    left_tail = table < -3
    right_tail = table > 3

    left_per_experiment = left_tail.transpose().sum()
    right_per_experiment = right_tail.transpose().sum()
    left_outliers = left_per_experiment[left_per_experiment > 3]
    right_outliers = right_per_experiment[right_per_experiment > 3]

    return data[~data.experiment_id.isin(left_outliers.index.to_list() + right_outliers.index.to_list())]


def filter_asymmetric_regions(data, plot=False):
    if plot:
        res = plot_symmetry_score(data, annotate_filtered=True)
    else:
        res = calculate_symmetry_score(data)

    res = pd.DataFrame(res.items(), columns=['region', 'score'])
    return data[data.region.isin(res[res.score <= SYMMETRY_SCORE_THRESHOLD].region.unique())]


def filter_small_regions(data):
    means = data.groupby('region').median()
    regions = means[(means['count'] > 500) & (means.volume > 0.3)].index.tolist()
    return data[data.region.isin(regions)]


def filter_dark_brains(data):
    valid_experiments = data[(data.region == 'grey') & (data['brightness|median'] > 25)].experiment_id.unique()
    return data[data.experiment_id.isin(valid_experiments)]


def determine_correlation_thresholds(data, param1, param2, correlation_threshold=CORRELATION_THRESHOLD):
    res = {p: data.groupby('region')[p].apply(np.array).to_dict()
           for p in (param1, param2)}
    max_threshold = data[param2].max()

    results = dict()
    for region in res[param1].keys():
        x = np.copy(res[param2][region])
        y = np.copy(res[param1][region])
        for p2_threshold in range(5, 200, 5):
            valid_indices = np.where(x > p2_threshold)
            x = x[valid_indices]
            y = y[valid_indices]
            if min(len(x), len(y)) < MIN_SAMPLE_COUNT:
                break
            correlation, pvalue = scipy.stats.pearsonr(x, y)
            if abs(correlation) < correlation_threshold:
                results[region] = (p2_threshold, data[(data.region == region) &
                                                      (data[param2] > p2_threshold)].experiment_id.unique().tolist())
                break
        if region not in results:
            results[region] = (max_threshold + 1, [])

    return results


def plot_symmetry_score(data, annotate_filtered=False):
    scores = calculate_symmetry_score(data)
    res = sorted(scores.items(), key=lambda e: e[1], reverse=True)
    x, y = np.array(list(zip(*res)))
    y = y.astype(np.double)
    fig, ax = get_subplots()

    for region, structs in main_regions.items():
        indices = np.where(np.isin(x.astype(str), structs))
        plot_scatter(ax, indices, y[indices], label=region)

    ax.set_xticks(list(range(len(x))[::50]))
    ax.set_xticklabels(x[::50], size=20)
    plot_grid(ax, [], [SYMMETRY_SCORE_THRESHOLD])

    if annotate_filtered:
        anns = np.where(y > SYMMETRY_SCORE_THRESHOLD)
        plot_annotations(ax, x[anns], np.arange(len(anns[0])), y[anns])

    produce_figure(ax, fig, "symmetry_score_linear", "", "", legend=True, format_xticks=False)
    ax.set_yscale('log')
    produce_figure(ax, fig, "symmetry_score_log", "", "", legend=False, format_xticks=False)
    return scores


def calculate_symmetry_score(data):
    mean_data = pd.DataFrame({'region': data.region,
                              'absdiff': (data.count_left - data.count_right).abs(),
                              'count': data['count']
                              }).groupby('region').mean()
    return (mean_data.absdiff / mean_data['count']).to_dict()


def find_inflection_point(xs, ys):
    diffs = np.abs(ys[:-1] - ys[1:])
    inf_pts = np.where(diffs > 2)[0]
    return inf_pts[-1] if len(inf_pts) > 0 else -1


def determine_correlation_intersection(ax, data, plot=False, correlation_threshold=CORRELATION_THRESHOLD):
    thresholds = determine_correlation_thresholds(data, 'count', 'brightness|median', correlation_threshold)
    # exps = sorted([(s, set(es)) for s, (t, es) in thresholds.items()], key=lambda s: len(s[1]), reverse=True)
    exp_thresholds = {e: {s for s, es in thresholds.items() if e in es[1]} for e in data.experiment_id.unique()}
    exps = sorted([(e, s) for e, s in exp_thresholds.items()], key=lambda s: len(s[1]), reverse=True)
    intersections = [tuple(zip(*exps[:i])) for i in range(1, len(exps))]
    intersections = [(set(s), set.intersection(*e)) for s, e in intersections]
    exp_sets, struct_sets = tuple(zip(*sorted(intersections, key=lambda t: len(t[1]), reverse=True)))
    ys = np.array([len(s) for s in struct_sets])
    xs = np.array([len(s) for s in exp_sets])
    inf_pt = find_inflection_point(xs, ys)
    if plot:
        plot_scatter(ax, xs, ys, label=str(correlation_threshold))
        plot_scatter(ax, xs[inf_pt], ys[inf_pt], s=100)
        plot_grid(ax, [xs[inf_pt]], [ys[inf_pt]])
    return thresholds, inf_pt, intersections, xs, ys


def plot_brightness_threshold_curves(data):
    fig, ax = get_subplots()
    for i in range(10, 51, 2):
        thresholds, inf_pt, intersections, xs, ys = determine_correlation_intersection(ax, data, True, i / 100)
        plot_annotations(ax, f'{i / 100} ({ys[inf_pt]} regions, {xs[inf_pt]} brains)', [xs[inf_pt]], [ys[inf_pt]])
    produce_figure(ax, fig, "valid_regions_vs_brains", "brains", "regions", legend=True)


def filter_brightness_correlated_data(data, plot=False):
    if plot:
        fig, ax = get_subplots()
    else:
        fig, ax = (None, None)

    # thresholds, inf_pt, intersections, xs, ys = determine_correlation_intersection(ax, data, plot=plot)
    thresholds = determine_correlation_thresholds(data, 'count', 'brightness|median', CORRELATION_THRESHOLD)

    if plot:
        produce_figure(ax, fig, "valid_regions_vs_brains", "brains", "regions")
        counts = data[data.region.isin(thresholds.keys())].groupby('region')['count'].mean().to_dict()
        regions = sorted([(k, sorted(list(g), key=lambda x: counts[x], reverse=True))
                          for k, g in itertools.groupby(sorted(thresholds.keys(),
                                                               key=lambda x: thresholds[x][0]),
                                                        lambda s: thresholds[s][0])], key=lambda x: x[0])

        fig, ax = get_subplots()
        x = data[data.region == regions[0][1][1]]['count'].dropna().to_numpy()
        y = data[data.region == regions[0][1][1]]['brightness|mean'].dropna().to_numpy()
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        plot_scatter(ax, x, y)
        ax.plot(x, reg.predict(x.reshape(-1, 1)), color='blue', linewidth=3)
        produce_figure(ax, fig, "valid_count_vs_brightness",
                       xlabel=f'{regions[0][1][1]}. corr: {scipy.stats.pearsonr(x, y)[0]:.2f}')

        fig, ax = get_subplots()
        x = data[data.region == regions[-1][1][0]]['count'].dropna().to_numpy()
        y = data[data.region == regions[-1][1][0]]['brightness|mean'].dropna().to_numpy()
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        plot_scatter(ax, x, y)
        ax.plot(x, reg.predict(x.reshape(-1, 1)), color='blue', linewidth=3)
        produce_figure(ax, fig, "invalid_count_vs_brightness",
                       xlabel=f'{regions[-1][1][0]}. corr: {scipy.stats.pearsonr(x, y)[0]:.2f}')

    nan_data = [[{'experiment_id': i, 'region': r, **{c: -12345 for c in data
                                                      if c not in ['experiment_id', 'region']}}
                 for i in set(data.experiment_id.unique()) - set(d[1])] for r, d in thresholds.items()]
    nan_data = [e for ee in nan_data for e in ee]
    nan_data = pd.DataFrame(nan_data).set_index(['experiment_id', 'region'])
    data.set_index(['experiment_id', 'region'], inplace=True)
    data.update(nan_data, overwrite=True)
    data[data < 0] = math.nan
    data.reset_index(inplace=True)

    # valid_structs = list(intersections[inf_pt][1])
    # valid_experiments = list(intersections[inf_pt][0])
    # valid_data = data[data.experiment_id.isin(valid_experiments) & data.region.isin(valid_structs)]

    return data
