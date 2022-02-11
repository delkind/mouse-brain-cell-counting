import numpy as np
import pandas as pd

from figures.clean_data import plot_symmetry_score
from figures.util import main_regions, produce_figure, produce_plot_data, get_subplots, fig_config, plot_scatter, \
    plot_annotations, density_scatter


def plot_observed_vs_technical(data):
    region_counts = pd.DataFrame({'region': data.region,
                                  'count': np.log10(data['count'])
                                  }).groupby('region')['count']
    regions = np.array(region_counts.mean().index)
    x = region_counts.mean().dropna().to_numpy()
    z = region_counts.std().dropna().to_numpy()

    y = pd.DataFrame({'region': data.region,
                      'diff': np.log10((data.count_left + 1) / (data.count_right + 1)).pow(2)
                      }).groupby('region')['diff'].sum().dropna().to_numpy()
    y = np.sqrt(y) / data.experiment_id.nunique()
    fig, ax = get_subplots()

    for region, structs in main_regions.items():
        indices = np.isin(regions, structs)
        plot_scatter(ax, x[indices], y[indices], label=f'diff_{region}')
        plot_scatter(ax, x[indices], z[indices], label=f'std_{region}')
    produce_figure(ax, fig, fpath="observed_vs_technical", xlabel='mean', legend=True)

    fig, ax = get_subplots()
    for region, structs in main_regions.items():
        indices = np.isin(regions, structs)
        ax.scatter(x[indices], (z - y)[indices], s=fig_config['dot_size'], label=region)

    ann = np.where(y > 1)
    ann_x = x[ann]
    ann_y = (z-y)[ann]
    ann_text = regions[ann]

    plot_annotations(ax, ann_text, ann_x, ann_y)
    produce_figure(ax, fig, fpath="observed_vs_technical_diff", xlabel='mean', ylabel='std-diff', legend=True)

    fig, ax = get_subplots()
    for region, structs in main_regions.items():
        indices = np.where(np.isin(regions, structs))
        ax.scatter(x[indices], (z / y)[indices], s=fig_config['dot_size'], label=region)

    # for i, key in enumerate(regions):
    #     y_coord = (y/z)[i]
    #     x_coord = x[i]
    #     if y_coord > 1.0:
    #         ax.annotate(regions[i], (x_coord, y_coord), fontsize='x-small')

    produce_figure(ax, fig, fpath="observed_vs_technical_ratio", xlabel='mean', ylabel='std/diff', legend=True)


def plot_logcount_vs_logdiff(data, remove_outliers):
    x = np.log10(data['count'].dropna().to_numpy())
    y = np.log10((data.count_left + 1).dropna().to_numpy() / (data.count_right + 1).dropna().to_numpy())

    if remove_outliers:
        lower_threshold = np.percentile(y, 1)
        upper_threshold = np.percentile(y, 99)
        show = (y > lower_threshold) & (y < upper_threshold)
        y = y[show]
        x = x[show]

    fig, ax = get_subplots()
    density_scatter(ax, x, y)
    produce_figure(ax, fig, fpath="logcount_vs_logdiff", xlabel='logcount', ylabel='log(L-R)', logscale=False,
                   format_xticks=False, format_yticks=False)


def plot_count_vs_diff(data):
    regions = data.region.unique()
    x = np.array([data.loc[data.region == r, 'count'].mean() for r in regions])
    lr_diff = data.count_left - data.count_right
    y = np.array([np.mean(lr_diff[data.region == r]) for r in regions])
    # z = np.array([np.std([e for e in res['count'][r]]) for r in regions])

    fig, ax = get_subplots()
    for region, structs in main_regions.items():
        indices = np.where(np.isin(regions, structs))
        plot_scatter(ax, x[indices], y[indices], label=region)
    produce_figure(ax, fig, fpath="count_vs_diff", xlabel='mean', ylabel='|l-r|', logscale=True, legend=True)


def plot_count_vs_brightness(data):
    count = np.log10(data['count'].dropna().to_numpy())
    brightness = data['brightness|mean'].dropna().to_numpy()
    fig, ax = get_subplots()
    density_scatter(ax, count, brightness)
    produce_figure(ax, fig, "count_vs_brightness", 'logcount', 'brightness')


def figure2(data, remove_outliers=False):
    plot_logcount_vs_logdiff(data, remove_outliers)
    plot_count_vs_diff(data)
    plot_observed_vs_technical(data)
    plot_symmetry_score(data)
    plot_count_vs_brightness(data)

    # plot_correlation_histogram(valid_data, 'brightness', 'count')
    # plot_correlation_histogram(valid_data, 'brightness', 'density')
    # plot_correlation_histogram(valid_data, "volume", "count")
    # plot_correlation_histogram(valid_data, "density", "count")
    # plot_correlation_histogram(valid_data, "coverage", "count")

# import os
# from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
# analyzed = set(os.listdir("output/full_brain/analyzed"))
# mcc = MouseConnectivityCache(manifest_file='mouse_connectivity/mouse_connectivity_manifest.json', resolution=25)
# cd1_exps = mcc.get_experiments(dataframe=True)
# cd1_exps = cd1_exps[cd1_exps.strain == 'FVB.CD1(ICR)']
# males = cd1_exps[cd1_exps.gender == 'M']
# females = cd1_exps[cd1_exps.gender == 'F']
# male_ids = set([str(i) for i in males.id.tolist()])
# female_ids = set([str(i) for i in females.id.tolist()])
#
# os.makedirs("output/cd1_males/input")
# for i in male_ids - analyzed:
#     os.rename(f"output/full_brain/input/{i}", f"output/cd1_males/input/{i}")
#
# os.makedirs("output/cd1_females/input")
# for i in female_ids - analyzed:
#     os.rename(f"output/full_brain/input/{i}", f"output/cd1_females/input/{i}")
