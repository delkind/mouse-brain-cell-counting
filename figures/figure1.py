import colorsys
import math

import numpy as np
import pandas as pd

from figures.util import get_subplots, plot_grid, produce_figure, plot_annotations, fig_config, mcc, plot_scatter, \
    to_col_name


def param_ranked_list(data, param):
    regions = data.region.unique()
    regions = regions[find_leaves(regions)]
    col_name = to_col_name(data, param)
    means = data[data.region.isin(regions)].groupby('region')[col_name].mean()
    param_data = means.dropna().to_numpy()
    regions = means.index.dropna().to_numpy()
    idx = np.argsort(param_data)[::-1]

    df = pd.DataFrame({
        'region': regions[idx],
        param: param_data[idx],
    })

    fig, ax = get_subplots()
    plot_scatter(ax, y=df[param], x=np.arange(len(df['region'])))
    plot_grid(ax, [], [np.median(param_data)])
    plot_annotations(ax,
                     np.concatenate([regions[idx][:10], regions[idx][-10:]]),
                     np.concatenate([np.arange(len(param_data))[:10], np.arange(len(param_data))[-10:]]),
                     np.concatenate([param_data[idx][:10], param_data[idx][-10:]]))
    ax.tick_params(axis='y', length=fig_config['tick_length'], labelsize=fig_config['tick_labelsize'])
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    produce_figure(ax, fig, f"{param}_ranked_list", format_yticks=False)


def find_leaves(regions):
    region_ids = [s['id'] for s in mcc.get_structure_tree().get_structures_by_acronym(regions)]
    child_ids = [set(i).intersection(set(region_ids)) for i in mcc.get_structure_tree().child_ids(region_ids)]
    return np.array([len(lv) == 0 for lv in child_ids])


def param_pie_chart(data, param='count', level=None, descend_from='grey'):
    regions = data.region.unique()
    regions = regions_by_level(level, regions, descend_from)
    means = data[data.region.isin(regions)].groupby('region')[to_col_name(data, param)].median()
    vals = means.dropna().to_numpy()
    regions = means.index.dropna().to_numpy()

    idx, colors = sort_by_color(regions)
    vals = vals[idx]
    regions = regions[idx]
    colors = [colors[r] for r in regions]

    fig, ax = get_subplots()
    ax.pie(x=vals, autopct="%.1f%%", explode=[0.1] * len(vals), labels=regions, pctdistance=0.5, colors=colors,
           textprops=dict(fontsize=5))

    if level is None:
        fname = f"{descend_from}_leaves_{param}_chart"
    else:
        fname = f"{descend_from}_level_{level}_{param}_chart"

    produce_figure(ax, fig, fname, format_yticks=False)


def color_key(r, g, b, repetitions=1):
    lum = math.sqrt(.241 * r + .691 * g + .068 * b)

    h, _, v = colorsys.rgb_to_hsv(r, g, b)

    h2 = int(h * repetitions)
    v2 = int(v * repetitions)

    if h2 % 2 == 1:
        v2 = repetitions - v2
        lum = repetitions - lum

    return h2, lum, v2


def regions_by_level(level, regions, descend_from):
    descend_from = mcc.get_structure_tree().get_structures_by_acronym([descend_from])[0]['id']
    paths = {r: s['structure_id_path'] for r, s in
             zip(regions, mcc.get_structure_tree().get_structures_by_acronym(regions))}
    regions = np.array(list(filter(lambda r: descend_from in paths[r], regions)))
    leaves = find_leaves(regions)
    levels = np.array([len(paths[r]) for r in regions])
    if level is not None:
        regions = regions[(levels == level) | ((levels <= level) & leaves)]
    else:
        regions = regions[leaves]

    return regions


def sort_by_color(regions):
    colors_dict = {r: tuple(np.array(s['rgb_triplet']) / 255) for r, s in
                   zip(regions, mcc.get_structure_tree().get_structures_by_acronym(regions))}
    colors = [colors_dict[r] for r in regions]
    color_keys = np.empty(len(colors), dtype=object)
    color_keys[:] = [color_key(*c, 8) for c in colors]
    idx = np.argsort(color_keys, axis=0)
    return idx, colors_dict


def figure1(data, valid_data):
    for param in ['diameter', 'density3d', 'coverage']:
        param_ranked_list(valid_data, param)

    for param in ['count', 'volume']:
        for level in [3, 4, 5, 6, 7, 8, 9, None]:
            param_pie_chart(data, param, level)
        param_pie_chart(data, param, None, "Isocortex")
