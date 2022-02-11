import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interpn

fig_config = dict(
    path='none',
    prefix='',
    dot_size=20,
    grid_linestyle='--',
    grid_color='r',
    annotation_text_offset=(1, 1),
    annotation_text_size=4,
    save_fig=False,
    display=True,
    tick_labelsize=4,
    tick_length=2,
    fig_size=(1.9, 1.9)
)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

mcc = MouseConnectivityCache(manifest_file='./.mouse_connectivity/mouse_connectivity_manifest.json', resolution=25)
experiments = mcc.get_experiments(dataframe=True)
experiments.drop_duplicates(subset="id", inplace=True)

main_regions = {
    **{r: [n['acronym']
           for n in mcc.get_structure_tree().get_structures_by_id(mcc.get_structure_tree().descendant_ids(
            [mcc.get_structure_tree().get_id_acronym_map()[r]])[0])]
       for r in ['HB', 'MB', 'IB', 'CH']}, "grey": {mcc.get_structure_tree().get_id_acronym_map()['grey']}}

region_to_upper = {
    **{s: r for r in ['HB', 'MB', 'IB', 'CH']
       for s in
       [n['acronym'] for n in mcc.get_structure_tree().get_structures_by_id(mcc.get_structure_tree().descendant_ids(
           [mcc.get_structure_tree().get_id_acronym_map()[r]])[0])]},
    mcc.get_structure_tree().get_id_acronym_map()['grey']: 'grey'
}


def load_data():
    data = pd.read_parquet('output/full_brain/stats.parquet')
    # data = pd.read_parquet('stats.parquet')
    return data


def produce_plot_data(data, mappers: dict, reducers: dict):
    assert set(mappers.keys()) == set(reducers.keys())
    keys = set(mappers.keys())

    intermediate_results = defaultdict(list)

    for experiment, exp_data in data.items():
        for k in keys:
            current = {s: mappers[k](d) for s, d in exp_data.items()}
            # current['id'] = experiment
            intermediate_results[k].append(current)

    results = defaultdict(dict)

    for k in intermediate_results.keys():
        for s in intermediate_results[k][0].keys():
            results[k][s] = [d[s] for d in intermediate_results[k]]

    for k in intermediate_results.keys():
        for s in intermediate_results[k][0].keys():
            results[k][s] = reducers[k](results[k][s])

    return results


def density_scatter(ax, x, y, fig=None, sort=True, bins=20, **kwargs):
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d",
                bounds_error=False, fill_value=0)

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    plot_scatter(ax, x, y, c=z, s=1)

    if fig is not None:
        norm = Normalize(vmin=np.min(z), vmax=np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
        cbar.ax.set_ylabel('Density')


def to_col_name(data, param, stat='median'):
    if param not in set(data):
        col_name = f'{param}|{stat}'
    else:
        col_name = param
    return col_name


def get_subplots():
    # plt.close()
    return plt.subplots(figsize=fig_config['fig_size'])


def plot_grid(ax, vert_lines, horiz_lines, color=fig_config['grid_color'], linestyle=fig_config['grid_linestyle']):
    grid = [
        (vert_lines, lambda _c: ax.axvline(x=_c, color=color, linestyle=linestyle)),
        (horiz_lines, lambda _c: ax.axhline(y=_c, color=color, linestyle=linestyle))
    ]

    for lines, func in grid:
        for c in lines:
            func(c)


def produce_figure(ax, fig, fpath, xlabel=None, ylabel=None, logscale=False, legend=False, format_xticks=True,
                   format_yticks=True, buf=None):
    if legend:
        ax.legend(fontsize=fig_config['tick_labelsize'])

    ax.set_xlabel(xlabel, fontsize=5)
    ax.set_ylabel(ylabel, fontsize=5)

    ax.tick_params(axis='y', length=fig_config['tick_length'], labelsize=fig_config['tick_labelsize'])
    if format_yticks:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

    ax.tick_params(axis='x', length=fig_config['tick_length'], labelsize=fig_config['tick_labelsize'])
    if format_xticks:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.0)

    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if fig_config['save_fig']:
        os.makedirs(f'output/figures/{fig_config["path"]}/', exist_ok=True)
        filename = f'output/figures/{fig_config["path"]}/{fig_config["prefix"]}{fpath}.pdf'
        fig.savefig(filename, dpi=100)
        print(f"Saving {filename}...")

    if buf is not None:
        fig.savefig(buf, format='pdf', dpi=100)

    if fig_config['display'] and buf is None:
        fig.suptitle(f'{fig_config["prefix"]}{fpath}.pdf', size=40)
        fig.show()
    else:
        plt.close('all')


def plot_annotations(ax, annotations, x_ann, y_ann, fontsize=fig_config['annotation_text_size'],
                     textoffset=fig_config['annotation_text_offset'], dot_color=None):
    for _x, _y, _text in zip(x_ann, y_ann, annotations):
        ax.annotate(
            _text,
            (_x, _y),
            size=fontsize,
            xycoords='data',
            xytext=textoffset,
            textcoords='offset points',
        )
    if dot_color is not None:
        plot_scatter(ax, x_ann, y_ann, color=dot_color)


def plot_scatter(ax, x, y, s=fig_config['dot_size'], label=None, color=None, **kwargs):
    if 'marker' in kwargs:
        ax.scatter(x, y, s=s, label=label, color=color, edgecolors='none', **kwargs)
    else:
        ax.scatter(x, y, s=s, label=label, color=color, marker='.', edgecolors='none', **kwargs)


def select_strain(data, strain):
    strain_ids = set(experiments[experiments.strain == strain].id.tolist())
    return {e: d for e, d in data.items() if int(e) in strain_ids}
