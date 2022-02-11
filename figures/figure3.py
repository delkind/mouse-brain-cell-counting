import itertools
import math

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.multitest

from figures.figure1 import find_leaves
from figures.util import experiments, mcc, get_subplots, produce_figure, plot_grid, plot_annotations, plot_scatter, \
    to_col_name


def generate_qval_data(data):
    regs = data.region.unique()
    leaves = regs[find_leaves(regs)]
    data = data[data.region.isin(leaves)]
    regs = leaves
    qvals = pd.DataFrame({'region': regs, 'region_name': [r['name'] for r in
                                                          mcc.get_structure_tree().get_structures_by_acronym(
                                                              regs)]}).set_index('region').sort_index()

    strains = {
        "BL6": 'C57BL/6J',
        "CD1": 'FVB.CD1(ICR)',
        "ALL": None
    }
    params = ['count3d', 'density3d', 'coverage', "volume"]
    genders = ['M', 'F']

    groups = {
        (g, s[0], p): data[(data.gender == g) & (data.strain == s[1])].groupby('region')[to_col_name(data, p)].apply(np.array).sort_index()
        for g, s, p in itertools.product(genders, strains.items(), params) if s[1] is not None
    }

    def mdn(a):
        return np.median(a[~np.isnan(a)])

    for p in params:
        for g in genders:
            local_data = pd.DataFrame({'left': groups[(g, 'BL6', p)], 'right': groups[(g, 'CD1', p)]}).sort_index()
            pvals = produce_pvals(local_data, scipy.stats.ranksums)
            pvals['sign'] = 1
            qvals[f"BL6_vs_CD1_{g}_{p}_ranksum_rejected"] = pvals.rejected
            qvals[f"BL6_vs_CD1_{g}_{p}_ranksum_qval"] = pvals.qval * pvals.sign
            qvals[f"BL6_vs_CD1_{g}_{p}_ranksum_pval"] = pvals.pval * pvals.sign
            qvals[f"BL6_vs_CD1_{g}_{p}_ranksum_log10(qval)"] = -np.log10(pvals.pval) * pvals.sign
            qvals[f"BL6_vs_CD1_{g}_{p}_effect"] = 2 * (local_data.left.apply(mdn) - local_data.right.apply(mdn)) / (local_data.left.apply(mdn) + local_data.right.apply(mdn))

    for strain_name, strain in strains.items():
        strain_data = data
        if strain is not None:
            strain_data = strain_data[strain_data.strain == strain]
        males = strain_data[strain_data.gender == 'M']
        females = strain_data[strain_data.gender == 'F']
        male_groups = males.groupby('region')
        female_groups = females.groupby('region')
        for param in params:
            male_groups_data = male_groups[to_col_name(strain_data, param)]
            female_groups_data = female_groups[to_col_name(strain_data, param)]
            male_data = male_groups_data.apply(np.array).sort_index()
            female_data = female_groups_data.apply(np.array).sort_index()
            qvals[f'{strain_name}_{param}_male_means'] = male_groups_data.median().sort_index()
            qvals[f'{strain_name}_{param}_female_means'] = female_groups_data.median().sort_index()
            qvals[f'{strain_name}_male_vs_female_{param}_effect'] = 2 * (
                    qvals[f'{strain_name}_{param}_male_means'] - qvals[f'{strain_name}_{param}_female_means']) / (
                    qvals[f'{strain_name}_{param}_male_means'] + qvals[f'{strain_name}_{param}_female_means'])
            for method_name, method in {'t-test': scipy.stats.ttest_ind, 'ranksum': scipy.stats.ranksums}.items():
                local_data = pd.DataFrame({'left': male_data, 'right': female_data}).sort_index()
                pvals = produce_pvals(local_data, method)
                pvals['sign'] = np.sign(
                    qvals[f'{strain_name}_{param}_male_means'] - qvals[f'{strain_name}_{param}_female_means'])
                qvals[f"{strain_name}_male_vs_female_{param}_{method_name}_rejected"] = pvals.rejected
                qvals[f"{strain_name}_male_vs_female_{param}_{method_name}_qval"] = pvals.qval * pvals.sign
                qvals[f"{strain_name}_male_vs_female_{param}_{method_name}_pval"] = pvals.pval * pvals.sign
                qvals[f"{strain_name}_male_vs_female_{param}_{method_name}_log10(qval)"] = -np.log10(
                    pvals.pval) * pvals.sign

        for gender, gender_groups in {'male': male_groups, 'female': female_groups}.items():
            for param in ['count', 'density', 'region_area']:
                left_data = gender_groups[to_col_name(strain_data, f'{param}_left')].apply(np.array).sort_index()
                right_data = gender_groups[to_col_name(strain_data, f'{param}_right')].apply(np.array).sort_index()
                qvals[f'{strain_name}_{param}_{gender}_left_means'] = gender_groups[to_col_name(strain_data,
                                                                                                f'{param}_left')].median().sort_index()
                qvals[f'{strain_name}_{param}_{gender}_right_means'] = gender_groups[to_col_name(strain_data,
                                                                                                 f'{param}_right')].median().sort_index()
                qvals[f'{strain_name}_{gender}_left_vs_right_{param}_effect'] = 2 * (
                        qvals[f'{strain_name}_{param}_{gender}_left_means'] - qvals[f'{strain_name}_{param}_{gender}_right_means']) / (
                                                                                qvals[f'{strain_name}_{param}_{gender}_left_means'] +
                                                                                qvals[f'{strain_name}_{param}_{gender}_right_means'])
                for method_name, method in {'t-test': scipy.stats.ttest_ind, 'ranksum': scipy.stats.ranksums}.items():
                    local_data = pd.DataFrame({'left': left_data, 'right': right_data}).sort_index()
                    pvals = produce_pvals(local_data, method)
                    pvals['sign'] = np.sign(
                        qvals[f'{strain_name}_{param}_{gender}_left_means']
                        - qvals[f'{strain_name}_{param}_{gender}_right_means'])
                    qvals[f"{strain_name}_left_{param}_{gender}_rejected"] = pvals.rejected
                    qvals[f"{strain_name}_left_vs_right_{param}_{gender}_{method_name}_rejected"] = pvals.rejected
                    qvals[f"{strain_name}_left_vs_right_{param}_{gender}_{method_name}_qval"] = pvals.qval * pvals.sign
                    qvals[f"{strain_name}_left_vs_right_{param}_{gender}_{method_name}_pval"] = pvals.pval * pvals.sign
                    qvals[f"{strain_name}_left_vs_right_{param}_{gender}_{method_name}_log10(qval)"] = -np.log10(
                        pvals.qval) * pvals.sign

    return qvals


def produce_pvals(local_data, method):
    regs, pvals = zip(*[(t.Index, method(t.left[~np.isnan(t.left)], t.right[~np.isnan(t.right)]).pvalue)
                        if min(t.left[~np.isnan(t.left)].shape[0], t.right[~np.isnan(t.right)].shape[0]) >= 20
                        else (t.Index, math.nan) for t in local_data.itertuples()])
    pvals = np.array(pvals)
    rejected, pval_corrected = statsmodels.stats.multitest.fdrcorrection(pvals[~np.isnan(pvals)], alpha=0.1)
    rej = np.ones_like(pvals)
    rej[~np.isnan(pvals)] = rejected
    rej[np.isnan(pvals)] = math.nan
    qvals = pvals.copy()
    qvals[~np.isnan(pvals)] = pval_corrected
    pvals = pd.DataFrame({'region': regs, 'pval': pvals, 'qval': qvals, 'rejected': rej}).set_index(
        'region').sort_index()
    return pvals


def plot_qval_vs_qval(qvals, strain, prefix, highlight_col, x_col, y_col, filename):
    regions = qvals.index.to_numpy()
    fig, ax = get_subplots()
    highlight = qvals[strain + '_' + prefix + "_" + highlight_col + '_qval'].to_numpy().copy()
    highlight_rejected = qvals[strain + '_' + prefix + "_" + highlight_col + '_rejected'].to_numpy()
    highlight[highlight_rejected == 0] = 1
    highlight[np.isnan(highlight)] = 1

    x = qvals[strain + '_' + prefix + "_" + x_col + '_log10(qval)'].to_numpy()
    y = qvals[strain + '_' + prefix + "_" + y_col + '_log10(qval)'].to_numpy()

    non_nans = (~np.isnan(x) & ~np.isnan(y))

    x = x[non_nans]
    y = y[non_nans]
    highlight = highlight[non_nans]
    regions = regions[non_nans]

    highlight = (np.abs(highlight) <= 0.01)
    plot_scatter(ax, x[~highlight], y[~highlight], label=f'{highlight_col} not significant')
    plot_annotations(ax, np.array(regions)[highlight], x[highlight], y[highlight], dot_color='r')
    plot_grid(ax, [0], [0])
    produce_figure(ax, fig, strain + '_' + prefix + '_' + filename)


def plot_effect_vs_effect(qvals, strain, prefix, x_col, y_col, filename):
    regions = qvals.index.to_numpy()
    fig, ax = get_subplots()

    highlights = dict()
    for col in [x_col, y_col]:
        highlight = qvals[strain + '_' + prefix + "_" + col + '_ranksum' + '_qval'].to_numpy().copy()
        highlight_rejected = qvals[strain + '_' + prefix + "_" + col + '_ranksum' + '_rejected'].to_numpy()
        highlight[highlight_rejected == 0] = 1
        highlight[np.isnan(highlight)] = 1
        highlight = (np.abs(highlight) <= 0.01)
        highlights[col] = highlight

    highlight = highlights[x_col] * 1 + highlights[y_col] * 2

    x = qvals[strain + '_' + prefix + "_" + x_col + '_effect'].to_numpy()
    y = qvals[strain + '_' + prefix + "_" + y_col + '_effect'].to_numpy()

    non_nans = (~np.isnan(x) & ~np.isnan(y))

    x = x[non_nans]
    y = y[non_nans]
    highlight = highlight[non_nans]
    regions = regions[non_nans]

    labels = ['nothing significant', f'{x_col} significant', f'{y_col} significant', f'{x_col} and {y_col} significant']
    markers = ['.', 's', 'P', '*']

    for i in np.unique(highlight).tolist():
        plot_scatter(ax, x[highlight == i], y[highlight == i], label=labels[i], color='C0', marker=markers[i])

    plot_annotations(ax, np.array(regions)[highlight != 0], x[highlight != 0], y[highlight != 0])
    plot_grid(ax, [0], [0])
    produce_figure(ax, fig, strain + '_' + prefix + '_' + filename)


def produce_volcano_plot(qvals, strain, prefix, x_col, y_col, filename, hide_rejected=False):
    fig, ax = get_subplots()
    if hide_rejected:
        qvals = qvals[qvals[strain + '_' + prefix + '_' + y_col + '_rejected'] == 1.0]
    regions = qvals.index.tolist()
    x_val = qvals[strain + '_' + x_col].to_numpy()
    y_val = qvals[strain + '_' + prefix + '_' + y_col + '_log10(qval)'].to_numpy()

    nans = np.isnan(x_val) | np.isnan(y_val)

    x_val = x_val[~nans]
    y_val = y_val[~nans]

    x = x_val
    y = np.abs(y_val)

    plot_scatter(ax, x, y)
    ann = np.where((np.abs(x) > 0.05) & (y > 2))
    plot_annotations(ax, np.array(regions)[ann], x[ann], y[ann], dot_color='r')
    plot_grid(ax, horiz_lines=[2], vert_lines=[0.05, -0.05])
    produce_figure(ax, fig, strain + '_' + prefix + '_' + filename)


def figure3(data):
    data = data.join(experiments[['strain', 'gender']], on='experiment_id')

    qvals = generate_qval_data(data)
    strains = ['BL6', 'CD1', 'ALL']
    genders = ['male', 'female']

    for strain in strains:
        plot_effect_vs_effect(qvals, strain, 'male_vs_female', 'density3d', 'volume', f'density_effect_vs_volume_effect')

        produce_volcano_plot(qvals, strain, 'male_vs_female',
                             f'male_vs_female_volume_effect',
                             f'volume_ranksum',
                             f'volume_qval_vs_normalized_M-F')

        for param, highlight_param in [('count3d', 'density3d'), ('density3d', 'count3d')]:
            produce_volcano_plot(qvals, strain, 'male_vs_female',
                                 f'male_vs_female_{param}_effect',
                                 f'{param}_ranksum',
                                 f'{param}_qval_vs_normalized_M-F')
            plot_qval_vs_qval(qvals, strain, 'male_vs_female',
                              f'{highlight_param}_ranksum',
                              f'{param}_ranksum',
                              'volume_ranksum',
                              f'{param}_qval_vs_volume_qval')

        plot_effect_vs_effect(qvals, strain, 'male_vs_female', 'density3d', 'volume', f'density_effect_vs_volume_effect')

        for (param, highlight_param), gender in itertools.product([('count', 'density'), ('density', 'count')],
                                                                  genders):
            produce_volcano_plot(qvals, strain, 'left_vs_right',
                                 f'{gender}_left_vs_right_{param}_effect',
                                 f'{param}_{gender}_ranksum',
                                 f"{param}_{gender}_qval_vs_normalized_M-F")
            plot_qval_vs_qval(qvals, strain, 'left_vs_right',
                              f'{highlight_param}_{gender}_ranksum',
                              f'{param}_{gender}_ranksum',
                              f'region_area_{gender}_ranksum',
                              f"{param}_{gender}_qval_vs_volume_qval")

    for param in ['count3d', 'density3d', 'volume']:
        x = qvals[f'BL6_male_vs_female_{param}_effect']
        y = qvals[f'CD1_male_vs_female_{param}_effect']
        qval_x = qvals[f"BL6_male_vs_female_{param}_ranksum_qval"]
        qval_y = qvals[f"CD1_male_vs_female_{param}_ranksum_qval"]
        produce_qvalue_marked_comparison(qval_x, qval_y, x, y, f'{param}_BL6_vs_CD1_M-F')

        x = qvals[f'BL6_vs_CD1_M_{param}_effect']
        y = qvals[f'BL6_vs_CD1_F_{param}_effect']
        qval_x = qvals[f"BL6_vs_CD1_M_{param}_ranksum_qval"]
        qval_y = qvals[f"BL6_vs_CD1_F_{param}_ranksum_qval"]
        produce_qvalue_marked_comparison(qval_x, qval_y, x, y, f'{param}_BL6-CD1_M_vs_F')


def produce_qvalue_marked_comparison(qval_x, qval_y, x, y, name):
    markers = {
        (False, False): ('b', '.'),
        (False, True): ('r', 'h'),
        (True, False): ('r', '*'),
        (True, True): ('r', 'd'),
    }

    non_nulls = ~qval_x.isnull() & ~qval_y.isnull()
    x = x[non_nulls]
    y = y[non_nulls]
    qval_x = qval_x[non_nulls] < 0.05
    qval_y = qval_y[non_nulls] < 0.05
    fig, ax = get_subplots()
    for qx, qy in itertools.product([True, False], [True, False]):
        px = x[(qval_x == qx) & (qval_y == qy)]
        py = y[(qval_x == qx) & (qval_y == qy)]
        plot_scatter(ax, px, py, color=markers[(qx, qy)][0], marker=markers[(qx, qy)][1])
        if qx or qy:
            plot_annotations(ax, px.index, px, py, fontsize=3)
    plot_grid(ax, [0], [0])
    xpoints = ypoints = ax.get_xlim()
    ax.plot(xpoints, ypoints, linestyle='-.', color='r', lw=1, scalex=False, scaley=False)
    ax.plot(xpoints, np.array(ypoints) - abs(ypoints[0] - ypoints[1]) / 10, linestyle='-.', color='r', lw=1,
            scalex=False, scaley=False)
    ax.plot(xpoints, np.array(ypoints) + abs(ypoints[0] - ypoints[1]) / 10, linestyle='-.', color='r', lw=1,
            scalex=False, scaley=False)
    produce_figure(ax, fig, name)
