import pickle
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.clean_data import CELL_COUNT_THRESHOLD, prepare_data
from scratch.clean_data_amit import trash_density3d_outliers, remove_correlated_regions
from figures.figure1 import find_leaves
from figures.util import mcc, load_data, to_col_name


def extract_data(s, param, statistic='median'):
    return s[param] if type(s[param]) != dict else s[param][statistic]


def filter_small_regions(data):
    regions = list(data[list(data.keys())[0]].keys())
    counts = {r: np.mean([data[e][r]['count'] for e in data.keys()]) for r in regions}
    data = {e: {s: d for s, d in ed.items() if counts[s] > CELL_COUNT_THRESHOLD} for e, ed in data.items()}
    return data


def process_experiment_section_data(experiment):
    experiment_data = pickle.load(open(f'./output/full_brain/stats-sections-{experiment}.pickle', 'rb'))
    data = list()
    for region, region_data in experiment_data.items():
        d = defaultdict(list)
        for section, section_data in region_data.items():
            for param, param_data in section_data.items():
                if isinstance(param_data, dict):
                    d[param].append(param_data['mean'])
                else:
                    d[param].append(param_data)
        reg_dict = dict()
        for col in d.keys():
            for i, val in enumerate(map(lambda x: x.mean() if x.shape[0] > 0 else 0, np.array_split(d[col], 10))):
                reg_dict[f'{col}_{i}'] = val
        reg_dict['experiment_id'] = experiment
        reg_dict['region'] = region
        data.append(reg_dict)
    return data


def produce_df(col_name, data, experiments):
    df = pd.pivot_table(data, columns=['experiment_id', 'region'])[col_name].unstack().join(
        experiments[['gender', 'strain', 'structure_id']], on='experiment_id').rename(
        columns={'structure_id': 'injection_site'})
    df['injection_site'] = [s['acronym'] for s in
                                 mcc.get_structure_tree().get_structures_by_id(df.injection_site.tolist())]
    return df


def build_csvs():
    experiments = mcc.get_experiments(dataframe=True)
    data = load_data()
    valid_data = prepare_data(data)
    params = ['count3d', 'density3d', 'volume', 'brightness', "diameter|percentile90"]
    regions = data.region.unique()
    build_region_map(regions)
    for name, data in {'unfiltered': data, 'filtered': valid_data}.items():
        regions = data.region.unique()
        leaves = regions[find_leaves(regions)]
        data = data[data.region.isin(leaves)]
        for param in params:
            col_name = to_col_name(data, param)
            df = produce_df(col_name, data, experiments)
            df.to_csv(f'{name}-{param}_data.tsv', sep='\t')

            col_name_left = to_col_name(data, param + "_left")
            col_name_right = to_col_name(data, param + "_right")

            if col_name_left in data.columns:
                df_left = produce_df(col_name_left, data, experiments)
                df_right = produce_df(col_name_right, data, experiments)

                df_left.to_csv(f'{name}-{param}_left_data.tsv', sep='\t')
                df_right.to_csv(f'{name}-{param}_right_data.tsv', sep='\t')


def build_region_map(regions=None):
    if regions is None:
        data = load_data()
        regions = data.region.unique()

    leaves = find_leaves(regions)

    reginfo = mcc.get_structure_tree().get_structures_by_acronym(regions)
    reginfo = [{**i, 'R': i['rgb_triplet'][0], 'G': i['rgb_triplet'][1], 'B': i['rgb_triplet'][2], 'leaf': leaves[n]} for n, i in enumerate(reginfo)]
    reginfo = pd.DataFrame(reginfo)
    reginfo[['acronym', 'id', 'name', 'leaf', 'R', 'G', 'B']].to_csv('regionmap.tsv', sep='\t')


def check_csvs():
    raw_data = load_data()
    regions = raw_data.region.unique()
    leaves = regions[find_leaves(regions)]
    raw_data = raw_data[raw_data.region.isin(leaves)]
    trashed_data = trash_density3d_outliers(raw_data)
    corr_data = remove_correlated_regions(trashed_data)

    data = pd.read_csv('unfiltered-density3d_data.tsv', sep='\t')
    numeric = data.select_dtypes([np.number]).drop(columns='experiment_id')
    deviation = ((numeric - numeric.median()) / (numeric.mad() * 1.4826)).abs()
    numeric[deviation > 2] = np.nan

    melted = numeric.join(data[['experiment_id']]).melt(id_vars=['experiment_id'], var_name='region',
                                                        value_name='density3d').set_index(['experiment_id', 'region'])
    joined = raw_data[['experiment_id', 'region']].join(melted, on=['experiment_id', 'region'])
    pass


def get_data(data_frame):
    data = defaultdict(dict)
    for ind, row in data_frame.iterrows():
        data[row.experiment_id][row.region] = get_region_data(row)

    return data


def get_region_data(row):
    region_data = defaultdict(dict)
    cols = [a for a in row.keys() if a not in {'region', 'experiment_id'}]
    for col in cols:
        split = col.split('|')
        if len(split) == 1:
            region_data[col] = row[col]
        else:
            region_data[split[0]][split[1]] = row[col]

    return region_data


def build_tree(start_node, data, leaves):
    regions = set(data.index.tolist())
    TreeNode = namedtuple('TreeNode', ['name', 'allen_volume', 'volume', 'child_volume', 'allen_child_volume',
                                       'children'])
    start_struct = mcc.get_structure_tree().get_structures_by_acronym([start_node])[0]
    children = [build_tree(c['acronym'], data, leaves) for c in mcc.get_structure_tree().children([start_struct['id']])[0]
                if c['acronym'] in regions]

    leaves.append({
        'region': start_node,
        'volume': data.volume[start_node],
        'allen_volume': data.allen_volume[start_node],
        'child_sum': sum(map(lambda n: n.volume, children)),
        'allen_child_sum': sum(map(lambda n: n.allen_volume, children)),
        'leaf': not children

    })

    return TreeNode(start_node, data.allen_volume[start_node], data.volume[start_node],
                    sum(map(lambda n: n.volume, children)), sum(map(lambda n: n.allen_volume, children)), children)


def compare_volumes():
    data = load_data()
    allen_data = pd.read_csv('~/Downloads/1-s2.0-S0092867420304025-mmc4.csv')
    joined = pd.DataFrame(data.groupby('region')['volume'].median()).join(
        allen_data[['abbreviation', 'Mean Volume (m)']].set_index('abbreviation'))
    joined = joined.rename(columns={'Mean Volume (m)': 'allen_volume'})

    leaves = []
    root = build_tree('grey', joined, leaves)
    joined = pd.DataFrame(leaves)

    joined_lim = joined[joined.leaf]
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.scatter(joined_lim.volume, joined_lim.allen_volume, s=1)
    for region in joined_lim.index.tolist():
        ax.annotate(region, (joined_lim.volume[region], joined_lim.allen_volume[region]), size=3)
    plt.savefig(f"volumes.pdf")
    plt.close()

    # lim = 0.125
    #
    # while lim < joined.volume.max():
    #     lim *= 2
    #     joined_lim = joined[joined.volume < lim]
    #     ax.scatter(joined_lim.volume, joined_lim.allen_volume, s=1)
    #     for region in joined_lim.index.tolist():
    #         ax.annotate(region, (joined_lim.volume[region], joined_lim.allen_volume[region]), size=3)
    #
    #     plt.savefig(f"volumes_{lim}.pdf")


def build_murakami_comparison():
    # import matplotlib.pyplot as plt
    data = load_data()
    experiments = mcc.get_experiments(dataframe=True)
    data = data.join(experiments[['gender', 'strain']], on='experiment_id')
    data = data[(data.gender == 'M') & (data.strain == 'C57BL/6J')]
    medians = pd.DataFrame(data.groupby('region')['count3d'].median())

    tree = pickle.load(open('./.mouse_connectivity/tree.pickle', 'rb'))

    murakami = pd.read_csv('/Users/delkind/Downloads/41593_2018_109_MOESM5_ESM.csv')
    murakami_counts = pd.DataFrame({'murakami_count': murakami.set_index(keys=['ABA ID'])[[
        '8wk01', '8wk02', '8wk03']].median(axis=1)})
    murakami_counts_dict = murakami_counts.murakami_count.to_dict()
    murakami_totals_dict = {tree[0][k]: sum([murakami_counts_dict.get(i, 0) for i in tree[1].get(k, {k})])
                            for k in murakami_counts_dict.keys()}
    murakami_counts = pd.DataFrame.from_dict(murakami_totals_dict, orient='index', columns=['murakami_count'])
    joined = medians.join(murakami_counts)
    joined.to_csv("murakami_comparison.tsv", sep='\t')


def build_region_and_grey_data():
    wa = pd.read_csv('./weights_ages.tsv', sep='\t').set_index('id')
    experiments = mcc.get_experiments(dataframe=True)
    data = load_data()
    data = data.join(experiments[['gender', 'strain']], on='experiment_id')
    region_data = data.groupby('region')[['volume', 'count3d', 'density3d']].median()
    region_data.to_csv('region_data.tsv', sep='\t')
    region_data = data[(data.gender == 'M') & (data.strain == 'C57BL/6J')].groupby('region')[['volume', 'count3d', 'density3d']].median()
    region_data.to_csv('region_data_bl6m.tsv', sep='\t')
    grey_data = data[data.region == 'grey'][['experiment_id', 'volume', 'count3d', 'density3d', 'gender', 'strain']].join(wa[['weight_grams', 'age_days']], on='experiment_id')
    grey_data.to_csv('greys_data.tsv', sep='\t')
    pass


build_csvs()