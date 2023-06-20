import itertools
import os
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from scipy import stats
from streamlit_tree_select import tree_select

from explorer.explorer_utils import hist

# Create an instance of the cache class
mcc = MouseConnectivityCache(manifest_file='./.mouse_connectivity/mouse_connectivity_manifest.json', resolution=25)
structure_tree = mcc.get_structure_tree()


# Function to load your dataframe, decorated with st.cache so it's only loaded once
@st.cache_data
def load_my_dataframe():
    def convert_node(node, regions):
        children = structure_tree.children([node['id']])[0]
        children = [convert_node(child, regions) for child in children]
        return {
            "label": node['acronym'],
            "value": node['acronym'],
            "children": [child for child in children if child['label'] in regions or child['children']]
        }

    mcc = MouseConnectivityCache(manifest_file='./.mouse_connectivity/mouse_connectivity_manifest.json', resolution=25)
    data_dir = './.mouse_connectivity'
    data_file = f'{data_dir}/stats.parquet'
    if not os.path.isfile(data_file):
        urllib.request.urlretrieve("https://storage.googleapis.com/www_zeisellab/allen_cellcounting_web/stats.parquet",
                                   data_file)
    stats = pd.read_parquet(data_file)
    exps = mcc.get_experiments(dataframe=True)
    exps = exps.drop_duplicates(['id']).set_index('id')
    joined = stats.join(exps, on='experiment_id').set_index("experiment_id")

    structure_tree = mcc.get_structure_tree()

    root = structure_tree.get_structures_by_acronym(['grey'])[0]
    nodes = [convert_node(root, set(joined.region.unique()))]

    return joined.reset_index(), nodes, stats.set_index(['experiment_id', 'region']).columns


df, nodes, parameters = load_my_dataframe()

# Sidebar selectors
st.sidebar.title("Brain Explorer")
st.sidebar.header('Basic Options')
parameters = list(filter(lambda s: '.' not in s, parameters))
parameters = sorted([p for p in parameters if '|' not in p]) + sorted([p for p in parameters if '|' in p])
selected_parameter = st.sidebar.selectbox('Parameter', parameters)
genders = df['gender'].unique()
selected_genders = st.sidebar.multiselect('Gender', genders, default=[])
if not selected_genders:
    selected_genders = genders

df = df[df.gender.isin(selected_genders)]
strains = df['strain'].unique()
selected_strains = st.sidebar.multiselect('Strain', strains, default=[])
if not selected_strains:
    selected_strains = strains

df = df[df.strain.isin(selected_strains)]
transgenic_lines = df['transgenic_line'].unique()
selected_transgenic_lines = st.sidebar.multiselect('Transgenic Line', transgenic_lines, default=[])
if not selected_transgenic_lines:
    selected_transgenic_lines = transgenic_lines

df = df[df.transgenic_line.isin(selected_transgenic_lines)]
ids = df['experiment_id'].unique()
selected_ids = st.sidebar.multiselect('ID', ids, default=[])
if not selected_ids:
    selected_ids = ids

df = df[df.experiment_id.isin(selected_ids)]

st.sidebar.header("Region")
with st.sidebar:
    selected_regions = tree_select(nodes, 'all', only_leaf_checkboxes=False, no_cascade=True)

df = df[df.region.isin(selected_regions['checked'])]

complete_selection = (len(df) > 0)

if 'region_group' not in st.session_state:
    st.session_state.region_group = dict()

title_values = [(selected_genders, genders), (selected_strains, strains),
        (selected_transgenic_lines, transgenic_lines), (selected_ids, ids), (selected_regions['checked'], None)]
title = ":".join([','.join(act) if act is not dv else "<All>" for act, dv in title_values] + [selected_parameter])

st.sidebar.text_input("Selection Title", value=title, disabled=not complete_selection)

st.sidebar.header("Selection Management")

# Button to add the selected region to the group
if st.sidebar.button('Add to Selected', disabled=not complete_selection):
    st.session_state.region_group[title] = df[selected_parameter]


def remove_selected():
    for g in selected_region_groups:
        del st.session_state.region_group[g]


def clear_selected():
    st.session_state.region_group = dict()


selected_region_groups = st.sidebar.multiselect('Selected', st.session_state.region_group.keys())
if not selected_region_groups:
    selected_region_groups = st.session_state.region_group.keys()

col1, col2 = st.sidebar.columns(2, gap='small')
col1.button('Remove from Selected', on_click=remove_selected, disabled=(selected_region_groups == st.session_state.region_group.keys()))
col2.button('Clear Selected', on_click=clear_selected)

st.sidebar.header("Histogram options")

bins = st.sidebar.slider("Histogram bins", min_value=10, max_value=50, value=20)
show_median = st.sidebar.checkbox("Show median", value=True)
show_steps = st.sidebar.checkbox("Show raw histogram (steps)", value=True)

if st.session_state.region_group:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for name in selected_region_groups:
        hist(ax, st.session_state.region_group[name], bins=bins, raw_hist=show_steps, median=show_median, label=name)
    ax.legend()
    st.header("Histogram")
    st.pyplot(fig)

    tests = {'T-Test': stats.ttest_ind, 'RankSum': stats.ranksums, 'KS-Test': stats.kstest}

    test_results = []
    medians = []

    for g in selected_region_groups:
        medians += [{'Group': g, 'Median': st.session_state.region_group[g].median()}]

    for l, r in set(itertools.combinations(list(selected_region_groups), 2)):
        for test_name, test in tests.items():
            result = test(st.session_state.region_group[l].dropna().to_numpy(),
                     st.session_state.region_group[r].dropna().to_numpy())
            test_results += [{**{'Description': f'{l}, {r}'}, 'Test': test_name,
                              "Statistic": result.statistic, 'P-Value': result.pvalue}]
    st.header("Medians")
    st.dataframe(medians)
    if test_results:
        st.header("Statistic Test Results")
        st.dataframe(pd.DataFrame(test_results).set_index(["Description", "Test"]))
