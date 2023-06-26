import itertools
import operator
import os
import urllib.request
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from scipy import stats
from streamlit_tree_select import tree_select

from explorer.explorer_utils import hist

FILTERING_CRITERIA = [('gender', 'Sex'), ('strain', 'Strain'), ('transgenic_line', 'Transgenic line')]

HIDDEN_PARAMETERS = {'count',
                     'density',
                     'section_count',
                     'brightness',
                     'region_area',
                     'injection'}

BOTH_HEMISPHERES = 'both'


def separate_hemispheres(parameters):
    parameter_dict = defaultdict(list)

    for param in parameters:
        if param.endswith('_left'):
            param_name = param[:-5]  # remove '_left'
            parameter_dict[param_name].append('left')
        elif param.endswith('_right'):
            param_name = param[:-6]  # remove '_right'
            parameter_dict[param_name].append('right')
        else:
            parameter_dict[param] += []

    return {k: ([], [BOTH_HEMISPHERES] + v) for k, v in parameter_dict.items()}


# Function to load your dataframe, decorated with st.cache so it's only loaded once
@st.cache_data
def load_data():
    def convert_node(node, regions, path, nodes_dict):
        children = structure_tree.children([node['id']])[0]
        children = [convert_node(child, regions, path + [child['acronym']], nodes_dict) for child in children]
        children = [child for child in children if child['label'] in regions or child['children']]
        for child in children:
            nodes_dict[child['label']] = path
        return {
            "label": node['acronym'],
            "value": node['acronym'],
            "children": children
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

    filtering_columns = list(map(operator.itemgetter(0), FILTERING_CRITERIA))
    exps = exps[filtering_columns].copy()
    for col in filtering_columns:
        exps[col][exps[col].isin([None])] = 'Unspecified'

    joined = stats.join(exps, on='experiment_id').set_index("experiment_id")

    structure_tree = mcc.get_structure_tree()
    root = structure_tree.get_structures_by_acronym(['grey'])[0]
    nodes_dict = {"": ['grey']}
    nodes = [convert_node(root, set(joined.region.unique()), ['grey'], nodes_dict)]

    return joined.reset_index(), nodes, nodes_dict, stats.set_index(['experiment_id', 'region']).columns


def remove_selected(selected_region_groups=None):
    for g in selected_region_groups:
        del st.session_state.region_group[g]


def clear_selected():
    st.session_state.region_group = dict()


def render_histogram_options():
    st.sidebar.header("Histogram options")
    bins = st.sidebar.slider("Histogram bins", min_value=10, max_value=50, value=20)
    show_median = st.sidebar.checkbox("Show median", value=True)
    show_steps = st.sidebar.checkbox("Show raw histogram (steps)", value=True)
    return bins, show_median, show_steps


def render_selection_management(complete_selection, df, selected_parameter, title):
    st.sidebar.header("Selection Management")
    # Button to add the selected region to the group
    if st.sidebar.button('Add to Selected', disabled=not complete_selection):
        st.session_state.region_group[title] = df[selected_parameter]
    selected_region_groups = st.sidebar.multiselect('Selected', st.session_state.region_group.keys())
    if not selected_region_groups:
        selected_region_groups = st.session_state.region_group.keys()
    col1, col2 = st.sidebar.columns(2, gap='small')
    col1.button('Remove from Selected', on_click=lambda: remove_selected(selected_region_groups),
                disabled=(selected_region_groups == st.session_state.region_group.keys()))
    col2.button('Clear All', on_click=clear_selected)
    return selected_region_groups


def render_region_selection(df, nodes, nodes_dict, selected_parameter, selected_criteria, unique_values):
    st.sidebar.header("Region")
    node = st.sidebar.selectbox("Search region", sorted(nodes_dict.keys()))
    with st.sidebar:
        selected_regions = tree_select(nodes,
                                       'all',
                                       expanded=list(nodes_dict[node]),
                                       checked=[node] if node != '' else [],
                                       only_leaf_checkboxes=False,
                                       no_cascade=True)
    df = df[df.region.isin(selected_regions['checked'])]
    complete_selection = (len(df) > 0)
    if 'region_group' not in st.session_state:
        st.session_state.region_group = dict()
    title_values = [(selected_criteria[c], unique_values[c]) for c in selected_criteria.keys()]
    title_values += [(selected_regions['checked'], None)]
    title = ":".join([','.join(act) if act is not dv else "<All>" for act, dv in title_values] + [selected_parameter])
    st.sidebar.text_input("Selection Title", value=title, disabled=not complete_selection)
    return complete_selection, df, title


def render_filtering_criteria(df, filtering_criteria):
    selected_criteria = dict()
    unique_values = dict()
    for col, title in filtering_criteria:
        unique_values[col] = df[col].unique()
        selected_criteria[col] = st.sidebar.multiselect(title, unique_values[col], default=[])

        if not selected_criteria[col]:
            selected_criteria[col] = unique_values[col]

        df = df[df[col].isin(selected_criteria[col])]
    return df, selected_criteria, unique_values


def render_basic_parameters(df, macroscopic, microscopic):
    st.sidebar.header('Basic Options')
    parameters = {**microscopic, **macroscopic}
    selected_param = st.sidebar.selectbox('Parameter', [v for v in (sorted(macroscopic.keys()) +
                                                        sorted(microscopic.keys())) if v not in HIDDEN_PARAMETERS])
    selected_hemisphere = st.sidebar.selectbox('Hemisphere', parameters[selected_param][1])
    selected_aggregation = st.sidebar.selectbox('Aggregation (for microscopic parameters)',
                                                parameters[selected_param][0])
    selected_parameter = '_'.join([v for v in [selected_param, selected_hemisphere] if v is not BOTH_HEMISPHERES])
    selected_parameter = '|'.join([v for v in [selected_parameter, selected_aggregation] if v is not None])

    df, selected_criteria, unique_values = render_filtering_criteria(df, FILTERING_CRITERIA)

    return df, selected_parameter, selected_criteria, unique_values


def render_sidebar(df, macroscopic, microscopic, nodes, nodes_dict):
    # Sidebar selectors
    st.sidebar.title('Brain Explorer')
    st.sidebar.markdown("See <a href='https://elifesciences.org/articles"
                        "/82376'>the article</a> for details", unsafe_allow_html=True)
    df, selected_parameter, selected_criteria, unique_values = render_basic_parameters(df, macroscopic, microscopic)
    complete_selection, df, title = render_region_selection(df, nodes, nodes_dict, selected_parameter,
                                                            selected_criteria, unique_values)
    selected_region_groups = render_selection_management(complete_selection, df, selected_parameter, title)
    bins, show_median, show_steps = render_histogram_options()
    return bins, selected_region_groups, show_median, show_steps


def render_manual():
    manual = requests.get('https://raw.githubusercontent.com/delkind/'
                          'mouse-brain-cell-counting/master/MANUAL.md').content.decode()
    manual = manual[manual.find("## Using the"):]
    st.markdown(manual)


def render_statistic_test_results(selected_region_groups):
    tests = {'T-Test': stats.ttest_ind, 'RankSum': stats.ranksums, 'KS-Test': stats.kstest}
    test_results = []
    for l, r in set(itertools.combinations(list(selected_region_groups), 2)):
        for test_name, test in tests.items():
            result = test(st.session_state.region_group[l].dropna().to_numpy(),
                          st.session_state.region_group[r].dropna().to_numpy())
            test_results += [{**{'Description': f'{l}, {r}'}, 'Test': test_name,
                              "Statistic": result.statistic, 'P-Value': result.pvalue}]
    if test_results:
        st.header("Statistic Test Results")
        st.dataframe(pd.DataFrame(test_results).set_index(["Description", "Test"]))


def render_medians(selected_region_groups):
    medians = []
    for g in selected_region_groups:
        medians += [{'Group': g, 'Median': st.session_state.region_group[g].median()}]
    st.header("Medians")
    st.dataframe(pd.DataFrame(medians).set_index('Group'), column_config={
        "Median": st.column_config.NumberColumn(
            "Median",
            format="%.3e",
        )
    })


def render_histogram(bins, selected_region_groups, show_median, show_steps):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for name in selected_region_groups:
        hist(ax, st.session_state.region_group[name], bins=bins, raw_hist=show_steps, median=show_median,
             label=name)
    ax.legend()
    st.header("Histogram")
    st.pyplot(fig)


def render_data_results(bins, selected_region_groups, show_median, show_steps):
    render_histogram(bins, selected_region_groups, show_median, show_steps)
    render_medians(selected_region_groups)
    render_statistic_test_results(selected_region_groups)


def render_main_screen(bins, selected_region_groups, show_median, show_steps):
    if st.session_state.region_group:
        render_data_results(bins, selected_region_groups, show_median, show_steps)
    else:
        render_manual()


def compute_parameters(parameters):
    parameters = list(filter(lambda s: '.' not in s, parameters))
    macroscopic = sorted([p for p in parameters if '|' not in p])
    microscopic = sorted([p for p in parameters if '|' in p])
    macroscopic = separate_hemispheres(macroscopic)
    microscopic = [(k.split('|')[0], k.split('|')[1]) for k in microscopic]
    microscopic = sorted(microscopic, key=operator.itemgetter(0))  # this might be unnecessary
    microscopic = {k: ([t[1] for t in v], [BOTH_HEMISPHERES]) for k, v in
                   itertools.groupby(microscopic, operator.itemgetter(0))}
    return macroscopic, microscopic


def main():
    df, nodes, nodes_dict, parameters = load_data()
    macroscopic, microscopic = compute_parameters(parameters)

    bins, selected_region_groups, show_median, show_steps = render_sidebar(df, macroscopic, microscopic, nodes,
                                                                           nodes_dict)
    render_main_screen(bins, selected_region_groups, show_median, show_steps)


main()
