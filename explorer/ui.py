import os

import ipytree
import ipywidgets as widgets
import numpy as np
from IPython.display import display, Markdown
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import matplotlib.pyplot as plt

from aggregate_cell_data import get_struct_aggregates, acronyms
from explorer.explorer_utils import retrieve_nested_path, DataFramesHolder


class StructureTreeNode(ipytree.Node):
    show_icon = False

    def __init__(self, name, struct_id, children):
        super().__init__(name=name, nodes=children)
        self.struct_id = struct_id
        self.opened = False
        self.parent = None


class StructureTree(widgets.VBox):
    def __init__(self, ids, multiple_selection):
        self.ids = ids
        self.mcc = MouseConnectivityCache(manifest_file=f'./.mouse_connectivity/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.structure_tree = self.mcc.get_structure_tree()
        self.nodes = dict()
        self.tree_widget = ipytree.Tree(nodes=[], multiple_selection=multiple_selection,
                                        layout=widgets.Layout(width='100%', overflow_y='scroll'))
        self.fill_tree()
        self.search = widgets.Dropdown(options=sorted(list(self.nodes.keys())))
        if not multiple_selection:
            self.search.observe(self.on_search, names=['value'])
        if not multiple_selection:
            super().__init__((self.search, self.tree_widget, ))
        else:
            super().__init__((self.tree_widget, ))

    def fill_tree(self):
        node = self.fill_node('grey')
        while len(node.nodes) < 2:
            node = node.nodes[0]
        self.tree_widget.add_node(node)

    def fill_node(self, start_node):
        start_struct = self.structure_tree.get_structures_by_acronym([start_node])[0]
        children = [self.fill_node(c['acronym']) for c in self.structure_tree.children(
            [self.structure_tree.get_id_acronym_map()[start_node]])[0]]
        children = [c for c in children if c is not None]

        if not children and start_struct['acronym'] not in self.ids:
            return None

        node = StructureTreeNode(start_struct['acronym'],
                                 start_struct['acronym'],
                                 children)
        for c in children:
            c.parent = node
        if start_struct['acronym'] not in self.ids:
            node.disabled = True
        else:
            if not self.tree_widget.multiple_selection:
                node.observe(self.on_tree_select, 'selected')

        self.nodes[start_struct['acronym']] = node

        return node

    def on_search(self, event):
        node = self.nodes.get(event['new'], None)
        if node is not None:
            if not self.tree_widget.multiple_selection:
                for n in self.nodes.values():
                    n.open = False
                    n.selected = False
            node.selected = True
            node.opened = False
            parent = node.parent
            while parent:
                parent.opened = True
                parent = parent.parent

    def on_tree_select(self, event):
        if event['new']:
            self.search.value = event['owner'].struct_id

    def get_selection(self):
        return self.search.value


class ExperimentsSelector(widgets.VBox):
    def __init__(self, available_brains=None):
        self.mcc = MouseConnectivityCache(manifest_file=f'./.mouse_connectivity/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.messages = widgets.Output()
        self.set_available_brains(available_brains)
        self.filter = {
            col: widgets.SelectMultiple(description=col, options=self.get_column_options(self.experiments[col]))
            for col in ['gender', 'strain', 'transgenic_line', 'id']}
        for c, f in self.filter.items():
            f.observe(lambda change, col=c: self.selection_changed(change, col))
        self.change_handler = None
        self.selection_changed({'name': 'value'}, 'gender')

        super().__init__((widgets.HBox(list(self.filter.values())), self.messages))

    def reset(self):
        for v in self.filter.values():
            v.value = []

    @staticmethod
    def get_column_options(df):
        return sorted(df.unique().tolist())

    def set_available_brains(self, available_brains):
        self.experiments = self.mcc.get_experiments(dataframe=True)
        self.experiments.at[self.experiments.strain.isin([None]), 'strain'] = '<none>'
        self.experiments.at[self.experiments.transgenic_line.isin([None]), 'transgenic_line'] = '<none>'
        if available_brains is not None:
            self.experiments = self.experiments[self.experiments.id.isin([int(e) for e in available_brains])]
        self.messages.clear_output()
        with self.messages:
            display(Markdown(f'Selected {len(self.get_selection())} brains'))

    def get_filter_value(self, col):
        selection = self.filter[col].value
        if selection is None or not selection:
            selection = self.filter[col].options
        return selection

    def selection_changed(self, change, col):
        if change['name'] == 'value':
            self.messages.clear_output()
            with self.messages:
                display(Markdown(f'Selected {len(self.get_selection())} brains'))

            selection = self.get_filter_value(col)

            if col == 'gender':
                oldval = self.filter['strain'].value
                self.filter['strain'].options = self.get_column_options(
                    self.experiments[self.experiments.gender.isin(selection)].strain)
                oldval = [v for v in oldval if v in self.filter['strain'].options]
                self.filter['strain'].value = oldval
                self.selection_changed({'name': 'value'}, 'strain')
            elif col == 'strain':
                oldval = self.filter['transgenic_line'].value
                self.filter['transgenic_line'].options = self.get_column_options(
                    self.experiments[self.experiments.gender.isin(self.get_filter_value('gender'))
                                     & self.experiments.strain.isin(selection)].transgenic_line)
                oldval = [v for v in oldval if v in self.filter['transgenic_line'].options]
                self.filter['transgenic_line'].value = oldval
                self.selection_changed({'name': 'value'}, 'transgenic_line')
            elif col == 'transgenic_line':
                oldval = self.filter['id'].value
                self.filter['id'].options = self.get_column_options(
                    self.experiments[self.experiments.gender.isin(self.get_filter_value('gender')) &
                                     self.experiments.strain.isin(self.get_filter_value('strain')) &
                                     self.experiments.transgenic_line.isin(selection)].id)
                oldval = [v for v in oldval if v in self.filter['id'].options]
                self.filter['id'].value = oldval

            if self.change_handler is not None:
                self.change_handler(change, col)

    def get_selection(self):
        selection = {col: sel.value for col, sel in self.filter.items()}
        query_string = ' and '.join([f'{col} == {val}' for col, val in selection.items() if val])
        if query_string:
            selection = self.experiments.query(query_string)
        else:
            selection = self.experiments

        return set(selection.id.unique().tolist())

    def get_selection_label(self):
        label = '.'.join([f'({",".join([str(v) for v in val.value])})'
                          for val in self.filter.values() if val.value])
        if label == '':
            label = '<all>'

        return label

    def on_selection_change(self, handler):
        self.change_handler = handler


class ResultsSelector(widgets.HBox):
    def __init__(self, data):
        self.data = data
        self.selector = widgets.Dropdown()
        self.selector.options = [d for d in data if d not in {'experiment_id', 'region'}]
        self.selector.value = self.selector.options[0]
        self.selector.observe(self.on_change)
        self.tree = StructureTree(data.region.unique(), multiple_selection=False)
        self.tree.layout.width = "100%"
        self.tree.layout.max_height = "240px"
        self.tree.layout.overflow_y = 'scroll'
        super().__init__([self.tree, self.selector])

    @staticmethod
    def enable_selector(selector, mode):
        if mode:
            selector.layout.visibility = 'visible'
        else:
            selector.value = None
            selector.options = []
            selector.layout.visibility = 'hidden'

    def get_available_brains(self):
        return list(self.data.experiment_id.unique())

    def on_change(self, change):
        if change['name'] == 'value' and self.selector.options and self.selector.value is not None:
            pass

    def get_selection(self, relevant_experiments):
        path = self.get_selection_path()

        if path is None:
            return None

        return np.array(self.data[self.data.experiment_id.isin(relevant_experiments) &
                                  (self.data.region == path[0])][path[1]].dropna())

    def get_selection_label(self):
        path = self.get_selection_path()

        if path is None:
            return None

        return '.'.join(str(p) for p in path)

    def get_selection_path(self):
        return self.tree.get_selection(), self.selector.value


class SectionDataResultsSelector(widgets.HBox):
    def __init__(self, data):
        self.data = data
        self.param_selector = widgets.Dropdown()
        self.param_selector.options = list(set(d[:-2] for d in data if d not in {'experiment_id', 'region'}))
        self.param_selector.value = self.param_selector.options[0]
        self.param_selector.observe(self.on_change)
        self.bin_selector = widgets.SelectMultiple()
        self.bin_selector.options = list(range(10))
        self.bin_selector.value = [self.bin_selector.options[0]]
        self.bin_selector.observe(self.on_change)
        self.tree = StructureTree(data.region.unique(), multiple_selection=False)
        self.tree.layout.width = "100%"
        self.tree.layout.max_height = "240px"
        self.tree.layout.overflow_y = 'scroll'
        super().__init__([self.tree, self.param_selector, self.bin_selector])

    @staticmethod
    def enable_selector(selector, mode):
        if mode:
            selector.layout.visibility = 'visible'
        else:
            selector.value = None
            selector.options = []
            selector.layout.visibility = 'hidden'

    def get_available_brains(self):
        return list(self.data.experiment_id.unique())

    def on_change(self, change):
        if change['name'] == 'value' and self.param_selector.options and self.param_selector.value is not None:
            pass

    def get_selection(self, relevant_experiments):
        path = self.get_selection_path()

        if path is None:
            return None

        return [np.array(self.data[self.data.experiment_id.isin(relevant_experiments) &
                                   (self.data.region == path[0])][f'{path[1]}_{b}']) for b in path[2]]

    def get_selection_label(self):
        path = self.get_selection_path()

        if path is None:
            return None

        return ['.'.join(str(p) for p in (path[:-1] + (b,))) for b in path[-1]]

    def get_selection_path(self):
        return self.tree.get_selection(), self.param_selector.value, [b for b in self.bin_selector.value]


class RawDataResultsSelector(widgets.VBox):
    def __init__(self, data_dir):
        self.data_frames = DataFramesHolder(data_dir)
        self.available_brains = [e for e in os.listdir(data_dir) if e.isdigit()]
        self.data_template = self.data_frames[self.available_brains[0]]
        self.messages = widgets.Output()
        self.mcc = MouseConnectivityCache(manifest_file=f'./.mouse_connectivity/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.structure_tree = self.mcc.get_structure_tree()
        self.aggregates = get_struct_aggregates(set(self.get_column_options(self.data_template['structure_id'])))

        self.tree = StructureTree([s['acronym'] for s in self.structure_tree.get_structures_by_id(
            list(self.aggregates.keys()))], multiple_selection=False)
        self.tree.layout.width = "100%"
        self.tree.layout.max_height = "240px"
        self.tree.layout.overflow_y = 'scroll'

        self.parameter_selector = widgets.Dropdown(description="Parameter", options=['coverage', 'area', 'perimeter'])

        self.change_handler = None
        super().__init__((widgets.HBox([self.tree, self.parameter_selector]), self.messages))

    def get_available_brains(self):
        return self.available_brains

    def reset(self):
        for v in self.filter.values():
            v.value = []

    @staticmethod
    def get_column_options(df):
        return sorted(df.unique().tolist())

    def get_selected_structs(self):
        selected_acronyms = [self.tree.get_selection()]
        id_sets = [self.aggregates[self.structure_tree.get_id_acronym_map()[s]] for s in selected_acronyms]
        ids = set.union(*id_sets)
        return [acronyms[i] for i in ids]

    def process_data_frame(self, df, structs):
        frame = df[df.structure_id.isin(structs)]
        d = frame[self.parameter_selector.value].to_numpy()
        return d

    def get_selection(self, relevant_experiments):
        structs = [self.structure_tree.get_id_acronym_map()[s] for s in self.get_selected_structs()]
        return np.concatenate([self.process_data_frame(self.data_frames[e], structs) for e in relevant_experiments])

    def get_selection_label(self):
        path = self.get_selection_path()

        if path is None:
            return None

        return '.'.join(str(p) for p in path)

    def get_selection_path(self):
        return self.tree.get_selection(), self.parameter_selector.value

    def on_selection_change(self, handler):
        self.change_handler = handler
