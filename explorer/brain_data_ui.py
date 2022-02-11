import base64
import io
import itertools

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from IPython.display import display, Markdown, HTML

from explorer.explorer_utils import hist, retrieve_nested_path
from explorer.ui import ExperimentsSelector, ResultsSelector
from figures.util import get_subplots, produce_figure


class DataSelector(widgets.VBox):
    def __init__(self, data_dir, results_selector):
        # self.experiment_selector = ExperimentsSelector([e for e in os.listdir(data_dir) if e.isdigit()])
        self.experiment_selector = ExperimentsSelector(results_selector.get_available_brains())
        self.results_selector = results_selector
        self.add_button = widgets.Button(description='Add')
        self.add_button.on_click(lambda b: self.add_data())
        self.remove_button = widgets.Button(description='Remove')
        self.remove_button.on_click(lambda b: self.remove_data())
        self.clear_button = widgets.Button(description='Reset')
        self.clear_button.on_click(lambda b: self.reset_data())
        self.output = widgets.Output()
        self.messages = widgets.Output()
        self.added = widgets.SelectMultiple(options=[])
        self.data = {}
        super().__init__((
            self.experiment_selector, self.results_selector,
            widgets.HBox((self.add_button, self.remove_button, self.clear_button), layout=widgets.Layout(width='auto')),
            self.messages,
            widgets.HBox((self.added,)),
            self.output))

    def output_message(self, message):
        self.messages.clear_output()
        with self.messages:
            display(Markdown(message))

    def reset_data(self):
        self.output.clear_output()
        self.added.options = ()
        self.messages.clear_output()
        self.data = {}

    def remove_data(self):
        if self.added.value:
            for v in self.added.value:
                del self.data[v]
            vals = [o for o in self.added.value]
            self.added.options = [o for o in self.added.options if o not in vals]
            self.messages.clear_output()
            with self.messages:
                display(Markdown("Selection removed"))

    def extract_values(self):
        if not self.data:
            self.output_message("Nothing to process")
            return {}

        if self.added.value:
            values = {k: self.data[k] for k in self.added.value}
        else:
            values = self.data

        return values

    def add_data(self):
        relevant_experiments = self.experiment_selector.get_selection()
        if len(relevant_experiments) == 0:
            self.output_message('Nothing to add, no relevant brains available')
        else:
            path = self.results_selector.get_selection_label()
            data = self.results_selector.get_selection(relevant_experiments)
            if isinstance(path, list):
                for p, d in zip(path, data):
                    if np.median(d) > 0:
                        self.add_data_item(d[d != 0], p, relevant_experiments)
            else:
                self.add_data_item(data, path, relevant_experiments)

    def add_data_item(self, data, path, relevant_experiments):
        if data is None:
            self.output_message(f'Nothing to add')
        label = f"{self.experiment_selector.get_selection_label()}.{path} ({len(relevant_experiments)}:{len(data)})"
        if label not in self.data:
            self.added.options += (label,)
            self.data[label] = data
            self.output_message(f'Added data for {len(relevant_experiments)} brains')
        else:
            self.output_message(f'Already added')


class BrainAggregatesHistogramPlot(widgets.VBox):
    def __init__(self, data_dir, raw_data_selector):
        # self.data_selector = DataSelector(data_dir, ResultsSelector(pickle.load(open(f'{data_dir}/../stats.pickle',
        #                                                                              'rb'))))
        self.data_selector = DataSelector(data_dir, raw_data_selector)
        self.show_median = widgets.Checkbox(value=True, description='Show median', indent=True)
        self.show_steps = widgets.Checkbox(value=True, description='Show raw histogram (steps)', indent=True)
        self.bins = widgets.IntSlider(min=10, max=100, value=50, description='Bins: ')
        self.plot_hist_button = widgets.Button(description='Plot histogram')
        self.plot_hist_button.on_click(lambda b: self.plot_data(self.do_histogram_plot))
        self.plot_violin_button = widgets.Button(description='Plot violin')
        self.plot_violin_button.on_click(lambda b: self.plot_data(self.do_violin_plot))
        self.ttest_button = widgets.Button(description='T-Test')
        self.ttest_button.on_click(lambda b: self.test(stats.ttest_ind))
        self.ranksum_button = widgets.Button(description='RankSum')
        self.ranksum_button.on_click(lambda b: self.test(stats.ranksums))
        self.kstest_button = widgets.Button(description='KS-Test')
        self.kstest_button.on_click(lambda b: self.test(stats.kstest))
        self.median_button = widgets.Button(description='Median')
        self.median_button.on_click(lambda b: self.median())
        self.output = widgets.Output()
        self.messages = widgets.Output()
        header = widgets.Output()
        with header:
            display(Markdown("----"))
            display(Markdown("## Histogram"), )
        super().__init__((
            header,
            self.data_selector,
            self.messages,
            widgets.HBox((self.show_median, self.show_steps)),
            widgets.HBox((self.bins, self.plot_hist_button, self.plot_violin_button, self.ttest_button,
                          self.kstest_button,
                          self.ranksum_button, self.median_button)),
            self.output))
        self.histograms = dict()

    def output_message(self, message):
        self.messages.clear_output()
        with self.messages:
            display(Markdown(message))

    def plot_data(self, plotter):
        values = self.data_selector.extract_values()

        if values:
            self.output.clear_output()

            with self.output:
                df = pd.DataFrame({k: pd.Series(v) for k, v in values.items()})
                csv = df.to_csv()
                b64 = base64.b64encode(csv.encode())
                payload_csv = b64.decode()

                fig, ax = get_subplots()
                plotter(values, ax)
                buf = io.BytesIO()
                produce_figure(ax, fig, "plot", buf=buf, format_xticks=False)
                plt.close('all')
                buf.seek(0)
                b64 = base64.b64encode(buf.read())
                payload_pdf = b64.decode()

                html = '''<a download="{filename_csv}" href="data:text/csv;base64,{payload_csv}" target="_blank">{title_csv}</a><BR>
                <a download="{filename_pdf}" href="data:application/pdf;base64,{payload_pdf}" target="_blank">{title_pdf}</a>
                '''
                html = html.format(payload_csv=payload_csv,
                                   title_csv="Click to download data",
                                   filename_csv='data.csv',
                                   payload_pdf=payload_pdf,
                                   title_pdf="Click to download PDF",
                                   filename_pdf='plot.pdf')
                display(HTML(html))
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                plotter(values, ax)
                plt.show()
        else:
            with self.messages:
                self.messages.clear_output()

    def do_histogram_plot(self, values, ax):
        for l, d in values.items():
            hist(ax, d, bins=self.bins.value, raw_hist=self.show_steps.value, median=self.show_median.value, label=l)
        ax.legend()

    @staticmethod
    def do_violin_plot(values, ax):
        # ax.violinplot(list(values.values()), showmeans=True, showmedians=True, showextrema=True)
        data = pd.DataFrame({k: pd.Series(v) for k, v in values.items()})
        sns.violinplot(data=data, color='0.8', orient='v', ax=ax)
        sns.stripplot(data=data, alpha=0.5, ax=ax)
        ax.xaxis.set_tick_params(direction='out', rotation=67)
        ax.xaxis.set_ticks_position('bottom')

    def median(self):
        self.messages.clear_output()
        values = self.data_selector.extract_values()
        self.messages.clear_output()
        for k in values.keys():
            with self.messages:
                display(Markdown(f'Median for ({k}): {np.median(values[k])}'))

    def test(self, test):
        self.messages.clear_output()
        values = self.data_selector.extract_values()
        keys = list(values.keys())
        self.messages.clear_output()
        for l, r in set(itertools.combinations(range(len(keys)), 2)):
            with self.messages:
                l_median = np.median(values[keys[r]])
                r_median = np.median(values[keys[l]])
                display(Markdown(
                    f'({2 * (l_median - r_median) / (l_median + r_median)})'
                    f' {keys[l]}, {keys[r]}: {str(test(values[keys[r]], values[keys[l]]))}'))
