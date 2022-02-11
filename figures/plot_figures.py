from figures.clean_data import prepare_data
from figures.figure1 import figure1
from figures.figure2 import figure2
from figures.figure3 import figure3
from figures.util import load_data, fig_config, get_subplots, produce_figure, plot_grid, plot_annotations, plot_scatter
import numpy as np
import pickle


def main():
    data = load_data()

    fig_config['save_fig'] = True
    fig_config['display'] = False

    fig_config['path'] = "2"
    fig_config['prefix'] = 'preclean_'
    figure2(data, False)

    fig_config['prefix'] = 'brightclean_'
    valid_data = prepare_data(data, plot=True)

    fig_config['prefix'] = 'postclean_'
    fig_config['path'] = "2"
    figure2(valid_data, True)

    fig_config['prefix'] = ''
    fig_config['path'] = "1"

    figure1(data, valid_data)

    fig_config['path'] = "3"
    figure3(valid_data)


def display():
    data = load_data()

    fig_config['save_fig'] = True
    fig_config['display'] = False

    fig_config['path'] = "2"
    fig_config['prefix'] = 'preclean_'
    # figure2(data, False)

    fig_config['prefix'] = 'brightclean_'
    valid_data = prepare_data(data, plot=False)

    fig_config['prefix'] = 'postclean_'
    fig_config['path'] = "2"
    # figure2(valid_data, True)

    fig_config['prefix'] = ''
    fig_config['path'] = "1"

    # figure1(data, valid_data)

    fig_config['path'] = "3"
    figure3(valid_data)


def test_figure():
    x, y = pickle.load(open('aaa.pickle', 'rb'))
    y = np.array(y)
    x = x * 100
    dots = np.where((np.abs(x) > 5) & (y > 2))
    x_ann = x[dots]
    y_ann = y[dots]
    fig, ax = get_subplots()
    plot_scatter(ax, x, y, label='aaa')
    plot_annotations(ax, ['bbb'] * len(dots[0]), x_ann, y_ann, dot_color='r')
    plot_grid(ax, horiz_lines=[2], vert_lines=[5, -5])
    fig_config['save_fig'] = True
    fig_config['display'] = False
    produce_figure(ax, fig, "test_pdf", "xlabel", "ylabel", legend=True)


if __name__ == '__main__':
    display()
