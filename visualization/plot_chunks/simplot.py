#! /home/toni/.pyenv/shims/python3
"""
Created on May 1, 2020

@author:toni
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from _collections_abc import generator

_root = os.getcwd()

_markers_full = ('.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8',
                 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|',
                 '_', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 'None')

linestyle_str = [
    ('solid', 'solid'),  # Same as (0, ()) or '-'
    ('dotted', 'dotted'),  # Same as (0, (1, 1)) or '.'
    ('dashed', 'dashed'),  # Same as '--'
    ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
    ('loosely dotted', (0, (1, 10))),
    ('dotted', (0, (1, 1))),
    ('densely dotted', (0, (1, 1))),

    ('loosely dashed', (0, (5, 10))),
    ('dashed', (0, (5, 5))),
    ('densely dashed', (0, (5, 1))),

    ('loosely dashdotted', (0, (3, 10, 1, 10))),
    ('dashdotted', (0, (3, 5, 1, 5))),
    ('densely dashdotted', (0, (3, 1, 1, 1))),

    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


def plot_linestyles(ax, linestyles):
    X, Y = np.linspace(0, 100, 10), np.zeros(10)
    yticklabels = []

    for i, (name, linestyle) in enumerate(linestyles):
        ax.plot(X, Y + i, linestyle=linestyle, linewidth=1.5, color='black')
        yticklabels.append(name)

    ax.set(xticks=[], ylim=(-0.5, len(linestyles) - 0.5),
           yticks=np.arange(len(linestyles)), yticklabels=yticklabels)

    # For each line style, add a text annotation with a small offset from
    # the reference point (0 in Axes coords, y tick value in Data coords).
    for i, (name, linestyle) in enumerate(linestyles):
        ax.annotate(repr(linestyle),
                    xy=(0.0, i), xycoords=ax.get_yaxis_transform(),
                    xytext=(-6, -12), textcoords='offset points',
                    color="blue", fontsize=8, ha="right", family="monospace")


fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]},
                               figsize=(10, 8))

plot_linestyles(ax0, linestyle_str[::-1])
plot_linestyles(ax1, linestyle_tuple[::-1])

plt.tight_layout()
plt.show()


def markers_test(markers):
    """Plot all markers for comparison."""
    mark = _generator(markers)
    x = list(range(-10, 10))  # [val for val in range(-10, 10)]
    n = len(markers)
    for i in range(n):
        y = [val + 100 * (n - i) for val in x]
        plt.plot(x, y, marker=next(mark), linewidth=2)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.show()


def plotter(data, xla, yla, data_label, figname):
    """
    This is a plotter.
    data - data set for plotting:
        x - list of floats, independent data set
        y - list of floats, dependent data set
    xla, yla - string, x axis label and y axis label
    data_label - legend name, type: string
    figname - string, name of the saved figure
    """
    (x, y) = data
    fig1 = plt.figure(1)
    plt.plot(x, y, 'gs-', label=data_label)
    plt.xlabel(xla)
    plt.ylabel(yla)
    plt.legend(loc='lower right')
    # to replace loosely dashed with dashed replace (0, (5, 10)) with '--'
    plt.grid(linestyle='--', linewidth=0.5)
    # turns interactive mode on, enables running in background
    # i.e. doesn't stop the execution when the plot shows
    # this is actually what you want, just save the plot to a .png
    plt.ion()
    plt.show()
    fig1.savefig(figname + '.png')
    plt.close(fig1)


def _generator(syms):
    """Yield a value from a provided sequence.

    syms -- sequence (list, tuple).
    """
    while True:
        for s in syms:
            yield s


def args_plotter(xlabel, ylabel, figname, *args):
    """
    Plot any number of data pairs and save the graphs in .png format.

    Parameters
    ----------
    xlabel -- string; Description of xlabel
    ylabel -- string; Description of ylabel
    figname -- string; Name of the saved figure without the .png extension
    *args -- tuples; Sets of data to be plotted.
        One set contains 3 values:
        x -- list of floats, independent data set
        y -- list of floats, dependent data set
        data_label -- string; curve name to be displayed in legend

    Function call example
    ---------------------
    args_plotter('Time, [s]', 'Displacement, [m]', 'displacement_history',
                 (x1, y1, 'curve1'), (x2, y2, 'curve2'), (xN, yN, 'curveN'))
    """

    # defines the default colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # markers = ['s', 'o', '*', 'v', '.']
    # mark = generator(markers)

    styles = ['-', '--', ':', '-.', ]
    style = _generator(styles)

    # optionally, pass marker=next(mark) as an argument to the plot function
    # uncomment the lines that defined the markers first!
    fig1 = plt.figure(1)
    for curve, _ in enumerate(args):
        plt.plot(args[curve][0], args[curve][1], color=colors[curve],
                 linestyle=next(style), linewidth=2, label=args[curve][2])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.ion()
    plt.show()
    fig1.savefig(figname + '.png')
    plt.close(fig1)


def path_change(path):
    """Replace backslash in Windows path with a forward slash.

    path -- string; path to the file
    """
    return path.replace('\\', '/')


def plot():
    """A basic plot."""
    # does not work if those commands are called from the interpreter
    squares = [1, 4, 9, 16, 25]
    plt.plot(squares)
    plt.show()


def bar_plot():
    """Plot a bar chart by using object oriented interface."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(x=['A', 'B', 'C'], height=[3.1, 7, 4.2], color='r')
    ax.set_xlabel(xlabel='X title', size=20)
    ax.set_ylabel(ylabel='Y title', color='b', size=20)
    plt.show()


# plt.hist()
# plt.pie()
# plt.scatter()
def main():
    bar_plot()
    plot()
    plt.plot([1, 2, 3], marker=11)
    plt.show()
    plt.plot([1, 2, 3], marker=mpl.markers.CARETDOWNBASE)
    plt.show()
    markers_test(_markers_full)


if __name__ == "__main__":
    main()
