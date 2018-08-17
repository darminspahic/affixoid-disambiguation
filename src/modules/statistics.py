#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Darmin Spahic <Spahic@stud.uni-heidelberg.de>
Project: Bachelorarbeit

Module name:
statistics

Short description:
This module displays various statistics

License: MIT License
Version: 1.0

"""

import matplotlib.pyplot as plt
import numpy as np


def plot_statistics(dict_1, dict_2, title):
    """ This function plots charts with affixoid statistics.

        Args:
            dict_1 (dict): Dictionary with Y instances
            dict_2 (dict): Dictionary with N instances
            title (str): Title of the chart

        Returns:
            Matplotlib images

        Example:
            >>> plot_statistics({'Bilderbuch': 10}, {'Bilderbuch': 5}, 'Prefixoids')

    """

    n = len(dict_1.keys())

    y_candidates = dict_1.values()

    ind = np.arange(n)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, y_candidates, width, color='0.85')

    n_candidates = dict_2.values()
    rects2 = ax.bar(ind + width, n_candidates, width, color='0.75')

    # adds text for labels, title and axes ticks
    ax.set_ylabel('Counts')
    ax.set_title('Counts per ' + title + ' candidate. Total: ' + str(sum(dict_1.values()) + sum(dict_2.values())) + '')
    ax.set_xticks(ind + width)
    ax.set_xticklabels((dict_1.keys()))

    ax.legend((rects1[0], rects2[0]), ('Y', 'N'))

    def autolabel(rects):
        # attaches text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.show()
