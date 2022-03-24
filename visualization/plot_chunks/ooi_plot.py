#! /home/toni/.pyenv/shims/python3
"""
Created on Nov 25, 2020

@author:toni

Plot by using the object oriented interface approach.
"""

# 2D kde plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# PLOT 1
# ----------------------------------------------------------------------------
np.random.seed(1)
numerical_1 = np.random.randn(100)

np.random.seed(2)
numerical_2 = np.random.randn(100)

# The key is the argument ax=ax. When running .kdeplot() method, seaborn
# would apply the changes to ax, an ‘axes’ object.
fig, ax = plt.subplots(figsize=(5, 5))
sns.kdeplot(x=numerical_1,
            y=numerical_2,
            ax=ax,
            shade=True,
            color="blue",
            bw_method=1)
plt.show()
# ----------------------------------------------------------------------------

# PLOT 2
# ----------------------------------------------------------------------------
df = pd.DataFrame(dict(categorical_1=['apple', 'banana', 'grapes',
                                      'apple', 'banana', 'grapes',
                                      'apple', 'banana', 'grapes'],
                       categorical_2=['A', 'A', 'A', 'B', 'B', 'B',
                                      'C', 'C', 'C'],
                       value=[10, 2, 5, 7, 3, 15, 1, 6, 8]))

pivot_table = df.pivot("categorical_1", "categorical_2", "value")

# try printing out pivot_table to see what it looks like!
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data=pivot_table,
            cmap=sns.color_palette("Blues"),
            ax=ax)
plt.show()
# ----------------------------------------------------------------------------

# PLOT 3
# ----------------------------------------------------------------------------
# overlay plot
fig, ax = plt.subplots(figsize=(4, 4))
sns.lineplot(x=['A', 'B', 'C', 'D'],
             y=[4, 2, 5, 3],
             color='r',
             ax=ax)
sns.lineplot(x=['A', 'B', 'C', 'D'],
             y=[1, 6, 2, 4],
             color='b',
             ax=ax)
ax.legend(['alpha', 'beta'], facecolor='w')
plt.show()
# ----------------------------------------------------------------------------

# PLOT 4
# ----------------------------------------------------------------------------
# because the two plots have different y-axis, we need to create another
# ‘axes’ object with the same x-axis (using .twinx()) and then plot on
# different ‘axes’. sns.set(…) is to set specific aesthetics for the current
# plot, and we run sns.set() in the end to set everything back to default
sns.set(style="white", rc={"lines.linewidth": 3})

fig, ax1 = plt.subplots(figsize=(4, 4))
ax2 = ax1.twinx()

sns.barplot(x=['A', 'B', 'C', 'D'],
            y=[100, 200, 135, 98],
            color='#004488',
            ax=ax1)

sns.lineplot(x=['A', 'B', 'C', 'D'],
             y=[4, 2, 5, 3],
             color='r',
             marker="o",
             ax=ax2)
plt.show()
sns.set()
# ----------------------------------------------------------------------------

# PLOT 5
# ----------------------------------------------------------------------------
categorical_1 = ['A', 'B', 'C', 'D']
colors = ['green', 'red', 'blue', 'orange']
numerical = [[6, 9, 2, 7],
             [6, 7, 3, 8],
             [9, 11, 13, 15],
             [3, 5, 9, 6]]

number_groups = len(categorical_1)
bin_width = 1.0 / (number_groups + 1)

fig, ax = plt.subplots(figsize=(6, 6))
for i in range(number_groups):
    ax.bar(x=np.arange(len(categorical_1)) + i * bin_width,
           height=numerical[i],
           width=bin_width,
           color=colors[i],
           align='center')

ax.set_xticks(
    np.arange(len(categorical_1)) + number_groups / (2 * (number_groups + 1)))

# number_groups/(2*(number_groups+1)): offset of xticklabel

ax.set_xticklabels(categorical_1)
ax.legend(categorical_1, facecolor='w')
plt.show()
# ----------------------------------------------------------------------------

# PLOT 6
# ----------------------------------------------------------------------------
tips = sns.load_dataset("tips")
ax = sns.scatterplot(x="total_bill", y="tip",
                     hue="size", size="size",
                     sizes=(20, 200), hue_norm=(0, 7),
                     legend="full", data=tips)
plt.show()
# ----------------------------------------------------------------------------

# PLOT 7
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(7, 7))
gs = mpl.gridspec.GridSpec(nrows=3,
                           ncols=3,
                           figure=fig,
                           width_ratios=[1, 1, 1],
                           height_ratios=[1, 1, 1],
                           wspace=0.3,
                           hspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax1.text(0.5, 0.5, 'ax1: gs[0, 0]', fontsize=12, fontweight="bold",
         va="center", ha="center")  # adding text to ax1

ax2 = fig.add_subplot(gs[0, 1:3])
ax2.text(0.5, 0.5, 'ax2: gs[0, 1:3]', fontsize=12, fontweight="bold",
         va="center", ha="center")

ax3 = fig.add_subplot(gs[1:3, 0:2])
ax3.text(0.5, 0.5, 'ax3: gs[1:3, 0:2]', fontsize=12, fontweight="bold",
         va="center", ha="center")

ax4 = fig.add_subplot(gs[1:3, 2])
ax4.text(0.5, 0.5, 'ax4: gs[1:3, 2]', fontsize=12, fontweight="bold",
         va="center", ha="center")
plt.show()
# ----------------------------------------------------------------------------


def main():
    pass


if __name__ == "__main__":
    main()
