#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:29:37 2021

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import random

eb1 = []
eb2 = []
eb3 = []
names = []
counter = 0
directory = ['/Users/emmabarash/Lab/data/eb4_data/stage_3','/Users/emmabarash/Lab/data/eb5_data/stage_3','/Users/emmabarash/Lab/data/eb6_data/stage_3']

for i in directory:
    for filename in os.listdir(i):
        if i[-16:-13] == "eb4":
            eb1.append(filename)
        eb1.sort()
        if i[-16:-13] == "eb5":
            eb2.append(filename)
        eb2.sort()
        if i[-16:-13] == "eb6":
            eb3.append(filename)
        eb3.sort()
    
def join_files(list, files):
    for i in directory:
        for name in list:
            f = os.path.join(i, name)
            if os.path.isfile(f):
                files.append(pd.read_csv(f))
                
eb1_files = []
join_files(eb1, eb1_files)
eb2_files = []
join_files(eb2, eb2_files)
eb3_files = []
join_files(eb3, eb3_files)

# convert data frame values from strings to bool
def convert(files):
    converted = []       
    for i in range(len(files)):
        converted.append(files[i].replace({"None": False, "suc": True, "nacl": True}))
    files = converted

convert(eb1_files)
convert(eb2_files)
convert(eb3_files)

# totals for the trigger and rewarder side activation
eb1_trig_counts = []
eb1_rew_counts = []

eb2_trig_counts = []
eb2_rew_counts = []

eb3_trig_counts = []
eb3_rew_counts = []

def count_in_trial(list):
    counter = 0
    prev_i = False
    for i in list:
        if i != prev_i and i != False:
            counter = counter + 1
        prev_i = i
    return counter


def setup_for_trial_counts(files, trig_counts, rew_counts):
    for f in files:
        trig_counts.append(count_in_trial(f["Line2"]))
        rew_counts.append(count_in_trial(f["Line1"]))

setup_for_trial_counts(eb1_files, eb1_trig_counts, eb1_rew_counts)
setup_for_trial_counts(eb2_files, eb2_trig_counts, eb2_rew_counts)
setup_for_trial_counts(eb3_files, eb3_trig_counts, eb3_rew_counts)

eb1_percentage = []
eb2_percentage = []
eb3_percentage = []
def find_percentage(trig_counts, rew_counts, percentage):
    # create a session-by-session comparison
    total = zip(trig_counts, rew_counts)
    # make into a mutable list
    conv_total = list(total)
    # get a percentage of reward choice for each session
    for val in conv_total:
        if val[0] != 0:
            percent = round((val[1]/val[0])*100, 2)
            if percent <= 100 and percent > 0:
                percentage.append(percent)

find_percentage(eb1_trig_counts, eb1_rew_counts, eb1_percentage)
find_percentage(eb2_trig_counts, eb2_rew_counts, eb2_percentage)
find_percentage(eb3_trig_counts, eb3_rew_counts, eb3_percentage)
        
percentage_all = [eb1_percentage, eb2_percentage, eb3_percentage]

first_half = []
second_half = []
for percent_list in percentage_all:
    first_half.append(len(percent_list)//2)
    second_half.append((len(percent_list)//2 + 1))

first_mean=[]; second_mean=[]
for perc,idx1,idx2 in zip(percentage_all,first_half,second_half):
    first_mean.append(np.mean(perc[:idx1]))
    second_mean.append(np.mean(perc[idx2:]))
    
first_std = np.std(np.array(first_mean))
first_mean = np.mean(np.array(first_mean))
second_std = np.std(np.array(second_mean))
second_mean = np.mean(np.array(second_mean))

# Create lists for the plot
halves = ['first half', 'second half']
x_pos = np.arange(len(halves))
CTEs = [first_mean, second_mean]
error = [first_std, second_std]

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('% of Successful Alternations')
ax.set_xticks(x_pos)
ax.set_xticklabels(halves)
ax.set_xlabel("Training Sessions for All Animals")
plt.axhline(70,c='red',linestyle=':')
#plt.legend(["70% threshold for learning", "% successful alternations"], facecolor="white", bbox_to_anchor=[1, 0.1], loc='center left')
ax.set_title('Standard Deviation of Means for First and Second Halves of Training')
ax.yaxis.grid(False)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()


        
