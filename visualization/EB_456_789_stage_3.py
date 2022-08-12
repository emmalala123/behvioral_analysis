#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:10:22 2021

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats
import seaborn as sns
import random


eb4 = []
eb5 = []
eb6 = []
names = []
counter = 0
directory = ['/Users/emmabarash/Lab/data/eb10_data','/Users/emmabarash/Lab/data/eb11_data',\
             '/Users/emmabarash/Lab/data/eb12_data']

for i in directory:
    for filename in os.listdir(i):

        if i[-9:-5] == "eb10":
            eb4.append(filename)
        eb4.sort()
        if i[-9:-5] == "eb11":
            eb5.append(filename)
        eb5.sort()
        if i[-9:-5] == "eb12":
            eb6.append(filename)
        eb6.sort()
    
def join_files(list, files):
    for i in directory:
        for name in list:
            f = os.path.join(i, name)
            if os.path.isfile(f):
                files.append(pd.read_csv(f))
                

eb4_files = []
join_files(eb4, eb4_files)
eb5_files = []
join_files(eb5, eb5_files)
eb6_files = []
join_files(eb6, eb6_files)

# convert data frame values from strings to bool
def convert(files):
    converted = []       
    for i in range(len(files)):
        converted.append(files[i].replace({"None": False, "suc": True, "qhcl": True}))
    files = converted

convert(eb4_files)
convert(eb5_files)
convert(eb6_files)

# totals for the trigger and rewarder side activation

eb4_trig_counts = []
eb4_rew_counts = []

eb5_trig_counts = []
eb5_rew_counts = []

eb6_trig_counts = []
eb6_rew_counts = []

rew_all = []

def count_in_trial(list):
    counter = 0
    prev_i = False
    for i in list:
        if i != prev_i and i != False:
            counter = counter + 1
        prev_i = i
    rew_all.append(counter)
    return counter


def setup_for_trial_counts(files, trig_counts, rew_counts):
    for f in files:
        trig_counts.append(count_in_trial(f["Line2"]))
        rew_counts.append(count_in_trial(f["Line1"]))

setup_for_trial_counts(eb4_files, eb4_trig_counts, eb4_rew_counts)
setup_for_trial_counts(eb5_files, eb5_trig_counts, eb5_rew_counts)
setup_for_trial_counts(eb6_files, eb6_trig_counts, eb6_rew_counts)


eb4_percentage = []
eb5_percentage = []
eb6_percentage = []
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


find_percentage(eb4_trig_counts, eb4_rew_counts, eb4_percentage)
find_percentage(eb5_trig_counts, eb5_rew_counts, eb5_percentage)
find_percentage(eb6_trig_counts, eb6_rew_counts, eb6_percentage)
        
percentage_all = [eb4_percentage, eb5_percentage ,eb6_percentage]

#pad the data to make exact same column (session) length
max_sess = np.max([len(x) for x in percentage_all])
#pad_matrix = [x for x in percentage_all if len(x)==max_sess\
              #else np.ones_like(max_sess-len(x))]
    
pad_matrix = np.asarray([x if len(x)==max_sess else\
                         np.concatenate([x,np.nan*np.ones(max_sess-len(x))]) for x in percentage_all])
    
#error = [eb1_percentage, eb2_percentage, eb3_percentage]
#np.concatenate(x,np.ones_like(max_sess-len(x)))
# cutoff at 6 days
#plt.ylim(0,105)
#plt.errorbar(range(max_sess),np.nanmean(pad_matrix,0))
plt.xlabel("Number of Sessions")
plt.ylabel("Performance after x-days of Additive Shaping")
#plt.legend(["90% threshold for learning", "% successful alternations"], facecolor="white", loc='lower right')
plt.title("All Animals: Percentage of Completed Alternations Per Session")
plt.ylim(0,105)
plt.xlim(0.5,8.5)
plt.errorbar(range(5),np.nanmean(pad_matrix,0)[0:8])
plt.scatter(range(max_sess),pad_matrix[0], c='blue', alpha=0.5, s=15) 
plt.scatter(range(max_sess),pad_matrix[1], c='purple', alpha=0.5, s=15) 
plt.scatter(range(max_sess),pad_matrix[2], c='goldenrod', alpha=0.5, s=15) 
plt.scatter(range(max_sess),pad_matrix[3], c='red', alpha=0.5, s=15) 
plt.scatter(range(max_sess),pad_matrix[4], c='green', alpha=0.5, s=15) 
plt.scatter(range(max_sess),pad_matrix[5], c='orange', alpha=0.5, s=15) 
plt.legend(["EB10", "EB11", "EB12"], facecolor="white", loc="lower right")
plt.axhline(90,c='black',linestyle=':')
plt.show()
#### confidence intervals
#sns.lineplot(range(max_sess),eb1_percentage)


