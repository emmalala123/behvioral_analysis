#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 20:42:53 2022

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats
import seaborn as sns
import random

eb1 = []
eb2 = []
eb3 = []
eb4 = []
eb5 = []
eb6 = []
names = []
counter = 0
directory = ['/Users/emmabarash/Lab/data/eb4_data/stage_3','/Users/emmabarash/Lab/data/eb5_data/stage_3',\
             '/Users/emmabarash/Lab/data/eb6_data/stage_3', '/Users/emmabarash/Lab/data/eb7_data/stage_3',\
                 '/Users/emmabarash/Lab/data/eb8_data/stage_3', '/Users/emmabarash/Lab/data/eb9_data/stage_3']

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
        if i[-16:-13] == "eb7":
            eb4.append(filename)
        eb4.sort()
        if i[-16:-13] == "eb8":
            eb5.append(filename)
        eb5.sort()
        if i[-16:-13] == "eb9":
            eb6.append(filename)
        eb6.sort()
    
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
        converted.append(files[i].replace({"None": False, "suc": True, "nacl": True}))
    files = converted

convert(eb1_files)
convert(eb2_files)
convert(eb3_files)
convert(eb4_files)
convert(eb5_files)
convert(eb6_files)

# totals for the trigger and rewarder side activation
eb1_trig_counts = []
eb1_rew_counts = []

eb2_trig_counts = []
eb2_rew_counts = []

eb3_trig_counts = []
eb3_rew_counts = []

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

setup_for_trial_counts(eb1_files, eb1_trig_counts, eb1_rew_counts)
setup_for_trial_counts(eb2_files, eb2_trig_counts, eb2_rew_counts)
setup_for_trial_counts(eb3_files, eb3_trig_counts, eb3_rew_counts)
setup_for_trial_counts(eb4_files, eb4_trig_counts, eb4_rew_counts)
setup_for_trial_counts(eb5_files, eb5_trig_counts, eb5_rew_counts)
setup_for_trial_counts(eb6_files, eb6_trig_counts, eb6_rew_counts)

eb1_percentage = []
eb2_percentage = []
eb3_percentage = []
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
            percent = round(((val[0]-val[1])/(val[1]+val[0]))*100, 2)
            if percent <= 100 and percent > 0:
                percentage.append(percent)

find_percentage(eb1_trig_counts, eb1_rew_counts, eb1_percentage)
find_percentage(eb2_trig_counts, eb2_rew_counts, eb2_percentage)
find_percentage(eb3_trig_counts, eb3_rew_counts, eb3_percentage)
find_percentage(eb4_trig_counts, eb4_rew_counts, eb4_percentage)
find_percentage(eb5_trig_counts, eb5_rew_counts, eb5_percentage)
find_percentage(eb6_trig_counts, eb6_rew_counts, eb6_percentage)
        
percentage_all = [eb1_percentage, eb2_percentage, eb3_percentage, eb4_percentage, eb5_percentage ,eb6_percentage]

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
plt.ylabel("% Error: Animals' Failure to Complete Alternations")
#plt.legend(["90% threshold for learning", "% successful alternations"], facecolor="white", loc='lower right')
plt.title("All Animals: Percentage of Failed Alternations Per Session")
plt.ylim(0,105)
plt.xlim(0.5,8.5)
plt.errorbar(range(1,9),np.nanmean(pad_matrix,0)[0:8])
plt.scatter(range(1,max_sess+1),pad_matrix[0], c='blue', alpha=0.5, s=15) 
plt.scatter(range(1,max_sess+1),pad_matrix[1], c='purple', alpha=0.5, s=15) 
plt.scatter(range(1,max_sess+1),pad_matrix[2], c='goldenrod', alpha=0.5, s=15) 
plt.scatter(range(1,max_sess+1),pad_matrix[3], c='red', alpha=0.5, s=15) 
plt.scatter(range(1,max_sess+1),pad_matrix[4], c='green', alpha=0.5, s=15) 
plt.scatter(range(1,max_sess+1),pad_matrix[5], c='orange', alpha=0.5, s=15) 
plt.legend(["EB4", "EB5", "EB6", "EB7", "EB8", "EB9"], facecolor="white", loc="upper right")
#plt.axhline(90,c='black',linestyle=':')
plt.show()
#### confidence intervals
#sns.lineplot(range(max_sess),eb1_percentage)

################################ FAILURES

