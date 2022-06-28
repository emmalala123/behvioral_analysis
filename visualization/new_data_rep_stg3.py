#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 20:54:20 2021

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
directory = ['/Users/emmabarash/Lab/data/eb1_data/stage_3','/Users/emmabarash/Lab/data/eb2_data/stage_3','/Users/emmabarash/Lab/data/eb3_data/stage_3']

for i in directory:
    for filename in os.listdir(i):
        if i[-16:-13] == "eb1":
            
            eb1.append(filename)
        eb1.sort()
        if i[-16:-13] == "eb2":
            eb2.append(filename)
        eb2.sort()
        if i[-16:-13] == "eb3":
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

#pad the data to make exact same column (session) length
max_sess = np.max([len(x) for x in percentage_all])
#pad_matrix = [x for x in percentage_all if len(x)==max_sess\
              #else np.ones_like(max_sess-len(x))]
    
pad_matrix = np.asarray([x if len(x)==max_sess else\
                         np.concatenate([x,np.nan*np.ones(max_sess-len(x))]) for x in percentage_all])
    
#error = [eb1_percentage, eb2_percentage, eb3_percentage]
#np.concatenate(x,np.ones_like(max_sess-len(x)))
plt.errorbar(range(max_sess),np.nanmean(pad_matrix,0), yerr=np.nanstd(pad_matrix,0))
plt.axhline(70,c='black',linestyle=':')
plt.xlabel("Number of sessions")
plt.ylabel("Percentage Successful Alternations")
plt.legend(["70% threshold for learning", "% successful alternations"], facecolor="white", loc="lower right")
plt.title("All Animals (n=3) Percentage of Completed Alternations Per Session")
