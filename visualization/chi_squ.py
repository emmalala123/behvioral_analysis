#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:20:06 2021

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import random
from scipy.stats import chi2_contingency

eb1 = []
eb2 = []
eb3 = []
names = []
counter = 0
directory = ['/Users/emmabarash/Lab/data/eb1_data','/Users/emmabarash/Lab/data/eb2_data','/Users/emmabarash/Lab/data/eb3_data']

for i in directory:
    for filename in os.listdir(i):
        if i[-8:-5] == "eb1":
            eb1.append(filename)
        eb1.sort()
        if i[-8:-5] == "eb2":
            eb2.append(filename)
        eb2.sort()
        if i[-8:-5] == "eb3":
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

trig_all = [eb1_trig_counts, eb2_trig_counts, eb3_trig_counts]
rew_all = [eb1_rew_counts, eb2_rew_counts, eb3_rew_counts]

def find_halves(list):
    first_half = []
    second_half = []
    for i in list:
        first_half.append(len(i)//2)
        second_half.append((len(i)//2 + 1))
    
    return first_half, second_half

trig_first, trig_second = find_halves(trig_all)
rew_first, rew_second = find_halves(rew_all)

def get_total(list_all, first, second):
    f_total = 0
    s_total = 0
    totals_1=[];totals_2=[]
    for item,idx1,idx2 in zip(list_all, first, second):
        print("total", f_total, "\n", item[:idx1])
        f_total = f_total + sum(item[:idx1])
        s_total = s_total + sum(item[idx2:])
        totals_1.append(f_total)
        totals_2.append(s_total)
    return f_total, s_total, totals_1, totals_2

first_trig_total, second_trig_total, trigs_1, trigs_2 = get_total(trig_all, trig_first, trig_second)
first_rew_total, second_rew_total, rews_1, rews_2 = get_total(rew_all, rew_first, rew_second)

incorrect_first = first_trig_total - first_rew_total
incorrect_second = second_trig_total - second_rew_total

chi_table = np.asarray([[first_rew_total, second_rew_total],\
                         [incorrect_first, incorrect_second]])
    
chi2 = chi2_contingency(chi_table)

incs_1 = [x-y for x,y in zip(trigs_1,rews_1)]
incs_2 = [x-y for x,y in zip(trigs_2,rews_2)]

success_df = pd.DataFrame(np.asarray([incs_1,incs_2,rews_1,rews_2]).flatten(),columns=['vals'])
success_df.insert(loc=0,column='animal',value=np.tile([x[-8:] for x in directory],4))
success_df.insert(loc=1,column='session',value=np.tile(np.repeat(['first','second'],3),2))
success_df.insert(loc=2,column='success',value=np.repeat(['no','yes'],6))


g = sns.catplot(x="animal", y="vals",
                hue="success", col="session",
                data=success_df, kind="bar",
                height=4, aspect=.7);

g = sns.barplot(x='animal',y='vals',)


x = np.arange(0, 100, 10)

#plot Chi-square distribution with 4 degrees of freedom
plt.plot(x, chi2)
    

