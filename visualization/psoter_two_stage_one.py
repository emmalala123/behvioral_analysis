#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 19:50:50 2022

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
from scipy import stats
from scipy.stats import sem

animal1 = []
animal2 = []
animal3 = []
names = []
counter = 0
directory = ['/Users/emmabarash/Lab/data/eb10_data']

for i in directory:
    for filename in os.listdir(i):
        if i[-9:-5] == "eb10":
            animal1.append(filename)
        animal1.sort()
        # if i[-8:-5] == "eb8":
        #     animal2.append(filename)
        # animal2.sort()
        # if i[-8:-5] == "eb9":
        #     animal3.append(filename)
        # animal3.sort()
    
def join_files(list, files):
    for i in directory:
        for name in list:
            f = os.path.join(i, name)
            if os.path.isfile(f):
                files.append(pd.read_csv(f))
                
animal1_files = []
join_files(animal1, animal1_files)
# animal2_files = []
# join_files(animal2, animal2_files)
# animal3_files = []
# join_files(animal3, animal3_files)

# convert data frame values from strings to bool
def convert(files):
    converted = []       
    for i in range(len(files)):
        converted.append(files[i].replace({"None": False, "suc": True, "nacl": True}))
    files = converted

convert(animal1_files)
# convert(animal2_files)
# convert(animal3_files)

# totals for the trigger and rewarder side activation
# animal1_trig_counts = []
animal1_rew_counts = []

# animal2_trig_counts = []
# animal2_rew_counts = []

# animal3_trig_counts = []
# animal3_rew_counts = []

def count_in_trial(list):
    counter = 0
    prev_i = False
    for i in list:
        if i != prev_i and i != False:
            counter = counter + 1
        prev_i = i
    return counter


def setup_for_trial_counts(files, rew_counts):
    for f in files:
        if not f.empty:
            rew_counts.append(count_in_trial(f["Line1"]))

setup_for_trial_counts(animal1_files, animal1_rew_counts)
# setup_for_trial_counts(animal2_files, animal2_rew_counts)
# setup_for_trial_counts(animal3_files, animal3_rew_counts)

# error = [sem(animal1_rew_counts), sem(animal2_rew_counts), sem(animal3_rew_counts)]
error = [sem(animal1_rew_counts)]
#plt.figure(figsize=(7,12))
plt.bar("EB10", animal1_rew_counts, width=0.5, align='center', color='purple', yerr=error[0])
# plt.bar("EB8", animal2_rew_counts, width=0.5, align='center', color='purple', yerr=error[1])
# plt.bar("EB9", animal3_rew_counts, width=0.5, align='center', color='purple', yerr=error[2])
plt.legend(["EB10"], loc="upper left")
plt.title("Stage One: Rewarder Activations")
plt.xlabel("Animals")
plt.ylabel("Total Rewarder Activations")


