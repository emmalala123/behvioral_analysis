#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:43:31 2022

@author: emmabarash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
from scipy import stats

animal_rep = []
names = []
counter = 0
directory = ['/Users/emmabarash/Lab/data/eb2_data/','/Users/emmabarash/Lab/data/eb2_data/stage_3']
animal_rep_files = []

for i in directory:
    for filename in os.listdir(i):
        if i[-9:-6] == "eb2" and not filename == "stage_3":
            animal_rep.append(filename)
        if i[-16:-13] == "eb2":
            animal_rep.append(filename)
        animal_rep.sort()
    
def join_files(list, files):
    for i in directory:
        for name in list:
            f = os.path.join(i, name)
            if os.path.isfile(f):
                files.append(pd.read_csv(f))
                
join_files(animal_rep, animal_rep_files)

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

rew_counts = []
trig_counts = []
def setup_for_trial_counts(files, trig_counts, rew_counts):
    for f in files:
        trig_counts.append(count_in_trial(f["Line2"]))
        rew_counts.append(count_in_trial(f["Line1"]))

setup_for_trial_counts(animal_rep_files, trig_counts, rew_counts)
#rew_counts = rew_counts[:28] + rew_counts[42:]
trig_counts1 = trig_counts[:28]
trig_counts2 = trig_counts[43:]

#x_vals = np.arange(-9,12)
x_vals = np.arange(-27,20)
plt.figure(figsize=(12,7))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(x_vals[:28], trig_counts1, c='cornflowerblue')
plt.plot(range(16,16+len(trig_counts2)), trig_counts2, c='cornflowerblue')
plt.plot(x_vals, rew_counts, c='darkorange')
plt.axvspan(xmin=-27,xmax=0, alpha=0.4, color="#FDDAB0")
plt.axvspan(xmin=0,xmax=19, alpha=0.4, color="#B0CBF1")
plt.legend(['R-side', '_nolegend_', 'L-Side', 'Both sides -> Reward', 'R-> L -> Reward'], loc="upper left")
plt.title('Additive training: representative animal')
plt.xticks(np.arange(-9,12))
plt.xlabel('sessions')
plt.ylabel('deliveries')

poke_counts = []
time = []

def count_in_trial(list):
    counter = 0
    prev_i = False
    for i in list:
        if i != prev_i and i != False:
            counter = counter + 1
        prev_i = i
    return counter

def setup_for_trial_counts(files, poke_counts):
    for f in files:
        poke_counts.append(count_in_trial(f["Poke1"]))
        
        
setup_for_trial_counts(animal_rep_files[29:43], poke_counts)
