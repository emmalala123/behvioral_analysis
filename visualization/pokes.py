#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:41:38 2022

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


####dan's garbage



######






#rew_counts = rew_counts[:28] + rew_counts[42:]
trig_counts1 = trig_counts[:28]
trig_counts2 = trig_counts[42:]

plt.plot(range(len(trig_counts1)), trig_counts1, c='cornflowerblue')
plt.plot(range(42,42+len(trig_counts2)), trig_counts2, c='cornflowerblue')
plt.plot(range(len(rew_counts)), rew_counts, c='darkorange')
plt.axvspan(xmin=0,xmax=28, alpha=0.4, color="#FDDAB0")
plt.axvspan(xmin=28,xmax=46, alpha=0.4, color="#B0CBF1")
plt.legend(['R-side -> Reward', '_nolegend_', 'L-side -> Reward', '1 step to reward', '2 steps to reward'], loc="upper left")
plt.title(directory[0][-9:-6]+ ': Representative Animal')
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

pokes_df = pd.concat(animal_rep_files[29:43])
prev_i = False
change_pokes = []
for idx,i in enumerate(pokes_df['Poke1']):
   if i != prev_i:
       change_pokes.append(idx)
   prev_i = i

pokes_in = change_pokes[::2]
pokes_out = change_pokes[1::2]
delta = pokes_out - pokes_in

diff = [a_i - b_i for a_i, b_i in zip(pokes_out, pokes_in)]


