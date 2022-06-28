#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 21:44:47 2021

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import random

files = []
names = []
counter = 0
directory = '/Users/emmabarash/Lab/data/eb3_data'


for filename in os.listdir(directory):
    names.append(filename)
names.sort()

for name in names:
    f = os.path.join(directory, name)
    if os.path.isfile(f):
        files.append(pd.read_csv(f))

# convert data frame values from strings to bool
converted = []       
for i in range(len(files)):
    converted.append(files[i].replace({"None": False, "suc": True, "nacl": True}))
files = converted

# totals for the trigger and rewarder side activation
trig_counts = []
rew_counts = []

def count_in_trial(list):
    counter = 0
    prev_i = False
    for i in list:
        if i != prev_i and i != False:
            counter = counter + 1
        prev_i = i
    return counter

for f in files:
    trig_counts.append(count_in_trial(f["Line2"]))
    rew_counts.append(count_in_trial(f["Line1"]))

# create a session-by-session comparison
total = zip(trig_counts, rew_counts)
# make into a mutable list
conv_total = list(total)
# get a percentage of reward choice for each session
percentage = []
for val in conv_total:
    if val[0] != 0:
        percent = round((val[1]/val[0])*100, 2)
        if percent <= 100 and percent > 0:
            percentage.append(percent)
        
# plot the percentage
plt.rcParams["figure.figsize"] = [8.00, 3.50]
plt.rcParams["figure.autolayout"] = True
y = percentage
nbins = 8

_, _, patches = plt.hist(y, bins=nbins, edgecolor='white')
# have a vertical bar at 70%
plt.axvline(x=70)
colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
for i in range(nbins)]
for patch in patches:
    patch.set_facecolor(colors[np.random.randint(100) % nbins])

plt.title(directory[-8:-5] + " Histogram of Completed Alternations Per Session")
plt.xlabel("Percentage Completed Alternations Per Session")
plt.ylabel("Number of Sessions")
plt.legend(['70% Accuracy Cutoff'], facecolor='white')
plt.show()