#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 00:48:01 2022

@author: emmabarash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats
#pyimport seaborn as sns
import random
import inflect


animal1 = []
names = []
counter = 0
directory = ['/Users/emmabarash/Lab/data/eb11_data']

for i in directory:
    print(i[-9:-5])
    for filename in os.listdir(i):
        if i[-9:-5] == "eb11":
            animal1.append(filename)
        animal1.sort()
    
def join_files(list, files):
    for i in directory:
        for name in list:
            f = os.path.join(i, name)
            if os.path.isfile(f):
                files.append(pd.read_csv(f))
                
animal1_files = []
join_files(animal1, animal1_files)


# convert data frame values from strings to bool
def convert(files):
    converted = []       
    for i in range(len(files)):
        converted.append(files[i].replace({"None": False, "suc": True, "nacl": True}))
    files = converted

convert(animal1_files)


# totals for the trigger and rewarder side activation
animal1_line1_counts = []
animal1_line2_counts = []
# animal1_line3_counts = []
#  animal1_line4_counts = []


def count_in_trial(list):
    counter = 0
    prev_i = False
    for i in list:
        if i != prev_i and i != False:
            counter = counter + 1
        prev_i = i
    return counter

counter = 0
trigger = []

def find_missed_trials(files, trigger):
    for f in files:
        trigger.append(count_in_trial(f["Poke1"]))
        
find_missed_trials(animal1_files, trigger)


def setup_for_trial_counts(files, line1, line2):
    for f in files:
        # line4.append(count_in_trial(f["Line4"]))
        # line3.append(count_in_trial(f["Line3"]))
        line2.append(count_in_trial(f["Line2"]))
        line1.append(count_in_trial(f["Line1"]))

setup_for_trial_counts(animal1_files, animal1_line1_counts,
                       animal1_line2_counts)
p = inflect.engine()

x = range(1,len(animal1_files)+1)
plt.ylim(0,80)
plt.title(filename[0:4] + ": deliveries across " + p.number_to_words(len(animal1_files)) + " taste session")
plt.xlabel("sessions")
plt.ylabel("deliveries")
plt.scatter(x, animal1_line1_counts, c='blue')
plt.plot(x, animal1_line1_counts, c='blue')
plt.scatter(x,animal1_line2_counts , c='red')
plt.plot(x, animal1_line2_counts, c='red')
# plt.scatter(x, animal1_line3_counts, c='green')
# plt.plot(x, animal1_line3_counts, c='green')
# plt.scatter(x, animal1_line4_counts)
# plt.plot(x, animal1_line4_counts)
plt.legend(["0.3M suc", '_nolegend_', "1mM QHCl", '_nolegend_' ])
plt.show()