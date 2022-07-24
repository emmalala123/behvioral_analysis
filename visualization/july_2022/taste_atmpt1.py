#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 14:29:38 2022

@author: emmabarash

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats
import seaborn as sns
import random
import inflect


animal1 = []
names = []
counter = 0
directory = ['/Users/emmabarash/lab/data/eb10_data/taste']

for i in directory:
    print(i[-14:-10])
    for filename in os.listdir(i):
        if i[-15:-11] == "eb10":
        # if i[-14:-10] == "eb10":
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
animal1_line3_counts = []
animal1_line4_counts = []
animal1_trigger = []


def count_in_trial(list):
    counter = 0
    prev_i = False
    for i in list:
        if i != prev_i and i != False:
            counter = counter + 1
        prev_i = i
    return counter

def setup_for_trial_counts(files, line1, line2, line3, line4, trigger):
    for f in files:
        line4.append(count_in_trial(f["Line4"]))
        line3.append(count_in_trial(f["Line3"]))
        line2.append(count_in_trial(f["Line2"]))
        line1.append(count_in_trial(f["Line1"]))
        trigger.append(count_in_trial(f["Poke1"]))

setup_for_trial_counts(animal1_files, animal1_line1_counts,
                       animal1_line2_counts, animal1_line3_counts, animal1_line4_counts, animal1_trigger)
p = inflect.engine()

plt.ylim(0,35)
x = range(1,len(animal1_files)+1)
plt.title("EB10: deliveries across first " + p.number_to_words(len(animal1_files)) + " taste sessions")
plt.xlabel("sessions")
plt.ylabel("deliveries")
plt.scatter(x, animal1_line1_counts)
plt.plot(x, animal1_line1_counts)
plt.scatter(x, animal1_line2_counts)
plt.plot(x, animal1_line2_counts)
plt.scatter(x, animal1_line3_counts)
plt.plot(x, animal1_line3_counts)
plt.scatter(x, animal1_line4_counts)
plt.plot(x, animal1_line4_counts)
plt.legend(["0.3M suc", '_nolegend_', "0.1M NaCl", '_nolegend_', "1M NaCl", 
            '_nolegend_', "0.3mM QHCl", '_nolegend_'])
plt.show()