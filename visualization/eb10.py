#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:03:39 2022

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import random

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

# animal1_percentage = []
# animal2_percentage = []
# animal3_percentage = []
# def find_percentage(trig_counts, rew_counts, percentage):
#     # create a session-by-session comparison
#     total = zip(trig_counts, rew_counts)
#     # make into a mutable list
#     conv_total = list(total)
#     # get a percentage of reward choice for each session
#     for val in conv_total:
#         if val[0] != 0:
#             percent = round((val[1]/val[0])*100, 2)
#             if percent <= 100 and percent > 0:
#                 percentage.append(percent)

# find_percentage(animal1_rew_counts, animal1_percentage)
# find_percentage(animal2_rew_counts, animal2_percentage)
# find_percentage(animal3_rew_counts, animal3_percentage)
        
# percentage_all = [animal1_percentage, animal2_percentage, animal3_percentage]
percentage_all = [animal1_rew_counts]

mean_activations = []
for animal in percentage_all:
    mean_activations.append(np.mean(animal))
    


first_half = []
second_half = []
for percent_list in percentage_all:
    first_half.append(len(percent_list)//2)
    second_half.append((len(percent_list)//2 + 1))

first_mean=[]; second_mean=[]
for perc,idx1,idx2 in zip(percentage_all,first_half,second_half):
    first_mean.append(np.mean(perc[:idx1]))
    second_mean.append(np.mean(perc[idx2:]))
    
first_std = np.std(np.array(first_mean))
first_mean = np.mean(np.array(first_mean))
second_std = np.std(np.array(second_mean))
second_mean = np.mean(np.array(second_mean))

# Create lists for the plot
halves = ['first half', 'second half']
x_pos = np.arange(len(halves))
CTEs = [first_mean, second_mean]
error = [first_std, second_std]

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('% of Successful Reward Deliveries')
ax.set_xticks(x_pos)
ax.set_xticklabels(halves)
ax.set_xlabel("Training Sessions for All Animals")
plt.axhline(30,c='red',linestyle=':')
#plt.legend(["70% threshold for learning", "% successful alternations"], facecolor="white", bbox_to_anchor=[1, 0.1], loc='center left')
ax.set_title('Standard Deviation of Means for First and Second Halves of Training')
ax.yaxis.grid(False)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()

##############################################
#pad the data to make exact same column (session) length
max_sess = np.max([len(x) for x in percentage_all])
#pad_matrix = [x for x in percentage_all if len(x)==max_sess\
              #else np.ones_like(max_sess-len(x))]
    
plt.figure(figsize=(12,7))
pad_matrix = np.asarray([x if len(x)==max_sess else\
                         np.concatenate([x,np.nan*np.ones(max_sess-len(x))]) for x in percentage_all])
    
#error = [eb1_percentage, eb2_percentage, eb3_percentage]
#np.concatenate(x,np.ones_like(max_sess-len(x)))
x_vals = [[1],[1],[1]]*np.arange(0,len(np.array(percentage_all).T))
plt.errorbar(range(max_sess),np.nanmean(pad_matrix,0), yerr=np.nanstd(pad_matrix,0))
#plt.scatter(x_vals,np.array(percentage_all), color="red")
# plt.axhline(70,c='black',linestyle=':')
plt.xlabel("Number of Sessions")
plt.ylabel("Number of Successful Deliveries")
# plt.legend(["30 deliveries threshold", "successful deliveries"], facecolor="white", loc="upper right")
plt.title("All Animals: Reward Pokes per Session")

#creating a new plotmax_sess = np.max([len(x) for x in percentage_all])
#pad_matrix = [x for x in percentage_all if len(x)==max_sess\
              #else np.ones_like(max_sess-len(x))]
    
pad_matrix = np.asarray([x if len(x)==max_sess else\
                         np.concatenate([x,np.nan*np.ones(max_sess-len(x))]) for x in percentage_all])
    
#error = [eb1_percentage, eb2_percentage, eb3_percentage]
#np.concatenate(x,np.ones_like(max_sess-len(x)))
x_vals = [[1],[1],[1]]*np.arange(0,len(np.array(percentage_all).T))
plt.figure(figsize=(12,7))
plt.plot(range(max_sess), np.nanmean(pad_matrix,0), color="red")
plt.plot(range(max_sess),np.array(animal1_rew_counts), color="darkgreen", linestyle="--")
# plt.plot(range(max_sess),np.array(animal2_rew_counts), color="purple", linestyle=":")
# plt.plot(range(max_sess),np.array(animal3_rew_counts), color="blue", linestyle="-.")
plt.title("All Animals: Reward Pokes per Session")
#plt.bar(7, 80, width=0.05, color="black")
plt.axhline(30,c='black',linestyle=':')
plt.xlabel("Number of Sessions")
plt.ylabel("Number of Successful Deliveries")
# plt.legend(['average','EB7', 'EB8', 'EB9', "30 deliveries", "added visual cue"], facecolor="white", loc="upper left")

