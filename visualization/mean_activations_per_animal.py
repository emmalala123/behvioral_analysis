"""
Created on Thu Sep 16 12:33:56 2021

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import scipy.stats as st
import random

eb1 = []
eb2 = []
eb3 = []
eb4 = []
eb5 = []
eb6 = []
eb7 = []
eb8 = []
eb9 = []
names = []
counter = 0
directory = ['/Users/emmabarash/Lab/data/eb1_data','/Users/emmabarash/Lab/data/eb2_data',\
             '/Users/emmabarash/Lab/data/eb3_data', '/Users/emmabarash/Lab/data/eb4_data',\
                 '/Users/emmabarash/Lab/data/eb5_data', '/Users/emmabarash/Lab/data/eb6_data',\
                     '/Users/emmabarash/Lab/data/eb7_data', '/Users/emmabarash/Lab/data/eb8_data',\
                         '/Users/emmabarash/Lab/data/eb9_data']

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
        if i[-8:-5] == "eb4":
            eb4.append(filename)
        eb4.sort()
        if i[-8:-5] == "eb5":
            eb5.append(filename)
        eb5.sort()
        if i[-8:-5] == "eb6":
            eb6.append(filename)
        eb6.sort()
        if i[-8:-5] == "eb7":
            eb7.append(filename)
        eb7.sort()
        if i[-8:-5] == "eb8":
            eb8.append(filename)
        eb8.sort()
        if i[-8:-5] == "eb9":
            eb9.append(filename)
        eb9.sort()
    
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
eb7_files = []
join_files(eb7, eb7_files)
eb8_files = []
join_files(eb8, eb8_files)
eb9_files = []
join_files(eb9, eb9_files)

#df stuff
eb1_files = [df.replace({"None": False, "suc": True, "nacl": True}) for df in eb1_files]
Df1 = pd.concat(eb1_files)
Df1['ID'] = "EB1"
Df1['group'] = "step 2"

eb2_files = [df.replace({"None": False, "suc": True, "nacl": True}) for df in eb2_files]
Df2 = pd.concat(eb2_files)
Df2['ID'] = "EB2"
Df2['group'] = "step 2"

eb3_files = [df.replace({"None": False, "suc": True, "nacl": True}) for df in eb3_files]
Df3 = pd.concat(eb3_files)
Df3['ID'] = "EB3"
Df3['group'] = "step 2"

Df4 = pd.concat(eb4_files)
Df4['ID'] = "EB4"
Df4['group'] = "step 1"

Df5 = pd.concat(eb5_files)
Df5['ID'] = "EB5"
Df5['group'] = "step 1"

Df6 = pd.concat(eb6_files)
Df6['ID'] = "EB6"
Df6['group'] = "step 1"

frames_prep1 = [Df1,Df2,Df3]
frames_prep2 = [Df4,Df5,Df6]
frames1 = pd.concat(frames_prep1)
frames2 = pd.concat(frames_prep2)

mean_activations_sns1 = frames1.groupby('ID').mean()
mean_activations_sns2 = frames2.groupby('ID').mean()

# convert data frame values from strings to bool
def convert(files):
    converted = []       
    for i in range(len(files)):
        converted.append(files[i].replace({ }))
    files = converted

convert(eb1_files)
convert(eb2_files)
convert(eb3_files)
convert(eb4_files)
convert(eb5_files)
convert(eb6_files)
convert(eb7_files)
convert(eb8_files)
convert(eb9_files)

# totals for the trigger and rewarder side activation
eb1_rew_counts = []
eb1_trig_counts = []

eb2_rew_counts = []
eb2_trig_counts = []

eb3_rew_counts = []
eb3_trig_counts = []

eb4_rew_counts = []

eb5_rew_counts = []

eb6_rew_counts = []

eb7_rew_counts = []

eb8_rew_counts = []

eb9_rew_counts = []

all_counts = []

def count_in_trial(list):
    counter = 0
    prev_i = False
    for i in list:
        if i != prev_i and i != False:
            counter = counter + 1
        prev_i = i
    all_counts.append(counter)
    return counter

def setup_for_all_counts(files, trig_counts, rew_counts):
    for f in files:
        trig_counts.append(count_in_trial(f["Line2"]))
        rew_counts.append(count_in_trial(f["Line1"]))

def setup_for_trial_counts(files, rew_counts):
    for f in files:
        rew_counts.append(count_in_trial(f["Line1"]))

setup_for_all_counts(eb1_files, eb1_trig_counts, eb1_rew_counts)
setup_for_all_counts(eb2_files, eb2_trig_counts, eb2_rew_counts)
setup_for_all_counts(eb3_files, eb3_trig_counts, eb3_rew_counts)

setup_for_trial_counts(eb4_files, eb4_rew_counts)
setup_for_trial_counts(eb5_files, eb5_rew_counts)
setup_for_trial_counts(eb6_files, eb6_rew_counts)
setup_for_trial_counts(eb7_files, eb7_rew_counts)
setup_for_trial_counts(eb8_files, eb8_rew_counts)
setup_for_trial_counts(eb9_files, eb9_rew_counts)
        
rew_all = [[eb1_rew_counts[i] + eb1_trig_counts[i] for i in np.arange(np.size(eb1_trig_counts))],\
           [eb2_rew_counts[i] + eb2_trig_counts[i] for i in np.arange(np.size(eb2_trig_counts))],\
               [eb3_rew_counts[i] + eb3_trig_counts[i] for i in np.arange(np.size(eb3_trig_counts))],\
                   eb4_rew_counts, eb5_rew_counts , eb6_rew_counts, eb7_rew_counts, eb8_rew_counts, eb9_rew_counts]
    

#####DAN WUZ HERE#####
eb1 = {'trig_counts':eb1_trig_counts, 'rew_counts':eb1_rew_counts}
eb1 = pd.DataFrame(eb1)
eb1['ID'] = 'eb1'
eb1['step_grp'] = '2 step'
eb2 = {'trig_counts':eb2_trig_counts, 'rew_counts':eb2_rew_counts}
eb2 = pd.DataFrame(eb2)
eb2['ID'] = 'eb2'
eb2['step_grp'] = '2 step'
eb3 = {'trig_counts':eb3_trig_counts, 'rew_counts':eb3_rew_counts}
eb3 = pd.DataFrame(eb3)
eb3['ID'] = 'eb3'
eb3['step_grp'] = '2 step'

eb4 = {'trig_counts':0, 'rew_counts':eb4_rew_counts}
eb4 = pd.DataFrame(eb4)
eb4['ID'] = 'eb4'
eb4['step_grp'] = '1 step'
eb5 = {'trig_counts':0, 'rew_counts':eb5_rew_counts}
eb5 = pd.DataFrame(eb5)
eb5['ID'] = 'eb5'
eb5['step_grp'] = '1 step'
eb6 = {'trig_counts':0, 'rew_counts':eb6_rew_counts}
eb6 = pd.DataFrame(eb6)
eb6['ID'] = 'eb6'
eb6['step_grp'] = '1 step'
eb7 = {'trig_counts':0, 'rew_counts':eb7_rew_counts}
eb7 = pd.DataFrame(eb7)
eb7['ID'] = 'eb7'
eb7['step_grp'] = '1 step'
eb8 = {'trig_counts':0, 'rew_counts':eb8_rew_counts}
eb8 = pd.DataFrame(eb8)
eb8['ID'] = 'eb8'
eb8['step_grp'] = '1 step'
eb9 = {'trig_counts':0, 'rew_counts':eb9_rew_counts}
eb9 = pd.DataFrame(eb9)
eb9['ID'] = 'eb9'
eb9['step_grp'] = '1 step'

ebdat = pd.concat([eb1,eb2,eb3,eb4,eb5,eb6, eb7, eb8, eb9])
cols = ['trig_counts','rew_counts']
ebdat['activations'] = ebdat[cols].sum(axis=1)
ebdat = ebdat.groupby('ID').mean()
ebdat['step_grp'] = ['2 step', '2 step', '2 step', '1 step', '1 step', '1 step', '1 step', '1 step', '1 step']
ebdat['ID'] = ['eb1', 'eb2', 'eb3', 'eb4', 'eb5', 'eb6', 'eb7', 'eb8', 'eb9']

plt.figure(figsize=(12,7))
ax = sns.barplot(x="step_grp", y='activations', data=ebdat)
ax = sns.swarmplot(x="step_grp", y="activations", hue='ID', dodge=True, marker='D', size =9, alpha=0.9, data=ebdat)

plt.ylim(0,1600)
ax = sns.boxplot(x='step_grp', y='activations', data=ebdat)
####END OF DAN'S MEDDLING
    
# get the mean of every session.

mean_activations1 = np.mean(np.mean(rew_all[0:3]))
mean_activations2 = np.mean(np.mean(rew_all[3:6]))

# not sure how to represent this in the plot
err = []
for animal in rew_all:
    err.append(np.mean(animal))

# data points for each animal's contribution to performance
# 95% CI of each stage mean from all data- height analogy-- 
# take each of the 20 data points (20 sessions) and plug them in
rew_all = np.array(rew_all)
rew_all1 = np.array(rew_all[0:3])
rew_all2 = np.array(rew_all[3:6])
mean_all = []
for i in rew_all.T:
    mean_all.append(np.mean(i))

low1, high1 = st.t.interval(alpha=0.95, df=len(mean_all)-1, loc=np.mean(mean_all), scale=st.sem(mean_all)) 
low2, high2 = st.t.interval(alpha=0.95, df=len(mean_all)-1, loc=np.mean(mean_all), scale=st.sem(mean_all)) 

yerr1 = high1 - low1
yerr2 = high2 - low2

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar('1 step', mean_activations2, color='goldenrod', yerr = yerr1)
#plt.scatter(mean_all[0:3], range(3))
ax.bar('2 step', mean_activations1, color='purple', yerr = yerr2)
#plt.scatter(mean_all[3:6], range(3))
plt.ylabel("mean rewards acquired per session")
plt.legend(['N=3', 'N=3'])
#
sns.swarmplot(data=mean_activations_sns2['Time'])
sns.swarmplot(data=mean_activations_sns1['Time'])
            
plt.boxplot(mean_activations2)
# plot a line graph for mean of sessions for each animal
#plt.plot(np.mean(rew_all[0,:]), range(len(rew_all[0])))
# 95% ci
# mean +- 1.96*(std/sqrt(n=num of sessions))
# bootstrap 95% ci
    
