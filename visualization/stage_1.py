#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:46:51 2021

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns

files = []
names = []
counter = 0
directory = '/Users/emmabarash/Lab/data/eb10_data'


for filename in os.listdir(directory):
    names.append(filename)
names.sort()

for name in names:
    f = os.path.join(directory, name)
    if os.path.isfile(f):
        files.append(pd.read_csv(f))
        
converted = []       
#if directory[-7:] == 'stage_3':
for i in range(len(files)):
    converted.append(files[i].replace({"None": False, "suc": True, "nacl": True}))
files = converted

sum_line1 = []
sum_line2 = []
def count_in_trial(list):
    counter = 0
    prev_i = False
    for i in list:
        if i != prev_i and i != False:
            counter = counter + 1
        prev_i = i
    return counter

for f in files:
    #if not f is files[24] and not f is files[3] and not f is files[14] and not f is files[17]:
    sum_line2.append(count_in_trial(f["Line2"]))
    sum_line1.append(count_in_trial(f["Line1"]))

counter = 0
for f in files:
    counter = counter + 1
    plt.figure(figsize=(13, 5))
    if directory[-7:] =="stage_3":
        plt.title(directory[-16:-13] + " Events vs Sessions " + str(counter))
    else:
        plt.title(directory[-8:-5] + " Events vs Sessions " + str(counter))
   # plt.title(directory[-8:-5] + " Events vs Time, Session: " + str(counter))
    plt.xlabel("time (sec)")
    plt.ylabel("Pokes")
        
    plt.scatter(f.Time, f.Line1)
    plt.scatter(f.Time, f.Line2*2)
    plt.plot(f.Time, f.Line1)
    plt.plot(f.Time, f.Line2*2)
    plt.legend(['rewarder','trigger'], loc="lower right", facecolor='white')

# weirdness with the last index, doesn't print if the whole program is run
counter = 0
for f in files:
    #if not f is files[0]:
    counter = counter + 1
    sum_df = f.sum(axis=0).reset_index() 
    sum_df.rename(columns={0:'total'},inplace=True)
    sum_df.insert(loc=0,column='session',value=0)
        
        
    fig,ax = plt.subplots(figsize=(8,5))
    g = sns.barplot(y='total',x='index', palette=['purple', (0.2, 0.4, 0.6, 0.6)],\
                data=sum_df.loc[sum_df['index'].isin(['Line1','Line2'])],ax=ax)
        
    if directory[-7:] =="stage_3":
        g.set_title(directory[-16:-13] + " Events vs Sessions")
    else:
        g.set_title(directory[-8:-5] + " Events vs Sessions")
   
    g.set_ylabel('deliveries')
    g.set_xlabel('one session')
    g.set_xticklabels(['rewarder side','trigger side'])
    
    g.legend(['rewarder', 'trigger'])
    
sum_line1 = []
sum_line2 = []
#for f in files:
        #if not f is files[2]:
    # sum_line1.append(f['Line1'].sum(axis=0))
    # sum_line2.append(f['Line2'].sum(axis=0))
 
plt.clf()
sum_lines = set(zip(sum_line1, sum_line2))
all_lines = pd.DataFrame(np.asarray([sum_line1,sum_line2]).T)
all_lines.insert(loc=0,column='session',value=range(len(all_lines)))

v2 = pd.DataFrame(np.asarray([sum_line1,sum_line2]).flatten())
v2.insert(loc=0,column='session',value=np.tile(range(len(sum_line1)),2))
v2.insert(loc=1,column='lines',value=np.repeat(range(1,3),len(sum_line1)))


g = sns.lineplot(x='session',y=0,hue='lines',data=v2, palette=(['orange', 'purple']))
if directory[-7:] =="stage_3":
    g.set_title(directory[-16:-13] + " Events vs Sessions")
else:
    g.set_title(directory[-8:-5] + " Events vs Sessions")
g.legend(['rewarder', 'trigger'])
g.set(ylabel='events')
    
#g.set(xticks=[0, 1, 2, 3, 4])
