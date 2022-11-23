#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:36:57 2022

@author: emmabarash
"""

# loop through each csv: make a df for poke1
# append each df made in loop to a main df
    # pass poke1 column through edge function
        # will return a column of entry edges and exit edges
        # for each column make a df with timestamps and AnID and Date
        # remove every row that does not have an edge (0 or False)
        # rename time column to poke1_time_entry or exit
        # merge the two sideways into one df
        # adjust so that the linger time is exit - entry
        # ^ repeat for poke2
            # will have a df for poke1 and 2, column with linger time
    # add columns that have a timestamp for every edge and AnID and Date
    # Time, Poke_entry/exit, AnID, Day 
    # exclude all rows that don't have edges
    # find the difference between timestamps - poke1 entry to the next poke1 exit 
    
    # sort the longest linger time for each poke and eliminate (note which days have large linger times)
    # groupby AnID and Date, then sum the linger time column, compare total linger time with session length
    
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

directory = '/Users/emmabarash/Lab/data'

filelist = glob.glob(os.path.join(directory,'*','*.csv'))

full_df = pd.DataFrame(columns = ['Time', 'poke1_entry', 'poke1_exit', 'Poke2', 'Line1', 'Line2', 'Line3', 'Line4', 'Cue1',
        'Cue2', 'Cue3', 'Cue4', 'TasteID', 'AnID', 'Date'])

# make a df for Poke1
Poke1_Entry = pd.DataFrame(columns = ['Time','poke1_timestamp', 'poke1_entry', 'TasteID', 'AnID', 'Day'])
Poke1_Exit = pd.DataFrame(columns = ['Time','poke1_timestamp', 'poke1_exit', 'TasteID', 'AnID', 'Day'])
filelist.sort()

def create_df(file):
        
    df = pd.read_csv(file)
    
    df['TasteID'] = None
    df['AnID'] = filelist[f][-22:-18]
    df['Date'] = filelist[f][-17:-11]
        
    return df

# loop through each csv
for f in range(len(filelist)):
# append each df made in loop to a main df
    full_df = full_df.append(create_df(filelist[f]))

# def create_edge_frame(copy):
#     copy['isdelivery'] = copy.Line1+copy.Line2
#     copy['del_open'], copy['del_close'] = process_edges(copy.isdelivery)
#     copy['Poke1_entry'],copy['Poke1_exit'] = process_edges(copy.Poke1)
#     copy['Poke2_entry'],copy['Poke2_exit'] = process_edges(copy.Poke2)
#     copy['Event_Edges'] = copy.del_open + copy.del_close+ copy.Poke1_entry+ copy.Poke1_exit+ copy.Poke2_entry+ copy.Poke2_exit
#     new_edge_df = copy.loc[copy['Event_Edges'] == 1]
#     return new_edge_df

def process_edges(col):
    fwd_shift = col.shift(1, fill_value=0)
    bwd_shift = col.shift(-1, fill_value=0)
    
    start_edges = col-fwd_shift
    start_edges = (start_edges==1)

    end_edges = col-bwd_shift
    end_edges = (end_edges==1)
    
    return start_edges, end_edges

def add_days_elapsed(finaldf):
   
    new_df = finaldf
        
    res = []
    for name, group in new_df.groupby('AnID'):
        i=1
        for n, g in group.groupby('Date'):
            print(g)
            bit = np.zeros(len(g))
            bit = bit + i
            res.extend(bit)
            i += 1
    new_df['Sessions'] = res

    return new_df

# for each column make a df with timestamps and AnID and Date
Poke1_Entry = add_days_elapsed(full_df)
Poke1_Exit = add_days_elapsed(full_df)
Poke1_Entry['poke1_entry'], Poke1_Exit['poke1_exit'] = process_edges(full_df.Poke2)
Poke1_Entry['Time'] = full_df.Time
Poke1_Exit['Time'] = full_df.Time

# put in timestamps for each true value in entry and exit
Poke1_Entry['poke1_entry'].loc[Poke1_Entry['poke1_entry'] == True] = Poke1_Entry.loc[Poke1_Entry['poke1_entry'] == True].Time
Poke1_Exit['poke1_exit'].loc[Poke1_Exit['poke1_exit'] == True] = Poke1_Exit.loc[Poke1_Exit['poke1_exit'] == True].Time


# remove every row that does not have an edge (0 or False)
Poke1_Entry = Poke1_Entry.loc[Poke1_Entry['poke1_entry'] != False]
Poke1_Exit = Poke1_Exit.loc[Poke1_Exit['poke1_exit'] != False]

# change type to float to derive durations
Poke1_Entry['poke1_entry'] = Poke1_Entry.poke1_entry.astype(float)
Poke1_Exit['poke1_exit'] = Poke1_Exit.poke1_exit.astype(float)

# reset indicies
Poke1_Entry = Poke1_Entry.reset_index()
Poke1_Exit = Poke1_Exit.reset_index()

#cleanup Poke1_Exit
# Poke1_Exit = Poke1_Exit.drop(columns=['Time', 'poke1_entry', 'AnID', 'TasteID', 'Sessions', 'Date'])
# Poke1_Entry = Poke1_Entry.drop(columns='poke1_exit')

# drop time and put the dfs together for Poke1
# Poke1_df = pd.concat([Poke1_Entry, Poke1_Exit], axis = 1)


Poke1_df = Poke1_Entry.drop(columns='Time')
Poke1_df['poke1_exit'] = Poke1_Exit['poke1_exit']

# get durations and save negative values in separate df
# exclude all the negative values (save the rows with negative values in a new df
Poke1_df['durations'] = Poke1_df['poke1_exit'] - Poke1_df['poke1_entry']
neg_vals = Poke1_df.loc[Poke1_df['durations'] < 0] 

# replot the hist without the 5-values
plt.hist(x=Poke1_df.loc[Poke1_df['durations'] > 0.05].durations, bins=100); 
plt.title('durations greater than 0sec in Poke1')

# sum the total poke durations for a session - make a new df for this.
poke1_sessions = Poke1_df
poke1_sessions = poke1_sessions.groupby(['Sessions', 'Date', 'AnID']).sum(sum)

# plot hist with the total amnt of time spent in the poke/session
plt.hist(x=poke1_sessions.loc[poke1_sessions['durations'] > 0].durations, bins=100); 
plt.title('durations greater than 0sec in Poke1 per session')
plt.xlabel('durations (sec) per session')

# exclude all the negative values (save the rows with negative values in a new df
# replot the hist without the 5-values
# see if there is a cutoff for lingering
# sum the total poke durations for a session - make a new df for this. 
# plot hist with the total amnt of time spent in the poke/session
plt.xticks(rotation = 45)





