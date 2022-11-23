#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:31:36 2022

@author: emmabarash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import glob
#import random
#import inflect
from scipy import stats
import re

if os.sep == '/':
    directory = '/Users/emmabarash/Lab/data'
else:
    directory = r'C:\Users\Emma_PC\Documents\data'

filelist = glob.glob(os.path.join(directory,'*','*.csv'))

finaldf = pd.DataFrame(columns = ['Time', 'Poke1', 'Poke2', 'Line1', 'Line2', 'Line3', 'Line4', 'Cue1',
       'Cue2', 'Cue3', 'Cue4', 'TasteID', 'AnID', 'Date', 'Taste_Delivery',
       'Delivery_Time', 'Latencies'])

final_edgedf = pd.DataFrame(columns = ['Time', 'Poke1', 'Poke2', 'Line1', 'Line2', 'Line3', 'Line4', 'Cue1',
       'Cue2', 'Cue3', 'Cue4', 'TasteID', 'AnID', 'Date', 'Taste_Delivery',
       'Delivery_Time', 'Latencies'])
trial_df = pd.DataFrame()
filelist.sort()

def create_edge_frame(copy):
    copy['isdelivery'] = copy.Line1+copy.Line2
    copy['del_open'], copy['del_close'] = process_edges(copy.isdelivery)
    copy['Poke1_entry'],copy['Poke1_exit'] = process_edges(copy.Poke1)
    copy['Poke2_entry'],copy['Poke2_exit'] = process_edges(copy.Poke2)
    copy['Event_Edges'] = copy.del_open + copy.del_close+ copy.Poke1_entry+ copy.Poke1_exit+ copy.Poke2_entry+ copy.Poke2_exit
    new_edge_df = copy.loc[copy['Event_Edges'] == 1]
    return new_edge_df

def process_edges(col):
    fwd_shift = col.shift(1, fill_value=0)
    bwd_shift = col.shift(-1, fill_value=0)
    
    start_edges = col-fwd_shift
    start_edges = (start_edges==1)

    end_edges = col-bwd_shift
    end_edges = (end_edges==1)
    
    return start_edges, end_edges

for f in range(len(filelist)):
    df = pd.read_csv(filelist[f])
    group = df
    col = ['Line1', 'Line2']
    
    def parse_edges(group,col):
        delivery_idx = []
        group['TasteID'] = None
        group['AnID'] = filelist[f][-22:-18]
        group['Date'] = filelist[f][-17:-11]
        for j in col:
            col = j
            if col == 'Line1': 
                taste = 'suc'
            if col == 'Line2':
                taste = 'qhcl'
                
            cols = ['Time']+[col]
            data = group[cols]
            try: edges = data[data[col].diff().fillna(False)]
            except: return None
            edgeON = edges[edges[col]==True]
            edgeON.col = True
            edgeON = edgeON.rename(columns={'Time':'TimeOn'})
            edgeON = edgeON.drop(col,axis=1)
            edgeON.index = np.arange(len(edgeON))
            edgeOFF = edges[edges[col]==False]
            edgeOFF = edgeOFF.rename(columns={'Time':'TimeOff'})
            edgeOFF = edgeOFF.drop(col,axis=1)
            edgeOFF.index = np.arange(len(edgeOFF))
            test = pd.merge(edgeON,edgeOFF,left_index=True,right_index=True)
            test['dt'] = test.TimeOff-test.TimeOn
    
            for i, row in test.iterrows():
                start = int(np.where(df['Time'] == test['TimeOn'][i])[0])
                stop = int(np.where(df['Time'] == test['TimeOff'][i])[0])
        
                group.loc[group.index[range(start,stop)],'TasteID'] = taste
                delivery_idx.append(start)
                
        return group, delivery_idx
    
    new_df, delivery_idx = parse_edges(df, ['Line1', 'Line2'])
    
    def find_poke_dat(copy, poke, delivery_idx):
        # instantiate new columns with null values for later use
        copy['Taste_Delivery'] = False
        copy['Delivery_Time'] = None
        # copy['Is_Delivery'] = None
        
        pokes = ['Time'] + [poke]
        data = copy[poke]
        try: edges = data[data[poke].diff().fillna(False)]
        except: return None
        edgeON = edges[edges[poke]==True].shift(1)
        edgeON.iloc[0] = copy['Time'][0]
        edgeON['Poke2'].iloc[0] = True
        edgeON.col = True
        edgeON = edgeON.rename(columns={'Time':'TimeOn'})
        edgeON = edgeON.drop(poke,axis=1)
        edgeON.index = np.arange(len(edgeON))
        edgeOFF = edges[edges[poke]==False]
        edgeOFF = edgeOFF.rename(columns={'Time':'TimeOff'})
        edgeOFF = edgeOFF.drop(poke,axis=1)
        edgeOFF.index = np.arange(len(edgeOFF))
        test = pd.merge(edgeON,edgeOFF,left_index=True,right_index=True)
        test['dt'] = test.TimeOff-test.TimeOn      
        
        delivery_time = []
        for i in delivery_idx:
            copy.loc[i,'Taste_Delivery'] = True
            copy.loc[i,'Delivery_Time'] = copy['Time'][i]
    
            # collect delivery time to erase Poke2 dat within 10 seconds of delivery
            delivery_time.append(copy['Time'][i])
    
        # generatees a new df with only delivery times (marked 'true')
        deliveries_only = copy.loc[copy['Taste_Delivery'] == True].reset_index(drop=True)
        
        second_copy = copy
        for i in delivery_time:
            second_copy = second_copy.loc[~((second_copy['Time'] > i) & (second_copy['Time'] < i+5)),:]
        
        for i, row in second_copy.iterrows():
            poke1 = np.where(second_copy['Taste_Delivery'] == True)[0]
            poke2 = poke1-1
            
        lat1 = second_copy['Time'].iloc[poke2].reset_index(drop=True)
        lat2 = second_copy['Time'].iloc[poke1].reset_index(drop=True)
        
        latencies = lat2.subtract(lat1) #look up how to subtract series from each other
        
        deliveries_only['Latencies'] = latencies
    
        return deliveries_only
     
    def get_preceding_true(idx,new_edge_df, colnames):
        preceding = new_edge_df.loc[new_edge_df['Time'] <= idx]
        times = []

        for colnm in colnames:
            have_true = preceding.loc[new_edge_df[colnm] == True]
            # colnames.append('TasteID')
            try: 
                lastrow = have_true.iloc[-1]
                time = lastrow.Time
                times.append(time)
            except:
                times.append('nan')

        colnames.append('del_open')
        time_val = new_edge_df.loc[new_edge_df['Time'] == idx].Time
        times.append(time_val.values[0])
        d = dict(zip(colnames,times))
        out = pd.DataFrame(d, index=[0])
        
        out['TasteID'] = preceding.loc[preceding.Time == out.del_open.values[0]].TasteID.values[0]
        if out.TasteID.values[0] == 'nan':
            raise Exception('no')
            
        out['AnID'] = preceding['AnID'].iloc[0]
        out['Date'] = preceding['Date'].iloc[0]  
        
        return out
    
    new_edge_df = create_edge_frame(df)
    taste_start = new_edge_df.loc[new_edge_df['del_open'] == True]
    colnames = ['Poke1','Poke1_entry','Poke1_exit','Poke2','Poke2_entry','Poke2_exit']

    for i, row in taste_start.iterrows():
        idx = row.Time
        out = get_preceding_true(idx,new_edge_df,colnames)
        trial_df = pd.concat([trial_df,out],axis = 0)
        
    deliveries_only = find_poke_dat(new_df,'Poke2', delivery_idx)
    final_edgedf = final_edgedf.append(new_edge_df)
    finaldf = finaldf.append(deliveries_only)


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

new_df = add_days_elapsed(finaldf)

# convert poke 1 to int
trial_df  = trial_df.reset_index()
test = trial_df.loc[trial_df.Poke1_entry == 'nan']

trial_df['Poke1_entry'].loc[trial_df.Poke1_entry == 'nan'] = trial_df.loc[trial_df.Poke1_entry == 'nan'].del_open
trial_df['Poke1_entry'] = trial_df.Poke1_entry.astype(int)

trial_df  = trial_df.reset_index()
test = trial_df.loc[trial_df.Poke1_exit == 'nan']

trial_df['Poke1_exit'].loc[trial_df.Poke1_exit == 'nan'] = trial_df.loc[trial_df.Poke1_exit == 'nan'].del_open
trial_df['Poke1_exit'] = trial_df.Poke1_exit.astype(int)

# plotting the duration in poke2
trial_df['poke2_dur'] = trial_df.Poke2_exit - trial_df.Poke2_entry
plt.hist(x=trial_df.poke2_dur,bins=30)
plt.title('Duration in Poke2')
plt.ylabel('Latency (s)')

# plotting the latency from enter Poke2 to enter Poke1
trial_df['poke2_to_poke1'] = trial_df.Poke1_entry - trial_df.Poke2_entry
plt.hist(x=trial_df.poke2_to_poke1,bins=30)
plt.title('Poke2 entry to Poke1 entry')
plt.ylabel('Latency (s)')

# plotting the latency from Poke2 exit to Poke1 entry
trial_df['poke2ex_poke1en'] = trial_df.Poke1_entry - trial_df.Poke2_exit
plt.hist(x=trial_df.poke2ex_poke1en,bins=30)
plt.title('Poke2 Exit to Poke1 Entry')
plt.ylabel('Latency (s)')

# delivery open - poke2
trial_df['del_from_poke2'] = trial_df.del_open - trial_df.Poke2 
plt.hist(x=trial_df.del_from_poke2,bins=150)

# poke2 != poke2_exit
trial_df['poke2_fail'] = trial_df.Poke2 != trial_df.Poke2_exit



