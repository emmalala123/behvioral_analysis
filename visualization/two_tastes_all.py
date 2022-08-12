#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:27:55 2022

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats
import seaborn as sns
import glob
import random
import inflect
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
filelist.sort()
# for f in filelist:
for f in range(len(filelist)):
    df = pd.read_csv(filelist[f])
    group = df
    col = ['Line1', 'Line2']
    
    # TODO: must figure out index start-1 for next taste in relation to current taste 
    # to mark activity specific to the delivery even after the taste has been delivered
    # so far this function marks the delivery identity in relation to when the valve is open 
    # during the session.
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
    
            # delivery_idx = []
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
        
        pokes = ['Time'] + [poke]
        data = copy[pokes]
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
    
    deliveries_only = find_poke_dat(new_df,'Poke2', delivery_idx)
    finaldf = finaldf.append(deliveries_only)
    
def add_days_elapsed(finaldf):
   
    new_df = finaldf
    
    res = []
    for name, group in new_df.groupby('AnID'):
        i=1
        # if name == 'eb10':
        #     i = 1
        # if name == 'eb11':
        #     i = 14
        # if name == 'eb12':
        #     i = 9
        for n, g in group.groupby('Date'):
            print(g)
            bit = np.zeros(len(g))
            bit = bit + i
            res.extend(bit)
            i += 1
    new_df['Sessions'] = res
    # def getElapsed(grp):
    #     startDate = grp.iloc[0]
    #     return ((grp - startDate) / np.timedelta64(1, 'D')).astype(int)
    
    # new_df['dcol'] = pd.to_datetime(new_df['Date'], format = '%m%d%y')
    # new_df['elapsed'] = finaldf.groupby('AnID').dcol.transform(getElapsed)

    return new_df

new_df = add_days_elapsed(finaldf)

# def offset_sessions(new_df):
#     df = new_df
    
#     for 

def cumulativedels(new_df):
    csum = new_df.groupby(['AnID','Sessions','TasteID']).Taste_Delivery.sum()
    csum = csum.reset_index()
    return csum

csum = cumulativedels(new_df)
means = csum.groupby(["TasteID","Sessions"]).Taste_Delivery.mean().reset_index()
fig, ax = plt.subplots(figsize=(12,7))
p1 = sns.scatterplot(data = csum, x = "Sessions", y = "Taste_Delivery", hue = "TasteID", style = "AnID")
p2 = sns.lineplot(data = means, x = "Sessions", y = "Taste_Delivery", hue = "TasteID")


##latency plot   
sns.barplot(deliveries_only['TasteID'], deliveries_only['Latencies'])
cmap = plt.get_cmap('tab10')
t = sns.catplot(
    data=new_df,
    x = 'TasteID',
    y='Latencies',
    col='Sessions',
    kind='bar',
    order = ['suc', 'qhcl'],
    color=cmap(0)
    )
t.fig.suptitle("All Animals Poke-to-Poke Latencies for Two Tastes", x=.80, fontsize=15)
[plt.setp(ax.get_xticklabels(), rotation=45) for ax in t.axes.flat]
t.fig.subplots_adjust(0.6,top=0.8, wspace=0.2)
##

cmap = plt.get_cmap('tab10')
t = sns.catplot(
    data=deliveries_only,
    x = None,
    y='Taste_Delivery',
    col='AnID',
    kind='count',
    # order = ['suc', 'qhcl'],
    color=cmap(0)
    )


stats.ttest_ind(finaldf['TasteID'],finaldf['Latencies'])