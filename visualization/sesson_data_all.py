#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:05:37 2022

@author: emmabarash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import scipy.stats
#import seaborn as sns
import glob
import random
import inflect
import re


animal1 = []
names = []
counter = 0

if os.sep == '/':
    directory = '/Users/emmabarash/Lab/data'
else:
    directory = r'C:\Users\Emma_PC\Documents\data'

filelist = glob.glob(os.path.join(directory,'*','*.csv'))
    
df = pd.read_csv(filelist[12])
group = df
col = ['Line1', 'Line2', 'Line3', 'Line4']


# TODO: must figure out index start-1 for next taste in relation to current taste 
# to mark activity specific to the delivery even after the taste has been delivered
# so far this function marks the delivery identity in relation to when the valve is open 
# during the session.
def parse_edges(group,col):
    delivery_idx = []
    group['TasteID'] = None
    for j in col:
        col = j
        if col == 'Line1': 
            taste = 'suc'
        if col == 'Line2':
            taste = 'low_NaCl'
        if col == 'Line3':
            taste = 'high_NaCl'
        if col == 'Line4':
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

new_df, delivery_idx = parse_edges(df, ['Line1', 'Line2', 'Line3', 'Line4'])

    
copy = new_df
poke = 'Poke2'
def find_poke_dat(copy, poke, delivery_idx):
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
    
    # for i, row in test.iterrows():
        # start = int(np.where(copy['Time'] == edgeON['TimeOn'][i])[0])
        # stop = int(np.where(copy['Time'] == edgeOFF['TimeOff'][i])[0])
    delivery_time = []
    for i in delivery_idx:
        copy.loc[i,'Taste_Delivery'] = True
        copy.loc[i,'Delivery_Time'] = copy['Time'][i]

        
        # collect delivery time to erase Poke2 dat within 10 seconds of delivery
        delivery_time.append(copy['Time'][i])

    
    # generatees a new df with only delivery times (marked 'true')
    deliveries_only = copy.loc[copy['Taste_Delivery'] == True]
    
    for i in delivery_time:
        if (edgeON['TimeOn'] - i['Time']) > 15:
            copy.loc[i == copy['Time']]
    
    for i in copy['Poke2']:
        if i == True and (copy['Time'] - copy) < 15:
            cleaned_pokes = copy.loc[i]
            
    cleaned_pokes = copy.loc[int(np.where(copy['Poke2'] == True and (15 - copy['Time']) < 15)[0])]
    
    for i in delivery_time:
        if copy['Poke2'] == True and (15 - i) < 15:
            print('hi')

def make_sess_data(df):
   suc = df.loc[df['Line1'] == True]
   low_Nacl = df.loc[df['Line2'] == True]
   high_Nacl = df.loc[df['Line3'] == True]
   qhcl = df.loc[df['Line4'] == True]
    
make_sess_data(df)