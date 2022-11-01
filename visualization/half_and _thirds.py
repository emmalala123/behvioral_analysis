#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:39:44 2022

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
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
        for n, g in group.groupby('Date'):
            print(g)
            bit = np.zeros(len(g))
            bit = bit + i
            res.extend(bit)
            i += 1
    new_df['Sessions'] = res

    return new_df

new_df = add_days_elapsed(finaldf)

def cumulativedels(new_df):
    csum = new_df.groupby(['AnID','Sessions','TasteID', 'Latencies']).Delivery_Time.sum()
    csum = csum.reset_index()
    return csum


csum = cumulativedels(new_df)
means = csum.groupby(["TasteID","Sessions"]).Delivery_Time.mean().reset_index()
fig, ax = plt.subplots(figsize=(10,5))
p1 = sns.scatterplot(data = csum, x = "Sessions", y = "Delivery_Time", hue = "TasteID", style = "AnID", s=65)
p2 = sns.lineplot(data = means, x = "Sessions", y = "Delivery_Time", hue = "TasteID")
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

csum = cumulativedels(new_df)
means = csum.groupby(["TasteID","Sessions"]).Latencies.mean().reset_index()
fig, ax = plt.subplots(figsize=(10,5))
p1 = sns.scatterplot(data = csum, x = "Sessions", y = "Latencies", hue = "TasteID", style = "AnID", s=65)
p2 = sns.lineplot(data = means, x = "Sessions", y = "Latencies", hue = "TasteID")
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


##latency plot  
sns.barplot(deliveries_only['TasteID'], deliveries_only['Latencies'])
cmap = plt.get_cmap('tab10')
t = sns.catplot(
    data=new_df,
    x = 'TasteID',
    y='Latencies',
    col='Sessions',
    kind='bar',
    hue = 'TasteID',
    order = ['suc', 'qhcl'],
    # color=cmap(0)
    )
###
sns.set_theme(style='white')
t = sns.catplot(
    data=new_df,
    kind='bar',
    x = 'Sessions',
    y='Latencies',
    #col='Sessions',
    hue = 'TasteID',
    #order = ['suc', 'qhcl']
    # color=cmap(0)
    # height = 8,
    aspect = 12/7
    )
t = sns.swarmplot(
    data=new_df,
    x = 'Sessions',
    y='Latencies',
    #col = 'Sessions',
    hue = "TasteID",
    #color = "TasteID",
    dodge= True,
    edgecolor = "white",
    linewidth = 1,
    alpha = 0.5,
    )
# fig = t.get_figure()
# fig.savefig('t.png', dpi=600)
###
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

copy = new_df
# finding halves
copy['Section'] = None
copy.loc[copy['Time'] <= 1800, "Section"] = "First_Half"
copy.loc[(copy['Time'] > 1800) & (copy['Time'] <= 3600), "Section"] = "Second_Half"

# labeling concentrations
copy['Concentration'] = None
copy.loc[(copy['Sessions'] < 15) & (copy['TasteID'] == 'qhcl'), "Concentration"] = "5mM"
copy.loc[(copy['Sessions'] >= 15) & (copy['TasteID'] == 'qhcl'), "Concentration"] = "10mM"
copy.loc[copy['TasteID'] == 'suc', "Concentration"] = "0.3M"

# test re-sums the deliveries without 'Section'
test = copy
test['Taste_Delivery'] = copy['Taste_Delivery'].astype(int)
test = copy.groupby(['AnID','Sessions','TasteID', 'Concentration']).agg(sum).reset_index()
fig, ax = plt.subplots(1,2, figsize=(17, 6))

# plot all deliveries for all sections
g = sns.lineplot(data = test.loc[test['AnID'] == 'eb11'],
            x = 'Sessions', y = 'Taste_Delivery', hue = 'TasteID', ax=ax[0]).set(title='eb11')
g = sns.lineplot(data = test.loc[test['AnID'] == 'eb12'],
            x = 'Sessions', y = 'Taste_Delivery', hue = 'TasteID', ax=ax[1]).set(title='eb12')

# plotting from 10mM qhcl
fig, ax = plt.subplots(1,2, figsize=(17, 6))
g = sns.lineplot(data = test.loc[(test['AnID'] == 'eb11') & (test['Sessions'] >= 15)],
            x = 'Sessions', y = 'Taste_Delivery', hue = 'TasteID', ax=ax[0]).set(title='eb11')
g = sns.lineplot(data = test.loc[(test['AnID'] == 'eb12') & (test['Sessions'] >= 15)],
            x = 'Sessions', y = 'Taste_Delivery', hue = 'TasteID', ax=ax[1]).set(title='eb12')


# plotting halves
#line
copybara = copy
copybara['Taste_Delivery'] = copy['Taste_Delivery'].astype(int)
copybara = copy.groupby(['Section','AnID','Sessions','TasteID']).agg(sum).reset_index()

# plot all deliveries for all sections
g = sns.relplot(data = copybara,kind = 'line',
            x = 'Sessions', y = 'Taste_Delivery', col = 'AnID', hue = 'TasteID')

# plot all deliveries for halves (All)
g = sns.relplot(data = copybara,kind = 'line',
            x = 'Sessions', y = 'Taste_Delivery', col = 'Section', hue = 'TasteID')

# plot all deliveries for halves (Individual)
g = sns.relplot(data = copybara,kind = 'line',
            x = 'Sessions', y = 'Taste_Delivery', row = 'Section', col='AnID', hue = 'TasteID')
# just the 10mM
g = sns.relplot(data = copybara.loc[copybara['Sessions'] >= 15],kind = 'line',
            x = 'Sessions', y = 'Taste_Delivery', row = 'Section', col='AnID', hue = 'TasteID')

#bar
g = sns.catplot(data = copybara.loc[copybara['Sessions'] >= 15],kind = "bar",
            x = 'Section', y = 'Taste_Delivery', hue = 'TasteID', order=['First_Half', "Second_Half"]).set(title="Deliveries across all sessions, N=2")

# take p values from ttest
first_qhcl = copybara.loc[(copybara['AnID'] == 'eb11') & (copybara['Sessions'] >= 15) & (copybara['TasteID'] == 'qhcl')]
first_suc = copybara.loc[(copybara['AnID'] == 'eb11') & (copybara['Sessions'] >= 15) & (copybara['TasteID'] == 'suc')]
stats.ttest_ind(first_qhcl['Taste_Delivery'], first_suc['Taste_Delivery'])

first_qhcl = copybara.loc[(copybara['AnID'] == 'eb12') & (copybara['Sessions'] >= 15) & (copybara['TasteID'] == 'qhcl')]
first_suc = copybara.loc[(copybara['AnID'] == 'eb12') & (copybara['Sessions'] >= 15) & (copybara['TasteID'] == 'suc')]
stats.ttest_ind(first_qhcl['Taste_Delivery'], first_suc['Taste_Delivery'])

# split sessions into thirds
thirds_df = new_df
thirds_df['Section'] = None
thirds_df.loc[thirds_df['Time'] <= 1200, "Section"] = "First_Third"
thirds_df.loc[(thirds_df['Time'] > 1200) & (copy['Time'] <= 2400), "Section"] = "Second_Third"
thirds_df.loc[(thirds_df['Time'] > 2400) & (copy['Time'] <= 3600), "Section"] = "Last_Third"

# plotting thirds
## altogether
#line
copybara = thirds_df
copybara['Taste_Delivery'] = copy['Taste_Delivery'].astype(int)
copybara = copy.groupby(['Section','AnID','Sessions','TasteID']).agg(sum).reset_index()
g = sns.relplot(data = copybara,kind = 'line',
            x = 'Sessions', y = 'Taste_Delivery', col = 'Section', hue = 'TasteID', col_order=(['First_Third', 'Second_Third', 'Last_Third']))
#bar
g = sns.catplot(data = copybara,kind = "bar",
            x = 'Section', y = 'Taste_Delivery', hue = 'TasteID', order=['First_Third', "Second_Third", "Last_Third"]).set(title="Deliveries across all sessions, N=2")

    
## separately
g = sns.relplot(data = copybara,kind = 'line',
            x = 'Sessions', y = 'Taste_Delivery', row='AnID', col = 'Section', hue = 'TasteID', col_order=(['First_Third', 'Second_Third', 'Last_Third']))
#bar
g = sns.catplot(data = copybara,kind = "bar",
            x = 'Section', y = 'Taste_Delivery', col='AnID', hue = 'TasteID', order=['First_Third', "Second_Third", "Last_Third"]).set(title="Deliveries across all sessions, N=2")
#box
g = sns.boxenplot(data = copybara,
            x = 'Section', y = 'Taste_Delivery', hue = 'TasteID', order=['First_Third', "Second_Third", "Last_Third"]).set(title="Deliveries across all sessions, N=2")

## separately -- just 10mM
g = sns.relplot(data = copybara.loc[copybara['Sessions'] >= 15],kind = 'line',
            x = 'Sessions', y = 'Taste_Delivery', row='AnID', col = 'Section', hue = 'TasteID', col_order=(['First_Third', 'Second_Third', 'Last_Third']))
#bar
g = sns.catplot(data = copybara.loc[copybara['Sessions'] >= 15],kind = "bar",
            x = 'Section', y = 'Taste_Delivery', col='AnID', hue = 'TasteID', order=['First_Third', "Second_Third", "Last_Third"]).set(title="Deliveries across all sessions, N=2")
#box
g = sns.boxenplot(data = copybara.loc[copybara['Sessions'] >= 15],
            x = 'Section', y = 'Taste_Delivery', hue = 'TasteID', order=['First_Third', "Second_Third", "Last_Third"]).set(title="Deliveries across all sessions, N=2")


