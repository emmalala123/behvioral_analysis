#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:41:38 2022

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
from scipy import stats
import glob
import re

###LOADING FILES
animal_rep = []
names = []
counter = 0
directory = ['/Users/emmabarash/Lab/data/eb2_data/','/Users/emmabarash/Lab/data/eb2_data/stage_3']
animal_rep_files = []

directory = '/Users/emmabarash/Lab/data/*/*'
pathlist = glob.glob(directory)

def filt_expdirs(pathlist):
    pat = re.compile("data/eb..")
    filtlist = [i for i in pathlist if pat.search(i)]
    pat = re.compile("stage_3")
    filtlist = [i for i in filtlist if not pat.search(i)]
    
    return filtlist
filtlist = filt_expdirs(pathlist)
fileframe = pd.DataFrame(filtlist, columns = ['rec_dir'])

fileframe = (fileframe
             .assign(expname = lambda x : x.rec_dir.str.split("/"))
             .assign(expname = lambda x : x.expname.str[6])
             .assign(date = lambda x : x.expname.str.split("_"))
             .assign(anID = lambda x : x.date.str[0])
             .assign(date = lambda x : x.date.str[1])
             )

directory = '/Users/emmabarash/Lab/data/*/*/*'
pathlist = glob.glob(directory)
def filt_stage_3(pathlist):
    pat = re.compile("stage_3")
    filtlist = [i for i in pathlist if pat.search(i)]
    return filtlist

stage_3_list = filt_stage_3(pathlist)

fileframe2 = pd.DataFrame(stage_3_list, columns = ['rec_dir'])

fileframe2 = (fileframe2
             .assign(expname = lambda x : x.rec_dir.str.split("/"))
             .assign(expname = lambda x : x.expname.str[7])
             .assign(date = lambda x : x.expname.str.split("_"))
             .assign(anID = lambda x : x.date.str[0])
             .assign(date = lambda x : x.date.str[1])
             )


dir_list = pd.concat([fileframe,fileframe2],axis=0)
###LOADING FILES

###insert code hre to filter bad files from dir_list

data = []
for index, row in dir_list.iterrows():
    dat = pd.read_csv(row.rec_dir)
    dat[['anID','date','rec_dir']] = row[['anID','date','rec_dir']]
    data.append(dat)
    
data = pd.concat(data,axis = 0)

#parse edges returns time of falling edge & dt since preceding rise
def parse_edges(group,col):
    cols = ['Time']+[col]
    data = group[cols]
    try: edges = data[data[col].diff().fillna(False)]
    except: return None
    edgeOFF = edges[edges[col]==False]
    edgeOFF[col] = True
    dt = edges['Time'].diff()
    dt = dt[edgeOFF.index]
    dtlab = col+'_dt'
    edgeOFF[dtlab] = dt
    # new_label = col+'_Time'
    # edgeOFF.pop(col)
    # edgeOFF = edgeOFF.rename(columns = {"Time":new_label})
    
    ## get count
    qsum = edgeOFF[col].cumsum()
    sum_label = col + "_sum"
    edgeOFF[sum_label] = qsum
    
    return edgeOFF

groupcols = ['anID','date', 'rec_dir']

edge_dat = []
for name, group in data.groupby(groupcols):
    p1 = parse_edges(group,'Poke1')
    p2 = parse_edges(group,'Poke2')
    l1 = parse_edges(group,'Line1')
    l2 = parse_edges(group,'Line2')
    res = pd.concat([p1,p2,l1,l2],axis = 0)
    res[groupcols] = name
    edge_dat.append(res)
    
edge_dat = pd.concat(edge_dat,axis=0)
#edge_dat['date'] = pd.to_datetime(edge_dat['date'], format="%m%d%y")
#calling rows by indicies: df.iloc[idx]
# edge_dat.iloc[eb2]
#calling rows by variable name: df.loc[df[col]==val]

subset = edge_dat[['rec_dir','date','anID']]
subset = subset.drop_duplicates()
subset['session']=subset.groupby(['anID']).cumcount()
subset = subset[['rec_dir','session']]

edge_dat =edge_dat.merge(subset,how='inner', on = 'rec_dir')
edge_dat = edge_dat.drop_duplicates()

#edge_dat['session'] = edge_dat.groupby(['anID']).cumcount()
eb2_dat = edge_dat.loc[edge_dat['anID']=='eb02']
eb2_dat['date'] = pd.to_datetime(eb2_dat['date'], format="%m%d%y")
eb2_subset = eb2_dat.loc[eb2_dat['session'] >= 28]
eb2_subset = eb2_subset.loc[eb2_subset['session'] <= 42]
eb2_subset = eb2_subset.loc[eb2_subset['Poke1_dt'] >= 1]

# get the pokes
eb2_rew_pokes = eb2_subset.loc[(eb2_subset['Poke1_dt'] >= 1)]

# I had some stupid stuff here, but deleted it becuase it didn't really work.




