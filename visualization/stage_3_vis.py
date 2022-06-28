#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:37:36 2021

@author: emmabarash
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 11:30:29 2021

@author: emmabarash
"""

import os 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

directory = '/Users/emmabarash/Lab/data/eb1_data'
files = []
names = []
for filename in os.listdir(directory):
    names.append(filename)
names.sort()

for name in names:
    f = os.path.join(directory, name)
    if os.path.isfile(f):
        files.append(pd.read_csv(f))

sum_line1 = []
sum_line2 = []
for f in files:
        #if not f is files[2]:
    sum_line1.append(f['Line1'].sum(axis=0))
    sum_line2.append(f['Line2'].sum(axis=0))
 
sum_lines = set(zip(sum_line1, sum_line2))
all_lines = pd.DataFrame(np.asarray([sum_line1,sum_line2]).T)
all_lines.insert(loc=0,column='session',value=range(len(all_lines)))

v2 = pd.DataFrame(np.asarray([sum_line1,sum_line2]).flatten())
v2.insert(loc=0,column='session',value=np.tile(range(len(sum_line1)),2))
v2.insert(loc=1,column='lines',value=np.repeat(range(1,3),len(sum_line1)))


g = sns.lineplot(x='session',y=0,hue='lines',data=v2, palette=(['steelblue', 'purple']))
if directory =="/Users/emmabarash/Lab/data/eb1_data/stage_3":
    g.set_title(directory[-16:-13] + " Events vs Sessions")
else:
    g.set_title(directory[-8:-5] + " Events vs Sessions")
    
#g.set(xticks=[0, 1, 2, 3, 4])
g.legend(['rewarder', 'trigger'])

# sum_df = df_5_14.sum(axis=0).reset_index() 
# sum_df.rename(columns={0:'total'},inplace=True)
# sum_df.insert(loc=0,column='session',value=0)