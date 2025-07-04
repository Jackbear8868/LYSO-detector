#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================
Created on 2025.04.07
@author: oyang
====================================
"""
import uproot
import math
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from datetime import datetime

fin = "LYSO4NTUstudent.root"
fTree = "LYSO"
LYSO = uproot.open(fin)[fTree]
print ("Read ROOT file:", fin)

#======================================
# read LYSO's leaves
#======================================
s1 = ak.to_numpy(LYSO.arrays()['S1'])
s2 = ak.to_numpy(LYSO.arrays()['S2'])
s3 = ak.to_numpy(LYSO.arrays()['S3'])
s4 = ak.to_numpy(LYSO.arrays()['S4'])
event = ak.to_numpy(LYSO.arrays()['EVENT'])
LYSO.close()

# plot
sgmin = 0.
sgmax = 2000.
nbin = 200       
plt.figure(20)
plt.title("S3 signal vs S2 signal")
plt.xlabel('S2')
plt.ylabel('S3')
plt.xlim([sgmin,sgmax])
plt.ylim([sgmin,sgmax])
plt.scatter(s2,s3,marker='.',s=1,linewidths=0)
plt.savefig('LYSO_s2s3.png')
plt.close()

import pandas as pd

df = pd.DataFrame({
    'EVENT': event,
    'S1': s1,
    'S2': s2,
    'S3': s3,
    'S4': s4
})

df[['S1', 'S2', 'S3', 'S4']] = df[['S1', 'S2', 'S3', 'S4']].astype(int)
df.to_csv("data.csv", index=False)
print("Saved as data.csv")
