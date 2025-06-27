#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================
Created on 2025.04.07
@author: oyang
====================================
"""
import uproot
import awkward as ak
import matplotlib.pyplot as plt

fin = "LYSO15x4NTUmentor.root"
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
s5 = ak.to_numpy(LYSO.arrays()['S5'])
s6 = ak.to_numpy(LYSO.arrays()['S6'])
ID = ak.to_numpy(LYSO.arrays()['ID'])
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
plt.savefig('S3vsS2.png')

import pandas as pd

df = pd.DataFrame({
    'EVENT': event,
    'S1': s1,
    'S2': s2,
    'S3': s3,
    'S4': s4,
    'S5': s5,
    'S6': s6,
    'ID': ID.astype(int)
})

df.to_csv("6d_data.csv", index=False)
print("Saved as 6d_data.csv")
