# read_csv.py

#===============================================================

import os

import csv
import numpy as np

import pandas as pd

#====================================================================

path_pred = '/home/mike/Jambo18/new_data/Predictor_Data/'
path_loss = '/home/mike/Jambo18/new_data/Loss_Data/Complete/'

# Files
files_loss = [f for f in os.listdir(path_loss) if f.endswith('.csv')]
files_pred = [f for f in os.listdir(path_pred) if f.endswith('.csv')]

nf_loss = len(files_loss)
nf_pred = len(files_pred)

#====================================================================

ids = pd.read_csv(path_loss+files_loss[0], sep=',',header=None,dtype='str')

print(ids)
print(np.shape(ids))
ids = np.array(float(ids))[1:,0]
print(ids)

df=pd.read_csv(path_pred+files_pred[0], sep=',',header=None,dtype='str')

df = np.array(df,dtype=str)

###########
###########

set1 = sorted(set(ids))
tally = np.array(len(set1))
ei=0
for element in set1:
	for i in range(0,np.shape(A,1)):
		if element == A[0,i]:
			tally[ei] += A[0,i]
	ei += 1

tally = A / 10000.

###########
###########

#print(df[1,:])


