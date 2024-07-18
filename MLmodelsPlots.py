#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:54:49 2023

@author: Georgios Georgalis
"""


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import math
import numpy as np
import numpy.linalg as nla
import pandas as pd
import re
import six
from os.path import join
from matplotlib import pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Path to test flights folder, we have 8 flights
test_folder = './../XC/JAISdata/testflights' 
flight1 = pd.read_csv(test_folder+'/'+'log_210820_152705_KCUH.csv', header = 2 ) #cessna
flight2 = pd.read_csv(test_folder+'/'+'log_190917_201335_KSWO.csv', header = 2 ) #cessna
flight3 = pd.read_csv(test_folder+'/'+'log_200131_225232_KSEP.csv', header = 2 ) #cirrus
flight4 = pd.read_csv(test_folder+'/'+'log_201210_145455_KSWO.csv', header = 2 ) #cirrus
flight5 = pd.read_csv(test_folder+'/'+'log_210331_124115_KSWO.csv', header = 2 ) #cessna
flight6 = pd.read_csv(test_folder+'/'+'log_211020_200656_KSWO.csv', header = 2 ) #cessna
flight7 = pd.read_csv(test_folder+'/'+'log_220118_174650_KSWO.csv', header = 2 ) #cirrus
flight8 = pd.read_csv(test_folder+'/'+'log_221006_200626_KSWO.csv', header = 2 ) #cirrus

#Each flight dataframes need some pre-processing and cleaning
#Flight 1
flight1 = flight1.apply(pd.to_numeric, errors='coerce').fillna(0)
flight1 = flight1[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
            ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
flight1['avg_CHT'] = flight1[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
flight1=flight1.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

flight1['avg_EGT'] = flight1[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
flight1=flight1.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

flight1.columns = flight1.columns.str.replace(' ', '')
flight1 = flight1[flight1.IAS > 0]
#flight1 = flight1[flight1.AltMSL>1000]
flight1 = flight1.reset_index(drop=True)
flight1['dAlt'] = flight1['AltMSL'].diff()
flight1['dAlt'][0] = 0
flight1['MA_VSpd'] = flight1['VSpd'].rolling(30).mean()
flight1['MA_VSpd'][0:30] = flight1['VSpd'][0:30]
keepalt1 = flight1['AltMSL']

#Flight 2
flight2 = flight2.apply(pd.to_numeric, errors='coerce').fillna(0)
flight2 = flight2[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
            ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
flight2['avg_CHT'] = flight2[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
flight2=flight2.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

flight2['avg_EGT'] = flight2[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
flight2=flight2.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

flight2.columns = flight2.columns.str.replace(' ', '')
flight2 = flight2[flight2.IAS > 0]
#flight2 = flight2[flight2.AltMSL>1000]
flight2 = flight2.reset_index(drop=True)
flight2['dAlt'] = flight2['AltMSL'].diff()
flight2['dAlt'][0] = 0
flight2['MA_VSpd'] = flight2['VSpd'].rolling(30).mean()
flight2['MA_VSpd'][0:30] = flight2['VSpd'][0:30]
keepalt2 = flight2['AltMSL']

#Flight 3

flight3 = flight3.apply(pd.to_numeric, errors='coerce').fillna(0)
flight3 = flight3[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
            ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
flight3['avg_CHT'] = flight3[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
flight3=flight3.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

flight3['avg_EGT'] = flight3[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
flight3=flight3.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

flight3.columns = flight3.columns.str.replace(' ', '')
flight3 = flight3[flight3.IAS > 0]
#flight3 = flight3[flight3.AltMSL>1000]
flight3 = flight3.reset_index(drop=True)
flight3['dAlt'] = flight3['AltMSL'].diff()
flight3['dAlt'][0] = 0
flight3['MA_VSpd'] = flight3['VSpd'].rolling(30).mean()
flight3['MA_VSpd'][0:30] = flight3['VSpd'][0:30]
keepalt3 = flight3['AltMSL']

#flight4
flight4 = flight4.apply(pd.to_numeric, errors='coerce').fillna(0)
flight4 = flight4[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
            ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
flight4['avg_CHT'] = flight4[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
flight4=flight4.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

flight4['avg_EGT'] = flight4[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
flight4=flight4.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

flight4.columns = flight4.columns.str.replace(' ', '')
flight4 = flight4[flight4.IAS > 0]
#flight4 = flight4[flight4.AltMSL>1000]
flight4 = flight4.reset_index(drop=True)
flight4['dAlt'] = flight4['AltMSL'].diff()
flight4['dAlt'][0] = 0
flight4['MA_VSpd'] = flight4['VSpd'].rolling(30).mean()
flight4['MA_VSpd'][0:30] = flight4['VSpd'][0:30]
keepalt4 = flight4['AltMSL']

#flight5

flight5 = flight5.apply(pd.to_numeric, errors='coerce').fillna(0)
flight5 = flight5[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
            ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
flight5['avg_CHT'] = flight5[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
flight5=flight5.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

flight5['avg_EGT'] = flight5[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
flight5=flight5.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

flight5.columns = flight5.columns.str.replace(' ', '')
flight5 = flight5[flight5.IAS > 0]
#flight5 = flight5[flight5.AltMSL>1000]
flight5 = flight5.reset_index(drop=True)
flight5['dAlt'] = flight5['AltMSL'].diff()
flight5['dAlt'][0] = 0
flight5['MA_VSpd'] = flight5['VSpd'].rolling(30).mean()
flight5['MA_VSpd'][0:30] = flight5['VSpd'][0:30]
keepalt5 = flight5['AltMSL']

#Flight 6

flight6 = flight6.apply(pd.to_numeric, errors='coerce').fillna(0)
flight6 = flight6[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
            ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
flight6['avg_CHT'] = flight6[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
flight6=flight6.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

flight6['avg_EGT'] = flight6[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
flight6=flight6.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

flight6.columns = flight6.columns.str.replace(' ', '')
flight6 = flight6[flight6.IAS > 0]
#flight6 = flight6[flight6.AltMSL>1000]
flight6 = flight6.reset_index(drop=True)
flight6['dAlt'] = flight6['AltMSL'].diff()
flight6['dAlt'][0] = 0
flight6['MA_VSpd'] = flight6['VSpd'].rolling(30).mean()
flight6['MA_VSpd'][0:30] = flight6['VSpd'][0:30]
keepalt6 = flight6['AltMSL']

#Flight 7

flight7 = flight7.apply(pd.to_numeric, errors='coerce').fillna(0)
flight7 = flight7[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
            ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
flight7['avg_CHT'] = flight7[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
flight7=flight7.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

flight7['avg_EGT'] = flight7[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
flight7=flight7.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

flight7.columns = flight7.columns.str.replace(' ', '')
flight7 = flight7[flight7.IAS > 0]
#flight7 = flight7[flight7.AltMSL>1000]
flight7 = flight7.reset_index(drop=True)
flight7['dAlt'] = flight7['AltMSL'].diff()
flight7['dAlt'][0] = 0
flight7['MA_VSpd'] = flight7['VSpd'].rolling(30).mean()
flight7['MA_VSpd'][0:30] = flight7['VSpd'][0:30]
keepalt7 = flight7['AltMSL']

#Flight 8

flight8 = flight8.apply(pd.to_numeric, errors='coerce').fillna(0)
flight8 = flight8[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
            ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
flight8['avg_CHT'] = flight8[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
flight8=flight8.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])

flight8['avg_EGT'] = flight8[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
flight8=flight8.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])

flight8.columns = flight8.columns.str.replace(' ', '')
flight8 = flight8[flight8.IAS > 0]
#flight8 = flight8[flight8.AltMSL>1000]
flight8 = flight8.reset_index(drop=True)
flight8['dAlt'] = flight8['AltMSL'].diff()
flight8['dAlt'][0] = 0
flight8['MA_VSpd'] = flight8['VSpd'].rolling(30).mean()
flight8['MA_VSpd'][0:30] = flight8['VSpd'][0:30]
keepalt8 = flight8['AltMSL']


#%%K-means clustering setup
kmeans = KMeans(
    init="k-means++",
    n_clusters=3,
    n_init=10,
    max_iter=400,
)

#GMM setup
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, random_state = 42, init_params='kmeans')

#PCA setup
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

flightdict = {'1':flight1, '2':flight2, '3':flight3, '4':flight4, '5':flight5, '6':flight6, '7':flight7, '8':flight8}  
minmaxscaler = MinMaxScaler()
sscaler = StandardScaler()

#Tables with scores of clustering quality for each method
#%% Low variance filter

sil_score_kmeans_lvf = []
ch_score_kmeans_lvf =[]
db_score_kmeans_lvf = []

sil_score_gmm_lvf = []
ch_score_gmm_lvf =[]
db_score_gmm_lvf =[]

for i in flightdict.items():
     flight_varscaled = minmaxscaler.fit_transform(i[1])
     flight_varscaled = pd.DataFrame(flight_varscaled, columns = ['AltMSL','IAS','VSpd','Pitch','avg_CHT','avg_EGT','E1RPM', 'dAlt', 'MA_VSpd'])
     columns = flight_varscaled.columns
     variable = []
     for j in range(0,9):
         if flight_varscaled.var()[j]>=flight_varscaled.var().median():
             variable.append(columns[j])
            

     flight = i[1].loc[:,variable]
     flight_scaled = sscaler.fit_transform(flight)  
     kmeans.fit(flight_scaled) 
     labels = kmeans.labels_
     sil_score_kmeans_lvf.append(silhouette_score(flight_scaled, labels, metric='euclidean'))
     ch_score_kmeans_lvf.append(calinski_harabasz_score(flight_scaled, labels))
     db_score_kmeans_lvf.append(davies_bouldin_score(flight_scaled, labels))
    
     gmm.fit(flight_scaled)
     labels = gmm.predict(flight_scaled)
     sil_score_gmm_lvf.append(silhouette_score(flight_scaled, labels, metric='euclidean'))
     ch_score_gmm_lvf.append(calinski_harabasz_score(flight_scaled, labels))
     db_score_gmm_lvf.append(davies_bouldin_score(flight_scaled, labels))

#Compile scores for LVF

df_lfv_scores = pd.DataFrame({'sil_kmeans': sil_score_kmeans_lvf, 'sil_gmm': sil_score_gmm_lvf, 'ch_kmeans': ch_score_kmeans_lvf, 'ch_gmm':ch_score_gmm_lvf, 'db_kmeans': db_score_kmeans_lvf, 'db_gmm':db_score_gmm_lvf })


#%%High Correlation Filter
 def hcf(dataset, threshold):
     col_corr = set()
     corr_matrix = dataset.corr()
     for i in range(len(corr_matrix.columns)):
         for j in range(i):
             if abs(corr_matrix.iloc[i,j]>threshold):
                 colname = corr_matrix.columns[i]
                 col_corr.add(colname)
     return col_corr

 sil_score_kmeans_hcf = []
 ch_score_kmeans_hcf =[]
 db_score_kmeans_hcf = []
 sil_score_gmm_hcf = []
 ch_score_gmm_hcf =[]
 db_score_gmm_hcf =[]
    
 for i in flightdict.items():
     flight_scaled = sscaler.fit_transform(i[1])
     flight_scaled = pd.DataFrame(flight_scaled, columns = ['AltMSL','IAS','VSpd','Pitch','avg_CHT','avg_EGT','E1RPM', 'dAlt', 'MA_VSpd'])
     cor_features = hcf(flight_scaled, 0.6) 
     flight_scaled = flight_scaled.drop(cor_features, axis = 1)
    
     kmeans.fit(flight_scaled) 
     labels = kmeans.labels_
     sil_score_kmeans_hcf.append(silhouette_score(flight_scaled, labels, metric='euclidean'))
     ch_score_kmeans_hcf.append(calinski_harabasz_score(flight_scaled, labels))
     db_score_kmeans_hcf.append(davies_bouldin_score(flight_scaled, labels))
   
     gmm.fit(flight_scaled)
     labels = gmm.predict(flight_scaled)
     sil_score_gmm_hcf.append(silhouette_score(flight_scaled, labels, metric='euclidean'))
     ch_score_gmm_hcf.append(calinski_harabasz_score(flight_scaled, labels))
     db_score_gmm_hcf.append(davies_bouldin_score(flight_scaled, labels))
   
 #Compile scores for HCF

df_hcf_scores = pd.DataFrame({'sil_kmeans': sil_score_kmeans_hcf, 'sil_gmm': sil_score_gmm_hcf, 'ch_kmeans': ch_score_kmeans_hcf, 'ch_gmm':ch_score_gmm_hcf, 'db_kmeans': db_score_kmeans_hcf, 'db_gmm':db_score_gmm_hcf })
   
    
 #%%PCA with 4 components    
    
 sil_score_kmeans_pca = []
 ch_score_kmeans_pca =[]
 db_score_kmeans_pca = []

 sil_score_gmm_pca = []
 ch_score_gmm_pca =[]
 db_score_gmm_pca =[]

 for i in flightdict.items():
     flight_scaled = sscaler.fit_transform(i[1])
     flight_scaled = pd.DataFrame(flight_scaled, columns = ['AltMSL','IAS','VSpd','Pitch','avg_CHT','avg_EGT','E1RPM', 'dAlt', 'MA_VSpd'])
     flight_pca = pca.fit_transform(flight_scaled)
  
     kmeans.fit(flight_pca) 
     labels = kmeans.labels_
     sil_score_kmeans_pca.append(silhouette_score(flight_pca, labels, metric='euclidean'))
     ch_score_kmeans_pca.append(calinski_harabasz_score(flight_pca, labels))
     db_score_kmeans_pca.append(davies_bouldin_score(flight_pca, labels))
  
     gmm.fit(flight_pca)
     labels = gmm.predict(flight_pca)
     sil_score_gmm_pca.append(silhouette_score(flight_pca, labels, metric='euclidean'))
     ch_score_gmm_pca.append(calinski_harabasz_score(flight_pca, labels))
     db_score_gmm_pca.append(davies_bouldin_score(flight_pca, labels))
# #Compile scores for PCA
df_pca_scores = pd.DataFrame({'sil_kmeans': sil_score_kmeans_pca, 'sil_gmm': sil_score_gmm_pca, 'ch_kmeans': ch_score_kmeans_pca, 'ch_gmm':ch_score_gmm_pca, 'db_kmeans': db_score_kmeans_pca, 'db_gmm':db_score_gmm_pca })
   
 #%% No-engine, no duplicates info
 sil_score_kmeans_noeng = []
 ch_score_kmeans_noeng =[]
 db_score_kmeans_noeng = []
 sil_score_gmm_noeng = []
 ch_score_gmm_noeng =[]
 db_score_gmm_noeng =[]
 for i in flightdict.items():
     flight_scaled = sscaler.fit_transform(i[1])
     flight_scaled = pd.DataFrame(flight_scaled, columns = ['AltMSL','IAS','VSpd','Pitch','avg_CHT','avg_EGT','E1RPM', 'dAlt', 'MA_VSpd'])
     flight_scaled = flight_scaled.drop(['AltMSL', 'VSpd' ,'avg_CHT', 'avg_EGT', 'E1RPM'], axis = 1)
 
     kmeans.fit(flight_scaled) 
     labels = kmeans.labels_
     sil_score_kmeans_noeng.append(silhouette_score(flight_scaled, labels, metric='euclidean'))
     ch_score_kmeans_noeng.append(calinski_harabasz_score(flight_scaled, labels))
     db_score_kmeans_noeng.append(davies_bouldin_score(flight_scaled, labels))
   
     gmm.fit(flight_scaled)
     labels = gmm.predict(flight_scaled)
     sil_score_gmm_noeng.append(silhouette_score(flight_scaled, labels, metric='euclidean'))
     ch_score_gmm_noeng.append(calinski_harabasz_score(flight_scaled, labels))
     db_score_gmm_noeng.append(davies_bouldin_score(flight_scaled, labels))

 #Compile scores for noeng
df_noeng_scores = pd.DataFrame({'sil_kmeans': sil_score_kmeans_noeng, 'sil_gmm': sil_score_gmm_noeng, 'ch_kmeans': ch_score_kmeans_noeng, 'ch_gmm':ch_score_gmm_noeng, 'db_kmeans': db_score_kmeans_noeng, 'db_gmm':db_score_gmm_noeng })
    
 #%%Doing Nothing
sil_score_kmeans = []
ch_score_kmeans  =[]
db_score_kmeans  = []
sil_score_gmm  = []
ch_score_gmm  =[]
db_score_gmm  =[]

 for i in flightdict.items():
     flight_scaled = sscaler.fit_transform(i[1])
     flight_scaled = pd.DataFrame(flight_scaled, columns = ['AltMSL','IAS','VSpd','Pitch','avg_CHT','avg_EGT','E1RPM', 'dAlt', 'MA_VSpd'])
   
     kmeans.fit(flight_scaled) 
     labels = kmeans.labels_
     sil_score_kmeans.append(silhouette_score(flight_scaled, labels, metric='euclidean'))
     ch_score_kmeans.append(calinski_harabasz_score(flight_scaled, labels))
     db_score_kmeans.append(davies_bouldin_score(flight_scaled, labels))
  
     gmm.fit(flight_scaled)
     labels = gmm.predict(flight_scaled)
     sil_score_gmm.append(silhouette_score(flight_scaled, labels, metric='euclidean'))
     ch_score_gmm.append(calinski_harabasz_score(flight_scaled, labels))
     db_score_gmm.append(davies_bouldin_score(flight_scaled, labels))
 #Compile scores for noeng
df_scores = pd.DataFrame({'sil_kmeans': sil_score_kmeans, 'sil_gmm': sil_score_gmm, 'ch_kmeans': ch_score_kmeans, 'ch_gmm':ch_score_gmm, 'db_kmeans': db_score_kmeans, 'db_gmm':db_score_gmm })
%%AE
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout

 ## Building the autoencoder
n_inputs=np.shape(flight1)[1] 

inputs = Input(shape=(n_inputs,))
#encoder block 0
e0 = Dense(9)(inputs)
e0 = BatchNormalization()(e0)
e0 = Activation('selu')(e0)
# #encoder block 1
e1 = Dense(8)(e0)
e1 = BatchNormalization()(e1)
e1 = Activation('selu')(e1)
# #encoder block 2
e2 = Dense(7)(e1)
e2 = BatchNormalization()(e2)
e2 = Activation('selu')(e2)
#encoder block 3
e3 = Dense(6)(e2)
e3 = BatchNormalization()(e3)
e3 = Activation('selu')(e3)
#encoder block 4
e4 = Dense(5)(e3)
e4 = BatchNormalization()(e4)
e4 = Activation('selu')(e4)

 #Bottleneck
h1 = Dense(4)(e4)
h1 = BatchNormalization()(h1)
h1 = Activation('selu')(h1)
#decoder block 0
d0 = Dense(5)(h1)
d0 = BatchNormalization()(d0)
d0 = Activation('selu')(d0)
#decoder block 1
d1 = Dense(6)(d0)
d1 = BatchNormalization()(d1)
d1 = Activation('selu')(d1)
#decoder block 2
d2 = Dense(7)(d1)
d2 = BatchNormalization()(d2)
d2 = Activation('selu')(d2)
#decoder block 3
d3 = Dense(8)(d2)
d3 = BatchNormalization()(d3)
d3 = Activation('selu')(d3)
#decoder block 4
d4 = Dense(8)(d3)
d4 = BatchNormalization()(d4)
d4 = Activation('selu')(d4)
outputs = Dense(n_inputs, activation = 'selu')(d4)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer=Adam(1e-5), loss='mse')
autoencoder.summary()

#encoder only
encoderonly = Model(inputs=inputs, outputs=h1)


#The autoencoder training needs to have happened before, so its weight can be read at this point
encoderonly.load_weights('encoder_98765456789.hdf5')
autoencoder.load_weights('model_params_98765456789.hdf5')



sil_score_kmeans_ae = []
ch_score_kmeans_ae  =[]
db_score_kmeans_ae  = []
sil_score_gmm_ae  = []
ch_score_gmm_ae  =[]
db_score_gmm_ae  =[]

for i in flightdict.items():
    flight_scaled = sscaler.fit_transform(i[1])
    representation = encoderonly.predict(flight_scaled)
    
  
    kmeans.fit(representation) 
    labels = kmeans.labels_
    sil_score_kmeans_ae.append(silhouette_score(representation, labels, metric='euclidean'))
    ch_score_kmeans_ae.append(calinski_harabasz_score(representation, labels))
    db_score_kmeans_ae.append(davies_bouldin_score(representation, labels))
   
    gmm.fit(representation)
    labels = gmm.predict(representation)
    sil_score_gmm_ae.append(silhouette_score(representation, labels, metric='euclidean'))
    ch_score_gmm_ae.append(calinski_harabasz_score(representation, labels))
    db_score_gmm_ae.append(davies_bouldin_score(representation, labels))

#Compile scores for noeng
df_scores_ae = pd.DataFrame({'sil_kmeans': sil_score_kmeans_ae, 'sil_gmm': sil_score_gmm_ae, 'ch_kmeans': ch_score_kmeans_ae, 'ch_gmm':ch_score_gmm_ae, 'db_kmeans': db_score_kmeans_ae, 'db_gmm':db_score_gmm_ae })



#%% SR-20 flights final qualitative testing and plots
sr20_folder = './../XC/SR-20/' 

combined = np.zeros([1,6])

for f in os.listdir(sr20_folder):
    if f.startswith('log'):
        print(f)
        flight = pd.read_csv(sr20_folder+'/'+f,header = 2)
        flight = flight.apply(pd.to_numeric, errors='coerce').fillna(0)
        flight = flight[['  AltMSL','    IAS','    VSpd','  Pitch',' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4',
                    ' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4',' E1 RPM']]
        flight['avg_CHT'] = flight[[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4']].mean(axis=1) #############!!!!!
        flight=flight.drop(columns=[' E1 CHT1',' E1 CHT2',' E1 CHT3',' E1 CHT4'])
        
        flight['avg_EGT'] = flight[[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4']].mean(axis=1) #############!!!!!
        flight=flight.drop(columns=[' E1 EGT1',' E1 EGT2',' E1 EGT3',' E1 EGT4'])
        
        flight.columns = flight.columns.str.replace(' ', '')
        flight = flight[flight.IAS > 0]
        #flight = flight[flight.AltMSL>1000]
        flight = flight.reset_index(drop=True)
        flight['dAlt'] = flight['AltMSL'].diff()
        flight['dAlt'][0] = 0
        flight['MA_VSpd'] = flight['VSpd'].rolling(20).mean()
        flight['MA_VSpd'][0:20] = flight['VSpd'][0:20]
        keepalt = flight['AltMSL']
        flight=flight.drop(columns=['AltMSL'])
        flight_s = sscaler.fit_transform(flight)
        
  
        
        combined = np.vstack((combined, flight_s[:,[0,1,2,5,6,7]]))
        
       
combined = np.delete(combined, (0), axis=0)
flight_gmm = gmm.fit(combined)

#%%Flight 3
temp = flight3.drop(['AltMSL', 'avg_CHT', 'E1RPM'], axis = 1)
temp = sscaler.fit_transform(temp)
#temp_gmm = gmm.fit(temp)
labels = gmm.predict(temp)
flight3 = flight3.assign(gmm = labels)
flight3 = flight3.assign(AltMSL = keepalt3)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight3, 0)-1):
    if flight3['gmm'][i] == flight3['gmm'][i+1]:
        j = j + 1
    if flight3['gmm'][i] != flight3['gmm'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight3['gmm'][i]])))#last point
        j = 0
    if i == np.size(flight3, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight3['gmm'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<80):
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 80), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 80), None)
        if (k2 == None) & (k1 !=None):
            flight3['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight3['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight3['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight3['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]


        
kmeans.fit(temp) 
labels = kmeans.labels_    
flight3 = flight3.assign(kmeans = labels)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight3, 0)-1):
    if flight3['kmeans'][i] == flight3['kmeans'][i+1]:
        j = j + 1
    if flight3['kmeans'][i] != flight3['kmeans'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight3['kmeans'][i]])))#last point
        j = 0
    if i == np.size(flight3, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight3['kmeans'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<80):
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 80), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 80), None)
        if (k2 == None) & (k1 !=None):
            flight3['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight3['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight3['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight3['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]



import colorcet as cc
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight3.index[flight3['gmm'] == 2].tolist() ,y=flight3.query('gmm==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight3.index[flight3['gmm'] == 0].tolist() ,y=flight3.query('gmm==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight3.index[flight3['gmm'] == 1].tolist(),y=flight3.query('gmm==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight3.index[flight3['kmeans'] == 2].tolist() ,y=flight3.query('kmeans==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight3.index[flight3['kmeans'] == 0].tolist() ,y=flight3.query('kmeans==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight3.index[flight3['kmeans'] == 1].tolist(),y=flight3.query('kmeans==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%%Flight 4
#flight4 = flight4.drop(['gmm', 'kmeans'], axis = 1)
temp = flight4.drop(['AltMSL', 'avg_CHT', 'E1RPM'], axis = 1)
temp = sscaler.fit_transform(temp)
#temp_gmm = gmm.fit(temp)
labels = gmm.predict(temp)
flight4 = flight4.assign(gmm = labels)
flight4 = flight4.assign(AltMSL = keepalt4)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight4, 0)-1):
    if flight4['gmm'][i] == flight4['gmm'][i+1]:
        j = j + 1
    if flight4['gmm'][i] != flight4['gmm'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight4['gmm'][i]])))#last point
        j = 0
    if i == np.size(flight4, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight4['gmm'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        print(i)
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight4['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight4['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight4['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight4['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]


        
kmeans.fit(temp) 
labels = kmeans.labels_    
flight4 = flight4.assign(kmeans = labels)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight4, 0)-1):
    if flight4['kmeans'][i] == flight4['kmeans'][i+1]:
        j = j + 1
    if flight4['kmeans'][i] != flight4['kmeans'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight4['kmeans'][i]])))#last point
        j = 0
    if i == np.size(flight4, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight4['kmeans'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight4['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight4['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight4['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight4['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]



import colorcet as cc
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight4.index[flight4['gmm'] == 2].tolist() ,y=flight4.query('gmm==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight4.index[flight4['gmm'] == 0].tolist() ,y=flight4.query('gmm==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight4.index[flight4['gmm'] == 1].tolist(),y=flight4.query('gmm==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight4.index[flight4['kmeans'] == 2].tolist() ,y=flight4.query('kmeans==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight4.index[flight4['kmeans'] == 0].tolist() ,y=flight4.query('kmeans==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight4.index[flight4['kmeans'] == 1].tolist(),y=flight4.query('kmeans==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%%Flight 7
flight7 = flight7.drop(['gmm', 'kmeans'], axis = 1)
temp = flight7.drop(['AltMSL', 'avg_CHT', 'E1RPM'], axis = 1)
temp = sscaler.fit_transform(temp)
#temp_gmm = gmm.fit(temp)
labels = gmm.predict(temp)
flight7 = flight7.assign(gmm = labels)
flight7 = flight7.assign(AltMSL = keepalt7)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight7, 0)-1):
    if flight7['gmm'][i] == flight7['gmm'][i+1]:
        j = j + 1
    if flight7['gmm'][i] != flight7['gmm'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight7['gmm'][i]])))#last point
        j = 0
    if i == np.size(flight7, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight7['gmm'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        print(i)
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight7['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight7['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight7['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight7['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]


        
kmeans.fit(temp) 
labels = kmeans.labels_    
flight7 = flight7.assign(kmeans = labels)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight7, 0)-1):
    if flight7['kmeans'][i] == flight7['kmeans'][i+1]:
        j = j + 1
    if flight7['kmeans'][i] != flight7['kmeans'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight7['kmeans'][i]])))#last point
        j = 0
    if i == np.size(flight7, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight7['kmeans'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight7['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight7['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight7['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight7['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]



import colorcet as cc
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight7.index[flight7['gmm'] == 2].tolist() ,y=flight7.query('gmm==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight7.index[flight7['gmm'] == 0].tolist() ,y=flight7.query('gmm==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight7.index[flight7['gmm'] == 1].tolist(),y=flight7.query('gmm==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight7.index[flight7['kmeans'] == 2].tolist() ,y=flight7.query('kmeans==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight7.index[flight7['kmeans'] == 0].tolist() ,y=flight7.query('kmeans==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight7.index[flight7['kmeans'] == 1].tolist(),y=flight7.query('kmeans==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%%Flight 8
#flight8 = flight8.drop(['gmm', 'kmeans'], axis = 1)
temp = flight8.drop(['AltMSL', 'avg_CHT', 'E1RPM'], axis = 1)
temp = sscaler.fit_transform(temp)
#temp_gmm = gmm.fit(temp)
labels = gmm.predict(temp)
flight8 = flight8.assign(gmm = labels)
flight8 = flight8.assign(AltMSL = keepalt8)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight8, 0)-1):
    if flight8['gmm'][i] == flight8['gmm'][i+1]:
        j = j + 1
    if flight8['gmm'][i] != flight8['gmm'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight8['gmm'][i]])))#last point
        j = 0
    if i == np.size(flight8, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight8['gmm'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        print(i)
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight8['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight8['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight8['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight8['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]


        
kmeans.fit(temp) 
labels = kmeans.labels_    
flight8 = flight8.assign(kmeans = labels)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight8, 0)-1):
    if flight8['kmeans'][i] == flight8['kmeans'][i+1]:
        j = j + 1
    if flight8['kmeans'][i] != flight8['kmeans'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight8['kmeans'][i]])))#last point
        j = 0
    if i == np.size(flight8, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight8['kmeans'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight8['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight8['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight8['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight8['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]



import colorcet as cc
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight8.index[flight8['gmm'] == 2].tolist() ,y=flight8.query('gmm==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight8.index[flight8['gmm'] == 0].tolist() ,y=flight8.query('gmm==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight8.index[flight8['gmm'] == 1].tolist(),y=flight8.query('gmm==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight8.index[flight8['kmeans'] == 2].tolist() ,y=flight8.query('kmeans==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight8.index[flight8['kmeans'] == 0].tolist() ,y=flight8.query('kmeans==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight8.index[flight8['kmeans'] == 1].tolist(),y=flight8.query('kmeans==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%%Flight 1
#flight1 = flight1.drop(['gmm', 'kmeans'], axis = 1)
temp = flight1.drop(['AltMSL', 'E1RPM', 'avg_CHT'], axis = 1)
temp = sscaler.fit_transform(temp)
temp_gmm = gmm.fit(temp)
labels = gmm.predict(temp)
flight1 = flight1.assign(gmm = labels)
flight1 = flight1.assign(AltMSL = keepalt1)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight1, 0)-1):
    if flight1['gmm'][i] == flight1['gmm'][i+1]:
        j = j + 1
    if flight1['gmm'][i] != flight1['gmm'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight1['gmm'][i]])))#last point
        j = 0
    if i == np.size(flight1, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight1['gmm'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        print(i)
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight1['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight1['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight1['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight1['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]


        
kmeans.fit(temp) 
labels = kmeans.labels_    
flight1 = flight1.assign(kmeans = labels)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight1, 0)-1):
    if flight1['kmeans'][i] == flight1['kmeans'][i+1]:
        j = j + 1
    if flight1['kmeans'][i] != flight1['kmeans'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight1['kmeans'][i]])))#last point
        j = 0
    if i == np.size(flight1, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight1['kmeans'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight1['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight1['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight1['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight1['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]



import colorcet as cc
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight1.index[flight1['gmm'] == 2].tolist() ,y=flight1.query('gmm==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight1.index[flight1['gmm'] == 0].tolist() ,y=flight1.query('gmm==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight1.index[flight1['gmm'] == 1].tolist(),y=flight1.query('gmm==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight1.index[flight1['kmeans'] == 2].tolist() ,y=flight1.query('kmeans==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight1.index[flight1['kmeans'] == 0].tolist() ,y=flight1.query('kmeans==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight1.index[flight1['kmeans'] == 1].tolist(),y=flight1.query('kmeans==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%%Flight 2
flight2 = flight2.drop(['gmm', 'kmeans'], axis = 1)
temp = flight2.drop(['AltMSL', 'E1RPM', 'avg_CHT'], axis = 1)
temp = sscaler.fit_transform(temp)
temp_gmm = gmm.fit(temp)
labels = gmm.predict(temp)
flight2 = flight2.assign(gmm = labels)
flight2 = flight2.assign(AltMSL = keepalt2)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight2, 0)-1):
    if flight2['gmm'][i] == flight2['gmm'][i+1]:
        j = j + 1
    if flight2['gmm'][i] != flight2['gmm'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight2['gmm'][i]])))#last point
        j = 0
    if i == np.size(flight2, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight2['gmm'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        print(i)
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight2['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight2['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight2['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight2['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]


        
kmeans.fit(temp) 
labels = kmeans.labels_    
flight2 = flight2.assign(kmeans = labels)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight2, 0)-1):
    if flight2['kmeans'][i] == flight2['kmeans'][i+1]:
        j = j + 1
    if flight2['kmeans'][i] != flight2['kmeans'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight2['kmeans'][i]])))#last point
        j = 0
    if i == np.size(flight2, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight2['kmeans'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight2['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight2['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight2['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight2['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]



import colorcet as cc
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight2.index[flight2['gmm'] == 2].tolist() ,y=flight2.query('gmm==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight2.index[flight2['gmm'] == 0].tolist() ,y=flight2.query('gmm==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight2.index[flight2['gmm'] == 1].tolist(),y=flight2.query('gmm==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight2.index[flight2['kmeans'] == 2].tolist() ,y=flight2.query('kmeans==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight2.index[flight2['kmeans'] == 0].tolist() ,y=flight2.query('kmeans==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight2.index[flight2['kmeans'] == 1].tolist(),y=flight2.query('kmeans==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%%Flight 5
flight5 = flight5.drop(['gmm', 'kmeans'], axis = 1)
temp = flight5.drop(['AltMSL', 'avg_CHT', 'avg_EGT'], axis = 1)
temp = sscaler.fit_transform(temp)
temp_gmm = gmm.fit(temp)
labels = gmm.predict(temp)
flight5 = flight5.assign(gmm = labels)
flight5 = flight5.assign(AltMSL = keepalt5)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight5, 0)-1):
    if flight5['gmm'][i] == flight5['gmm'][i+1]:
        j = j + 1
    if flight5['gmm'][i] != flight5['gmm'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight5['gmm'][i]])))#last point
        j = 0
    if i == np.size(flight5, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight5['gmm'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        print(i)
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight5['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight5['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight5['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight5['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]


        
kmeans.fit(temp) 
labels = kmeans.labels_    
flight5 = flight5.assign(kmeans = labels)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight5, 0)-1):
    if flight5['kmeans'][i] == flight5['kmeans'][i+1]:
        j = j + 1
    if flight5['kmeans'][i] != flight5['kmeans'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight5['kmeans'][i]])))#last point
        j = 0
    if i == np.size(flight5, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight5['kmeans'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight5['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight5['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight5['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight5['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]



import colorcet as cc
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight5.index[flight5['gmm'] == 2].tolist() ,y=flight5.query('gmm==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight5.index[flight5['gmm'] == 0].tolist() ,y=flight5.query('gmm==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight5.index[flight5['gmm'] == 1].tolist(),y=flight5.query('gmm==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight5.index[flight5['kmeans'] == 2].tolist() ,y=flight5.query('kmeans==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight5.index[flight5['kmeans'] == 0].tolist() ,y=flight5.query('kmeans==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight5.index[flight5['kmeans'] == 1].tolist(),y=flight5.query('kmeans==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#%%Flight 6
flight6 = flight6.drop(['gmm', 'kmeans'], axis = 1)
temp = flight6.drop(['AltMSL', 'avg_CHT', 'E1RPM'], axis = 1)
temp = sscaler.fit_transform(temp)
temp_gmm = gmm.fit(temp)
labels = gmm.predict(temp)
flight6 = flight6.assign(gmm = labels)
flight6 = flight6.assign(AltMSL = keepalt6)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight6, 0)-1):
    if flight6['gmm'][i] == flight6['gmm'][i+1]:
        j = j + 1
    if flight6['gmm'][i] != flight6['gmm'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight6['gmm'][i]])))#last point
        j = 0
    if i == np.size(flight6, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight6['gmm'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        print(i)
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight6['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight6['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight6['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight6['gmm'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]


        
kmeans.fit(temp) 
labels = kmeans.labels_    
flight6 = flight6.assign(kmeans = labels)

segment_points = np.zeros([1,4])
j = 0

for i in range(0, np.size(flight6, 0)-1):
    if flight6['kmeans'][i] == flight6['kmeans'][i+1]:
        j = j + 1
    if flight6['kmeans'][i] != flight6['kmeans'][i+1]:
        segment_points = np.vstack((segment_points,np.array([i-j, i,j+1, flight6['kmeans'][i]])))#last point
        j = 0
    if i == np.size(flight6, 0)-2:
         segment_points = np.vstack((segment_points,np.array([i-j+1, i+1,j+1, flight6['kmeans'][i]])))#last point
         segment_points = np.delete(segment_points, 0, axis = 0)

segment_points = segment_points.astype(int)

for i in range(0, np.size(segment_points, 0)):
    if (segment_points[i,2]<50):
        k1 = next((k1 for k1 in range(i, np.size(segment_points, 0)) if segment_points[k1,2] >= 50), None)
        k2 = next((k2 for k2 in range(i,-1,-1) if segment_points[k2,2] >= 50), None)
        if (k2 == None) & (k1 !=None):
            flight6['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k1,3]
        if (k1 == None) & (k2 != None):
            flight6['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
        if (k1 != None) & (k2 != None):
            if segment_points[k1,3] == segment_points[k2,3]:
                flight6['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[k2,3]
            if segment_points[k1,3] != segment_points[k2,3]:
                findmaximum = np.max([k1,k2])
                flight6['kmeans'][range(segment_points[i,0], segment_points[i,1]+1)]  = segment_points[findmaximum,3]



import colorcet as cc
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight6.index[flight6['gmm'] == 2].tolist() ,y=flight6.query('gmm==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight6.index[flight6['gmm'] == 0].tolist() ,y=flight6.query('gmm==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight6.index[flight6['gmm'] == 1].tolist(),y=flight6.query('gmm==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x = flight6.index[flight6['kmeans'] == 2].tolist() ,y=flight6.query('kmeans==2')['AltMSL'],c='#8da0cb',s=5, label = 'cluster 1')
scatter = ax.scatter(x = flight6.index[flight6['kmeans'] == 0].tolist() ,y=flight6.query('kmeans==0')['AltMSL'],c='#66c2a5',s=5,  label = 'cluster 2')
scatter = ax.scatter(x = flight6.index[flight6['kmeans'] == 1].tolist(),y=flight6.query('kmeans==1')['AltMSL'],c='#fc8d62',s=5, label = 'cluster 3')
ax.set_xlabel(r'$\Delta T$ since takeoff (s)', fontsize = 20)
ax.set_ylabel(r'Altitude MSL (ft)',fontsize = 20)
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_yticklabels(ax.get_yticks(), size = 18)
ax.set_xticklabels(ax.get_xticks(), size = 18)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
