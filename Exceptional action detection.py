import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
from utils_hs import *
import time
import pyts
import scipy
import scikitplot as skplt
import subprocess
from sklearn.cluster import KMeans
from pyts.approximation import PiecewiseAggregateApproximation
from statsmodels.tsa.seasonal import seasonal_decompose
from tslearn.metrics import dtw_path_from_metric
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import argparse
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

class timeseriesdatapreparation:
    def __init__(self, f, n, fx, alpha, df):
        self.f = f
        self.n = n
        self.fx = fx
        self.alpha = alpha
        self.df = df
        self.w = self._calculate_w(f, n, fx)
        self.w2 = int(self.w/2)
        self.wind = np.arange(0, len(df[0]), 500)
    
    def _calculate_w(self, f, n, fx):
        w = int(f*fx/n)
        return w
    
    def read_dataset(self):
        df = self.df
        return np.array(df.T)
    
    def read_and_normalize_dataset(self):
        df_init = self.read_dataset()
        df_normd = np.apply_along_axis(utils.MaxMinNorm, 1, df_init)
        return df_normd
    
    def data_compression(self, df_normd):
        paa = PiecewiseAggregateApproximation(window_size=self.n)
        df_normd_paa = paa.transform(df_normd)
        return df_normd_paa
        
    def trend_removal(self, df_normd_paa, df_normd):
        data_hs = df_normd_paa.copy()
        for i in range(len(df_normd)):
            res = seasonal_decompose(df_normd_paa[i], model='additive', period=self.w, extrapolate_trend='freq')
            new = df_normd_paa[i]/res.trend
            data_hs[i] = new
        return data_hs
        
    def calculate_distance_matrix(self, data_hs):
        x1 = utils.Recc_ED_sm_1(df=data_hs, sm=self.w)
        return x1
    
    def calculate_distance_matrix(self, data_hs):
        x1 = utils.Recc_ED_sm_1(df=data_hs, sm=self.w)
        return x1
        
    def calculate_lrec(self, x4):
        LREC = np.array([])
        w = self.w
        for p in range(w-1, len(x4)):
            LREC = np.append(LREC, np.mean(x4[p-w+1:(p+1), p-w+1:(p+1)]))
        return LREC
    
    def calculate_distance_matrix_all_sub(self, LREC):
        sub_id = np.array([0])
        start = 0
        for z in range(10_000):
            ww = self.w
            if start+ww >= len(LREC):
                sub_id = np.append(sub_id, len(LREC))
                break
            res = LREC[start:start+ww]
            a = np.where(res==np.max(res))
            b = np.where(res==np.min(res))
            c = np.max(np.append(a, b)) 
            sub_id = np.append(sub_id, c+start)
            start = (c+start).copy()
        subseq = []
        for i in range(1, len(sub_id)):
            subseq.append(LREC[(sub_id[i-1]):(sub_id[i])])
        sub_id = sub_id + self.w - 1
        sub_id_original = sub_id * self.n
        return subseq, sub_id, sub_id_original
    
    def distmat_alt(self, data_hs):
        x1 = euclidean_distances(data_hs[0].reshape(-1,1)).round(2)
        for i in range(1, len(data_hs)):
            x1 += euclidean_distances(data_hs[i].reshape(-1,1))
        x1 = x1/np.max(x1)
        return(x1)

    
def merge(cid, centers, maxLen):
    no = 0
    distances = []
    ij = []
    cid_list = np.unique(cid)
    for i in range(len(centers)):
        for j in range(len(centers)):
            if i != j :
                distances.append( dtw_path_from_metric(centers[i], centers[j])[1] )
                ij.append([cid_list[i], cid_list[j]])
    mean_distances = np.mean(distances)
    which = np.argmin(distances)
    if distances[which] < mean_distances/2 :
        cid[cid==ij[which][0]] = ij[which][1]
        cid_list = np.unique(cid)

        centers = np.empty((len(cid_list), maxLen, 1))
        model = TimeSeriesKMeans(n_clusters=1, metric="dtw", max_iter=1)
        for k in range(len(cid_list)):
            dat = data[ cid==cid_list[k] ]
            model.fit(data[ cid==cid_list[k] ])
            res = list(np.ravel(model.cluster_centers_)) 
            res = np.array(res + [np.nan]*((maxLen)-len(res))).reshape(1,maxLen,1)
            centers[k] = res
    else:
        no += 1
    return(cid, centers, no)

def split(cid, centers, maxLen):
    no = 0
    distances = []
    ij = []
    for i in range(len(centers)):
        for j in range(len(centers)):
            if i != j :
                distances.append( dtw_path_from_metric(centers[i], centers[j])[1] )
                ij.append([i,j])
    mean_distances = np.mean(distances)
    
    cid_list = np.unique(cid)
    for i in range(len(cid_list)):
        dat = data[cid==cid_list[i]]
        if len(dat>1):
            distance = []
            for j in range(len(dat)):
                distance.append(dtw_path_from_metric(dat[j], centers[i])[1])
            d_intra = np.max(distance)+np.min(distance)
            if d_intra <= mean_distances/2:
                no += 1
            else :
                X = data[cid==cid_list[i]] #data
                model = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=1000)
                model.fit(X)
                
                new_centers = np.empty((2, maxLen, 1))
                new_label = model.labels_+np.max(cid)+1
                for k in range(2):
                    res = list(np.ravel(model.cluster_centers_[k]))
                    res = np.array(res + [np.nan]*((maxLen)-len(res))).reshape(1,maxLen,1)
                    new_centers[k] = res
                centers = np.delete(centers, i, axis=0)
                centers = np.vstack((centers, new_centers))
                cid[cid==cid_list[i]] = new_label
                break
    return (cid, centers, no)



iterations = 100
total_bestacc = []
acc_iter=[]
TP_iter=[]
FP_iter=[]
TN_iter=[]
FN_iter=[]

name = ['filtered1']#"filtered1" 0.1'filtered2' 0.15 for low pass,entropy_v2
best_acc = 0
entropy_1 = np.load("./entropy/hammering_1_entropy.npy")
entropy_2 = np.load("./entropy/hammering_1_entropy_v2.npy")

if 'filtered1' in name[0]:
    b, a = scipy.signal.butter(3, 0.1)
    entropy_1_filtered = scipy.signal.filtfilt(b, a, entropy_1)
    entropy_2_filtered = scipy.signal.filtfilt(b, a, entropy_2)


elif 'filtered2' in name[0]:
    for cutoff in [.15]:
        b, a = scipy.signal.butter(3, cutoff)
        entropy_1_filtered = scipy.signal.filtfilt(b, a, entropy_1)
        entropy_2_filtered = scipy.signal.filtfilt(b, a, entropy_2)

df = pd.DataFrame()        
for i in range(2):
    if 'filtered1' in name[0]:
        df[0] = entropy_1_filtered[3:(len(entropy_1)-64)]
        print('filtered1')
    elif 'filtered2' in name[0]:
        df[0] = entropy_1_filtered[3:(len(entropy_1)-64)]
        print('filtered2')
    if 'entropy_v2' in name[0]:
        df[0] = entropy_1[3:(len(entropy_2)-64)]
        print('entropy_v2')
    if 'combined' in name[0]:
        df[0] = entropy_1_filtered[3:(len(entropy_1)-64)]
        df[1] = entropy_2_filtered[3:(len(entropy_2)-64)]
        print('combined')
    else:
        continue
label_df = pd.read_csv('./label/hammering_1.csv',encoding= 'unicode_escape')
start = label_df[label_df[label_df.columns[3]]=='others'].iloc[0].Frame
end = label_df[label_df[label_df.columns[3]]=='others'].iloc[-1].Frame
discords = []
discords = [[start-3, end-3]]
real_disc = np.array([])
for k in range(len(discords)):
    x = np.arange(discords[k][0],discords[k][1],1)
    real_disc = np.append(real_disc, x)
real_disc = real_disc.astype(int)


fx = 58


n = 1
f = 1
alpha = 0.5

for iteration in range(iterations):
    print('Iterations========================='+str(iteration)+'======================')
    test = timeseriesdatapreparation(f, n, fx, alpha, df)
    df_normd = test.read_and_normalize_dataset()
    df_normd_paa = test.data_compression(df_normd)
    data_hs = test.trend_removal(df_normd_paa, df_normd)
    x4 = test.distmat_alt(data_hs)
    LREC = test.calculate_lrec(x4)
    subseq, sub_id, sub_id_original = test.calculate_distance_matrix_all_sub(LREC)

    maxLen = 0
    for j in range(len(subseq)):
        if len(subseq[j]) > maxLen:
            maxLen = len(subseq[j])

    data = np.empty((len(subseq), maxLen, 1))
    for i in range(len(subseq)):
        if len(subseq[i]) < maxLen :
            elements = np.array( list(subseq[i]) + [np.nan]*((maxLen)-len(subseq[i])) )
        else :
            elements = np.array( list(subseq[i]) )
        data[i].T[0] = elements
    cid = np.array(range(len(data)))
    centers = data.copy()

    for a in range(300):
        count = 0
        cid, centers, no = merge(cid, centers, maxLen)
        cid, centers, no1 = split(cid, centers, maxLen)
        if no+no1 == 2 :
            count += 1
        if count > 3 :
            break
    ids, counts = np.unique(cid, return_counts=True)

    y = np.array(cid.copy())

    for i in range(len(ids)):
        y[y==ids[i]] = (1 - (counts[i]/np.sum(counts)))*10000
    y = y/10000
    y_01 = np.array([0]*len(y))
    y_01[y>=0.8] += 1

    pred = np.zeros([len(df)])
    for i in range(len(y_01)):
        if i == len(y_01)-1 and y_01[i] == 0:

            pred[sub_id[i]:sub_id[i+1]] = 0

        elif i == 0 and y_01[i] == 1:
            pred[sub_id[i]:sub_id[i+1]] = 1

        else:
            if y_01[i] == 0:
                pred[sub_id[i]:sub_id[i+1]] = 0

            elif y_01[i] == 1:
                pred[sub_id[i]:sub_id[i+1]] = 1

  
    labels = np.zeros([len(df)])
    labels[real_disc] = 1
    
    
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(sub_id[0],len(labels)):
        if labels[i]==0 and pred[i]==0:
            TP+=1
        elif labels[i]==1 and pred[i]==1:
            TN+=1
        elif labels[i]==1 and pred[i]==0:
            FN += 1
        elif labels[i]==0 and pred[i]==1:
            FP += 1
    acc = (TP+TN)/(TP+FP+FN+TN)

    print('-------------------ACCURACY: ',acc)

    if acc > best_acc:
        best_acc = acc
        print('-------------------Best ACCURACY: ',best_acc)
        print('\nBetter\n')
    ft1, ft2 = 25, 20
    x = np.arange(df_normd.shape[1])+1  
    total_bestacc.append(best_acc)
    acc_iter.append(acc)
    TP_iter.append(TP)
    FP_iter.append(FP)
    TN_iter.append(TN)
    FN_iter.append(FN)
    
result=pd.DataFrame({"total_bestacc":pd.Series(total_bestacc),"acc_iter":pd.Series(acc_iter),"TP_iter":pd.Series(TP_iter),"FP_iter":pd.Series(FP_iter),"TN_iter":pd.Series(TN_iter),"FN_iter":pd.Series(FN_iter)})
result.to_csv("comparation_result.csv")








