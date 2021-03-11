# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 13:47:29 2019

@author: tadam
"""
import mne
import pandas as pd
import numpy as np

import scipy
import sklearn

import os
from os import listdir, getcwd
from os.path import isfile, join

import matplotlib.pyplot as plt

from mne.minimum_norm import (apply_inverse_epochs)
from mne.event import make_fixed_length_events
from mne.connectivity import spectral_connectivity
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import time

from scipy.signal import hilbert

import networkx as nx
import community

#%%
#main_folder = 'C:/Users/tadam/OneDrive/PythonProjects/Connectivity/'
main_folder = 'F:/OneDrive/PythonProjects/Connectivity/'

folder_input = 'F:/OneDrive/PythonProjects/Speech/Preprocessed_raws/' #main_folder+'Network Analysis/Input/'
folder_input = main_folder+'Network Analysis/Input/'


#%%
montage = mne.channels.read_montage(main_folder+"/Network Analysis/Auxiliary/64-cap_new.bvef", unit = 'mm')
#montage = mne.channels.read_montage('F:/OneDrive/PythonProjects/Speech/BrainVision_Montage.bvef', unit ='auto')

#Путь к файлу с forward solution. 
fwd = mne.read_forward_solution(main_folder+"/Network Analysis/Auxiliary/forward_solution_1026-fwd.fif")
#Путь к папке с обратными операторами
inverse_operator_folder = main_folder+"/Network Analysis/Inverse_operators/"

#%%
onlyfiles = [f for f in listdir(folder_input) if isfile(join(folder_input, f))]
onlyfiles= pd.DataFrame(onlyfiles)
onlyfiles = onlyfiles[0].str.split('_')
people = [x[0] for x in onlyfiles]
people = pd.unique(people)


#%%
def artifacts_supression (raw,art):
    onset = (art[' Position'][art[' Description']!=' Blink'])/raw.info['sfreq']
    duration = (art[' Length'][art[' Description']!=' Blink'])/raw.info['sfreq']
    annotations = mne.Annotations(onset = onset, duration = duration, description = 'bad')
    raw.set_annotations(annotations)
    return(raw)

#Режет файл на эпохи
def epoching(raw, duration, overlap = 0):
    tmin = 0
    tmax = duration
    duration = duration - overlap
    events=mne.make_fixed_length_events(raw,id=1,duration=duration)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False)
    epochs = mne.Epochs(raw, events=events, tmin=tmin,
                    tmax=tmax, baseline = (0,0), picks=picks, reject_by_annotation=True)
    epochs.drop_bad()
    return epochs   
    
#%%
#Synchrotools functions for Kuramoto order
    
def global_order_parameter_spacetime_slow(spacetime):
    """Returns the Kuramoto order parameter for an NxT numpy array.
    Parameters:
    -----------
    spacetime: NxT numpy array (N: number of nodes, T: time)
               Each row of represents a timeseries of the corresponding node.
    Returns:
    --------
    r: Global order parameter (Scalar number)
    """
    sum_theta = np.sum(np.exp(0.0 + 1j*spacetime), axis=0)
    r = abs(sum_theta) / (1.0*spacetime.shape[0])
    return r


def global_order_parameter_spacetime(spacetime):
    """Returns the Kuramoto order parameter for an NxT numpy array.
    It is faster than the global_order_parameter_slow
    Parameters:
    -----------
    spacetime: NxT numpy array (N: number of nodes, T: time).
               Each row represents a timeseries of the corresponding node.
    Returns:
    --------
    r: Global order parameter (Scalar number)
    """
    scos = np.cos(spacetime).sum(axis=0)
    ssin = np.sin(spacetime).sum(axis=0)
    r = np.sqrt((scos*scos + ssin*ssin)) / (1.0*spacetime.shape[0])
    return r


def local_order_parameter_space(space, order):
    """Returns the Kuramoto order parameter for an Nx1 numpy array
    within a certain neighborhood.
    Parameters:
    -----------
    space: Nx1 numpy array (N: number of nodes)
           It contains the phases of the N nodes.
    order: Number of neighboring nodes to the left and to the right
           taken into account in the calculation of the order parameter.
    Returns:
    --------
    local_r: Local order parameter (Vector with N entries)
    """
    N = space.shape[0]
    local_r = np.zeros(N)
    list_i = np.array([i for i in range(N)])

    for i in list_i:
        sum_theta = np.exp(0.0 + 1j * np.zeros_like(space))
        j_delta = 0
        list_j = np.array([j for j in range(i-order, i+order)])
        for j in list_j:
            # check the boundaries
            if (j < 0):
                j_delta = j + N
            elif (j > (N-1)):
                j_delta = j - N
            else:
                j_delta = j
            # summarize all the neighbors
            if (i != j):
                sum_theta += np.exp(0.0 + 1j*space[j_delta])
            # take the absolute value
            local_r[i] = (1.0/(2.0*order)) * abs(np.sum(sum_theta, axis=0))
    return local_r


def local_order_parameter_spacetime(spacetime, order):
    """Returns the Kuramoto order parameter for a NxT numpy array
    within a certain neighborhood.
    Parameters:
    -----------
    spacetime: NxT numpy array (N: Number of nodes, T: time)
               Each row represents a timeseries of the corresponding node.
    order: Number of neighboring nodes to the left and to the right
           taken into account in the calculation of the order parameter.
    Returns:
    --------
    local_r: Local order parameter for each time step (NxT numpy array)
    """
    space, time = spacetime.shape
    local_r = np.zeros(space)
    times = np.array([t for t in range(time)])
    local_r = np.zeros_like(spacetime)
    for t in times:
        local_r[:, t] = local_order_parameter_space(spacetime[:, t], order)
    return local_r

#%%
# synchtools chimera_index
def chimera_index(spacetime, membership):
    """Chimera-like index.
    spacetime: NxT numpy array (N: Number of the nodes , T: Time).
               Each row represents a timeseries of the corresponding node.
    membership: 1xN numpy array with the indices of the communities,
                e.g. [0,0,3,3,3,3,2,0,0,2,0,1,2,1,2,2,3]
    Nodes in spacetime and membership must have the same ordering
    Returns:
    --------
    chimera_val: Chimera-like index (scalar value)
    """
    # Find the unique elements of membership. They are the communities indices
    community_index = np.unique(membership)

    # dictionary for the index of the nodes in each community
    # keys  : the community index
    # values: the index of the nodes
    partition = {}
    for m in community_index:
        partition[m] = list(np.argwhere(np.array(membership) == m).flatten())

    # dictionary for the order parameters of each community
    # keys  : the community index
    # values: the order parameters
    order_parameter = {}
    for m in community_index:
        order_parameter[m] = global_order_parameter_spacetime(spacetime[partition[m]])

    # dictionary for the mean (over communities) order parameters
    # for each time step
    # keys  : the time snapshots
    # values: the mean order parameters
    mean_order_parameter = {}
    for t in range(len(order_parameter[0])):
        # mean over communities for each time step t
        mean_over_communities = 0.0
        for m in community_index:
            mean_over_communities += order_parameter[m][t]
        mean_order_parameter[t] = mean_over_communities / len(community_index)

    # dictionary for the (chimera-like) variance at each time step t
    # keys  : the community index
    # values: the chimera-like index
    sigma_chimera = {}
    for t in range(len(order_parameter[0])):
        tmp = 0.0
        for m in community_index:
            tmp += (order_parameter[m][t] - mean_order_parameter[t])**2
        sigma_chimera[t] = tmp / len(community_index)

    # The chimera index is the mean of the sigma_chimera over all communities
    chimera_val = np.mean(np.fromiter(sigma_chimera.values(), dtype=float))
    return chimera_val    

# synchtools metastability index    
def metastability_index(spacetime, membership):
    """Metastability index.
    Parameters:
    -----------
    spacetime: NxT numpy array (N: Number of the nodes , T: Time).
               Each row represents a timeseries of the corresponding node.
    membership: 1xN numpy array with the indices of the communities,
                e.g. [0,0,3,3,3,3,2,0,0,2,0,1,2,1,2,2,3]
    Nodes in spacetime and membership must have the same ordering
    Returns:
    --------
    metastability_val: Metastabiity index (scalar value)
    """
    # Find the unique elements of membership.
    # They are the indices of the communities.
    community_index = np.unique(membership)

    # dictionary for the index of the nodes in each community
    # keys  : Community index
    # values: Index of the nodes
    partition = {}
    for m in community_index:
        partition[m] = list(np.argwhere(np.array(membership) == m).flatten())

    # dictionary for the order parameters of each community
    # keys  : Community index
    # values: Order parameters
    order_parameter = {}
    for m in community_index:
        order_parameter[m] = global_order_parameter_spacetime(spacetime[partition[m]])

    # dictionary for the (metastability) variance of the order_parameter
    # in each community
    # keys  : Community index
    # values: Metastability index, i.e. the variance of the order parameters
    sigma_metastable = {}
    for m in community_index:
        sigma_metastable[m] = np.var(order_parameter[m])
        # sigma_metastable[m] = np.sum((order_parameter[m] -
        #                               np.mean(order_parameter[m])
        #                               )**2
        #                              ) / (len(order_parameter[m])-1)

    # The metastability index is the mean of the sigma_metastable
    # over all communities
    metastability_val = np.mean(np.fromiter(sigma_metastable.values(), dtype=float))
    return metastability_val    

#%%
def met_and_chim (spacetime, membership, data_raw = True):
    """Chimera-like index and metastability. Handmade!
    spacetime: NxT numpy array (N: Number of the nodes , T: Time).
               Each row represents a timeseries of the corresponding node.
    membership: 1xN numpy array with the indices of the communities,
                e.g. [0,0,3,3,3,3,2,0,0,2,0,1,2,1,2,2,3]
    Nodes in spacetime and membership must have the same ordering
    
    data_raw: set to False if data is already in radians
    
    Returns:
    --------
    chimera_val: Chimera-like index (scalar value)
    metastability_val: Metastabiity index (scalar value)
       
    """
    spacetime = np.unwrap(np.angle(hilbert(spacetime)))
    
    # Find the unique elements of membership. They are the communities indices
    community_index = np.unique(membership)

    # dictionary for the index of the nodes in each community
    # keys  : the community index
    # values: the index of the nodes
    partition = {}
    for m in community_index:
        partition[m] = list(np.argwhere(np.array(membership) == m).flatten())

    # dictionary for the order parameters of each community
    # keys  : the community index
    # values: the order parameters
    order_parameter = {}
    for m in community_index:
        order_parameter[m] = global_order_parameter_spacetime(spacetime[partition[m]])
    
    syncdf = np.empty((0, spacetime.shape[1]))
    for m in community_index:
        syncdf = np.vstack((syncdf, order_parameter[m]))    
    
    ch = np.mean(np.var(syncdf, axis = 0))
    met = np.mean(np.var(syncdf, axis = 1))


    ch_norm = ch/(5/36)
    met_norm = met/(1/12)
    
    return(ch, met, ch_norm, met_norm)

#%%
# functional coupling score

def coupling(data,window):

    """

        creates a functional coupling metric from 'data'

        data: should be organized in 'time x nodes' matrix

        smooth: smoothing parameter for dynamic coupling score

    """
    #define variables
    [tr,nodes] = data.shape
    der = tr-1
    td = np.zeros((der,nodes))
    td_std = np.zeros((der,nodes))
    data_std = np.zeros(nodes)
    mtd = np.zeros((der,nodes,nodes))
    sma = np.zeros((der,nodes*nodes))

    #calculate temporal derivative
    for i in range(0,nodes):
        for t in range(0,der):
            td[t,i] = data.iloc[t+1,i] - data.iloc[t,i]
    #standardize data
    for i in range(0,nodes):
        data_std[i] = np.std(td[:,i])
    td_std = td / data_std

    #functional coupling score
    for t in range(0,der):
        for i in range(0,nodes):
            for j in range(0,nodes):
                mtd[t,i,j] = td_std[t,i] * td_std[t,j]

    #temporal smoothing
    temp = pd.DataFrame(np.reshape(mtd,[der,nodes*nodes]))
    sma = temp.rolling(window).mean()
    sma = np.reshape(sma.as_matrix(),[der,nodes,nodes])
    
    return (mtd, sma)


#%%
def calc_phase_sync (data):
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
        
    def phase_sync(x,y):
        return(1-np.sin(np.abs(x-y)/2))
    
    aux = data.shape
    res = np.empty((aux[1], aux[1], aux[0]))
    
    data = np.apply_along_axis(hilbert, 0, data)
    data = np.apply_along_axis(np.angle, 0, data) 

    for i in range(0, aux[1]):
        for j in range(0, aux[1]):
            res[i, j, :] = phase_sync(data[:,i], data[:,j])
                
    return(res)


#%%
def global_metastability(data):
    return(np.nanvar(data))

#%%
#mtd, sma = coupling(data, 16)


#%%

# =============================================================================
# Main cycle
# =============================================================================
#%%
#Подсчет сенсоров, источников или обоих. Допустимые варианты: sensors, sources или both

con_method = 'wpli' #метод подсчета connectivity. В данной случае- weighted PLI
method = 'eLORETA' #метод локализации источников. В данном случае - dSPM

#Параметры деления на эпохи:
epochs_length = [0.5, 1.] # [2., 4., 6., 8., 10.] #6 #длина эпохи
overlap = 0 #перекрытие между эпохами

snr = 3. #Signal-to-noise ratio. Нужно для подсчета коэффициента регуляризации при подсчете источников
lambda2 = 1. / snr ** 2 #Коэффициент регуляризации

# Частоты для подсчетов
con_freqs = ['4-8', '8-13', '13-20','20-30', '4-30']
#%%
current = list(globals().keys())
current = [x for x in current if 'asdf' in x]
for x in current:
    del globals()[x]
    
all_asdf1 = [0, 600]

#ZG_asdf1 = [0, 120]
#ZG_asdf2 = [240, 360]
#ZG_asdf3 = [480, 600]

#OG_asdf1 = [120, 240]
#OG_asdf2 = [360, 482]

#%%
variabls = globals()
variabls = list(variabls.keys())
variabls = [x for x in variabls if 'asdf' in x]

var_split = [x.split('_')[0] for x in variabls]
uniques = list(Counter(var_split).keys())
uniquesv = list(Counter(var_split).values())

freqs_min = [int(x.split('-')[0]) for x in con_freqs]
freqs_max = [int(x.split('-')[1]) for x in con_freqs]
freqs = list(zip(freqs_min, freqs_max))

#%%
labels = mne.read_labels_from_annot('fsaverage')
labels = labels[:-1]
label_names = [label.name for label in labels]

#%%
output_filename = 'resting_state_chimera_and_metastability_each_epoch_cluster_new_freqs.csv'
results = pd.DataFrame(columns=['name','freq', 'epochs_length' ,'n_epoch', 'chimera', 'metastability', 'chimera_norm', 'metastability_norm'])
results.to_csv(output_filename)

#indices = np.tril_indices(63)


for person in people:
    try:
        #person = people[i]
        fname = folder_input + str(person)+ '_Connectivity.fif'  #+ '_Connectivity.fif'
        
        #folder_input + str(person) + '_Connectivity.fif'
        #path = folder_input + str(person) + '_Markers'
        # read EEG-file into raw, artifacts file into art
        raw = mne.io.read_raw_fif(fname, preload = True) # mne.io.read_raw_edf(fname, preload=True)
        
        #raw.set_montage(montage) #применяем схему монтажа к нашим товарищам
        raw.set_eeg_reference("average", projection=False) 
        
        # Supress artifacts
        #art = pd.read_csv(path, header=1) #читаем файл с маркерами
        #raw = artifacts_supression(raw, art) 
        
        elements = {}      
        for number in range(0,len(uniques)): #Создаем словарь с элементами записи
            aux = list()
            uni = uniques[number]
            univ = uniquesv[number]
            for number2 in range(1, univ+1):
                times = eval(uni+'_asdf'+str(number2)) 
                aux_raw = raw.copy().crop(times[0], times[1])
                aux.append(aux_raw)
            aux_full = aux[0]
            [aux_full.append(part) for part in aux[1:]]
            elements[uni] = aux_full  
            del aux_full, aux_raw     
            
        for part in elements:
            reeg = elements[part]
            
            for freq in freqs:
                reeg = reeg.filter(freq[0], freq[1])
    
                for length in epochs_length:
                
                    epochs = epoching(reeg, length)
                    
                    #people_res = pd.DataFrame()
                    for epoch in enumerate(epochs):
                        print('Now: ', str(part), str(length) ,str(epoch[0]))
                        
                        data =  epoch[1] #.to_data_frame()
                        
                        adj_matrix = calc_phase_sync(data.T)   
                        adj_matrix = np.mean(adj_matrix , axis = 2)
                        
                        indices = np.tril_indices(adj_matrix.shape[0])
                        adj_matrix[indices] = 0
                        flat = adj_matrix.flatten()
                        
                        t = np.percentile(flat[flat>0], 75)
                        adj_matrix[adj_matrix < t] = 0
                        
                        graph = nx.from_numpy_matrix(adj_matrix)
                        
                        partitions = np.fromiter(community.best_partition(graph).values(), dtype=float)
                       
                        
                        #data = data #np.array(data.T)
                        ch, met, ch_norm, met_norm = met_and_chim(data, partitions)
    
                        auxres = pd.DataFrame({'name':person, 'freq' :[str(freq[0])+'-'+str(freq[1])],
                                               'epochs_length':[length] ,'n_epoch':epoch[0], 
                                               'chimera': [ch], 'metastability':[met], 
                                               'chimera_norm':[ch_norm], 'metastability_norm': [met_norm]})
                        #people_res = pd.concat([people_res, auxres], axis = 1)
                        auxres.to_csv(output_filename, mode='a', header=False)
                    
            
        #people_res.to_csv('resting_state_chimera_and_metastability.csv', mode='a', header=False)
        #with open('resting_state_chimera_and_metastability.csv', 'a') as csv_file:
        #    df.to_csv(csv_file, header=False)
            
    except Exception as e:
        print(str(e))
        continue
    
    
#%%
for person in people[0:5]:
    try:
        #person = people[i]
        fname = folder_input + str(person) + '_Connectivity.edf'
        path = folder_input + str(person) + '_Markers'
        # read EEG-file into raw, artifacts file into art
        raw = mne.io.read_raw_edf(fname, preload=True)
        
        raw.set_montage(montage) #применяем схему монтажа к нашим товарищам
        raw.set_eeg_reference("average", projection=True) #новый референс ####CHECK
        
        # Supress artifacts
        art = pd.read_csv(path, header=1) #читаем файл с маркерами
        raw = artifacts_supression(raw, art) #разбираемся с артефа
        
        raw.to_data_frame().to_csv('raw/'+person+'.csv')
    except Exception as e:
        print(str(e))
        continue
    
    
#%%    
    
#%%
# =============================================================================
# Debug
# =============================================================================
    
#%%
indices = np.tril_indices(64)
#%%
person = people[0]    

fname = folder_input + str(person)+ '_Connectivity.fif'  #+ '_Connectivity.fif'
        
        #folder_input + str(person) + '_Connectivity.fif'
        #path = folder_input + str(person) + '_Markers'
        # read EEG-file into raw, artifacts file into art
raw = mne.io.read_raw_fif(fname, preload = True) # mne.io.read_raw_edf(fname, preload=Tru

#%%
raw.set_eeg_reference("average", projection=False) 

#%%
reeg = elements[part]

#%%
reeg = reeg.filter(freq[0], freq[1])
#%%

fname = folder_input + str(person) + '_Connectivity.edf'
path = folder_input + str(person) + '_Markers'
# read EEG-file into raw, artifacts file into art
raw = mne.io.read_raw_edf(fname, preload=True)

raw.set_montage(montage) #применяем схему монтажа к нашим товарищам
raw.set_eeg_reference("average", projection=True) #новый референс ####CHECK

# Supress artifacts
art = pd.read_csv(path, header=1) #читаем файл с маркерами
raw = artifacts_supression(raw, art) #разбираемся с артефа

epochs = epoching(raw, 10.)

people_res = pd.DataFrame()
#%%
for item in enumerate(epochs):
    data = item[1] #.to_data_frame()
#%%
#data = epoch[1].T
adj_matrix = calc_phase_sync(data.T)
adj_matrix = np.mean(adj_matrix , axis = 2)
adj_matrix[indices] = 0
flat = adj_matrix.flatten()
t = np.percentile(flat[flat>0], 75)
adj_matrix[adj_matrix < t] = 0

graph = nx.from_numpy_matrix(adj_matrix)
partitions = np.fromiter(community.best_partition(graph).values(), dtype=float)

#%%
membership = partitions
spacetime = hilbert(data.copy())
spacetime = np.unwrap(np.angle(spacetime))

#%%
ch, met, ch_norm, met_norm = met_and_chim(spacetime, partitions)

#%%
# =============================================================================
# Create Synch_affinity clustering
# =============================================================================

data = raw.copy().crop(20., 30.).to_data_frame().T
adj_matrix = calc_phase_sync(data.T)
#%%
threshold = np.mean(adj_matrix[adj_matrix<1])

#%%




#%%%%
data = data  #np.array(data.T)
ch, met, ch_norm, met_norm = met_and_chim(data, partitions)
chim = chimera_index(data, partitions)
metastab = metastability_index(data, partitions)

#%%
partition = community.best_partition(graph)
pos = nx.spring_layout(graph)  # compute graph layout
plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
plt.axis('off')
nx.draw_networkx_nodes(graph, pos, node_size=600, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
nx.draw_networkx_edges(graph, pos, alpha=0.3)
plt.show(graph)

#%%
auxres = pd.DataFrame({'name':person, 'n_epoch':epoch[0], 
                       'chimera': [ch], 'metastability':[met], 
                       'chimera_norm':[ch_norm], 'metastability_norm': [met_norm], 
                       'chimera_synchtools': [chim], 'metastability_synchtools': [metastab]})
people_res = pd.concat([people_res, auxres], axis = 1)

#%%
folder_output_conn = 'F:/OneDrive/PythonProjects/Connectivity/Network Analysis/Sources_PCA_output/Connectivity/'
folder_output_ts = 'F:/OneDrive/PythonProjects/Connectivity/Network Analysis/Sources_PCA_output/TimeSeries/'


start = time.time()
for person in people:
    try:
        #person = people[i]
        fname = folder_input + str(person) + '_Connectivity.edf'
        path = folder_input + str(person) + '_Markers'
        # read EEG-file into raw, artifacts file into art
        raw = mne.io.read_raw_edf(fname, preload=True)
        
        raw.set_montage(montage) #применяем схему монтажа к нашим товарищам
        raw.set_eeg_reference("average", projection=True) #новый референс ####CHECK
        
        # Supress artifacts
        art = pd.read_csv(path, header=1) #читаем файл с маркерами
        raw = artifacts_supression(raw, art) #разбираемся с артефа
        
        elements = {}      
        for number in range(0,len(uniques)): #Создаем словарь с элементами записи
            aux = list()
            uni = uniques[number]
            univ = uniquesv[number]
            for number2 in range(1, univ+1):
                times = eval(uni+'_asdf'+str(number2)) 
                aux_raw = raw.copy().crop(times[0], times[1])
                aux.append(aux_raw)
            aux_full = aux[0]
            [aux_full.append(part) for part in aux[1:]]
            elements[uni] = aux_full  
            del aux_full, aux_raw     
        
        for part in elements:
            reeg = elements[part]
            epochs = epoching(reeg, epochs_length) #делим на эпохи. Эпохи перекрываются
            
    except Exception as e:
        print(str(e))
        continue
    
print(time.time() - start) #Итоговое время

#%%

# =============================================================================
# Some things
# =============================================================================
#%%
chimera_members = chimera_index(np.array(data.T), partition)
metastab = metastability_index(np.array(data.T), partition)

#%%
#partition = 64*[0]
#chimera_glob = chimera_index(np.array(data.T), partition)
#metastab_glob = metastability_index(np.array(data.T), partition)


#%%
data_sync = data.iloc[:, 4]
data_sync = pd.concat([data_sync, data_sync], axis = 1)
for i in range(0,5):
    data_sync = pd.concat([data_sync, data_sync], axis = 1)
data_sync = data_sync.T
data_async = np.random.uniform(0.001, 0.99, (64, 5121))

#%%
chimera_snc = chimera_index(np.array(data_sync.T), partition)
metastab_snc = metastability_index(np.array(data_sync.T), partition)

#%%
chimera_asnc = chimera_index(data_async, partition)
metastab_asnc = metastability_index(data_async, partition)

#%%
data_chim = np.vstack((np.array(data_sync.iloc[0:32, :]),data_async[0:32, :]))   #.concat([data_sync.iloc[0:32, :], data_async[0:32, :]], axis = 0)

#%%
partitionc = [0]*32
partitioncd = [1]*32
partitionc = [partitionc, partitioncd]


chimera_chim = chimera_index(data_async, partitionc)
metastab_chim = metastability_index(data_async, partitionc)
