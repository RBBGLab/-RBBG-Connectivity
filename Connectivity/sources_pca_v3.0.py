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

import itertools

import os
from os import listdir, getcwd
from os.path import isfile, join


from mne.minimum_norm import (apply_inverse_epochs)
from mne.event import make_fixed_length_events
from mne.connectivity import spectral_connectivity
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import time

#%%
#main_folder = 'C:/Users/tadam/OneDrive/PythonProjects/Connectivity/'
main_folder = 'F:/OneDrive/PythonProjects/Connectivity'

folder_input = main_folder+'/Network Analysis/Input_old/'

folder_output_conn = folder_output = main_folder+'/Network Analysis/Sources_PCA_output/'

#%%
montage = mne.channels.read_montage(main_folder+"/Network Analysis/Auxiliary/64-cap_new.bvef", unit = 'mm')
#Путь к файлу с forward solution. 
fwd = mne.read_forward_solution(main_folder+"/Network Analysis/Auxiliary/forward_solution_1026-fwd.fif")
#Путь к папке с обратными операторами
inverse_operator_folder = main_folder+"/Network Analysis/Inverse_operators/"

#%%
#Подсчет сенсоров, источников или обоих. Допустимые варианты: sensors, sources или both

con_method = 'wpli' #метод подсчета connectivity. В данной случае- weighted PLI
method = 'dSPM' #метод локализации источников. В данном случае - dSPM

#Параметры деления на эпохи:
epochs_length = 6 #длина эпохи
overlap = 0.5 #перекрытие между эпохами

snr = 3. #Signal-to-noise ratio. Нужно для подсчета коэффициента регуляризации при подсчете источников
lambda2 = 1. / snr ** 2 #Коэффициент регуляризации

# Частоты для подсчетов
con_freqs = ['4-8', '8-13', '13-20','20-30', '4-30']

# Нужны ли матрицы коннективити? True- да,  False - нет
calculation_of_con_matrix = True

# Сохранять ли time-series? True- да,  False - нет
save_time_series = True

#%%
onlyfiles = [f for f in listdir(folder_input) if isfile(join(folder_input, f))]
onlyfiles= pd.DataFrame(onlyfiles)
onlyfiles = onlyfiles[0].str.split('_')
people = [x[0] for x in onlyfiles]
people = pd.unique(people)

#%%
existing_files = [f for f in listdir(folder_output) if isfile(join(folder_output, f))]
existing_files = [f for f in existing_files if con_method in f]
existing_files = [f for f in existing_files if str(epochs_length) in f]
existing_files = pd.DataFrame(existing_files)
existing_files = existing_files[0].str.split('_')
existing_people = [x[0] for x in existing_files]
existing_people = dict(Counter(existing_people))
existing_people = [k for k,v in existing_people.items() if int(v) == scipy.stats.mode(list(existing_people.values()))[0]]
people = set(people)-set(existing_people)

#%%
existing_files = [f for f in listdir(folder_output) if isfile(join(folder_output, f))]
files_to_remove = [[x for x in existing_files if k in x] for k in people]
files_to_remove = list(itertools.chain(*files_to_remove))
#%%
[os.remove(join(folder_output, filename)) for filename in files_to_remove]

#%%
def artifacts_supression (raw,art):
    onset = (art[' Position'])/raw.info['sfreq']
    duration = (art[' Length'])/raw.info['sfreq']
    art[' Description'][art[' Description']==' Userdefined'] = 'bad'
    
    annotations = mne.Annotations(onset = onset, duration = duration, description = art[' Description'])
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

# =============================================================================
# Main cycle
# =============================================================================

#%%
current = list(globals().keys())
current = [x for x in current if 'asdf' in x]
for x in current:
    del globals()[x]

ZG_asdf1 = [0, 120]
ZG_asdf2 = [240, 360]
ZG_asdf3 = [480, 600]

OG_asdf1 = [120, 240]
OG_asdf2 = [360, 480]

#%%
variabls = globals()
variabls = list(variabls.keys())
variabls = [x for x in variabls if 'asdf' in x]

var_split = [x.split('_')[0] for x in variabls]
uniques = list(Counter(var_split).keys())
uniquesv = list(Counter(var_split).values())

freqs_min = [int(x.split('-')[0]) for x in con_freqs]
freqs_max = [int(x.split('-')[1]) for x in con_freqs]

freqs = list(zip(freqs_max, freqs_min))
#%%
labels = mne.read_labels_from_annot('fsaverage')
labels = labels[:-1]
label_names = [label.name for label in labels]
#%%
#folder_output_conn = main_folder+'/Network Analysis/Sources_PCA_output/Connectivity/'
#folder_output_ts = main_folder+'/Network Analysis/Sources_PCA_output/TimeSeries/'

#%%
if calculation_of_con_matrix == False and save_time_series == False:
    import warnings
    warnings.warn("Script won`t do anything. Both calculation of connectivity matrices and extiraction of time series are disabled")


#%%
start = time.time()
for person in people[0:1]:
    try:
        #person = people[i]
        fname = folder_input + str(person) + '_Connectivity.edf'
        path = folder_input + str(person) + '_Markers'
        # read EEG-file into raw, artifacts file into art
        raw = mne.io.read_raw_edf(fname, preload=True)
        
        raw.set_montage(montage) #применяем схему монтажа к нашим товарищам
        raw.set_eeg_reference("average", projection=True) 
        
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
        
        inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_folder+str(person)+'-inv.fif')
        
        for part in elements:
            reeg = elements[part]
            for freq in freqs:
                reeg.filter(freq[1], freq[0])
                
                
                if save_time_series == True:
                    stcs = mne.minimum_norm.apply_inverse_raw(reeg, inverse_operator, lambda2, method,
                                                       pick_ori="normal")
                
                    labels_ts = mne.extract_label_time_course(stcs, labels, inverse_operator['src'], mode = 'pca_flip')

                    np.savetxt(folder_output+ str(person)+ '_' + str(part)+ '_' + str(freq[1])+ '-'+str(freq[0])+'_ts.csv', 
                           labels_ts, delimiter=',') 
                
                
                                
                if calculation_of_con_matrix == True:
                    epochs = epoching(reeg, epochs_length)
                    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                                                       pick_ori="normal")
                
                    labels_ts = mne.extract_label_time_course(stcs, labels, inverse_operator['src'], mode = 'pca_flip')

                    con, _, _, _, _ = spectral_connectivity(labels_ts, method=con_method, mode='multitaper', sfreq=raw.info['sfreq'], 
                                        fmin = [freq[1]], fmax = [freq[0]], faverage=True, mt_adaptive=True, n_jobs=1)
                    np.savetxt(folder_output_conn + str(person)+ '_' + str(part)+ '_' + str(epochs_length)+ '-sec_' + str(con_method)+'_sources-epochs_' + 
                                str(freq[1])+ '-'+str(freq[0]) + '.csv', np.array(con[:, :, 0]), delimiter=',')
                
                
                
    
    except Exception as e:
        print(str(e))
        continue
#%%
