
# coding: utf-8

# In[15]:


from os import listdir, getcwd
from os.path import isfile, join
import pandas as pd
import mne
from mne.minimum_norm import (apply_inverse_epochs)
from mne.event import make_fixed_length_events

import time

import numpy as np

# In[16]:
#main_folder = os.getcwd()

#%%
#Определяем переменные

#В этой папке лежат начальные данные: файл ЭЭГ и файл с маркерами.
folder_input = main_folder+'/Network Analysis/Input/'

#Оба файла должны иметь определенною схему наименования
#Первый файл - файл ЭЭГ .edf. Наименование строится, как: [уникальный идентифкатор испытуемого]_Connectivity.edf
#Второй файл - файл с маркерами. Схема наименования: [уникальный идентификатор испытуемого]_Markers
#Схема наименования важна для того, чтобы впоследствии корректно сформировался список испытуемых

#Эта папка вывода. В этой папке будет лежать файл с метриками и папки с коннективити-таблицами
folder_output = main_folder+'/Network Analysis/Inverse_operators/'

#Путь к файлу с монтажом. 
montage = mne.channels.read_montage(main_folder+'/Network Analysis/Auxiliary/64-cap_new.bvef', unit = 'auto')

#%%
#Путь к файлу с inverse operator. 
#inverse_operator = mne.minimum_norm.read_inverse_operator('/data/inverseoperator-inv.fif')

#Разные параметры
duration = 2.5 #Нужна для деления записи на эпохи
tmin=-3 #Нужна для деления записи на эпохи
tmax=3 #Нужна для деления записи на эпохи

# In[17]:
# =============================================================================
# This section is created for the creation of forward operator
# Skip if you already have one
# In order to calculate forward operator we need at least one raw file
# =============================================================================
#%%
subjects_dir = mne.datasets.fetch_fsaverage() 
subjects_dir = subjects_dir[:-9]
subject = 'fsaverage'
src = mne.setup_source_space(subject, spacing='oct5', subjects_dir=subjects_dir, add_dist=False)

conductivity = (0.3, 0.006, 0.3)
model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)

bem = mne.make_bem_solution(model)

onlyfiles = [f for f in listdir(folder_input) if isfile(join(folder_input, f))]
raw = mne.io.read_raw_edf(folder_input+'/'+onlyfiles[0], preload=True)

fwd = mne.make_forward_solution(raw.info, trans = 'fsaverage', src=src, 
                                bem=bem, meg=False, eeg=True)

mne.write_forward_solution(main_folder+'/Auxiliary/forward_solution_1026-fwd.fif', fwd)

#%%
# =============================================================================
# Here you can specify the path to the forward operator
# =============================================================================
#Путь к файлу с forward solution. 
fwd = mne.read_forward_solution(main_folder+'/Network Analysis/Auxiliary/forward_solution_1026-fwd.fif')

#%%
#Эта часть скрипта подгружает список участников
onlyfiles = [f for f in listdir(folder_input) if isfile(join(folder_input, f))]
onlyfiles= pd.DataFrame(onlyfiles)
onlyfiles = onlyfiles[0].str.split('_')
people = [x[0] for x in onlyfiles]
people = pd.unique(people)
#%%
existing_files = [f for f in listdir(folder_output) if isfile(join(folder_output, f))]
existing_files = pd.DataFrame(existing_files)
existing_files = existing_files[0].str.split('-')
existing_people = [x[0] for x in existing_files]
people = set(people) - set(existing_people)


# In[19]:
#Functions for including markers and creating epochs

#Давит артефакты на основе файла с маркерами
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

# In[22]:

start = time.time() #Начинаем
for i in people:
    #функция будет работать, пропуская ошибку, так что аккуратнее
    try:
        #choose person from list
        person = i
        # show path to EEG-file (fname) & artifatc
        fname = folder_input + str(person) + '_Connectivity.edf'
        path = folder_input + str(person) + '_Markers'
        # read EEG-file into raw, artifacts file into art
        raw = mne.io.read_raw_edf(fname, preload=True)

        raw.set_montage(montage) #применяем схему монтажа к нашим товарищам
        raw.set_eeg_reference("average", projection=True) #новый референс
        
        # Supress artifacts
        art = pd.read_csv(path, header=1) #читаем файл с маркерами
        raw = artifacts_supression(raw, art) #разбираемся с артефактами
        
        #epochs, averaging and noise_covariance matrix
        epochs = epoching(raw, 6.) 
        noise_cov = mne.compute_covariance(epochs, method = 'shrunk')
        evoked= epochs.average()
        
        inverse_operator = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)
        mne.minimum_norm.write_inverse_operator(folder_output+'/'+str(person)+'-inv.fif', inverse_operator)
    except Exception as e:
        print(str(e))
        continue

            
print(time.time() - start) #Итоговое время