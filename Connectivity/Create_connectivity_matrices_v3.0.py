import os
import inspect
import sys


from os import listdir, getcwd
from os.path import isfile, join
from collections import Counter
import itertools

import pandas as pd

import mne
from mne.minimum_norm import (apply_inverse_epochs)
from mne.event import make_fixed_length_events
from mne.connectivity import spectral_connectivity

import time

import numpy as np

#main_folder = path_to_script = sys.path[0]

main_folder = getcwd()
#%%
#Определяем переменные

#В этой папке лежат начальные данные: файл ЭЭГ и файл с маркерами.
folder_input = main_folder+'/Input/'

#Оба файла должны иметь определенною схему наименования
#Первый файл - файл ЭЭГ .edf. Наименование строится, как: [уникальный идентифкатор испытуемого]_Connectivity.edf
#Второй файл - файл с маркерами. Схема наименования: [уникальный идентификатор испытуемого]_Markers
#Схема наименования важна для того, чтобы впоследствии корректно сформировался список испытуемых

#Эта папка вывода. В этой папке будут лежать файлы с коннективити-таблицами
folder_output = main_folder+'/Output/'

#Путь к файлу с монтажом. 
montage = mne.channels.read_custom_montage(main_folder+"/64-cap_new.bvef")
#Путь к файлу с forward solution. 
#fwd = mne.read_forward_solution(main_folder+"/Network Analysis/Auxiliary/forward_solution_1026-fwd.fif")
#Путь к папке с обратными операторами
#inverse_operator_folder = main_folder+"/Network Analysis/Inverse_operators/"

#%%
#Подсчет сенсоров, источников или обоих. Допустимые варианты: sensors, sources или both
sensors_or_sources = 'sensors'

con_method = 'wpli' #метод подсчета connectivity. В данной случае- weighted PLI
method = 'eLORETA' #метод локализации источников. В данном случае - dSPM

#Параметры деления на эпохи:
epochs_length = 6 #длина эпохи
overlap = 0.5 #перекрытие между эпохами

# Частоты для подсчетов
con_freqs = ['4-8', '8-13', '13-20','20-30', '4-30']
#%%
current = list(globals().keys())
current = [x for x in current if 'asdf' in x]
for x in current:
    del globals()[x]

#################################################
#В этом окне есть возможность указать интервалы для деления записи.
# =============================================================================
# Каждый элемент указывается следующим образом:
#     1. Название отрезка
#     2. Кодовый элемент
#     3. Номер куска в отрезке
#     
# Для каждого элемента прописывается время начала и время окончания интересующего отрезка в записи.
# 
# Пример:
#     zg_asdf1 - отрезок называется zg, этот кусок будет первым
# 
# Все куски в рамках одного отрезка соединяются в один в том порядке, в каком они указаны:
#     zg_asdf1 и zg_asdf2 будут вырезаны из сырой записи, соеденены в один отрезок zg, сначала zg_asdf_1, потом zg_asdf2
# 
# Количество интересующих интервалов и количество элементов в интервалах может быть любым
# =============================================================================

part1_asdf1 = [0, 60]
part1_asdf2 = [61, 120]

part2_asdf1 = [121, 180]

part3_asdf1 = [123, 145]




#%%
fo_exist = (os.path.exists(folder_output))

if not fo_exist:
    dir_name = main_folder +'/Output/'
    if not os.path.exists(dir_name): 
        os.mkdir(dir_name)
    folder_output = dir_name

#%%
#Эта часть скрипта подгружает список участников
onlyfiles = [f for f in listdir(folder_input) if isfile(join(folder_input, f))]
onlyfiles= pd.DataFrame(onlyfiles)
onlyfiles = onlyfiles[0].str.split('_')
people = [x[0] for x in onlyfiles]
people = pd.unique(people)

# In[18]:
people #Проверим правильность подгрузки, все ли на месте и нет ли дублеров

#%%
existing_files = [f for f in listdir(folder_output) if isfile(join(folder_output, f))]
existing_files = pd.DataFrame(existing_files)
existing_files = existing_files[0].str.split('_')
existing_people = [x[0] for x in existing_files]
existing_people = dict(Counter(existing_people))
existing_people = [k for k,v in existing_people.items() if int(v) == 20]
people = set(people)-set(existing_people)

#%%
existing_files = [f for f in listdir(folder_output) if isfile(join(folder_output, f))]
files_to_remove = [[x for x in existing_files if k in x] for k in people]
files_to_remove = list(itertools.chain(*files_to_remove))
[os.remove(join(folder_output, filename)) for filename in files_to_remove]
        
#%%
if sensors_or_sources == 'both' or sensors_or_sources == 'sources':    
    fo_exist = (os.path.exists(inverse_operator_folder))
    files_in_iof = len([f for f in listdir(inverse_operator_folder) if isfile(join(inverse_operator_folder, f))])
    
    if not (os.path.exists(inverse_operator_folder)) or files_in_iof != 0:
        print ('Folder with inverse operators is missing or contain no files')
        print ('Current folder:   ', inverse_operator_folder)
        print ('Do you wish to proceed?')
        answer = input ('y/n?  ')
        if str(answer)[0].lower() == 'n':
            print('Please, specify correct folder with inverse operators')

# In[19]:
#Давит артефакты на основе файла с маркерами
def artifacts_supression (raw,art):
    onset = (art[' Position'])/raw.info['sfreq']
    duration = (art[' Length'])/raw.info['sfreq']
    art[' Description'][art[' Description']==' Userdefined'] = 'bad'
    
    annotations = mne.Annotations(onset = onset, duration = duration, description = art[' Description'])
    raw.set_annotations(annotations)
    return(raw)

#Режет файл на эпохи
def epoching(raw, duration):
    global overlap
    tmin = 0
    tmax = duration
    duration = duration - overlap
    events=mne.make_fixed_length_events(raw,id=1,duration=duration)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False)
    epochs = mne.Epochs(raw, events=events, tmin=tmin,
                    tmax=tmax, picks=picks, reject_by_annotation=True)
    epochs.drop_bad()
    return epochs   
    
#%%%
#Эта часть подготавливает словарь с вырезанными из записи элементами и подготавливает списки минимальных и максимальных частот в частотных диапазонах 
variabls = globals()
variabls = list(variabls.keys())
variabls = [x for x in variabls if 'asdf' in x]

var_split = [x.split('_')[0] for x in variabls]
uniques = list(Counter(var_split).keys())
uniquesv = list(Counter(var_split).values())

freqs_min = [x.split('-')[0] for x in con_freqs]
freqs_max = [x.split('-')[1] for x in con_freqs]

freqs_min = [int(x) for x in freqs_min]
freqs_max = [int(x) for x in freqs_max]

#%%
snr = 3. #Signal-to-noise ratio. Нужно для подсчета коэффициента регуляризации при подсчете источников
lambda2 = 1. / snr ** 2 #Коэффициент регуляризации
  
#%%
start = time.time()
for person in people:
    try:
        #person = people[i]
        fname = folder_input + str(person) + '_Connectivity.edf'
        path = folder_input + str(person) + '_Markers'
        # read EEG-file into raw, artifacts file into art
        raw = mne.io.read_raw_edf(fname, preload=True)
        
        raw.set_montage(montage) #применяем схему монтажа к нашим товарищам
        raw.set_eeg_reference("average", projection=False) #новый референс ####CHECK
        
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
            #del aux_full, aux_raw     
        
        for part in elements:
            reeg = elements[part]
            epochs = epoching(reeg, epochs_length) #делим на эпохи. Эпохи перекрываются
            
            if sensors_or_sources == 'sensors' or sensors_or_sources == 'both':
                con, freqs, times, n_epochs, n_tapers = spectral_connectivity(epochs, method=con_method, mode='multitaper', sfreq=raw.info['sfreq'], 
                                                                                                             fmin=freqs_min, fmax=freqs_max, 
                                                                                                             faverage=True, mt_adaptive=True, n_jobs=1)
                
                for shp in range(0, con.shape[2]):
                    np.savetxt(folder_output + str(person)+ '_' + str(part)+ '_' + str(epochs_length)+ '-sec_sensors_' + 
                               str(con_method)+ '_' + str(con_freqs[shp]) + '.csv', np.array(con[:, :, shp]), delimiter=',')
    
        
            if sensors_or_sources == 'sources' or sensors_or_sources == 'both':
                inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_folder+str(person)+'-inv.fif')
                stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                                                   pick_ori="normal", return_generator=True)
                
                con, freqs, times, n_epochs, n_tapers = spectral_connectivity(stcs, method=con_method, mode='multitaper', sfreq=raw.info['sfreq'], 
                                                                                                                 fmin=freqs_min, fmax=freqs_max, 
                                                                                                                 faverage=True, mt_adaptive=True, n_jobs=1)
        
                for shp in range(0, con.shape[2]):
                    np.savetxt(folder_output + str(person)+ '_' + str(part)+ '_' + str(epochs_length)+ '-sec_sources_' + 
                               str(con_method)+ '_'+ str(con_freqs[shp]) + '.csv', np.array(con[:, :, shp]), delimiter=',')
    except Exception as e:
        print(str(e))
        continue
    
print(time.time() - start) #Итоговое время

