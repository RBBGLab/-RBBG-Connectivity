# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:55:09 2019

@author: tadam
"""
import pathlib

import random

import mne
import pandas as pd

import autoreject

from autoreject import AutoReject, Ransac

ar = AutoReject()

import logging
import sys
import typing
import functools
import datetime

import warnings


logger = logging.getLogger(__name__)


def echo_elapsed_time(echofunc:
                      typing.Optional[typing.Callable] = None
                      ) -> typing.Callable:
    def _echo_elapsed_time(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        def wrapper(*args: typing.Any, **kargs: typing.Any
                    ) -> typing.Any:
            start = datetime.datetime.now()
            retval = func(*args, **kargs)
            difftime = datetime.datetime.now() - start

            msg = f'Elapsed time for <{func.__name__}>: <{difftime}>.'
            if echofunc is None:
                print(msg)
            else:
                echofunc(msg)

            return retval

        return wrapper
    return _echo_elapsed_time


def suppress_errors(echofunc:
                    typing.Optional[typing.Callable] = None
                    ) -> typing.Callable:

    def _suppress_errors(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        def wrapper(*args: typing.Any, **kargs: typing.Any
                    ) -> typing.Any:
            try:
                retval = func(*args, **kargs)
            except Exception as e:
                msg = f'Suppressed error <{e}> for <{func.__name__}>.'
                if echofunc is None:
                    print(msg)
                else:
                    echofunc(msg)

                retval = None

            return retval

        return wrapper
    return _suppress_errors


##
#Individual aplha - from philistine.mne import savgol_iaf
#> help(savgol_iaf)
#>
# > savgol_iaf(raw)
#import mne_bids


#%%
main_folder = pathlib.Path().cwd()
montage = main_folder / 'Auxiliary' / '64-cap_new.bvef' #'BrainVision_Montage.bvef' #
montage = mne.channels.read_custom_montage(montage)

fwd = mne.read_forward_solution(main_folder/'Auxiliary/forward_solution_1026-fwd.fif')

folder = main_folder / pathlib.Path('RAW')

folder_output = pathlib.Path('P:/EEG-Data/output_epochs_ma_tomsk_moscow/')  #pathlib.Path('S:/OneDrive/PythonProjects/ArtifactsRejection/output_v3/')
#%%
# =============================================================================
# folder = pathlib.Path('P:/EEG_raw/')
# folders = [folder for folder in folder.iterdir() if folder.is_dir()]
# 
# files = []
# for folder in folders:
#     files_fold = [file for file in folder.iterdir() if file.is_file() and file.suffix == '.vhdr']
#     files.extend(files_fold)
# 
# =============================================================================

#%%
folder = pathlib.Path('P:/EEG-Data/EEG_raw/MA_Tomsk+Moscow/')
files = [file for file in folder.iterdir() if file.is_file() and file.suffix == '.vhdr']

#%%
#check files
good_files = []
bad_files = []
for file in files:
    print(f'Now: {file}')
    try:
        raw = mne.io.read_raw_brainvision(file, preload=True)
        good_files.append(file)
    except Exception as e:
        bad_files.append(file)
        print(e)
        continue
        
print(f'Good files: {len(good_files)} \n Bad files: {len(bad_files)}')
files = good_files

#%%
duration = 6.

reference = 'average'#'REST'

n_components = 48 #Количество компонент. Не может быть больше, чем количество каналов\n",
method = 'infomax' 
ica = mne.preprocessing.ICA(n_components=n_components, method=method)

channel_names = montage.ch_names
#%%
indices = {'zg': [[0, 120], 
              [240, 360],
              [480, 600]],
           'og': [[120, 240], 
              [260, 480]]}

#%%
@echo_elapsed_time(logger.debug)
#@suppress_errors(logger.warning)
def handle_file(file):
    ar = AutoReject() #set autoreject with step 1 - 1 second epoch interval
    tstep = 1.
    
    raw = mne.io.read_raw_brainvision(file, preload=True) #read file
    
    raw.drop_channels(list(set(raw.ch_names)-set(channel_names))) #there are some additional channels, VEOG or/and AUX. Deleting them 
    
    raw.set_montage(montage)
        
    raw.filter(0.1, 30) #filtering
    raw.resample(500.) #resample #check number of cores aailable here
    
    if reference != 'REST':
        raw.set_eeg_reference(reference, projection = True)
        raw.apply_proj()
    else:
        raw.set_eeg_reference(reference, forward = fwd)
    
    raw_length = raw.times.max()
    raw_parts = {}
    FILE_TOO_SHORT = False
    
    for key, time_cuts in indices.items():
        parts = []
        for cut in time_cuts:
            if cut[1]<raw_length:
            	part = raw.copy().crop(cut[0], cut[1])
            	parts.append(part)
            else:
                break
        
        if len(parts) == 0:
            warnings.warn('File is too short')
            FILE_TOO_SHORT = True
            break
        
        cut_raw = parts[0]
        for x in range(1, len(parts)):
        	cut_raw.append(parts[x])
            
        raw_parts[key] = cut_raw

    if FILE_TOO_SHORT == True:
        log = pd.DataFrame({'filename':file.stem,'Part':['too short'],'New_reference':['0'],'rejected_epochs':['0'],'missing_channels':['0']})
        return(log)

    file_log = pd.DataFrame(columns = ['filename','part','new_reference','rejected_epochs','missing_channels'])
    for name, raw_part in raw_parts.items():
        events=mne.make_fixed_length_events(raw_part,id=1,duration=1.)
        picks = mne.pick_types(raw_part.info, meg=False, eeg=True, eog=True)    
    
        
        if 'zg' not in name.lower():
            print ('Now ICA')
            epochs = mne.Epochs(raw_part, events=events, tmin=-0.3, tmax=1., 
                                baseline = (-0.3, 0.), picks=picks, preload = True)
            
            reject = autoreject.get_rejection_threshold(epochs)
            
            ica = mne.preprocessing.ICA(n_components = len(picks), method = 'infomax') 
    
            ica.fit(epochs, reject=reject, tstep=tstep)
            
            #Find ICA components. I apply ICA back to raw data instead of epochs, because of possible necessity to create epochs of different length 
            
            if 'IO' in raw.info['ch_names']:
                eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'ΙΟ')
            else:
                eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'Fp2')
            ica.exclude.extend(eog_inds)
            eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'AF8')  
            ica.exclude.extend(eog_inds)
            ica.apply(raw_part, exclude = eog_inds)
        
        channels_to_interpolate = list(set(channel_names)-set(raw_part.info['ch_names']))
        print(f'Interpolating channels: {channels_to_interpolate}')
        
        if len(channels_to_interpolate) > 0:
            dummy = raw_part.copy().pick_channels(random.sample(raw.info['ch_names'], len(channels_to_interpolate)))
    
            rename_dist = dict(zip(dummy.info['ch_names'], channels_to_interpolate))
            dummy.rename_channels(rename_dist)
            
            raw_part.add_channels([dummy])
            raw_part.info['bads'] = channels_to_interpolate
            raw_part.set_montage(montage)
            raw_part.interpolate_bads()
        
        events=mne.make_fixed_length_events(raw_part,id=1,duration=duration)
        picks = mne.pick_types(raw_part.info, meg=False, eeg=True, eog=False)
        epochs = mne.Epochs(raw_part, events=events, tmin=-0.3,
                    tmax=duration, baseline = (-0.3, 0.), picks=picks, preload = True)
        
        #Rejection of bad epochs
        ar = AutoReject(thresh_method = 'bayesian_optimization', verbose = False) #n_jobs = -1,
        rsc = Ransac(verbose = False ) #n_jobs = -1,
    
        epochs = rsc.fit_transform(epochs)  
        epochs_clean = ar.fit_transform(epochs) 
    
        rej_epochs = epochs_clean.drop_log_stats()
        
        output_filename = folder_output/f'{file.parts[-2]}_{file.stem}_{name}_{duration}-sec_epo.fif'
        epochs_clean.save(output_filename, overwrite = True)
        
        flog = pd.DataFrame({'filename':output_filename.stem, 'part': name, 'new_reference':reference, 'rejected_epochs': rej_epochs, 'missing_channels':[channels_to_interpolate]})
        file_log = pd.concat([file_log, flog])         
                
    return file_log
        


#%%

if __name__ == '__main__':
    progname = 'autoreject_ma_tomsk_moscow' #pathlib.Path(sys.argv[0]).stem
    logging.basicConfig(filename=f'{progname}.log',
                        level=logging.INFO,
                        format=('%(asctime)s '
                                ':: %(levelname)s '
                                ':: %(message)s'))

    mne.set_log_file(f'{progname}_mne.log',
                     output_format=('%(asctime)s '
                                    ':: %(levelname)s '
                                    ':: %(message)s'))
    
    
    log_name = 'log_autoreject_ma_tomsk_moscow_desc.csv'
    log = pd.DataFrame(columns = ['filename','part','new_reference','rejected_epochs','missing_channels'])
    log.to_csv(log_name, index=False)

    
    
    for file in files:
        #epochs_clean, epochs_clean_name, log = handle_file(file)
        #output = handle_file(file)
        #epochs_clean.save(epochs_clean_name, overwrite = True)
        print(f'Now file: {file.stem}')
        
        file_log = handle_file(file)
                
        logger.info(f'Handled {file.parts[-2]}_{file.stem}')
        file_log.to_csv(log_name, mode='a', header=False, index = False)
#%%
file = files[57]
ar = AutoReject() #set autoreject with step 1 - 1 second epoch interval
tstep = 1.

raw = mne.io.read_raw_brainvision(file, preload=True) #read file

raw.drop_channels(list(set(raw.ch_names)-set(channel_names))) #there are some additional channels, VEOG or/and AUX. Deleting them 

raw.set_montage(montage)

raw.filter(0.1, 30) #filtering
raw.resample(500.) #resample

if reference != 'REST':
    raw.set_eeg_reference(reference, projection = True)
    raw.apply_proj()
else:
    raw.set_eeg_reference(reference, forward = fwd)

raw_length = raw.times.max()
raw_parts = {}
FILE_TOO_SHORT = False

for key, time_cuts in indices.items():
    parts = []
    for cut in time_cuts:
        if cut[1]<raw_length:
        	part = raw.copy().crop(cut[0], cut[1])
        	parts.append(part)
        else:
            break

if len(parts) == 0:
    warnings.warn('File is too short')
    FILE_TOO_SHORT = True

cut_raw = parts[0]
for x in range(1, len(parts)):
	cut_raw.append(parts[x])
    
raw_parts[key] = cut_raw

if FILE_TOO_SHORT == True:
    log = pd.DataFrame({'filename':file.stem,'Part':['too short'],'New_reference':['0'],'rejected_epochs':['0'],'missing_channels':['0']})

file_log = pd.DataFrame(columns = ['filename','part','new_reference','rejected_epochs','missing_channels'])
for name, raw_part in raw_parts.items():
    events=mne.make_fixed_length_events(raw_part,id=1,duration=1.)
    picks = mne.pick_types(raw_part.info, meg=False, eeg=True, eog=True)    


    if 'zg' not in name.lower():
        print ('Now ICA')
        epochs = mne.Epochs(raw_part, events=events, tmin=-0.3, tmax=1., 
                            baseline = (-0.3, 0.), picks=picks, preload = True)
        
        reject = autoreject.get_rejection_threshold(epochs)
    
        ica.fit(epochs, reject=reject, tstep=tstep)
        
        #Find ICA components. I apply ICA back to raw data instead of epochs, because of possible necessity to create epochs of different length 
        
        if 'IO' in raw.info['ch_names']:
            eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'ΙΟ')
        else:
            eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'Fp2')
        ica.exclude.extend(eog_inds)
        eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'AF8')  
        ica.exclude.extend(eog_inds)
        ica.apply(raw_part, exclude = eog_inds)
    
    channels_to_interpolate = list(set(channel_names)-set(raw_part.info['ch_names']))
    print(f'Interpolating channels: {channels_to_interpolate}')
    
    if len(channels_to_interpolate) > 0:
        dummy = raw_part.copy().pick_channels(random.sample(raw.info['ch_names'], len(channels_to_interpolate)))
    
        rename_dist = dict(zip(dummy.info['ch_names'], channels_to_interpolate))
        dummy.rename_channels(rename_dist)
        
        raw_part.add_channels([dummy])
        raw_part.info['bads'] = channels_to_interpolate
        raw_part.set_montage(montage)
        raw_part.interpolate_bads()
    
    events=mne.make_fixed_length_events(raw_part,id=1,duration=duration)
    picks = mne.pick_types(raw_part.info, meg=False, eeg=True, eog=False)
    epochs = mne.Epochs(raw_part, events=events, tmin=-0.3,
                tmax=duration, baseline = (-0.3, 0.), picks=picks, preload = True)
    
    #Rejection of bad epochs
    ar = AutoReject(thresh_method = 'bayesian_optimization', verbose = False) #n_jobs = -1,
    rsc = Ransac(verbose = False ) #n_jobs = -1,
    
    epochs = rsc.fit_transform(epochs)  
    epochs_clean = ar.fit_transform(epochs) 
    
    rej_epochs = epochs_clean.drop_log_stats()
    
    output_filename = folder_output/f'{file.parts[-2]}_{file.stem}_{name}_{duration}-sec_epo.fif'
    epochs_clean.save(output_filename, overwrite = True)
    print(f'{output_filename} saved!')
    
    flog = pd.DataFrame({'filename':output_filename.stem, 'part': name, 'new_reference':reference, 'rejected_epochs': rej_epochs, 'missing_channels':[channels_to_interpolate]})
    file_log = pd.concat([file_log, flog])   


        
        #%%
  
#%%
# =============================================================================
# log = pd.DataFrame(columns = ['filename','Part','New_reference','rejected_epochs','missing_channels'])
# log.to_csv('log_autoreject.csv', index=False)
# 
# for file in files:    
#     ar = AutoReject() #set autoreject with step 1 - 1 second epoch interval
#     tstep = 1.
#     
#     raw = mne.io.read_raw_brainvision(file, preload=True) #read file
#     
#     raw.drop_channels(list(set(raw.ch_names)-set(channel_names))) #there are some additional channels, VEOG or/and AUX. Deleting them 
#     
#     raw.set_montage(montage)
#         
#     raw.resample(500.) #resample
#     raw.filter(0.1, 30) #filtering
#     
#     if reference != 'REST':
#         raw.set_eeg_reference(reference, projection = True)
#         raw.apply_proj()
#     else:
#         raw.set_eeg_reference(reference, forward = fwd)
#     
#     raw_length = raw.times.max()
#     raw_parts = {}
#     for key, time_cuts in indices.items():
#         parts = []
#         for cut in time_cuts:
#             if cut[1]<raw_length:
#             	cut = raw.copy().crop(cut[0], cut[1])
#             	parts.append(cut)
#             else:
#                 break
#         
#         cut_raw = parts[0]
#         for x in range(1, len(parts)):
#         	cut_raw.append(parts[x])
#             
#         raw_parts[key] = cut_raw
# 
# 
#     for name, raw_part in raw_parts.items():
#         events=mne.make_fixed_length_events(raw,id=1,duration=1.)
#         picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False)
#         
#                 
#         if 'zg' not in name.lower():
#             print ('Now ICA')
#             epochs = mne.Epochs(raw_part, events=events, tmin=-0.3, tmax=1., 
#                                 baseline = (-0.3, 0.), picks=picks, preload = True)
#             
#             reject = autoreject.get_rejection_threshold(epochs)
#     
#             ica.fit(epochs, reject=reject, tstep=tstep)
#             
#             #Find ICA components. I apply ICA back to raw data instead of epochs, because of possible necessity to create epochs of different length 
#             
#             if 'IO' in raw.info['ch_names']:
#                 eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'ΙΟ')
#             else:
#                 eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'Fp2')
#             ica.exclude.extend(eog_inds)
#             eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'AF8')  
#             ica.exclude.extend(eog_inds)
#             ica.apply(raw_part, exclude = eog_inds)
#         
#         channels_to_interpolate = list(set(channel_names)-set(raw_part.info['ch_names']))
#         
#         if len(channels_to_interpolate) > 0:
#             dummy = raw_part.copy().pick_channels(random.sample(raw.info['ch_names'], len(channels_to_interpolate)))
#     
#             rename_dist = dict(zip(dummy.info['ch_names'], channels_to_interpolate))
#             dummy.rename_channels(rename_dist)
#             
#             raw_part.add_channels([dummy])
#             raw_part.info['bads'] = channels_to_interpolate
#             raw_part.set_montage(montage)
#             raw_part.interpolate_bads()
#         
#         events=mne.make_fixed_length_events(raw_part,id=1,duration=duration)
#         picks = mne.pick_types(raw_part.info, meg=False, eeg=True, eog=False)
#         epochs = mne.Epochs(raw_part, events=events, tmin=-0.3,
#                     tmax=duration, baseline = (-0.3, 0.), picks=picks, preload = True)
#         
#         #Rejection of bad epochs
#         ar = AutoReject(thresh_method = 'bayesian_optimization', verbose = False) #n_jobs = -1,
#         rsc = Ransac(verbose = False ) #n_jobs = -1,
#     
#         epochs = rsc.fit_transform(epochs)  
#         epochs_clean = ar.fit_transform(epochs) 
# 
#         rej_epochs = epochs_clean.drop_log_stats()
#         
#         output_filename = folder_output/f'{file.parts[-2]}_{file.stem}_{name}_{duration}-sec_epo.fif'
#         epochs_clean.save(output_filename, overwrite = True)
#         
#         file_log = pd.DataFrame({'filename':file.stem, 'Part': name, 'New_reference':reference, 'rejected_epochs': rej_epochs, 'missing_channels':[channels_to_interpolate]})
#         log = pd.concat([log, file_log])
#         log.to_csv('log_autoreject.csv', mode='a', header=False)
# 
#             
# =============================================================================



#%%
# =============================================================================
# OLD
# =============================================================================
# =============================================================================
# 
# 
# #%%
# # =============================================================================
# # Main cycle
# # =============================================================================
# #%%
# for file in files:
#     ar = AutoReject() #set autoreject with step 1 - 1 second epoch interval
#     tstep = 1.
#     
#     raw = mne.io.read_raw_brainvision(file, preload=True) #read file
#     
#     raw.drop_channels(list(set(raw.ch_names)-set(channel_names))) #there are some additional channels, VEOG or/and AUX. Deleting them 
#     
#     raw.set_montage(montage)
#         
#     raw.resample(500.) #resample
#     raw.filter(0.1, 30) #filtering
#     
#     raw.set_eeg_reference(reference, projection = False)
#     
#     print('First reject!')
#     
#     #Here we need to set first, global threshold to remove the worst artifacts in our data before the ICA
#     #Global threshold is set on 1-sec intervals
#     events=mne.make_fixed_length_events(raw,id=1,duration=1.)
#     picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False)
#     epochs = mne.Epochs(raw, events=events, tmin=-0.3,
#                     tmax=1., baseline = (-0.3, 0.), picks=picks, preload = True)
#     
#     print('Now ICA')
#     #Here we computing the global threshold and fit the ICA
#     reject = autoreject.get_rejection_threshold(epochs)
#     ica.fit(epochs, reject=reject, tstep=tstep)
#     
#     #Find ICA components. I apply ICA back to raw data instead of epochs, because of possible necessity to create epochs of different length 
#     eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'ΙΟ')  
#     ica.exclude.extend(eog_inds)
#     eog_inds, _ = ica.find_bads_eog(epochs, ch_name = 'AF8')  
#     ica.exclude.extend(eog_inds)
#     ica.apply(raw, exclude = eog_inds)
#     
#     print('ICA complete!')
#     
#     #In some files we have missing channels - one or two. We need to interpolate them
#     #To interpolate channels, missing in the oroginal data, we need to create an injection - add this channels to the original data
#     #Here I create this injection as subset of one or two channels from the original data (there is no particular choise of channels,
#     #   we could create them manually, but this is easier)
#     #Then I add this subset back into the original data with names of the missing channels
#     #And then we have our channels, we could interpolate them as normal bad channels
#     
#     channels_to_interpolate = list(set(channel_names)-set(raw.info['ch_names']))
#     
#     if len(channels_to_interpolate) ==2:
#         dummy = raw.copy().pick_channels(['Fp1', 'T7'])
#     else:
#         dummy = raw.copy().pick_channels(['Fp1'])
# 
#     rename_dist = dict(zip(dummy.info['ch_names'], channels_to_interpolate))
#     dummy.rename_channels(rename_dist)
#     
#     raw.add_channels([dummy])
#     raw.info['bads'] = channels_to_interpolate
#     raw.set_montage(montage)
#     raw.interpolate_bads()
#     
#     #Here I remake epochs data with our needed duration 
#     events=mne.make_fixed_length_events(raw,id=1,duration=duration)
#     picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False)
#     epochs = mne.Epochs(raw, events=events, tmin=-0.3,
#                 tmax=duration, baseline = (-0.3, 0.), picks=picks, preload = True)
#     
#     #Rejection of bad epochs
#     ar = AutoReject()
#     epochs_clean = ar.fit_transform(epochs) 
#     
#     
#     #and save file
#     output_filename = folder_output/f'{file.stem}_{duration}-sec_epo.fif'
#     epochs_clean.save(output_filename, overwrite = True)
#     
#     
#     ####
#     #Here we have a little trick to return and save raw data simullaneously with epochs
#     #We could mark all bad epochs on the original raw data
#     # first, we identify indices of bad epochs
#     indices_of_rejected_epochs = list(set(epochs.selection) - set(epochs_clean.selection))
#     
#     #second - we calculate start mark for each bad epoch
#     onset = [x *duration for x in  indices_of_rejected_epochs]
#     #duration_ev = [1]*len(indices_of_rejected_epochs)\
#     
#     #third - we apply new annotations to our raw data
#     annotations = mne.Annotations(onset = onset, duration = duration, description = 'bad')
#     raw.set_annotations(annotations)
#     
#     #and save raw file
#     output_filename = folder_output/f'{file.stem}-raw.fif'
#     raw.save(output_filename)
#     
# =============================================================================
