# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:33:45 2020

@author: PC
"""

import pathlib

import random

import mne
import pandas as pd
import numpy as np

import networkx as nx
import community as community_louvain

#%%
#Upload files
main_folder = pathlib.Path().cwd()

folder_input = pathlib.Path('P:/chcon_output/')#pathlib.Path('S:/OneDrive/PythonProjects/ArtifactsRejection/output_v3/')
folder_output =  folder_output = pathlib.Path('P:/chcon_conn/')#main_folder/'con_matrices' 

files = [x for x in folder_input.iterdir() if x.is_file()]

#%%
sensors_output_filename = 'sensors_measures2-1.csv'
sources_output_filename = 'sources_measures2-1.csv'

method = 'eLORETA' 

con_method = 'imcoh' 
con_freqs = ['1-4', '4-8', '8-13', '8-10', '10-13', '13-20', '20-30', '1-30']

thresholds = [0.5, 0.6, 0.7, 0.8]

bem = mne.read_bem_solution('Auxiliary/bem_oct6')
src = mne.read_source_spaces('Auxiliary/src_oct6')

#‘aparc’ or ‘aparc.a2009s’. 'HCPMMP1',
labels = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s') #aparc - 68 labels, aparc 2009 - 250

#%%

freqs_min = [int(x.split('-')[0]) for x in con_freqs]
freqs_max = [int(x.split('-')[1]) for x in con_freqs]


def quantile_threshold(adj, quantile = 0.5, binarization = False):
    #if isinstance(data, (np.array)) == False:
    #    adj = np.array(adj)
    import numpy as np
    
    adj = abs(adj)
    np.fill_diagonal(adj, 0)
    adj = adj+np.transpose(adj)#np.triu(adj)
    threshold = np.quantile(adj[adj != 0], quantile)
    adj[adj<threshold] = 0
    
    #if binarization == True:
    #    adj[adj>0] = 1
    
    return(adj)

# =========
def extract_cpl (path_generator):
    aux = list(path_generator)
    
    list_of_paths = []
    for x in aux:
        paths = list(x[1].values())
        list_of_paths.extend(paths[1:])
        
    return(np.median(list_of_paths))

def participation_coefficient(G, module_partition):
    '''
    Computes the participation coefficient of nodes of G with partition
    defined by module_partition.
    (Guimera et al. 2005).

    Parameters
    ----------
    G : :class:`networkx.Graph`
    module_partition : dict
        a dictionary mapping each community name to a list of nodes in G

    Returns
    -------
    dict
        a dictionary mapping the nodes of G to their participation coefficient
        under the participation specified by module_partition.
    '''
    # Initialise dictionary for the participation coefficients
    pc_dict = {}

    # Loop over modules to calculate participation coefficient for each node
    for m in module_partition.keys():
        # Create module subgraph
        M = set(module_partition[m])
        for v in M:
            # Calculate the degree of v in G
            degree = float(nx.degree(G=G, nbunch=v))

            # Calculate the number of intramodule degree of v
            wm_degree = float(sum([1 for u in M if (u, v) in G.edges()]))

            # The participation coeficient is 1 - the square of
            # the ratio of the within module degree and the total degree
            pc_dict[v] = 1 - ((float(wm_degree) / float(degree))**2)

    return pc_dict


def participation_coef(W, ci):
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    
    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot(np.squeeze(np.asarray((W != 0))), np.squeeze(np.asarray(np.diag(ci))))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors
    
    for i in range(int(np.max(ci))):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))
    
    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0
    
    return P


def nodes_in_partition(partition):
    N = max(partition.values())
    output = []
    for x in range(N):
        nodes = dict(filter(lambda elem: elem[1] == x, partition.items())).keys()
        output.append(set(nodes))
    return(output)


def modularity(G, communities, weight="weight"):
    out_degree = dict(G.degree(weight=weight))
    deg_sum = sum(out_degree.values())
    m = deg_sum / 2
    norm = 1 / deg_sum ** 2

    def community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum = out_degree_sum

        return L_c / m - out_degree_sum * in_degree_sum * norm

    return sum(map(community_contribution, communities))

def calculate_GCS_and_LCS(adj_matrix):
    GCS = np.mean(adj_matrix[adj_matrix>0])
    
    adj_matrix = (adj_matrix+adj_matrix.T)/2
    
    adj_list = [[y for y in x if y != 0] for x in np.transpose(adj_matrix.tolist())]
    LCS = [np.average(x) for x in adj_list]
        
    return (GCS, LCS)


def calculate_measures(graph):
    measures = {}
    cpl = nx.shortest_path_length(graph, weight = 'weight')
    measures['cpl'] = extract_cpl(cpl)
    measures['clust_coef_mean'] = nx.algorithms.cluster.average_clustering(graph, weight = 'weight')
    
    partition = community_louvain.best_partition(graph)
    partition_nodes = nodes_in_partition(partition)
    
    measures['modularity'] = modularity(graph, partition_nodes)
    
    measures['part_index_mean'] = np.mean(participation_coef(nx.to_numpy_array(graph), list(partition.values())))
    measures['betw_centrality_mean'] = np.mean(list(nx.algorithms.centrality.betweenness_centrality(graph, weight = 'weight').values()))
    measures['eigenvector_centrality_mean'] = np.mean(list(nx.algorithms.centrality.eigenvector_centrality(graph, max_iter=1000, weight = 'weight').values()))
    
    
    measures = pd.DataFrame(measures, index = [0])
    
    
    partition = community_louvain.best_partition(graph)
    partition_nodes = nodes_in_partition(partition)
    partition_nodes = dict(zip(range(0, 4), partition_nodes))
    
    part_coef = pd.DataFrame.from_dict(participation_coefficient(graph, partition_nodes), orient='index' , columns = ['participation-coef'])
    betw_centr = pd.DataFrame.from_dict(nx.algorithms.centrality.betweenness_centrality(graph, weight = 'weight'), orient='index' , columns = ['betwenness-centrality'])
    eigen_centr = pd.DataFrame.from_dict(nx.algorithms.centrality.eigenvector_centrality(graph, max_iter=1000, weight = 'weight'), orient='index' , columns = ['eigenvector-centrality'])
    
    
    degrees = list(graph.degree())
    degrees = pd.DataFrame.from_dict(dict((x, y) for x, y in degrees), orient='index', columns = ['node-degree'])
    
    clustering_coef = pd.DataFrame.from_dict(nx.algorithms.cluster.clustering(graph, weight = 'weight'), orient='index' , columns = ['clust-coef'])
    kcore = pd.DataFrame.from_dict(nx.algorithms.core.core_number(graph), orient='index', columns = ['kcore'])
    
    local_df = pd.concat([part_coef, betw_centr, eigen_centr, clustering_coef, kcore, degrees], axis=1)
    local_df.columns = ['participation_coef', 'betwenness-centr', 'eigenvector-centr', 'clust-coef', 'kcore', 'node-degree']
    
    local_df.index = clustering_coef.index
    
    local_df = local_df.unstack().to_frame().sort_index(level=1).T
    local_df.columns = local_df.columns.map('{0[0]}_{0[1]}'.format)  #res_df_new.columns.map('_'.join)

    measures = pd.concat([measures, local_df], axis = 1)
    
    return(measures)

def graph_measures_from_con_matrices(matrices, thresholds, node_names, con_freqs, file, folder__output):
    if len(node_names) == 64:
        sen_or_sour = 'sensors'
    elif len(node_names) == 150 or len(node_names) == 68:
        sen_or_sour = 'sources'
    else:
        print(f'Length of node names does not match number of sensors or sources') #Увеличить читаемость
        
    
    one_file_measures = pd.DataFrame()
    for shp in range(0, matrices.shape[2]):
        output_filename = pathlib.Path(f'{file.stem[:-4]}_{sen_or_sour}_{con_method}_{con_freqs[shp]}.csv')
        
        print(f'Now frequency: {con_freqs[shp]}')
        
        #print(f'Now file: {file.stem}')
        ts_conn = con[:, :, shp]
        np.savetxt(folder_output/output_filename, ts_conn) #rework
        
        for threshold in thresholds:
            print(f'Now threshold: {threshold}')
            ts_adj = quantile_threshold(ts_conn, threshold)
            
            #names = epochs.info['ch_names']
            
            adj_pd = pd.DataFrame(ts_adj, index = node_names, columns = node_names)
            graph = nx.from_pandas_adjacency(adj_pd)       
            
            graph_was_cut = False
            if len(list(nx.connected_components(graph))) != 1:
                print('Main graph have two components, extracting bigger one')
                graph_was_cut = True
                Gc = max(nx.connected_components(graph), key=len)
                graph = graph.subgraph(Gc)
            
            
            #partition = np.fromiter(community_louvain.best_partition(graph).values(), dtype=float)
            
            GCS, LCS = calculate_GCS_and_LCS(np.array(adj_pd))
            names = ['LCS_'+x for x in node_names]
            LCS = dict(zip(names, LCS))
            
            G_LCS = pd.DataFrame({'GCS':GCS}, index = [0])
            LCS = pd.DataFrame.from_dict(LCS, orient = 'index').T
            
            G_LCS = pd.concat([G_LCS, LCS], axis = 1)
            
            output = pd.DataFrame({'file':output_filename, 'threshold':[threshold], 'isolated_nodes_cut':graph_was_cut}, index = [0])
            output = pd.concat([output, G_LCS], axis = 1)
            
            graph_measures = calculate_measures(graph)
            
            output = pd.concat([output, graph_measures], axis = 1)
            #result = pd.concat([graph_measures, output], axis = 1) 
            one_file_measures = pd.concat([one_file_measures, output])
    
    
    return(one_file_measures)


snr = 3.
lambda2 = 1. / snr ** 2 

#labels = mne.read_labels_from_annot('fsaverage', 'aparc')
#, ‘aparc’ or ‘aparc.a2009s’. 'HCPMMP1',
labels = mne.read_labels_from_annot('fsaverage', 'aparc.a2009s')
#Destrieux Atlas
#labels = labels[:-1] #only if aparc
label_names = [label.name for label in labels] 

#%%
#calculation of whole-epoch connectivity on sensors
conn_measures_sensors = pd.DataFrame()
conn_measures_sources = pd.DataFrame()

for file in files:
    epochs = mne.read_epochs(file)
    
    con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epochs, 
                                                                  method=con_method, mode='multitaper', sfreq=epochs.info['sfreq'], 
                                                                  fmin=freqs_min, fmax=freqs_max, faverage=True, 
                                                                  mt_adaptive=True, n_jobs=-1)
    
        
    sensors_measures = graph_measures_from_con_matrices(con, thresholds, epochs.info['ch_names'], con_freqs, file, folder_output)
    
    fwd = mne.make_forward_solution(epochs.info, trans = 'fsaverage', src=src, 
                                bem=bem, meg=False, eeg=True)  


    noise_cov = mne.compute_covariance(epochs, method = 'shrunk')
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov,
                                                 loose=0.2, depth=0.8) 
    
    epochs.stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2, method, pick_ori="normal")
    
    labels_ts = mne.extract_label_time_course(epochs.stcs, labels = labels, src = src, allow_empty=True,
                                                  mode = 'pca_flip', trans = 'fsaverage')

    con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(labels_ts, 
                                                                  method=con_method, mode='multitaper', sfreq=epochs.info['sfreq'], 
                                                                  fmin=freqs_min, fmax=freqs_max, faverage=True, 
                                                                  mt_adaptive=True, n_jobs=-1)

    sources_measures = graph_measures_from_con_matrices(con, thresholds, label_names, con_freqs, file, folder_output)
    
    conn_measures_sensors = pd.concat([conn_measures_sensors, sensors_measures])
    conn_measures_sources = pd.concat([conn_measures_sensors, sources_measures])

conn_measures_sensors.to_csv(sensors_output_filename, index = False)
conn_measures_sources.to_csv(sources_output_filename, index = False)
