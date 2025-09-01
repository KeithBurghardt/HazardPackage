import os
#from utils import mergeNetworks
import json
import os
from glob import glob
import pandas as pd
import numpy as np
import networkx as nx
import time
from coURL import coURL
from coRetweet import coRetweet
from fastRetweet import fastRetweet
from hashtagSeq import hashSeq
from textSimilarity import textSim
from coSharing import coSharing
from utils import mergeNetworks, computeCentrality_nx

# create dict: ID 2 timestamp
# calculate time diff = data['timePublished'] - id2timestamp[parentId]
def retweet_diff(df):
    id2timestamp = {tweet_id:time for tweet_id,time in df[['tweetId','timePublished']].values}
    all_ids = set(list(id2timestamp.keys()))
    rt_parent_ts = []
    for parent,engagement in df[['engagementParentId','engagementType']].values:
        ts = np.nan
        if engagement == 'retweet':
            if parent in all_ids:
                ts = id2timestamp[parent]
        rt_parent_ts.append(ts)
    rt_parent_ts = np.array(rt_parent_ts)
    rt_diff = df['timePublished'].values - rt_parent_ts
    # in our datasets, time is in milliseconds, so we convert to seconds
    rt_diff /= 1000
    df['rt_diff'] = rt_diff
    return df


for ii,data_name in enumerate(['french']):# 'hamas',
    #data_name = ['hamas','french'][ii]
    #df = df.sample(100000)
    print(data_name)
    if 'df' in globals():
        del df
    if data_name == 'hamas':
        df = pd.concat([pd.read_csv('hamas_israel_challenge_problem_all_20240229_simplified.csv',sep='\t'),
pd.read_csv('challenge_problem_two_21NOV_simplified.csv',sep='\t')])
        df = df.drop_duplicates(subset='tweetId',ignore_index=True)
    elif data_name == 'french':
        df = pd.concat([pd.read_csv('TA1-USCISI-Slice1-20221108_simplified.csv',sep='\t'),
pd.read_csv('TA1-USCISI-Slice2-20221109_simplified.csv',sep='\t')])
        df = df.drop_duplicates(subset='tweetId',ignore_index=True)
    # retweet diff
    print('HAMAS LENGTH: ',len(df))
    print('retweet time diff')
    start= time.time()
    df = retweet_diff(df)
    print(time.time()-start)
    print('FAST RT')
    start= time.time()
    outfile = 'fastRT_'+data_name+'.gexf'
    if not os.path.exists(outfile):
        G_fastRT = fastRetweet(df)
        nx.write_gexf(G_fastRT,outfile)
    else:
        G_fastRT = nx.read_gexf(outfile)
        G_fastRT.remove_edges_from(list(nx.selfloop_edges(G_fastRT)))
    print(time.time()-start)
    print('CO-HASHTAG SEQUENCE')
    start= time.time()
    outfile = 'hashtagSeq_'+data_name+'.gexf'
    if not os.path.exists(outfile):
        G_hashSeq = hashSeq(df)
        nx.write_gexf(G_hashSeq,outfile)
    else:
        G_hashSeq = nx.read_gexf(outfile)
        G_hashSeq.remove_edges_from(list(nx.selfloop_edges(G_hashSeq)))
    print(time.time()-start)
    print('CO URL')
    start= time.time()
    outfile = 'coURL_'+data_name+'.gexf'
    if not os.path.exists(outfile):
        G_coURL = coURL(df)
        nx.write_gexf(G_coURL,outfile)
    else:
        G_coURL = nx.read_gexf(outfile)
        G_coURL.remove_edges_from(list(nx.selfloop_edges(G_coURL)))
    print('CO RT')
    start= time.time()
    outfile = 'coRT_'+data_name+'.gexf'
    if not os.path.exists(outfile):
        G_coRT = coRetweet(df)
        nx.write_gexf(G_coRT,outfile)
    else:
        G_coRT = nx.read_gexf(outfile)
        G_coRT.remove_edges_from(list(nx.selfloop_edges(G_coRT)))
    print(time.time()-start)
    print('TEXT SIMILARITY')
    timeWindow = 365
    start= time.time()
    outfile = 'textSim_'+data_name+'.gexf'
    threshold = 0.9
    if not os.path.exists(outfile):
        G_textsim = textSim(df, timeWindow,threshold)
        nx.write_gexf(G_textsim,outfile)
    else:
        G_textsim = nx.read_gexf(outfile)
        G_textsim.remove_edges_from(list(nx.selfloop_edges(G_textsim)))
    print(time.time()-start)
    try:
        singleFeatureNets = [G_coURL,G_coRT,G_fastRT,G_hashSeq,G_textsim]
        for jj,G in enumerate(singleFeatureNets):
            print(['G_coURL','G_coRT','G_fastRT','G_hashSeq','G_textsim'][jj])
            print(len(G.nodes()))
            print(len(list(nx.connected_components(G))))
        M = mergeNetworks(singleFeatureNets, weighted=False)
        M.remove_edges_from(nx.selfloop_edges(M))
        M.remove_nodes_from(list(nx.isolates(M)))
        eigenvector_dict = nx.eigenvector_centrality(M, max_iter=500)
        nx.set_node_attributes(M, eigenvector_dict, 'eigenvectorCentr')
        df2 = pd.DataFrame(M.nodes(data='eigenvectorCentr'))
        df2.columns = ['userid', 'eigenvectorCentr']
        df2['userid'] = df2['userid'].astype(str)
        df2.to_csv(data_name+'_evectcent.csv')
    except:
        continue
