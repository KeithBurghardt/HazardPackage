import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

# Data assumptions:
#   - Pandas dataframe
#     - 'twitterAuthorScreenname' (tweet author ID), 'url' (all urls in a tweet), 'tweetId'


def coURL(df):
    control = df.copy(deep=True)
    control.dropna(inplace=True)
    control['userid'] = control['twitterAuthorScreenname']
    control['urls'] = control['url'] #control['entities'].apply(lambda x: dict(x)['urls'])
    # minimum number of tweets to count user
    min_tweets = 10 
    # all tweets for each user
    filt = control[['userid', 'tweetId']].groupby(['userid'],as_index=False).count()
    # find all users that make at least minimum number of tweets (e.g., 20+ tweets)
    filt = list(filt.loc[filt['tweetId'] >= min_tweets]['userid'])
    # filter by these users who are active
    control = control.loc[control['userid'].isin(filt)]
    
    control = control[['userid', 'urls']].explode('urls')
    control.dropna(inplace=True)
    cum = control[['userid', 'urls']].dropna() #pd.concat([control, treated])[['userid', 'urls']].dropna()
    cum.drop_duplicates(inplace=True)

    temp = cum.groupby('urls', as_index=False).count()
    cum = cum.loc[cum['urls'].isin(temp.loc[temp['userid']>1]['urls'].to_list())]
    cum['value'] = 1
    urls = dict(zip(list(cum.urls.unique()), list(range(cum.urls.unique().shape[0]))))
    cum['urls'] = cum['urls'].apply(lambda x: urls[x]).astype(int)
    del urls
    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)
    
    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.urls.unique()), ordered=True)
    
    row = cum.userid.astype(person_c).cat.codes
    col = cum.urls.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    if sparse_matrix.shape[0] ==0 or sparse_matrix.shape[1] == 0:
        return nx.Graph()
    del row, col, person_c, thing_c
    
    vectorizer = TfidfTransformer()
    tfidf_matrix = vectorizer.fit_transform(sparse_matrix)
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)

    df_adj = pd.DataFrame(similarities.toarray())
    del similarities
    df_adj.index = userid.keys()
    df_adj.columns = userid.keys()
    G = nx.from_pandas_adjacency(df_adj)
    del df_adj
    to_remove = [(a,b) for a,b,d in G.edges(data=True) if a==b or d['weight'] < 0.9]
    G.remove_edges_from(to_remove)
    
    G.remove_nodes_from(list(nx.isolates(G)))

    return G
