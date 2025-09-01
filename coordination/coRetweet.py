import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

# Data assumptions:
#   - Pandas dataframe
#     - dataset includes columns ['tweetId' (tweet ID), 'twitterAuthorScreenname' (author ID), 'engagementParentId' (tweet parent ID), 'engagementType' (is the tweet a retweet, quote, original tweet, or reply)]

def coRetweet(df):
    control = df.copy(deep=True)
    control['id'] = control['tweetId']
    control['retweet_id'] = [tid  if eng == 'retweet' else np.nan for tid,eng in control[['engagementParentId','engagementType']].values] #control['retweeted_status'].apply(lambda x: int(dict(x)['id']))
    control['userid'] = control['twitterAuthorScreenname'] #.apply(lambda x: int(dict(x)['id']))
    control = control[['id', 'userid', 'retweet_id']]
    control.columns = ['tweetid', 'userid', 'retweet_tweetid']
    control.dropna(inplace=True)

    # minimum number of tweets to count user
    min_tweets = 10 # 20
    cum = control.copy() #pd.concat([treated, control])
    # all tweets for each user
    filt = cum[['userid', 'tweetid']].groupby(['userid'],as_index=False).count()
    # find all users that make at least minimum number of tweets (e.g., 20+ tweets)
    filt = list(filt.loc[filt['tweetid'] >= min_tweets]['userid'])
    # filter by these users who are active
    cum = cum.loc[cum['userid'].isin(filt)]
    # get all unique retweet IDs and users among those who are active
    cum = cum[['userid', 'retweet_tweetid']].drop_duplicates().dropna()
    # count the number of (active) users for each retweeted tweet
    temp = cum.groupby('retweet_tweetid', as_index=False).count()
    # prune to remove tweets retweeted only once
    cum = cum.loc[cum['retweet_tweetid'].isin(temp.loc[temp['userid']>1]['retweet_tweetid'].to_list())]
    cum['value'] = 1
    # IDs mapping retweet tweet ID to a number
    ids = dict(zip(list(cum.retweet_tweetid.unique()), list(range(cum.retweet_tweetid.unique().shape[0]))))
    cum['retweet_tweetid'] = cum['retweet_tweetid'].apply(lambda x: ids[x]).astype(int)
    del ids
    # user ID to a number
    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)
    # order users, retweets
    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.retweet_tweetid.unique()), ordered=True)
    # faster way of making RT ids parsable
    row = cum.userid.astype(person_c).cat.codes
    col = cum.retweet_tweetid.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    if sparse_matrix.shape[0] == 0 or sparse_matrix.shape[1] == 0:
        return nx.Graph()
    del row, col, person_c, thing_c
    # TF-IDF vector of RTs
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
