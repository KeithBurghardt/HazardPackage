import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

# Data assumptions:
#   - Pandas dataframe
#      dataset includes at least columns ['twitterAuthorScreenname', 'engagementType', 'engagementParentId','contentText']
# minHashtags: minimum number of hashtags inside a tweet

def hashSeq(df, minHashtags = 5):
    control = df.copy(deep=True)
    control_filt = control[['twitterAuthorScreenname', 'engagementType', 'engagementParentId','contentText']]
    del control
    cum = control_filt.copy() #pd.concat([control_filt, treated_filt])
    del control_filt
    cum = cum.loc[cum['engagementType'] != 'retweet']
    cum['hashtag_seq'] = ['__'.join([tag.strip("#") for tag in tweet.split() if tag.startswith("#")]) for tweet in cum['contentText'].values.astype(str)]
    cum.drop('contentText', axis=1, inplace=True)
    # look at all hashtag sequences > minHashtags (e.g., 5)
    cum = cum[['twitterAuthorScreenname', 'hashtag_seq']].loc[cum['hashtag_seq'].apply(lambda x: len(x.split('__'))) >= minHashtags]    
    cum.drop_duplicates(inplace=True)
    temp = cum.groupby('hashtag_seq', as_index=False).count()
    cum = cum.loc[cum['hashtag_seq'].isin(temp.loc[temp['twitterAuthorScreenname']>1]['hashtag_seq'].to_list())]
    cum['value'] = 1
    hashs = dict(zip(list(cum.hashtag_seq.unique()), list(range(cum.hashtag_seq.unique().shape[0]))))
    cum['hashtag_seq'] = cum['hashtag_seq'].apply(lambda x: hashs[x]).astype(int)
    del hashs
    userid = dict(zip(list(cum.twitterAuthorScreenname.astype(str).unique()), list(range(cum.twitterAuthorScreenname.unique().shape[0]))))
    cum['twitterAuthorScreenname'] = cum['twitterAuthorScreenname'].astype(str).apply(lambda x: userid[x]).astype(int)
    person_c = CategoricalDtype(sorted(cum.twitterAuthorScreenname.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.hashtag_seq.unique()), ordered=True)
    row = cum.twitterAuthorScreenname.astype(person_c).cat.codes
    col = cum.hashtag_seq.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["value"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
    if sparse_matrix.shape[0] == 0 or sparse_matrix.shape[1] == 0:
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
    
    G.remove_nodes_from(list(nx.isolates(G)))

    return G
