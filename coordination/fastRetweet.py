import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from datetime import timedelta
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

# retrieves tweet's timestamp from its ID
def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp/1000)
        return utcdttime
    except:
        return None   


# Data assumptions:
#   - Pandas dataframe
#     - contains 'tweetId', 'contentText' (tweet text), 'language' (tweet language), 'twitterAuthorScreenname' (tweet author screenname), 'timePublished' (time tweet was published), 'rt_diff' (time in seconds between current tweet and original message retweeted), 'engagementType'
# timeInterval: time distance in seconds between retweet and original tweet under which a retweet is considered fast


def fastRetweet(df, timeInterval = 10):
    control = df.copy(deep=True)
    tweet2user = {id:userid for userid,id in control[['twitterAuthorScreenname','tweetId']].dropna().values}
    all_tweets = set(list(tweet2user.keys()))
    control['retweet_id'] = [id  if eng == 'retweet' else np.nan for id,eng in control[['engagementParentId','engagementType']].values]
    control['retweet_userid'] = [tweet2user[id] if id in all_tweets else np.nan for id  in control['retweet_id'].values ]#if eng == 'retweet'] #control['retweeted_status'].apply(lambda x: int(dict(dict(x)['user'])['id']))
    control['userid'] = control['twitterAuthorScreenname'] #.apply(lambda x: int(dict(x)['id']))
    control = control[['tweetId', 'userid', 'retweet_id', 'retweet_userid','rt_diff']]
    control.columns = ['tweetid', 'userid', 'retweet_tweetid',  'retweet_userid','delta']
    control.dropna(inplace=True)
    cumulative =  control[['userid','retweet_userid', 'delta']] #pd.concat([treated[['userid', 'retweet_userid', 'delta']], control[['userid','retweet_userid', 'delta']]])
    # convert user IDs to ints
    try:
        converted_to_int =  cumulative['userid'].astype(int).astype(str)
        cumulative['userid'] = converted_to_int
    except:
        pass
    # look at all fast RTs
    cumulative = cumulative.loc[cumulative['delta'] <= timeInterval,]
    # group by user ID, RT user ID
    # find all tweets that take more than 1 second
    cumulative = cumulative.groupby(['userid', 'retweet_userid'],as_index=False).count()
    cum = cumulative.loc[cumulative['delta'] > 1]
    # convert RT user ID to int
    urls = dict(zip(list(cum.retweet_userid.unique()), list(range(cum.retweet_userid.unique().shape[0]))))
    cum['retweet_userid'] = cum['retweet_userid'].apply(lambda x: urls[x]).astype(int)
    del urls
    # convert User ID to int
    userid = dict(zip(list(cum.userid.astype(str).unique()), list(range(cum.userid.unique().shape[0]))))
    cum['userid'] = cum['userid'].astype(str).apply(lambda x: userid[x]).astype(int)
    # find unusual number of common tweets RT'd for each user
    person_c = CategoricalDtype(sorted(cum.userid.unique()), ordered=True)
    thing_c = CategoricalDtype(sorted(cum.retweet_userid.unique()), ordered=True)
    
    row = cum.userid.astype(person_c).cat.codes
    col = cum.retweet_userid.astype(thing_c).cat.codes
    sparse_matrix = csr_matrix((cum["delta"], (row, col)), shape=(person_c.categories.size, thing_c.categories.size))
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
    
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G
