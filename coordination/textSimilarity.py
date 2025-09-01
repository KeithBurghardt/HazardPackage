from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import pandas as pd
import numpy as np
import os
from os import listdir
from nltk.corpus import stopwords
import nltk
import re
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from datetime import timedelta
import networkx as nx

def get_tweet_timestamp(tid):
    try:
        offset = 1288834974657
        tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tstamp/1000)
        return utcdttime
    except:
        return None  

def get_positive_data(pos_df):
    pos_df = process_data(pos_df)
    pos_df = pos_df.loc[pos_df['engagementType']!='retweet',]
    pos_df = pos_df[['tweetid','userid','tweet_time','tweet_language','tweet_text']]
    pos_df['tweet_time'] = pos_df['tweetid'].apply(lambda x: get_tweet_timestamp(x))
    
    return pos_df

#Downloading Stopwords
nltk.download('stopwords')

#Load English Stop Words
stopword = stopwords.words('english')

def preprocess_text(df):
    # Cleaning tweets in en language
    # Removing RT Word from Messages
    df['tweet_text']=df['tweet_text'].str.lstrip('RT')
    # Removing selected punctuation marks from Messages
    df['tweet_text']=df['tweet_text'].str.replace( ":",'')
    df['tweet_text']=df['tweet_text'].str.replace( ";",'')
    df['tweet_text']=df['tweet_text'].str.replace( ".",'')
    df['tweet_text']=df['tweet_text'].str.replace( ",",'')
    df['tweet_text']=df['tweet_text'].str.replace( "!",'')
    df['tweet_text']=df['tweet_text'].str.replace( "&",'')
    df['tweet_text']=df['tweet_text'].str.replace( "-",'')
    df['tweet_text']=df['tweet_text'].str.replace( "_",'')
    df['tweet_text']=df['tweet_text'].str.replace( "$",'')
    df['tweet_text']=df['tweet_text'].str.replace( "/",'')
    df['tweet_text']=df['tweet_text'].str.replace( "?",'')
    df['tweet_text']=df['tweet_text'].str.replace( "''",'')
    # Lowercase
    df['tweet_text']=df['tweet_text'].str.lower()
    return df

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

#Message Clean Function
def msg_clean(msg):
    #Remove URL
    msg = re.sub(r'https?://\S+|www\.\S+', " ", msg)

    #Remove Mentions
    msg = re.sub(r'@\w+',' ',msg)

    #Remove Digits
    msg = re.sub(r'\d+', ' ', msg)

    #Remove HTML tags
    msg = re.sub('r<.*?>',' ', msg)
    
    #Remove HTML tags
    msg = re.sub('r<.*?>',' ', msg)
    
    #Remove Emoji from text
    msg = remove_emoji(msg)

    # Remove Stop Words 
    msg = msg.split()
    
    msg = " ".join([word for word in msg if word not in stopword])

    return msg

def create_sim_score_df(lims,D,I,search_query1,combined_tweets_df):
    source_idx = []
    target_idx = []
    sim_score = []

    for i in range(len(search_query1)):
        idx = I[lims[i]:lims[i+1]]
        sim = D[lims[i]:lims[i+1]]
        for j in range(len(idx)):
            source_idx.append(i)
            target_idx.append(idx[j])
            sim_score.append(sim[j])

    sim_score_df = pd.DataFrame(list(zip(source_idx, target_idx, sim_score)), columns=['source_idx', 'target_idx', 'sim_score'])
    del source_idx
    del target_idx
    del sim_score
    sim_score_df = sim_score_df.query("source_idx != target_idx")
    sim_score_df['combined_idx'] = sim_score_df[['source_idx', 'target_idx']].apply(tuple, axis=1)
    sim_score_df['combined_idx'] = sim_score_df['combined_idx'].apply(sorted)
    sim_score_df['combined_idx'] = sim_score_df['combined_idx'].transform(lambda k: tuple(k))
    sim_score_df = sim_score_df.drop_duplicates(subset=['combined_idx'], keep='first')
    sim_score_df.reset_index(inplace=True)
    sim_score_df = sim_score_df.loc[:, ~sim_score_df.columns.str.contains('index')]
    sim_score_df.drop(['combined_idx'], inplace = True, axis=1)

    df_join = pd.merge(pd.merge(sim_score_df,combined_tweets_df, left_on='source_idx', right_on='my_idx', how='inner'),combined_tweets_df,left_on='target_idx',right_on='my_idx',how='inner')

    result = df_join[['userid_x','userid_y','clean_tweet_x','clean_tweet_y','sim_score']]
    result = result.rename(columns = {'userid_x':'source_user',
                                     'userid_y':'target_user',
                                     'clean_tweet_x':'source_text',
                                     'clean_tweet_y':'target_text'})
    return result

# MAIN FUNCTION
# Data assumptions:
#   - dataframe containing 'tweetId', 'contentText', 'language', 'twitterAuthorScreenname', 'timePublished','engagementType'
#   - timeWindow: time window to compare tweets (in days)

def textSim(df_global,  timeWindow,threshold):
    old_columns = ['tweetId', 'contentText', 'language', 'twitterAuthorScreenname', 'timePublished','engagementType']
    control = df_global[old_columns].copy(deep=True)
    # we ONLY look at uniquely written text, not retweets
    control = control.loc[control.engagementType != 'retweet']
    new_columns = ['id', 'tweet_text', 'lang', 'userid', 'tweet_time','engagementType']
    control.columns = new_columns
    print('control size for text sim: ',len(control))
    neg_en_df_all = preprocess_text(control)
    del control
    
    neg_en_df_all['tweet_text']  = neg_en_df_all['tweet_text'].replace(',', '')
    neg_en_df_all['clean_tweet'] = neg_en_df_all['tweet_text'].astype(str).apply(lambda x: msg_clean(x))
    
    neg_en_df_all = neg_en_df_all[neg_en_df_all['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]
    # assumes time is in milliseconds
    neg_en_df_all['tweet_time'] = pd.to_datetime(neg_en_df_all['tweet_time'], unit='ms')

    #neg_en_df_all['tweet_time'] = neg_en_df_all['id'].apply(lambda x: get_tweet_timestamp(x))
    
    date = neg_en_df_all['tweet_time'].min().date()
    finalDate = max(date,neg_en_df_all['tweet_time'].max().date()-timedelta(days=timeWindow))
    print([date,finalDate])
    i = 1
    all_Gs = []
    # make graphs up until +1 year and no further
    while date <= finalDate:
        
        #pos_en_df = pos_en_df_all.loc[(pos_en_df_all['tweet_time'].dt.date >= date)&(pos_en_df_all['tweet_time'].dt.date < date+timedelta(days=timeWindow))]
        neg_en_df = neg_en_df_all.loc[(neg_en_df_all['tweet_time'].dt.date >= date)&(neg_en_df_all['tweet_time'].dt.date < date+timedelta(days=timeWindow))]
    
        #actual_pos_user = pos_en_df.userid.unique()
        actual_neg_user = neg_en_df.userid.unique()
    
        combined_tweets_df = neg_en_df.copy() #pd.concat([pos_en_df, neg_en_df], axis=0)
        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.loc[:, ~combined_tweets_df.columns.str.contains('index')]
    
        #del pos_en_df
        del neg_en_df
    
        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.rename(columns = {'index':'my_idx'})
    
        sentences = combined_tweets_df.clean_tweet.tolist()
        print('SENTENCES TO EMBED: ',len(sentences))
        embed_model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
        #encoder = SentenceTransformer('stsb-xlm-r-multilingual')
        encoder = SentenceTransformer(embed_model_name)
        # Process in smaller batches
        batch_size = 2**15  # Adjust this number based on your available memory
        plot_embeddings = []
        for i in range(0, len(sentences), batch_size):
            if i % 10 == 0:
                print(str(i)+'/'+str(len(sentences)))
            try:
                batch = sentences[i:i + batch_size]
                batch_embeddings = encoder.encode(batch, show_progress_bar=True)
            except:
                print('BAD BATCH',i*batch_size,'-',(i+1)*batch_size)
                continue
            plot_embeddings+=list(batch_embeddings)    
        #plot_embeddings = encoder.encode(sentences)    

        plot_embeddings = np.array(plot_embeddings)

        try:
            dim = plot_embeddings.shape[1]  # vector dimension
        except:
            date = date+timedelta(days=1)
            continue
        plot_embeddings = np.ascontiguousarray(plot_embeddings.copy()).astype(np.float32)
        random_int = np.random.randint(10000000)
        db_vectors1 = plot_embeddings.copy().astype(np.float32)
        a = [i for i in range(plot_embeddings.shape[0])]
        db_ids1 = np.array(a, dtype=np.int64)
        faiss.normalize_L2(db_vectors1)
    
        index1 = faiss.IndexFlatIP(dim)
        index1 = faiss.IndexIDMap(index1)  # mapping df index as id
        index1.add_with_ids(db_vectors1, db_ids1)
    
        search_query1 = plot_embeddings.copy().astype(np.float32)
        print(search_query1.shape)
        faiss.normalize_L2(search_query1)
        result_plot_thres = []
        result_plot_score = []
        result_plot_metrics = []
        print('SEARCHING 2')
        lims, D, I = index1.range_search(x=search_query1, thresh=threshold)
        print('Retrieved results of index search')
    
        sim_score_df = create_sim_score_df(lims,D,I,search_query1,combined_tweets_df)
        print('Generated Similarity Score DataFrame')
    
        del combined_tweets_df
        all_text_sim_networks = []
        #for threshold in [0.7]: #np.arange(0.7,1.01,0.05):
        #print("Threshold: ", threshold)
        sim_score_temp_df = sim_score_df.copy()#[(sim_score_df.sim_score >= threshold)&(sim_score_df.sim_score < threshold+0.05)]
        text_sim_network = sim_score_temp_df[['source_user','target_user']]
        text_sim_network = text_sim_network.loc[text_sim_network.source_user != text_sim_network.target_user,]
        text_sim_network = pd.DataFrame(text_sim_network.value_counts(subset=(['source_user','target_user'])))
        
        text_sim_network.reset_index(inplace=True)
        text_sim_network.columns = ['source_user','target_user', 'count']
        all_text_sim_networks.append(text_sim_network)
        outputfile = 'threshold_' + str(threshold) + '_'+str(i)+'.csv'
        text_sim_network.to_csv(outputfile)
        date = date+timedelta(days=1)
        i += 1
        G = getSimilarityNetwork(all_text_sim_networks)
        all_Gs.append(G)
    return G

# to run after the textSim function
# inputDir: path of the directory containing the similarity files; it corresponds to the outputDir used in the textSim function
def getSimilarityNetwork(inputDir):
    files = [f for f in listdir(inputDir)]
    files.sort()

    d = {'threshold_1.00':[],
        'threshold_0.90':[],
        'threshold_0.95':[],
        'threshold_0.85':[],
        'threshold_0.8':[],
        'threshold_0.75':[],
        'threshold_0.7':[]}
    
    for f in files:
        if f[:9]=='threshold':
            d['_'.join(f[:-4].split('_')[:2])[:14]].append(f)

    i = 0

    for fil in d.keys():
        thr = float(fil.split('_')[-1][:4])
        
        l = d[fil]
        if i == 0:
            combined = pd.read_csv(path+l[0])
            combined['weight'] = thr
            i += 1
            for o in l[1:]:
                temp = pd.read_csv(path+o)
                temp['weight'] = thr
                combined = pd.concat([combined, temp])
        else:
            for o in l:
                temp = pd.read_csv(path+o)
                temp['weight'] = thr
                combined = pd.concat([combined, temp])
    
    combined = combined.groupby(['source_user','target_user','weight'], as_index=False).sum()
    combined['weight'] = combined['weight']*combined['count']
    combined = combined.groupby(['source_user','target_user'], as_index=False).sum()
    combined['weight'] = combined['weight']/combined['count']
    
    G = nx.from_pandas_edgelist(combined[['source_user','target_user','weight']], source='source_user', target='target_user', edge_attr=['weight'])
            
    return G


def getSimilarityNetwork(dfs):
    # df = df[['source_user','target_user']]
    union = set()
    df_dict = {}
    st_keys = set()
    for i,df in enumerate(dfs):
        for s,t in df[['source_user','target_user']].values:
            key = str(s)+'_|_'+str(t)
            if key not in st_keys:
                df_dict[key] = 1
                st_keys.add(key)
            else:
                df_dict[key] += 1
    #df_dict = pd.DataFrame(df_dict)
    combined = {}
    combined['source_user'] = [st.split('_|_')[0] for st in df_dict.keys()]
    combined['target_user'] = [st.split('_|_')[1] for st in df_dict.keys()]
    
    combined['weight'] = [1]*len(df_dict.keys()) #[df_dict[st] for st in df_dict.keys()]
    combined = pd.DataFrame(combined)
    #combined = combined.groupby(['source_user','target_user','weight'], as_index=False).sum()
    #combined['weight'] = combined['weight']*combined['count']
    #combined = combined.groupby(['source_user','target_user'], as_index=False).sum()
    #combined['weight'] = combined['weight']/combined['count']
    
    G = nx.from_pandas_edgelist(combined[['source_user','target_user','weight']], source='source_user', target='target_user', edge_attr=['weight'])
            
    return G
        
