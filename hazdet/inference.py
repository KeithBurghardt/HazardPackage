import pandas as pd
import numpy as np
import pickle as pk
import os
from sentence_transformers import SentenceTransformer

def read_any(file):
    if file.endswith('.csv') or file.endswith('.tsv') :
        df = pd.read_csv(file)
    elif file.endswith('.json'):
        df = pd.read_json(file)
    elif file.endswith('.xml'):
        df = pd.read_xml(file)
    elif file.endswith('.xls') or file.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.endswith('.hdf'):
        df = pd.read_hdf(file)           
    elif file.endswith('.sql'):
        df = pd.read_sql(file)
    else:
        raise ValueError(f'ERROR: Unsupported filetype: {file}')
    return df

def load_data(file,text_col='text'):
    data = read_any(file)
    # find text column
    if text_col not in data.columns:
        raise Exception('ERROR: Could not find a text column. Please input the correct text column.')
    return data,text_col

def inference(file,model_file,text_col='text',sentence_tf= 'stsb-xlm-r-multilingual'):
    # if 
    data,text_col = load_data(file,text_col)
    if data is None or len(data) == 0:
        raise Exception('ERROR: data file is blank: ',file)
    if not os.path.exists(model_file):
        raise Exception('ERROR: model file is missing: ',model_file)
    try:
        clf = pk.load(open(model_file,'rb'))
    except:
        raise Exception('ERROR: model file could not be open (perhaps it is corrupted or in the wrong format): ',model_file)
    try:
        model = SentenceTransformer(sentence_tf)
    except:
        raise Exception('Error loading sentence transformer: ',sentence_tf)
    print('Using ',sentence_tf,': make sure this is the best transformer for your data.')
    data[text_col] = data[text_col].astype("string")
    all_pred = []
    tweets = []
    num_tweets_parsed = 1000
    for ii,tweet in enumerate(data[text_col].values):
        tweets.append(str(tweet))
        if ii % num_tweets_parsed == 0:
            try:
                embeddings = model.encode(tweets)
                pred_prob = list(clf.predict_proba(embeddings)[:,1])
                all_pred+=pred_prob
                tweets = []
            except:
                raise Exception('ERROR: one or more text could not be parsed for hazards. Look around index ',data.index.values[ii],'-',data.index.values[(ii+1)*num_tweets_parsed])
        # catching any leftovers
    try:
        embeddings = model.encode(tweets)
        pred_prob = list(clf.predict_proba(embeddings)[:,1])
        all_pred+=pred_prob
        tweets = []
    except:
        raise Exception('ERROR: one or more text could not be parsed for hazards. Look around index ',data.index.values[ii],'-',data.index.values[(ii+1)])

    data['hazard'] = all_pred
    return data

            
