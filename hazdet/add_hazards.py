import sys
sys.path.insert(0,'./myenv/lib/python3.11/site-packages')
sys.path.insert(0,'/project2/jambitem_1194/burghard_975/burghard_975/hazards/myenv/lib/python3.11/site-packages')
import pandas as pd
import pickle as pk
import os
import demoji
from sentence_transformers import SentenceTransformer
import json

embed_model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
model = SentenceTransformer(embed_model_name)

model_file = 'sentence-transformers_distiluse-base-multilingual-cased-v2best_svc_model.pkl'
clf = pk.load(open(model_file,'rb'))
#data_files = ['hamas_israel_challenge_problem_all_20240229.jsonl'] 
#data_files =['TA1-USCISI-Slice2-20221109.jsonl']
#data_files =['TA1-USCISI-Slice1-20221108.jsonl']
data_files = ['challenge_problem_two_21NOV.jsonl']
#data_files =['TA1-USCISI-Slice2-20221109.jsonl', 'TA1-USCISI-Slice1-20221108.jsonl']
for data_file in data_files:
  annotated_data = {'id':[],'url':[],'twitterAuthorScreenname':[],'tweetId':[],'contentText':[],'hazard':[]}
  count = 0
  index = 0
  num_lines = 100000
  for ii,line in enumerate( open(data_file, 'r')):
      data = json.loads(line)
      if data['mediaType'] == 'Twitter':
        count +=1
        dummy_index = int(count/num_lines)
        outfile = data_file.replace('.jsonl','_annotations_'+str(dummy_index)+'.pkl')
        if os.path.exists(outfile) or dummy_index < 19:
            if count % 10000 == 0:
                print([dummy_index,count])
            continue
        try:
            print('works?')
            id_val = data['id']
            url_val = data['url']
            author = data['mediaTypeAttributes']['twitterData']['twitterAuthorScreenname']
            tweet_id = data['mediaTypeAttributes']['twitterData']['tweetId']
            tweet_text = str(data['contentText'])
            embedding = model.encode(tweet_text)
            hazard_conf = clf.predict_proba([embedding])[0,1]
            print('works!')
        except:
            continue
        try:
            annotated_data['id'].append(id_val)
            annotated_data['url'].append(url_val)
            annotated_data['twitterAuthorScreenname'].append(author)
            annotated_data['tweetId'].append(tweet_id)
            annotated_data['contentText'].append(tweet_text)
            annotated_data['hazard'].append(hazard_conf)
        except:
            continue
        ex_outfile = data_file.replace('.jsonl','_annotations_'+str(dummy_index)+'.pkl')
        if not os.path.exists(ex_outfile) and len(annotated_data['hazard']) == num_lines:
            print(ex_outfile)
            pd.DataFrame(annotated_data).to_csv(data_file.replace('.jsonl','_annotations_'+str(dummy_index)+'.csv'),sep='\t',index=False)
            pk.dump(annotated_data,open(data_file.replace('.jsonl','_annotations_'+str(dummy_index)+'.pkl'),'wb'))
            annotated_data = {'id':[],'url':[],'twitterAuthorScreenname':[],'tweetId':[],'contentText':[],'hazard':[]}
  ex_outfile = data_file.replace('.jsonl','_annotations_'+str(dummy_index)+'.pkl')
  if not os.path.exists(ex_outfile):
      print(ex_outfile)
      pd.DataFrame(annotated_data).to_csv(data_file.replace('.jsonl','_annotations_'+str(dummy_index)+'.csv'),index=False)
      pk.dump(annotated_data,open(data_file.replace('.jsonl','_annotations_'+str(dummy_index)+'.pkl'),'wb'))
