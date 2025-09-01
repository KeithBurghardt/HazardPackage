

from glob import glob
import json
import pandas as pd
import pickle as pk
import numpy as np

ih = glob('/Volumes/Keith_Burghardt_EHD/INCAS/INCASdatasets/Challenge/*.jsonl')
french = glob('/Volumes/Keith_Burghardt_EHD/INCAS/INCASdatasets/Phase1B/*.jsonl')

hazard_text0 = pd.read_csv('threat.txt',header=None)
hazard_text0.columns = ['word']
hazard_text0 = set(hazard_text0['word'].tolist())
hazard_text2 = set(pd.read_csv('comprehensive_words_n=2.csv')['word'].tolist())
hazard_text3 = set(pd.read_csv('comprehensive_words_n=3.csv')['word'].tolist())
hazard_text5 = set(pd.read_csv('comprehensive_words_n=5.csv')['word'].tolist())


for ii,files in enumerate([ih,french]):
    hazard_dict = {'author':[],'id':[],'tweetId':[],'contentText':[],'translatedContentText':[]}
    for n in [0,2,3,5]:
        set_name = 'hazard_text'+str(n)
        hazard_dict[set_name]=[]
    for file in files:
        with open(file) as r:
            for line in r:
                line = json.loads(line)
                if line['mediaType'] == 'Twitter':
                    try:
                        # author
                        author = line['mediaTypeAttributes']['twitterData']['twitterAuthorScreenname']
                        # tweet id
                        tweetId = line['mediaTypeAttributes']['twitterData']['tweetId']
                        # text 
                        text = line['contentText']
                        translated_text = line['translatedContentText']
                        # id
                        id = line['id']
                        hazard_labels = {}
                        for n in [0,2,3,5]:
                            set_name = 'hazard_text'+str(n)
                            hazard_words = globals()[set_name]
                            if translated_text is None:
                                haz_text = text[:]
                            else:
                                haz_text = translated_text[:]

                            hazard_labels[set_name] = np.any([word in hazard_words for word in haz_text.split(' ')])
                        hazard_dict['author'].append(author)
                        hazard_dict['id'].append(id)
                        hazard_dict['tweetId'].append(tweetId)
                        hazard_dict['contentText'].append(text)
                        hazard_dict['translatedContentText'].append(translated_text)
                        for key in hazard_labels.keys():
                            hazard_dict[key].append(int(hazard_labels[key]))
                    except:
                        continue
                    #print(len(hazard_dict['tweetId']))
    with open('/Volumes/Keith_Burghardt_EHD/INCAS/dict_hazards_'+['ih','french'][ii]+'.pkl', 'wb') as file:
        pk.dump(hazard_dict, file)
    hazard_dict = pd.DataFrame(hazard_dict).to_csv('/Volumes/Keith_Burghardt_EHD/INCAS/dict_hazards_'+['ih','french'][ii]+'.tsv',sep='\t')

