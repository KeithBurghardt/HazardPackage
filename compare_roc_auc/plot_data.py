import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import random
import pickle as pk
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import  train_test_split

GT_labels = pd.read_csv('TweetGT.csv')
text_train, text_test = train_test_split(GT_labels['text'].tolist(),train_size=0.9, random_state=42)
set_text_train = set(text_train)

text2hazard = {t:h for t,h in GT_labels[['text','hazard']].values}
mean_std = {}
for gpt_file in ['tweet_responses_2025.csv','tweet_responses_2025_few-shot_n=2.csv','tweet_responses_2025_few-shot_n=5.csv']:
    gpt_responses = pd.read_csv(gpt_file)
    
    for col in gpt_responses.columns:
        #if 'auc' in col:
        if 'gpt' not in col:
            continue
        text2response = {t:h for t,h in gpt_responses[['text',col]].values}
        response = np.array([1 if 'yes' in str(text2response[o]).lower() else 0 for o in text_train])
        gt = np.array([round(text2hazard[t]) for t in text_train])    
        boot_roc = []
        for i in range(50):
            boot_indices = np.random.randint(0,len(response),len(response)) 
            boot_roc.append(roc_auc_score(gt[boot_indices],response[boot_indices]))
        mean_std[gpt_file.replace('.csv','')+'_'+col] = [np.mean(boot_roc),np.std(boot_roc)]

for gpt_file in ['tweet_responses_2025.csv','tweet_responses_2025_few-shot_n=2.csv','tweet_responses_2025_few-shot_n=5.csv']:
    gpt_responses = pd.read_csv(gpt_file)
    
    for col in gpt_responses.columns:
        #if 'auc' in col:
        if 'gpt' not in col:
            continue
        print(col)
        text2response = {t:h for t,h in gpt_responses[['text',col]].values}
        response = np.array([1 if 'yes' in str(text2response[o]).lower() else 0 for o in text_train])
        gt = np.array([round(text2hazard[t]) for t in text_train])    
        boot_roc = []
        for i in range(50):
            boot_indices = np.random.randint(0,len(response),len(response)) 
            boot_roc.append(roc_auc_score(gt[boot_indices],response[boot_indices]))
        mean_std[gpt_file.replace('.csv','')+'_'+col] = [np.mean(boot_roc),np.std(boot_roc),len(boot_roc)]

        
for embed_file in ['metrics_gpt_vs_gpt_s_new_sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_correct_split.csv','metrics_gpt_vs_gpt_s_new_Qwen_Qwen3-Embedding-0.6B_correct_split.csv','metrics_gpt_vs_gpt_s_new_stsb-xlm-r-multilingual_correct_split.csv','metrics_gpt_vs_gpt_s_new_sentence-transformers_distiluse-base-multilingual-cased-v2_correct_split.csv']:
    embed_results = pd.read_csv(embed_file)
    for col in embed_results.columns:
        #if 'auc' in col:
        if '_auc' not in col:
            continue
        print(col)
        mean_std[embed_file.replace('.csv','')+'_'+col] = [embed_results[col].mean(),embed_results[col].std(),len(embed_results)]


matplotlib.rcParams.update({'font.size': 15})
labelfonts = {'fontname':'Arial','fontsize':15}
#metrics = pd.read_csv('metrics_gpt_vs_gpt_s.csv')
cols = list(mean_std.keys())
bars = [mean_std[c][0] for c in cols]#'GPT3.5-SocCons_auc','GPT3.5-LibCons_auc','GPT4_auc',
stds = [mean_std[c][1] for c in cols]
cols = np.array(cols)[np.argsort(bars)]
stds = np.array(stds)[np.argsort(bars)]
bars = np.array(bars)[np.argsort(bars)]


ax,fig = plt.subplots(1,1,figsize=(15,10))
errors = [mean_std[c][1]/np.sqrt(mean_std[c][2]) for c in cols]# 'GPT3.5-SocCons_auc','GPT3.5-LibCons_auc','GPT4_auc',
jet = plt.get_cmap('jet')
plt.bar(list(range(len(cols))),bars,color=[jet(i/(len(cols))) for i in range(len(mean_std.keys()))])
plt.errorbar(list(range(len(cols))),bars,yerr=stds,color='gray',linestyle='',linewidth=3,label='St. Dev.')

#plt.errorbar(list(range(len(cols))),bars,yerr=errors,color='k',linewidth=8,alpha=1,linestyle='',label='St. Errors')
ticks = [c.replace('tweet_responses_2025_','').replace('metrics_gpt_vs_gpt_s_new_','').replace('Qwen_','').replace('_n=2','Two-shot').replace('_n=5','Five-shot').replace('few-shot','').replace('_correct_split','').replace('sentence-transformers_','').replace('_10','').replace('_1','').replace('_2','').replace('_3','').replace('_auc','').replace('_',' ').replace('gpt','GPT').replace('Gpt','GPT') for c in cols]
print(len(ticks))
plt.xticks(list(range(len(cols))),ticks,rotation=60,ha='right')# 'GPT3.5-SocCons_auc','GPT3.5-LibCons_auc','GPT4_auc',
plt.ylabel('ROC-AUC')
#plt.legend(fontsize=14,loc='upper left')
plt.ylim([0.5,0.85])
plt.tight_layout()
plt.savefig('AUC_comparisons_new_all3.pdf')
#plt.show()
