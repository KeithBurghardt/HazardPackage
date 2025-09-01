import os
import openai
import pandas as pd
import numpy as np
import networkx as nx
import time
from openai import OpenAI


openai_api_key = ""
client = OpenAI(
    api_key=openai_api_key,
)

annotations = pd.read_csv('TweetGT.csv')
unique_text = annotations['text'].drop_duplicates().values
# social conservative
pre_prompt = 'Does the tweet describe a hazard (something that could impose harm or other costs on the author of the tweet or on others)? Please answer "yes" or "no" and explain your thought process.\n'


responses={'text':unique_text}
if False:
    for model_engine in ["gpt-3.5-turbo","gpt-4","gpt-5"]:
        print(model_engine)
        responses[model_engine] = []
        for ii,text in enumerate(unique_text):
            if ii % 100 == 0:
                print(round(ii/len(unique_text)*100,2))
            prompt = pre_prompt + text
            response = np.nan
            completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": prompt}])
            response = completion.choices[0].message.content
            responses[model_engine].append(response)
    try:
        responses = pd.DataFrame(responses)
        responses.to_csv('tweet_responses_2025.csv')
    except:
        import pickle as pk
        with open('tweet_responses_2025.pkl','wb') as w:
            pk.dump(responses,w)


for num_shot in [2,5,10]:
    simple_hazard = annotations.loc[annotations['hazard']==1.0,'text'].sample(num_shot)
    simple_non_hazard = annotations.loc[annotations['hazard']==0.0,'text'].sample(num_shot)

    few_shot_pre_prompt = pre_prompt+"For example:\n"
    for haz in simple_hazard:
        few_shot_pre_prompt += "Text: "+haz+'\n'
        few_shot_pre_prompt += "Result: Yes, because the text describes a harm or other cost on the author of the tweet or on others.\n\n"

    for haz in simple_non_hazard:
        few_shot_pre_prompt += "Text: "+haz+'\n'
        few_shot_pre_prompt += "Result: No, because the text describes a benign event with no harm or significant cost to the author of the tweet or others.\n\n"

    responses={'text':unique_text}
    pause_count = 20
    
    for model_engine in ["gpt-3.5-turbo","gpt-4","gpt-5"]:
        print(model_engine)
        responses[model_engine] = []
        pause_iter = 0
        for ii,text in enumerate(unique_text):
            if ii % 100 == 0:
                print(round(ii/len(unique_text)*100,2))
            prompt = few_shot_pre_prompt + "Text: "+text+"\nHazard: "
            response = np.nan
            try:
                completion = client.chat.completions.create(model=model_engine,messages=[{"role": "user", "content": prompt}])
                response = completion.choices[0].message.content
            except:
                print('Pausing 60 seconds')
                pause_iter +=1
                if pause_iter > pause_count:
                    break
                time.sleep(60)
            responses[model_engine].append(response)
    try:
        responses = pd.DataFrame(responses)
        responses.to_csv('tweet_responses_2025_few-shot_n='+str(num_shot)+'.csv')
    except:
        import pickle as pk
        with open('tweet_responses_2025_few-shot_n='+str(num_shot)+'.pkl','wb') as w:
            pk.dump(responses,w)

