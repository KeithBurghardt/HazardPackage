import requests
import pandas as pd
import json
import time
tcount=0
nexttoken=0
request_number=0
file_number=0
next_token=None
filename='random_tweets_'
def create_url(keyword, start_date, max_results = 500):
    search_url = "https://api.twitter.com/2/tweets/search/all"
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source,entities,possibly_sensitive',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)
def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token
    response = requests.request("GET", url, headers = headers, params = params)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()
# data from https://www.wordfrequency.info/samples.asp, lemmas
freqs=pd.read_excel('wordFrequency.xlsx',sheet_name='1 lemmas')
terms = freqs['lemma'].values[:200]
print(terms)
terms=[t.replace('-','') for t in terms]
terms=[t.lower() for t in terms]
def check_term(x):
    if ' ' in x:
        return '\"'+x+'\"'
    return x
terms=[check_term(t) for t in terms]
keyword=''
for t in range(len(terms)):
    if t<len(terms)-1:
        if len(keyword+terms[t]+' OR ') >1024:
            keyword=keyword[:-4]
            break
        else:
            keyword+=terms[t]+' OR '
    else:
        if len(keyword+terms[t]) >1024:
            keyword=keyword[:-4]
            break
        else:
            keyword+=terms[t]
print(keyword)
while tcount<=300000:
    bearer_token= ""
    start_time="2006-03-21T00:00:00.000Z" #Format of start date 2022-08-20T00:00:00.000Z
    url = create_url(keyword,start_time, 500)
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    try:
        if tcount==0:
            json_response = connect_to_endpoint(url[0], headers, url[1])
        else:
            json_response = connect_to_endpoint(url[0], headers, url[1],nexttoken)
    except Exception as e:
        print(e)
        time.sleep(900)
        if tcount==0:
            json_response = connect_to_endpoint(url[0], headers, url[1])
        else:
            json_response = connect_to_endpoint(url[0], headers, url[1],nexttoken)
    tcount+=500
    time.sleep(1)
    json_data = json.dumps(json_response,sort_keys=True)
    nexttoken=json_response['meta']['next_token']
    request_number+=1
    print("Req: ",request_number,nexttoken)
    with open(filename+str(file_number)+".json", "a+") as outfile:
        outfile.write(json_data)
    if request_number>250:
        file_number+=1
        time.sleep(900) #sleep for 15 minutes to respect Twitter rate limits. 300 requests every 15 minutes
        request_number=0
