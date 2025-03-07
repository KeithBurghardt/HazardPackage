import requests
import pandas as pd
import json
import time
import random
from datetime import datetime, timedelta


tcount=0
nexttoken=0
request_number=0
file_number=8
next_token=None
filename='threat_tweets_'




# or a function
def gen_datetime(min_year=1900, max_year=datetime.now().year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()



def create_url(keyword, start_date,end_date, max_results = 500):
    search_url = "https://api.twitter.com/2/tweets/search/all"
    query_params = {'query': keyword,
                    'start_time': start_date,
                    #'tweet_mode':'extended',
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source,entities,possibly_sensitive',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'is':'retweet',
                    'next_token': {}}
    return (search_url, query_params)
def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token
    response = requests.request("GET", url, headers = headers, params = params)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def check_term(x):
    if ' ' in x:
        return '\"'+x+'\"'
    return x


df=pd.read_csv("threat.csv") #Replace "terms.csv" with the keyword file. Preferably have the .csv file with keywords under a column titled "word"

def gen_keywords(num_terms):
    terms=df['words'].sample(num_terms).tolist()
    terms=[t.replace('-','') for t in terms]
    terms=[t.lower() for t in terms]
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
    return keyword


# Overall idea: extract 500 tweets from random points in time with a random subsample of threat keywords
while tcount<=1000000:
    bearer_token= ""
    start_time = datetime(2006,3,21)
    start_time_str = str(start_time).replace(' ','T')+'.000Z'
    start_time_str="2006-03-21T00:00:00Z" #Format of start date 2022-08-20T00:00:00.000Z
    end_time = gen_datetime(min_year=2006, max_year=datetime.now().year)
    end_time_str = str(end_time)[:-7].replace(' ','T')+'.000Z'
    if end_time < start_time or end_time > datetime.now():
        continue
    num_terms = 120
    keyword = gen_keywords(num_terms)
    url = create_url(keyword,start_time_str,end_time_str, 500)
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    print(url)
    try:
        #if tcount==0:
        json_response = connect_to_endpoint(url[0], headers, url[1])        
        print(json_response)
        #else:
        #    json_response = connect_to_endpoint(url[0], headers, url[1],nexttoken)
    except Exception as e:
        print(e)
        time.sleep(900)
        #if tcount==0:
        json_response = connect_to_endpoint(url[0], headers, url[1])
        #else:
        #    json_response = connect_to_endpoint(url[0], headers, url[1],nexttoken)
    tcount+=500
    time.sleep(2)
    json_data = json.dumps(json_response,sort_keys=True)
    #nexttoken=json_response['meta']['next_token']
    request_number+=1
    print("Req: ",request_number,nexttoken)
    with open(filename+str(file_number)+".json", "a+") as outfile:
        outfile.write(json_data)
        outfile.write('\n')
    if request_number>250:
        file_number+=1
        time.sleep(800) #sleep for 15 minutes to respect Twitter rate limits. 300 requests every 15 minutes
        request_number=0
