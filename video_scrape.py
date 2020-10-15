# -*- coding: utf-8 -*-
"""
This program will collect all the revelant information from youtube's api
"""

import json 
import urllib3
import isodate
import re
import os

#grabbing the key as from my environmental variables
api_key = os.environ.get('YOUTUBE_KEY')

def api_scrape(url: str):
    #prepare to use the youtube api
    id_ = re.split('=', url)[1]#grabbing the id that the youtube api will use
    api_url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id={id_}&key={api_key}'
    
    http = urllib3.PoolManager()
    
    #get json file of video data
    r = http.request('GET', api_url)
    data = json.loads(r.data.decode('utf-8'))
    
    
    #grabbing the info we can from the api scrape
    title = data['items'][0]['snippet']['title']
    description = data['items'][0]['snippet']['description']
    tags = data['items'][0]['snippet']['tags']
    dislike_count = int(data['items'][0]['statistics']['dislikeCount'])
    like_count = int(data['items'][0]['statistics']['likeCount'])
    view_count = int(data['items'][0]['statistics']['viewCount'])
    duration = isodate.parse_duration(data['items'][0]['contentDetails']['duration']).seconds
    comment_count = int(data['items'][0]['statistics']['commentCount'])

    
    return(title, description, tags, view_count, like_count, dislike_count, duration, comment_count)
   


