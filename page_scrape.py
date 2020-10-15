
"""
This program will scrape all video data from the Anthony Fantano's youtube channel"Theneedledrop"
1.) it uses selenium's driver to deal with youtube's infinite scrolling feature.
2.) It will then scrape the links to every video on his channel using beautiful soup
3.) it will call another program (video_scrape) which applies youtube's api to collect the information from every video
"""

from bs4 import BeautifulSoup
from selenium import webdriver
import requests
from video_scrape import api_scrape
import time
import collections
import pandas as pd


url = 'https://www.youtube.com/c/theneedledrop/videos'
d = collections.defaultdict(list)
def main():
    driver = webdriver.Chrome()
    driver.get(url)
    
    
    #scrolling the driver to the bottom of the channel
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0,document.documentElement.scrollHeight);")
    
        # Wait to load page
        time.sleep(3)
    
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
           print("break")
           break
        last_height = new_height
        
        
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(5)
    
    content = driver.page_source.encode('utf-8').strip()
    driver.close()
    
    #use beautiful soup to collect all the video links
    soup = BeautifulSoup(content, 'html.parser')
    titles = soup.findAll('a', id='video-title')
    
    
    video_urls = []
    i=0 #counter for grabbing urls
    for title in titles:
        video_urls.append(titles[i].get('href'))
        i+=1
    
    video_df = pd.DataFrame(video_urls)
    video_df.to_csv('Video_urls.csv')
    #call the api_scrape function to collect the data from the videos
    for video_url in video_urls:
        try:
            title, description, tags, view_count, like_count, dislike_count, duration, comment_count = api_scrape(video_url)
            d['title'].append(title)
            d['description'].append(description)
            d['tags'].append(tags)
            d['view_count'].append(view_count)
            d['like_count'].append(like_count)
            d['dislike_count'].append(dislike_count)
            d['duration'].append(duration)
            d['commentCount'].append(comment_count)
        except:
            print('video analytics not found for url')
    #print(d)
    #create the data frame that the analysis will be conducted on
    data_frame = pd.DataFrame(d)
    
    #create the dataframe
    data_frame.to_csv('fantano_dataset.csv', index = False)
main()
