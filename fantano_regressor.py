# -*- coding: utf-8 -*-
"""
Now that we've collected all the info from theneedledrop's channel we are going to use a regressor to predict album scores
This program has a main function which:
    calls the cleaning function
    employs train_test_split to train a model
    improves the model through cross validation
    has the model predict scores for data it was not trained upon, and provides the MSE on the predictions

The cleaning function:
    cleans and transposes the data to where we could conduct analysis
    it also adds several important predictors by looking through several of the long strings that
    youtube's api provides
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
import re
from typing import List
from pandas.core.common import flatten
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from collections import Counter
import seaborn as sn
import matplotlib.pyplot as plt

def main(): 
    #importing the data
    data = pd.read_csv('fantano_dataset.csv')
    cleaned_data = cleaning(data)
    
    #establishing the predictors and the values to be predicted
    y= cleaned_data.scores
    X = cleaned_data.drop(['scores', 'description',
                           'tags', 'title'], axis=1)
    
    X_test = pd.read_csv('fantano_test.csv')#the test data set will be used to test the model with data outside of what it's trained upon
    
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,
                                                          train_size = .8,
                                                          test_size = .2,
                                                          random_state=0)
    
    titles = X_test.title
    test_scores = X_test.scores
    X_test = X_test.drop(['title','scores','description','tags', 'Unnamed: 0'], axis=1)
    print(X_test.columns)
    print(X.columns)
    
    
    #use cross validation to test potential models
    results= {}
    def get_score(n_estimators):
        model = XGBRegressor(n_estimators = n_estimators,
                             random_state = 0,
                             objective='reg:squarederror')
        scores = -1* cross_val_score(model, X_train, y_train, cv=5, scoring = 'neg_mean_absolute_error')
        
        return scores.mean()
    
    #create a loop to find the best number of estimators
    for i in range(1,10):
        results[10*i]= get_score(10*i)
    
    #plotting the mse against the number of estimators
    plt.plot(list(results.keys()), list(results.values()))
    plt.show()
    
    #get the correct number of estimators
    index = min(results.values())
    for key, value in results.items():
        if value == index:
            index = key
    print(index)
    
    
    
    
    #creating the best model and record the predictions
    model = XGBRegressor(n_estimators = index,
                         random_state=0,
                         objective='reg:squarederror')
    
    #fit the model
    model.fit(X_train, y_train)
    
    
    #get predictions
    predictions = model.predict(X_valid)
    final_predictions = model.predict(X_test)
    
    
    print("Mean Absolute Error: " + str(mean_absolute_error(final_predictions, test_scores)))
    
    
    
    output = pd.DataFrame({'Title': titles,
                           'Predicted Score': final_predictions,
                           'Acutal score': test_scores})
    output.to_csv('Fantano_RGBRegressor.csv', index=False)
    print('results have been saved')
    
    
    
    """
    
    END OF ML CODE
    
    """
    
    
    
    
def cleaning(data):#function that will clean all the data
    #finding only the album reviews
    regex = 'album review'
    for i in range(1,len(data)):
        if not bool(re.search(regex, data['title'][i].lower())):
            data = data.drop(index = i, axis=0)        
    #print(data.shape)#checking to see that rows have been dropped
    
    
    #creating a function to obtain the scores from the description
    def scores(row):
        regex = '([0-9]*/10)'
        score = re.findall(regex, row.description.lower())
        if len(score) == 1:
            return score
        elif len(score) == 2: #if there is more than one score provided in the description, Theneedledrop provides his second
            return score[1]
        else:
            return 'NaN'
        
    #using apply to get all the scores from the description
    #and converting scores from a list of lists to just a list
    scores = list(flatten(data.apply(scores, axis='columns')))
    
    #add the scores column
    data['scores'] = scores
    
    #finding the "not good" reviews and the "classic" reviews which are reviews with a score of /10, and NaN scores
    indexes = data.scores.loc[data.scores=='/10'].index
    data = data.drop(indexes)
    indexes = data.scores.loc[data.scores=='NaN'].index
    data = data.drop(indexes)
    #print(data.shape)
    
    #converting the scores column to float values
    def score_float(row):
        score = re.split('/', row.scores)[0]
        return float(score)/10
    data.scores = data.apply(score_float, axis='columns')
    
    #remove the scores that are not within 0 and 1
    indexes = data.scores.loc[(data.scores>1) | (data.scores<0.0)].index
    data = data.drop(indexes)
    #print(data.shape)
    #check to see if there are any more null values in the data set
    #print(pd.isnull(data).sum())#there aren't
    
    
    """
    we need to find the most common and least common tags for a review.
    The intuition behind this is that particular words that are more common or less common could be associated with
    the score from the album.
    """
    #converting the tags from a string to a list of strings
    def make_list(row):
        row = row[1:-1]
        return row.split(',')
    data['tags'] = data.tags.apply(make_list)
    
    
    
    
    #creating a flattened list
    #removing the bracket that is inside the stings at each row from the tags list
    words = []
    for row in data.tags:
        words.append(row)
    #print(words)
    
    #create a list of all the words    
    series_words = list(flatten(words))
    #print(series_words)
    
    #cleaning all of the tags and the list of all tags
    cleaned_words = []
    def cleaning_words(words):
        cleaned_words = [re.sub('[^A-Za-z0-9]+', '', word) for word in words]
        return(cleaned_words)
    
    series_words = cleaning_words(series_words)
    #print(series_words)
    
    data['tags'] = data.tags.apply(cleaning_words)
    #print(data.tags[1])
    
    
    """
    We want words that are descriptive of the review, these do not include the words he includes in every list of
    tags like "review". We also do not want words that only appear once since they would 
    likely cause our model to overfit
    """
    
    most_common_list = (Counter(series_words).most_common())
    #print(most_common_list)#looking at the full list to get a good idea of what range of counts we want to include

    """
    pop seems to the most commonly mentioned genre descriptor at 559 apperances, which we will use as the ceiling
    and shoegaze appears to be the last genre descriptor at 13 counts which we will use as the floor.
    my first idea is to create dummies that pertain to ranges of frequency, and then checking to see which 
    albums contain those tags.
    Since the frequency of words follow a zipfian distribution these ranges will not be a constant size. This means
    the range of most frequent words must be the largest, and the least frequent range must be the smallest
    """
    most_frequent_words = []
    somewhat_frequent_words = []
    least_frequent_words = []
    for item in (most_common_list):
        if item[1] >=35 and item[1]<=533:
            most_frequent_words.append(item[0])
        elif item[1]<35 and item[1]>20:
            somewhat_frequent_words.append(item[0])
        elif item[1]>=13 and item[1]<=20:
            least_frequent_words.append(item[0])
            
    
    #checking that each range contains roughly the same number of words
    '''
    print(len(most_frequent_words))
    print(len(somewhat_frequent_words))
    print(len(least_frequent_words))
    '''


    #now we will create these dummy columns
    data['most_frequent_dummy'] = 0
    data['somewhat_frequent_dummy'] = 0
    data['least_frequent_dummy'] = 0
    
   
    #populating the dummy columns      
    #function to search for most common words in tags
    def most_frequent_check(row):
        for word in most_frequent_words:
            for tag in row.tags:
                if word==tag:                
                    return 1
        return 0
    
    #function to search the somewhat frequent
    def somewhat_frequent_check(row):
        for word in somewhat_frequent_words:
            for tag in row.tags:
                if word==tag:                
                    return 1
        return 0
    
    #function to search for the least frequent words
    def least_frequent_check(row):
        for word in least_frequent_words:
            for tag in row.tags:
                if word==tag:  
                    #print(word, tag)
                    return 1
        return 0
    
    #populating the columns
    data.most_frequent_dummy = data.apply(most_frequent_check, axis = 'columns')
    data.somewhat_frequent_dummy = data.apply(somewhat_frequent_check, axis = 'columns')
    data.least_frequent_dummy = data.apply(least_frequent_check, axis = 'columns')
    
    
    
    
    #creating the counts for the fav and least fav tracks
    data['fav_track']=0
    data['Least_fav_track']=0
    
    
    
    #finding the fav tracks
    def fav_track_count(row): #this function is much easier to write since the list of fav tracks is always followed by the least fav tracks
        for row in data.description:
            words = row.split()
            try:
                start_index = words.index('FAV')+2
                stop_index = words.index('LEAST')
                return len(words[start_index:stop_index])
            
            except:
                return 0
    
                
    def least_fav_count(row):
        title = row.title.split()#the title follows his list of least favorite tracks so we need it here
        for row in data.description:
            words = row.split()
            for i in range(len(words)-1):
                if words[i]=='LEAST' and words[i+1]=='FAV' and words[i+2]=='TRACK:':
                    start_index = i+3
                    
                    #print(words[start_index])
                    potential_words = words[start_index:]
                    for word in potential_words:        
                        if word.lower() == title[0].lower():
                            #print(word.lower(), title[0].lower())
                            stop_index = words.index(word)
                            return len(words[start_index:stop_index])
                            break            
        return 0
        #print(potential_words, title[0])
        
    #populate the columns 
    data['fav_track'] = data.apply(fav_track_count, axis='columns')
    data['least_fav_count'] = data.apply(least_fav_count, axis='columns')
    data['fav_least_ratio']= data.fav_track/data.least_fav_count
    
    
    
    #creating the ratios of likes to dislikes
    data['like_view'] = data.like_count/data.view_count
    data['dislike_view'] = data.dislike_count/data.view_count
    data['like_dislike'] = data.like_count/data.dislike_count
    
    
    
    #checking the correlation between these columns to prevent multicollinearity
    #corrMatrix = data.loc[:,['like_view', 'dislike_view', 'like_dislike']].corr()    
    #sn.heatmap(corrMatrix, annot=True)
    #plt.show()#we can see that these three predictors are not very correlated
    #and as such could all be used in the model
    
    #saving about a 1/6th of the data to a data set we will test with
    test_length = int(len(data)/6)
    train_length = len(data) - test_length
    test_subset = data[:][-test_length:]
    test_subset.to_csv('fantano_test.csv')
    train_subset = data[:][:train_length]
    
    #print(len(data))
    #print(len(train_subset))
    #print(len(test_subset))

    return(train_subset)#we only want to pass the training data to then main function
main()
