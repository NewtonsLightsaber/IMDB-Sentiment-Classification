import pandas as pd
import pickle
import json
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

def main():
    project_dir = Path(__file__).resolve().parents[2]
    raw_path = project_dir / 'data' / 'raw'
    interim_path = project_dir / 'data' / 'interim'
    processed_path = project_dir / 'data' / 'processed'
    
    with open("IMDB-Sentiment-Classification/data/interim/test.json") as fp:
        data = json.load(fp)
        
    i = 0
    trainData = []
    sentiment = []
    
    while i < 25000:
        data_point = data[i]
        for info_name, info_value in data_point.items():
            if info_name == "text":
                processedInfo_Value = (info_value.lower()).split()
                trainData.append(" ".join(processedInfo_Value))
            elif info_name == "sentiment":
                sentiment.append(info_value)
        i += 1
    
    #Test data cleaning
    with open("IMDB-Sentiment-Classification/data/interim/train.json") as fp:
        data = json.load(fp)
        
    i = 0
    testData = []
    
    while i < 25000:
        data_point = data[i]
        for info_name, info_value in data_point.items():
            if info_name == "text":
                processedInfo_Value = (info_value.lower()).split()
                testData.append(" ".join(processedInfo_Value))
        i += 1
    
    #Define TfidfVectorizer
    print("Defining TFIDF")        
    
    newTFIDF = TFIDF(       
      stop_words = 'english', # 'english' is currently the only supported string value.
      min_df = 3, # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
      max_features = 3000, # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
      strip_accents = 'unicode', # Remove accents during the preprocessing step. 'ascii' is a fast method that only works on characters that have an direct ASCII mapping.
      analyzer = 'word', #Word n-grams
      token_pattern = r'\w{1,}', # Regular expression denoting what constitutes a "token", only used if analyzer == 'word'. 
      ngram_range = (1,5), # The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.
      use_idf = 1, # Enable inverse-document-frequency reweighting.
      smooth_idf = 1, # Smooth idf weights by adding one to document frequencies, prevents zero divisions
      sublinear_tf = 1 # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
      )
 
    allData = trainData + testData
    
    #Only training set to fit data
    print("Fitting Data")
    newTFIDF.fit(trainData)
    
    #Transforming data all data
    print("Transforming Data")
    allData = newTFIDF.transform(allData)
    
    print("Fitting and transforming finished")
    #Separates back into train and test
    trainAfter = allData[:len(trainData)]
    testAfter = allData[len(trainData):]
