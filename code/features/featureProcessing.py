import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
    
def featureProcessing ():
    #Training and test lists
    X_train = []
    X_test = []
    sentiment = []
    #Training data cleaning
    with open("train.json") as fp:
        data = json.load(fp)
        
    i = 0
    while i < 25000:
        data_point = data[i]
        for info_name, info_value in data_point.items():
            if info_name == "text":
                processedInfo_Value = (info_value.lower()).split()
                X_train.append(" ".join(processedInfo_Value))
            elif info_name == "sentiment":
                sentiment.append(info_value)
        i += 1
      
    #Test data cleaning
    with open("test.json") as fp:
        data = json.load(fp)
        
    i = 0
    while i < 25000:
        data_point = data[i]
        for info_name, info_value in data_point.items():
            if info_name == "text":
                processedInfo_Value = (info_value.lower()).split()
                X_test.append(" ".join(processedInfo_Value))
        i += 1
    
    #Bag of Words implementation, very large matrix 25 000 by # of unique words in corpus
    count_vect = CountVectorizer().fit(X_train)
    X_train_counts = count_vect.transform(X_train)
    X_test_counts = count_vect.transform(X_test) 
    
    #Implement Tdidf
    print("TDIF start")
    print("Fitting Training Data")
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    print("Transforming Data")
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    print("Fitting and Transforming done")
    
    #Default l2 normalization
    normalizer_tranformer = Normalizer().fit(X=X_train_tfidf)
    X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
    X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)
    
    return (X_train_normalized, X_test_normalized, sentiment)
