from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text
import json
import re
import os
from scipy import sparse, io
import load_data


# 生成词向量并进行存储
def vector_word():
    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/RawData/train_content.json', 'r') as f:
        content = json.load(f)
    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/RawData/train_label.json', 'r') as f:
        label = json.load(f)
    '''
        vec_count = MessageCountVectorizer(min_df=2, max_df=0.8)
        data_count = vec_count.fit_transform(content)
        name_count_feature = vec_count.get_feature_names()
    '''
    content_sub = content[0:10000]
    label_sub = label[0:10000]

    #分割测试数据和训练数据
    testfrom = int(len(content_sub)-len(content_sub)*0.1)
    #train_x, test_x, train_y, test_y = train_test_split(content_sub, label_sub, test_size=0.1)
    #vec_tfidf = TfidfVectorizer(min_df=2, max_df=0.8,max_features=40,use_idf=True)
    vec_tfidf = TfidfVectorizer(max_features=500)
    #获取词频率
    dataSet = vec_tfidf.fit_transform(content_sub)
    #print(train_x.toarray().shape)
    feature_train = vec_tfidf.get_feature_names()
    feature_train_bak = vec_tfidf.get_feature_names()
    train_x = dataSet[0:testfrom]
    train_y = label_sub[0:testfrom]
    test_x = dataSet[testfrom:]
    test_y = label_sub[testfrom:]
    #test_x = vec_tfidf.fit_transform(test_x)
    #print(test_x.toarray().shape)
    #return train_x,train_y,test_x,test_y



    return train_x,test_x,train_y,test_y,feature_train,feature_train_bak
   # print(name_tfidf_feature)
    #return
    #DecisionTreeClassifyTfidf(data_tfidf,,name_tfidf_feature)
    io.mmwrite('I:\MeachineLearnProject\SpamMessage-LR-Twt/Data/word_vector_sub.mtx', train_x)

    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/Data/train_label_sub.json', 'w') as f:
        json.dump(label, f)
    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/Data/vector_type_sub.json', 'w') as f:
        json.dump(name_tfidf_feature, f)


if '__main__' == __name__:
   vector_word()