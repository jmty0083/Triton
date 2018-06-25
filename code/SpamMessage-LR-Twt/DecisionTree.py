from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
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

# 用TFID生成对应词向量
class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        #analyzer = super(TfidfVectorizer, self).build_analyzer()
        index = 0
        with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/RawData/stopword.json', 'r') as f:
            stoplist = json.load(f)
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x' and w.word not in stoplist and len(w.word)>1)
            words = jieba.cut(new_doc)
            print(index)
            return words
        return analyzer

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
    content_sub = content[0:100000]
    label_sub = label[0:100000]
    return content_sub,label_sub
    #print(content_sub)
    #vec_tfidf = TfidfVectorizer(min_df=2, max_df=0.8)
    #vec_tfidf = TfidfVectorizer()
    #data_tfidf = vec_tfidf.fit_transform(content_sub)
    #weight = vec_tfidf.fit_transform(content_sub).toarray()
    #print(data_tfidf)
    #name_tfidf_feature = vec_tfidf.get_feature_names()
    #print(name_tfidf_feature)

    #DecisionTreeClassifyTfidf(data_tfidf,,name_tfidf_feature)
    io.mmwrite('I:\MeachineLearnProject\SpamMessage-LR-Twt/Data/word_vector_sub.mtx', data_tfidf)

    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/Data/train_label_sub.json', 'w') as f:
        json.dump(label, f)
    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/Data/vector_type_sub.json', 'w') as f:
        json.dump(name_tfidf_feature, f)

def DecisionTreeClassifyTfidf(content,label):
    DecisionTree = DecisionTreeClassifier(criterion="entropy",
                                          splitter="best",
                                          max_depth=None,
                                          min_samples_split=2,
                                          min_samples_leaf=2)
    #分割测试数据和训练数据
    train_x, test_x, train_y, test_y = train_test_split(content, label, test_size=0.1)
    vec_tfidf = TfidfVectorizer(min_df=2, max_df=0.8,max_features=1000,use_idf=True)
    #vec_tfidf = TfidfVectorizer(max_features=5)
    #获取词频率
    #print(train_x)
    train_x = vec_tfidf.fit_transform(train_x)
    test_x = vec_tfidf.fit_transform(test_x)

    #return
    # 训练加上测评
    DecisionTree.fit(train_x, train_y)
    print(DecisionTree.score(test_x, test_y))


    print( classification_report(test_y, DecisionTree.predict(test_x)))


    from sklearn.tree import export_graphviz
    export_graphviz(DecisionTree, open("tree.dot", "w"))
    #dot -Tpng tree.dot -o tree.png
    return
    # 特征提取，提取词汇

    #


    name_tfidf_feature = vec_tfidf.get_feature_names()



if '__main__' == __name__:
    content,label = vector_word()
    DecisionTreeClassifyTfidf(content,label)
    print ('OK ')