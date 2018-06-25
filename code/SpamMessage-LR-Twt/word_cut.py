# -*- coding: utf-8 -*-
import numpy as np
import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text
import json
import re
import os
from scipy import sparse, io
import load_data



# 将连续的数字转变为长度的维度
def process_cont_numbers(content):
    digits_features = np.zeros((len(content), 16))
    for i, line in enumerate(content):
        for digits in re.findall(r'\d+', line):
            length = len(digits)
            if 0 < length <= 15:
                digits_features[i, length-1] += 1
            elif length > 15:
                digits_features[i, 15] += 1
    return process_cont_numbers


# 正常分词，非TFID
class MessageCountVectorizer(sklearn.feature_extraction.text.CountVectorizer):
    def build_analyzer(self):
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words
        return analyzer


# 用TFID生成对应词向量
class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):

    def build_analyzer(self):
        #analyzer = super(TfidfVectorizer, self).build_analyzer()
        index = 0
        #stoplist = {'xxx', '等', '的', 'xxxxxxxxxxx','x','是','你','了','有','到','在','xx','我们','和'}
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
    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/RawData/stopword.json', 'r') as f:
        stopword = json.load(f)
    '''
        vec_count = MessageCountVectorizer(min_df=2, max_df=0.8)
        data_count = vec_count.fit_transform(content)
        name_count_feature = vec_count.get_feature_names()
    '''

    content_sub = content[0:20000]
    label_sub = label[0:20000]

    # 分割测试数据和训练数据
    testfrom = int(len(content_sub) - len(content_sub) * 0.1)
    # train_x, test_x, train_y, test_y = train_test_split(content_sub, label_sub, test_size=0.1)
    vec_tfidf = TfidfVectorizer(min_df=2, max_df=0.8,max_features=200,use_idf=True)
    #vec_tfidf = TfidfVectorizer(max_features=5)
    # 获取词频率
    dataSet = vec_tfidf.fit_transform(content_sub)
    feature_train = vec_tfidf.get_feature_names()
    print(feature_train)
    feature_train_bak = vec_tfidf.get_feature_names()
    train_x = dataSet[0:testfrom]
    train_y = label_sub[0:testfrom]
    test_x = dataSet[testfrom:]
    test_y = label_sub[testfrom:]
    # test_x = vec_tfidf.fit_transform(test_x)
    # return train_x,train_y,test_x,test_y

    return train_x, test_x, train_y, test_y, feature_train, feature_train_bak
    #
    # return
    # DecisionTreeClassifyTfidf(data_tfidf,,name_tfidf_feature)
    io.mmwrite('I:\MeachineLearnProject\SpamMessage-LR-Twt/Data/word_vector_sub.mtx', train_x)

    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/Data/train_label_sub.json', 'w') as f:
        json.dump(label, f)
    with open('I:\MeachineLearnProject\SpamMessage-LR-Twt/Data/vector_type_sub.json', 'w') as f:
        json.dump(name_tfidf_feature, f)

if '__main__' == __name__:
    vector_word()
    print ('OK ')
