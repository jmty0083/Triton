import numpy as np
import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text
import json
import re
from scipy import sparse, io

# 加载原始数据，进行分割
def load_message():
    content = []
    label = []
    lines = []
    with open('RawData/message.txt') as fr:
        for i in range(10000):
            line = fr.readline()
            lines.append(line)
        num = len(lines)
        for i in range(num):
            message = lines[i].split('\t')
            label.append(message[0])
            content.append(message[1])
    return num, content, label

# 将分割后的原始数据存到json
def data_storage(content, label):
    with open('RawData/train_content.json', 'w') as f:
        json.dump(content, f)
    with open('RawData/train_label.json', 'w') as f:
        json.dump(label, f)



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


# 用TFID生成对应词向量
class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        #analyzer = super(TfidfVectorizer, self).build_analyzer()
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words
        return analyzer

# 生成词向量并进行存储
def vector_word():
#    with open('RawData/train_content.json', 'r') as f:
#        content = json.load(f)
#    with open('RawData/train_label.json', 'r') as f:
#        label = json.load(f)
    '''
        vec_count = MessageCountVectorizer(min_df=2, max_df=0.8)
        data_count = vec_count.fit_transform(content)
        name_count_feature = vec_count.get_feature_names()
    '''

    vec_tfidf = TfidfVectorizer(min_df=2, max_df=0.8)
    data_tfidf = vec_tfidf.fit_transform(content)
    name_tfidf_feature = vec_tfidf.get_feature_names()

    io.mmwrite('Data/word_vector.mtx', data_tfidf)

#    with open('Data/train_label.json', 'w') as f:
#        json.dump(label, f)
#    with open('Data/vector_type.json', 'w') as f:
#        json.dump(name_tfidf_feature, f)

#if '__main__' == __name__:
    #vector_word()
    #print ' OK '