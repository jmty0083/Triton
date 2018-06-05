import numpy as np
import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text
from sklearn import preprocessing
from scipy import sparse, io
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import re
from scipy import sparse, io


# read data from file path
def load_data(file):
    c = []
    la = []
    lines = []
    with open(file) as fr:
        for i in range(10000):
            line = fr.readline()
            lines.append(line)
        n = len(lines)
        for i in range(n):
            message = lines[i].split('\t')
            la.append(message[0])
            c.append(message[1])
    return n, c, la


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


def split_data(c, la):
    training_data, test_data, training_target, test_target = train_test_split(
        c, la, test_size=0.1, random_state=0)
    return training_data, test_data, training_target, test_target

def standardized_data(c, la):
    training_data, test_data, training_target, test_target = split_data(content, label)
    scalar = preprocessing.StandardScaler().fit(training_data)
    training_data_transformed = scalar.transform(training_data)
    test_data_transformed = scalar.transform(test_data)
    return training_data_transformed, test_data_transformed, training_target, test_target


# 用TF-ID生成对应词向量
class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        # analyzer = super(TfidfVectorizer, self).build_analyzer()
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words
        return analyzer


num, content, label = load_data('data/spams.txt')
vec_tf_idf = TfidfVectorizer(min_df=2, max_df=0.8)
data_tf_idf = vec_tf_idf.fit_transform(content)
name_tf_idf_feature = vec_tf_idf.get_feature_names()


io.mmwrite('output/word_vector.mtx', data_tf_idf)


#import inspect, os
#print(os.path.abspath("data/spams.txt"))
#print(inspect.getfile(inspect.currentframe())) # script filename (usually with path)
#print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))# script directory