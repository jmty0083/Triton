import re
import time

import jieba
import jieba.posseg as pseg
import numpy as np
import sklearn.feature_extraction.text
from scipy import sparse, io
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from src import MLPTrainer


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
def process_cont_numbers(c):
    digits_features = np.zeros((len(c), 16))
    for i, line in enumerate(c):
        for digits in re.findall(r'\d+', line):
            length = len(digits)
            if 0 < length <= 15:
                digits_features[i, length-1] += 1
            elif length > 15:
                digits_features[i, 15] += 1
    return process_cont_numbers


def split_data(c, la):
    ta, ya, tb, yb = train_test_split(c, la, test_size=0.1, random_state=0)
    return ta, ya, tb, yb


def standardized_data(c, la):
    ta, ya, tb, yb = split_data(c, la)
    scalar = preprocessing.StandardScaler().fit(ta)
    training_data_transformed = scalar.transform(ta)
    test_data_transformed = scalar.transform(ya)
    return training_data_transformed, test_data_transformed, tb, yb


def dimensionality_reduction(td, yd):
    n_components = 1000
    t0 = time.time()
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    pca.fit(td)
    print("done in %0.3fs" % (time.time() - t0))
    t0 = time.time()
    training_data_transform = sparse.csr_matrix(pca.transform(td))
    test_data_transform = sparse.csr_matrix(pca.transform(yd))
    print("done in %0.3fs" % (time.time() - t0))
    return training_data_transform, test_data_transform


# 用TF-ID生成对应词向量
class TfidfVector(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        # analyzer = super(TfidfVector, self).build_analyzer()
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words
        return analyzer


num, content, label = load_data('../data/spams.txt')
vec_tf_idf = TfidfVector(min_df=2, max_df=0.8)
data_tf_idf = vec_tf_idf.fit_transform(content)
name_tf_idf_feature = vec_tf_idf.get_feature_names()


training_data, test_data, training_target, test_target = split_data(data_tf_idf, label)
training_data, test_data = dimensionality_reduction(training_data.todense(), test_data.todense())

trainer = MLPTrainer.MLPTrainer(training_data, test_data)
trainer.train_classifier()

#import inspect, os
#print(os.path.abspath("data/spams.txt"))
#print(inspect.getfile(inspect.currentframe())) # script filename (usually with path)
#print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))# script directory