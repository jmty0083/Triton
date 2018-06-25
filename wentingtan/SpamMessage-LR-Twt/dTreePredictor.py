from sklearn import metrics


class Predictor:
    def __init__(self, test_result, test_target):
        self.test_result = test_result
        self.test_target = test_target

    def sample_predict(self):
        #test_result = clf.predict(self.test_data)
        print(metrics.classification_report(self.test_target, self.test_result))
        print(metrics.confusion_matrix(self.test_target, self.test_result))

    def new_predict(self):
        #test_result = clf.predict(self.test_data)
        with open('result/predict_label.txt', 'wt') as f:
            for i in range(len(self.test_result)):
                f.writelines(self.test_result[i])
        self.test_target = self.test_result
        print('write over')
