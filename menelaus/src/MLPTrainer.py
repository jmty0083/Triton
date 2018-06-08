from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import metrics


class MLPTrainer:
    def __init__(self, training_data, training_target):
        self.training_data = training_data
        self.training_target = training_target
        self.clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999,
                                 early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(10, 2),
                                 learning_rate='constant', learning_rate_init=0.001, max_iter=200, momentum=0.9,
                                 nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='lbfgs',
                                 tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)

    def train_classifier(self):
        self.clf.fit(self.training_data, self.training_target)
        training_result = self.clf.predict(self.training_data)
        print(metrics.classification_report(self.training_target, training_result))

    def cross_validation(self):
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=20)
        scores = cross_val_score(self.clf, self.training_data, self.training_target, cv=cv, scoring='f1_macro')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def predict(self, test_data):
        return self.clf.predict(test_data)