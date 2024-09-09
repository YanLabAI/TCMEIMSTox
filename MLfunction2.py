from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score as BACC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def scores_names(y_true, t_pred, scoresFun=None, reg_cla=None):
    if scoresFun==None:
        if reg_cla=='regressor':
            scoresFun = [r2_score, MSE, MAE]
        elif reg_cla=='classifier':
            scoresFun = [ACC, BACC]
        else:
            print('Whong Model Type')
            return
    return  [fun(y_true, t_pred) for fun in scoresFun], [fun.__name__ for fun in scoresFun]

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.metrics import roc_curve, auc

class cross_val():

    def __init__(self, model, X, y, random_seed=None, scoresFun=None, cv=5):

        if random_seed is not None:
            index = np.arange(len(y))
            np.random.seed(random_seed)
            np.random.shuffle(index)
        else:
            index = np.arange(len(y))

        step = int(len(y) / cv)
        self.scores = []
        self.pred_all = np.zeros(len(y), dtype=float)
        self.obs_all = np.zeros(len(y), dtype=float)
        self.indexes = []
        self.probas = []  # 用于存储预测概率
        addNum = len(y) % cv
        length = [step + 1] * addNum + [step] * (cv - addNum)

        for i in range(cv):
            index_train = np.concatenate([index[:sum(length[:i])], index[sum(length[:i + 1]):]], axis=0)
            index_val = index[sum(length[:i]):sum(length[:i + 1])]
            self.indexes.append([index_train, index_val])
            X_train = X[index_train]
            y_train = y[index_train]
            X_val = X[index_val]
            y_val = y[index_val]

            model.fit(X_train, y_train)
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_val)
            else:
                probas = model.decision_function(X_val)
                probas = np.vstack([1 - probas, probas]).T

            pred = np.argmax(probas, axis=1)  # 对于多分类问题选择预测的类别

            self.pred_all[index_val] = pred
            self.obs_all[index_val] = y_val
            self.probas.append(probas)  # 存储每一折的预测概率

            scores_oneCV, funNames = scores_names(y_val, pred, reg_cla='classifier', scoresFun=scoresFun)
            self.scores.append(scores_oneCV)

        self.scores = pd.DataFrame(self.scores, columns=funNames)

def scores_names(y_true, y_pred, reg_cla, scoresFun):
    # Placeholder for actual implementation of score functions
    scores = {score.__name__: score(y_true, y_pred) for score in scoresFun}
    funNames = [score.__name__ for score in scoresFun]
    return scores, funNames



class parmaOptimaze():
    def __init__(self, model, data, params, random_seed=None, scoresFun=None, cv=5):
        if len(data)==4:
            X_train, y_train, X_test, y_test = data
        elif len(data)==2:
            X_train, y_train = data
        else:
            print('Whong Number of Data')
            return
        reg_cla = model._estimator_type
                    
        cvBegin = cross_val(model, X_train, y_train, random_seed=random_seed, cv=cv)
        # paramKeys = list(params.keys())
        print('Scores Begin:')
        print(cvBegin.scores.mean())
        self.parmasBest = {}
        for i, (key, value) in enumerate(params.items()):
            print('='*100)
            print(f'{i+1}. Optimaze Parma: {key}  range:{value}')
            print('Original Model', model)
            model, parmaBest = parmaOptimatzeOne(model, 
                                                [X_train, y_train], 
                                                dict([[key, value]]), 
                                                random_seed=random_seed, 
                                                scoresFun=scoresFun, 
                                                cv=cv)
            self.parmasBest.update(parmaBest)
            print('Optimazed Model', model)
            cvOpt = cross_val(model, X_train, y_train, random_seed=random_seed, cv=cv)
            print('Best Scores:')
            print(cvOpt.scores.mean())
            if len(data)==4:
                print('Test Scores:')
                self.test_pred = model.fit(X_train, y_train).predict(X_test)
                scores_oneParam, funNames = scores_names(y_test, self.test_pred, reg_cla=reg_cla)
                print(pd.DataFrame([[scores_now] for scores_now in scores_oneParam], index=funNames)[0])

        self.cvEnd = cross_val(model, X_train, y_train, random_seed=random_seed, cv=cv)
        print('='*100)
        print(pd.DataFrame([cvBegin.scores.mean(), self.cvEnd.scores.mean()], index=['Before', 'After']).T)
        print('Best Model for test:', model)
        self.model = model
        

def continu2Binary(label, threshold="mean"):
    if threshold=="mean":
        threshold = np.array(label).mean()
    elif threshold=="median":
        threshold = np.median(label)
    outputs = np.array([int(i<threshold) for i in label])
    print(threshold, "\n", outputs.sum()/len(outputs), "\n")
    return outputs

def quzheng(num, up=True):
    if num==int(num):
        return num
    else:
        if num > 0:
            return int(num) + up
        else:
            return int(num)-1+up
