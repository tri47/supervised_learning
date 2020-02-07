import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import mlrose

from sklearn.base import is_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

def return_stratified_cv_results(clf,x_data,y_data):
    y_data = y_data.to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    train_scores, test_scores = [], []
    train_times, test_times = [], []
    for train_index, test_index in skf.split(x_data, y_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
       # print(y_data[test_index])
       # xt, xv, yt, yv = X[train, :], X[test, :], y[train], y[test]
        start_time = time.time()
        clf.fit(x_train, y_train)
        train_times.append(time.time()-start_time)
        y_train_pred = clf.predict(x_train)
        #y_train_pred = pd.Series(y_train_pred)
        start_time = time.time()
        y_test_pred = clf.predict(x_test)
        test_times.append(time.time()-start_time)
        #y_test_pred = pd.Series(y_test_pred)
        train_score =f1_score(y_train, y_train_pred) 
        test_score =f1_score(y_test, y_test_pred) 

        train_scores.append(train_score)
        test_scores.append(test_score)
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    train_times = np.array(train_times)
    test_times = np.array(test_times)        
  
    return train_scores.mean(), test_scores.mean(),train_times.mean(),test_times.mean()


def get_DT_results(x_data, y_data, leaf_size, split_method):
    clf = DecisionTreeClassifier(min_samples_leaf=leaf_size,
                                        splitter = split_method,
                                        criterion='gini')
    scores = cross_validate(clf,x_data,y_data, scoring ='f1'
        , return_train_score = True, cv = 5)
    train_scores = scores['train_score']
    test_scores = scores['test_score']
    train_time = scores['fit_time']
    test_time = scores['score_time']
    return train_scores.mean(), test_scores.mean(), train_time.mean(), test_time.mean()

def run_DT_hyper_test(x_data, y_data):

    random_train_score = [] 
    random_test_score = [] 
    random_train_time = []
    random_test_time = []
    best_train_score = []
    best_test_score = []
    best_train_time = []
    best_test_time = []
    leaf_sizes = [i for i in range(1,40,2)]
    split_methods = ['random','best']
    for split_method in split_methods:
        for leaf_size in leaf_sizes:
            train_score, test_score, train_time, test_time = get_DT_results(x_data, y_data, leaf_size, split_method)
            if split_method == 'random':
                random_train_score.append(train_score)
                random_test_score.append(test_score)
                random_train_time.append(train_time)
                random_test_time.append(test_time)
            if split_method == 'best':
                best_train_score.append(train_score)
                best_test_score.append(test_score)
                best_train_time.append(train_time)
                best_test_time.append(test_time)
    plt.figure(1)
    plt.plot(leaf_sizes, random_train_score,'-o',color='black')
    plt.plot(leaf_sizes, best_train_score,'-o',color='green')
    plt.plot(leaf_sizes, random_test_score,'-',color='black')
    plt.plot(leaf_sizes, best_test_score,'-',color='green')
    plt.legend(['random split train score','best split train score', 'random split CV score', 'best split CV score'],fontsize=12)
    plt.title('Decision Trees (best split vs random split) score',fontsize=12)
    plt.xlabel('Min leaf size',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.figure(2)
    plt.plot(leaf_sizes, random_train_time,'-o',color='black')
    plt.plot(leaf_sizes, best_train_time,'-o',color='green')   
    plt.plot(leaf_sizes, random_test_time,'-',color='black')
    plt.plot(leaf_sizes, best_test_time,'-',color='green')  
    plt.legend(['random','best'],fontsize=12)
    plt.title('Decision Trees (best split vs random split) time performance',fontsize=12)
    plt.xlabel('Min leaf size',fontsize=12)
    plt.ylabel('Time (s)',fontsize=12)
    plt.show()

def run_Boost_hyper_test(x_data, y_data, leaf_sizes=[9,40], split_method='best',boost_method = 'Ada'):
    random_state = 100
    train_results = [] 
    test_results = [] 
    n_estimators = [i for i in range(1,40,2)]
    criterion='gini'
    for leaf_size in leaf_sizes:
        train_scores = []
        test_scores = []
        for n in n_estimators:
            print('Number of estimators: ', n)
            if boost_method == 'Ada':
                clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=leaf_size,
                    splitter = split_method,
                    criterion=criterion),
                    n_estimators=n,learning_rate=0.8,
                    algorithm="SAMME")
            elif boost_method == 'Bagging':
                clf = BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=leaf_size,
                    splitter = split_method,
                    criterion=criterion),
                    n_estimators=n)       
            scores = cross_validate(clf,x_data,y_data, scoring ='f1', cv = 5, 
                return_train_score=True)
            train_score = scores['train_score'].mean()
            test_score = scores['test_score'].mean()
            train_scores.append(train_score)
            test_scores.append(test_score)
        train_results.append(train_scores)
        test_results.append(test_scores)
    plt.figure(1)
    legends = []
    for i in range(0, len(leaf_sizes)):
        plt.plot(n_estimators, train_results[i],'-o')
        plt.plot(n_estimators, test_results[i],'-')
        legends += ['Train score. Min leaf size = ' + str(leaf_sizes[i]),
                    'Test score. Min leaf size = ' + str(leaf_sizes[i])]
    plt.legend(legends,fontsize=12)
    if boost_method == 'Ada':
        plt.title('Decision Tree with AdaBoost',fontsize=12)
    elif boost_method == 'Bagging':
        plt.title('Decision Tree with Bagging',fontsize=12)
    plt.xlabel('Number of trees',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.show()

def get_KNN_results(x_data, y_data, k_size, method):
    clf = KNeighborsClassifier(n_neighbors=k_size, weights = method)
    scores = cross_validate(clf,x_data,y_data, scoring ='f1'
        , return_train_score = True, cv = 5)
    train_scores = scores['train_score']
    test_scores = scores['test_score']
    train_time = scores['fit_time']
    test_time = scores['score_time']
    return train_scores.mean(), test_scores.mean(), train_time.mean(), test_time.mean()


def run_KNN_hyper_test(x_data, y_data):
    k_sizes = [i for i in range(1,25,2)] ## 1 to 22
    methods = ['uniform','distance']
    uniform_train_score = [] 
    uniform_test_score = [] 
    uniform_train_time = []
    uniform_test_time = []
    distance_train_score = []
    distance_test_score = []
    distance_train_time = []
    distance_test_time = []

    for method in methods:
        for k in k_sizes:
            print('k :', k)
            train_score, test_score, train_time, test_time = get_KNN_results(x_data, y_data, k, method)
            if method == 'uniform':
                uniform_train_score.append(train_score)
                uniform_test_score.append(test_score)
                uniform_train_time.append(train_time)
                uniform_test_time.append(test_time)
            if method == 'distance':
                distance_train_score.append(train_score)
                distance_test_score.append(test_score)
                distance_train_time.append(train_time)
                distance_test_time.append(test_time)
           
    plt.figure(1)
    plt.plot(k_sizes, uniform_train_score,'-o',color='black')
    plt.plot(k_sizes, distance_train_score,'-o',color='green')
    plt.plot(k_sizes, uniform_test_score,'-',color='black')
    plt.plot(k_sizes, distance_test_score,'-',color='green')
    plt.legend(['uniform train','distance CV', 'uniform CV', 'distance CV'],fontsize=12)
    plt.title('KNN performance',fontsize=12)
    plt.xlabel('K',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.xticks(k_sizes)
    plt.figure(2)
    plt.plot(k_sizes, uniform_train_time,'-o',color='black')
    plt.plot(k_sizes, distance_train_time,'-o',color='green')   
    plt.plot(k_sizes, uniform_test_time,'-',color='black')
    plt.plot(k_sizes, distance_test_time,'-',color='green')    
    plt.legend(['uniform train','distance train','uniform CV', 'distance CV'],fontsize=12)
    plt.title('KNN time performance',fontsize=12)
    plt.xlabel('K',fontsize=12)
    plt.ylabel('Time (s)',fontsize=12)
    plt.xticks(k_sizes)
    plt.show()

def get_SVM_results(x_data, y_data, kernel, C):
    clf = SVC(kernel=kernel, C=C, gamma='auto')
    scores = cross_validate(clf,x_data,y_data, scoring ='f1'
        , return_train_score = True, cv = 5, n_jobs = -1)
    train_scores = scores['train_score']
    test_scores = scores['test_score']
    train_time = scores['fit_time']
    test_time = scores['score_time']
    return train_scores.mean(), test_scores.mean(), train_time.mean(), test_time.mean()

def run_SVM_hyper_test(x_data, y_data, verbose = False):
    Cs = [0.0001, 0.001, 0.01,0.1,1,10,100,1000] ## 1 to 22
    kernels = ['linear','rbf']
    linear_train_score = [] 
    linear_test_score = [] 
    linear_train_time = []
    linear_test_time = []
    rbf_train_score = []
    rbf_test_score = []
    rbf_train_time = []
    rbf_test_time = []

    for kernel in kernels:
        for C in Cs:
            print('C :', C)
            train_score, test_score, train_time, test_time = get_SVM_results(x_data, y_data, kernel, C)
            if kernel == 'linear':
                linear_train_score.append(train_score)
                linear_test_score.append(test_score)
                linear_train_time.append(train_time)
                linear_test_time.append(test_time)
            if kernel == 'rbf':
                rbf_train_score.append(train_score)
                rbf_test_score.append(test_score)
                rbf_train_time.append(train_time)
                rbf_test_time.append(test_time)
           
    plt.figure(1)
    plt.plot(Cs, linear_train_score,'-o',color='black')
    plt.plot(Cs, rbf_train_score,'-o',color='green')
    plt.plot(Cs, linear_test_score,'-',color='black')
    plt.plot(Cs, rbf_test_score,'-',color='green')
    plt.legend(['linear train','rbf train', 'linear CV', 'rbf CV'],fontsize=12)
    plt.title('SVM performance',fontsize=12)
    plt.xlabel('C',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.xticks(Cs)
    plt.xscale('log')
    plt.figure(2)
    plt.plot(Cs, linear_train_time,'-o',color='black')
    plt.plot(Cs, rbf_train_time,'-o',color='green')   
    plt.plot(Cs, linear_test_time,'-',color='black')
    plt.plot(Cs, rbf_test_time,'-',color='green')    
    plt.legend(['linear train','rbf train','linear CV', 'rbf CV'],fontsize=12)
    plt.title('SVM time performance',fontsize=12)
    plt.xlabel('C',fontsize=12)
    plt.ylabel('Time (s)',fontsize=12)
    plt.xticks(Cs)
    plt.xscale('log')
    plt.show()

def get_NN_results(x_data, y_data, hidden_nodes, activation, verbose = False):
   # x_data.reset_index(inplace=True)
    y_data = y_data.to_numpy()
    clf = mlrose.NeuralNetwork(
        hidden_nodes = hidden_nodes, activation = activation, \
        algorithm = 'gradient_descent', max_iters = 1000, \
        bias = True, is_classifier = True, learning_rate = 0.0001, \
        early_stopping = True, clip_max = 5, max_attempts = 100, \
        random_state = 30)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    train_scores, test_scores = [], []
    train_times, test_times = [], []
    for train_index, test_index in skf.split(x_data, y_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
       # print(y_data[test_index])
       # xt, xv, yt, yv = X[train, :], X[test, :], y[train], y[test]
        start_time = time.time()
        clf.fit(x_train, y_train)
        train_times.append(time.time()-start_time)
        y_train_pred = clf.predict(x_train)
        #y_train_pred = pd.Series(y_train_pred)
        start_time = time.time()
        y_test_pred = clf.predict(x_test)
        test_times.append(time.time()-start_time)
        #y_test_pred = pd.Series(y_test_pred)
        train_score =f1_score(y_train, y_train_pred) 
        test_score =f1_score(y_test, y_test_pred) 

        train_scores.append(train_score)
        test_scores.append(test_score)
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    train_times = np.array(train_times)
    test_times = np.array(test_times)        
  
    return train_scores.mean(), test_scores.mean(),train_times.mean(),test_times.mean()

def get_NN_results_2(x_data, y_data, hidden_nodes, activation, verbose = False):
    clf = mlrose.NeuralNetwork(
        hidden_nodes = hidden_nodes, activation = activation, \
        algorithm = 'gradient_descent', max_iters = 1000, \
        bias = True, is_classifier = True, learning_rate = 0.0001, \
        early_stopping = True, clip_max = 5, max_attempts = 100, \
        random_state = 30)
    #print(clf._estimator_type)
    #clf._estimator_type = "classifier"
    scores = cross_validate(clf,x_data,y_data, scoring ='f1'
        , return_train_score = True, cv = 5)

    train_scores = scores['train_score']
   # print(train_scores)
    test_scores = scores['test_score']
    train_time = scores['fit_time']
    test_time = scores['score_time']
    return train_scores.mean(), test_scores.mean(), train_time.mean(), test_time.mean()


def run_1_layer_NN_hyper_test(x_data,y_data):
    random_state = 100
    single_train_scores = []
    single_test_scores = []

    nodes_plot = []
    for i in range(1,10):
        print('Nodes: ', i)
        hidden_nodes = []
        hidden_nodes.append(i)
        nodes_plot.append(i)
        train_score, test_score, train_time, test_time = get_NN_results_2(x_data, y_data, hidden_nodes, 'relu')
        single_train_scores.append(train_score)
        single_test_scores.append(test_score)

    plt.figure(1)
    plt.plot(nodes_plot, single_train_scores,'-o',color='black')
    plt.plot(nodes_plot, single_test_scores,'-',color='green')
    plt.legend(['Train score','CV score'],fontsize=12)
    plt.title('Single hidden layer Neural Network performance',fontsize=12)
    plt.xlabel('Nodes in the hidden layer',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.show()

def run_multi_layer_NN_hyper_test(x_data,y_data, node_count):
    layers = [1,2,3,4,5] ## 1 to 22
    activations = ['relu','sigmoid']
    relu_train_score = [] 
    relu_test_score = [] 
    relu_train_time = []
    relu_test_time = []
    sigmoid_train_score = []
    sigmoid_test_score = []
    sigmoid_train_time = []
    sigmoid_test_time = []

    for activation in activations:
        for layer in layers:
            hidden_nodes = [node_count]*layer
            print('Hidden nodes :', hidden_nodes)
            train_score, test_score, train_time, test_time = get_NN_results(x_data, y_data, hidden_nodes, activation)
            if activation == 'relu':
                relu_train_score.append(train_score)
                relu_test_score.append(test_score)
                relu_train_time.append(train_time)
                relu_test_time.append(test_time)
            if activation == 'sigmoid':
                sigmoid_train_score.append(train_score)
                sigmoid_test_score.append(test_score)
                sigmoid_train_time.append(train_time)
                sigmoid_test_time.append(test_time)
           
    plt.figure(1)
    plt.plot(layers, relu_train_score,'-o', dashes = [6,2],color='black')
    plt.plot(layers, sigmoid_train_score,'-o',color='green')
    plt.plot(layers, relu_test_score,'-', dashes = [6,2],color='black')
    plt.plot(layers, sigmoid_test_score,'-',color='green')
    plt.legend(['relu train','sigmoid train', 'relu CV', 'sigmoid CV'],fontsize=12)
    plt.title('Neural Network performance',fontsize=12)
    plt.xlabel('Hidden layers',fontsize=12)
    plt.ylabel('Score',fontsize=12)
    plt.xticks(layers)
    plt.figure(2)
    plt.plot(layers, relu_train_time,'-o',dashes = [6,2],color='black')
    plt.plot(layers, sigmoid_train_time,'-o',color='green')   
    plt.plot(layers, relu_test_time,'-',dashes = [6,2],color='black')
    plt.plot(layers, sigmoid_test_time,'-',color='green')    
    plt.legend(['relu train','sigmoid train','relu CV', 'sigmoid CV'],fontsize=12)
    plt.title('Neural Network time performance',fontsize=12)
    plt.xlabel('Hidden layers',fontsize=12)
    plt.ylabel('Time (s)',fontsize=12)
    plt.xticks(layers)
    plt.show()

def run_learning_curve(X,y):
    print('Run learning curve ...')
    estimators = []
    names = ['DT','AdaBoost','KNN','SVM','NN']
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    estimator = DecisionTreeClassifier()
    estimators.append(estimator)
    estimator = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 10)
    estimators.append(estimator)
    estimator = KNeighborsClassifier()
    estimators.append(estimator)
    estimator = SVC(gamma='auto')
    estimators.append(estimator)
    estimator = mlrose.NeuralNetwork(
        algorithm = 'gradient_descent', max_iters = 1000, \
        bias = True, is_classifier = True, learning_rate = 0.0001, \
        early_stopping = True, clip_max = 5, max_attempts = 100, \
        random_state = 3)
    estimators.append(estimator)

    plt.figure(0)
    for  i in range (0,len(estimators)):
        print('Learning curve for ', names[i])
        train_sizes_abs, train_scores, test_scores = learning_curve(estimators[i], X, y, 
            groups=y, 
            train_sizes=np.array([0.1,0.3, 0.55, 0.78, 1. ]), cv = 5, shuffle=True,
            scoring='f1', n_jobs=4, random_state=100)
        train_scores = train_scores.mean(axis=1)
        test_scores = test_scores.mean(axis=1)
        plt.plot(train_sizes_abs,train_scores,'-o',color=colors[i])
        plt.plot(train_sizes_abs,test_scores,color =colors[i])

    plt.legend(['DT train','DT test',
                'AdaBoost train', 'AdaBoost test',
                'KNN train','KNN test',
                'SVM train','SVM test',
                'NN train','NN test'],fontsize=10)
    plt.title('Learning curves for five algorithms',fontsize=10)
    plt.xlabel('Training size',fontsize=10)
    plt.ylabel('Score',fontsize=10)
    plt.show()

# ********************************************
# ************ RUN EXPERIMENTS ***************
# ********************************************   

def main(problem, learning_curve_flag, DT_flag, Boost_flag, Bagging_flag ,KNN_flag, SVM_flag, NN_multi_flag):
    NN_flag = 0
    if problem == 'bank':
        # PREPROCESS BANK DATA
        data = pd.read_csv('bank-full.csv',sep=';')
        data.drop(['day','month'],axis=1,inplace=True)
        data['y'].replace(['no'],0,inplace=True)
        data['y'].replace(['yes'],1,inplace=True)
        # convert data to numeric where possible
        data = data.apply(pd.to_numeric, errors='ignore', downcast='float')
        x_data = data.loc[:, data.columns != "y"]
        y_data = data.loc[:, "y"]
        numerical_features = x_data.dtypes == 'float32'
        categorical_features = ~numerical_features
        #onehotencoder = OneHotEncoder(categorical_features = [0]) 
        #data = onehotencoder.fit_transform(data).toarray() 
        random_state = 100
        preprocess = make_column_transformer(
            (OneHotEncoder(),categorical_features), 
            (StandardScaler(), numerical_features),
            remainder="passthrough")
        x_data = preprocess.fit_transform(x_data)


        #plt.hist(y_data)
        #plt.show()

        #one_hot = OneHotEncoder()
        #y_data = one_hot.fit_transform(y_data.resshape(1,-1)).todense()
        # RUN BANK DATA EXPERIMENT

        print("Running Bank Marketing problem ....")
        if learning_curve_flag:
            'Generating learning curves ...'
            run_learning_curve(x_data,y_data)

        # Hold out test set for final performance measure
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.3, random_state=random_state, shuffle=True, stratify=y_data)

        if DT_flag: 
            print('Run DT analysis ....')
            run_DT_hyper_test(x_train, y_train)
            print('Calculating test score: ')
            clf = DecisionTreeClassifier(min_samples_leaf=25,
                                                splitter = 'best',
                                                criterion='gini')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final score: ", final_score)
            print('\n')

        if Boost_flag: 
            print('Run AdaBoost analysis ...')
            run_Boost_hyper_test(x_train, y_train, leaf_sizes=[25,40], split_method = 'best',boost_method = 'Ada')
            print('Calculating test score: ')
            clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=25,
                    splitter = 'best',
                    criterion='gini'),
                    n_estimators=1)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

        if Bagging_flag: 
            print('Run Bagging analysis ...')
            run_Boost_hyper_test(x_train, y_train, leaf_sizes=[25,40], split_method = 'best', boost_method = 'Bagging')
            print('Calculating test score: ')
            clf = BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=25,
                    splitter = 'best',
                    criterion='gini'),
                    n_estimators=13)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

        if KNN_flag: 
            print('Run KNN analysis ....')
            run_KNN_hyper_test(x_train, y_train)
            print('Calculating test score: ')
            clf = KNeighborsClassifier(n_neighbors=3, weights = 'uniform')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

        if SVM_flag: 
            print('Run SVM analysis ....')
            run_SVM_hyper_test(x_train, y_train)
            print('Calculating test score: ')
            clf = SVC( kernel ='rbf',C = 10, gamma='auto')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

        if NN_flag:
            print('Neural Network ...')
            run_1_layer_NN_hyper_test(x_data,y_data)

        if NN_multi_flag:
            print('Run NN (multi layer) analysis ...')
            run_multi_layer_NN_hyper_test(x_train, y_train, 6)
            print('Calculating test score: ')
            clf = mlrose.NeuralNetwork(
                hidden_nodes = [6], activation = 'relu', \
                algorithm = 'gradient_descent', max_iters = 1000, \
                bias = True, is_classifier = True, learning_rate = 0.0001, \
                early_stopping = True, clip_max = 5, max_attempts = 100, \
                random_state = 30)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

    if problem == 'tree':
        # PREPROCESS WILT DATA
        data = pd.read_csv('wilt_full.csv')
        data['class'].replace(['n'],0,inplace=True)
        data['class'].replace(['w'],1,inplace=True)
        x_data = data.loc[:, data.columns != 'class']
        y_data = data.loc[:,'class']
        scaler = StandardScaler()
        x_data = scaler.fit_transform(x_data)

        random_state = 100

        #
        #plt.hist(y_data)
        #plt.show()

        # RUN WILT DATA EXPERIMENT
        print("Running Diseased Trees problem ....")
        if learning_curve_flag:
            run_learning_curve(x_data,y_data)

        # Hold out test set for final performance measure
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.3, random_state=random_state, shuffle=True, stratify=y_data)

        if DT_flag: 
            print('Run DT analysis ....')
            run_DT_hyper_test(x_train, y_train)
            print('Calculating test score: ')
            clf = DecisionTreeClassifier(min_samples_leaf=3,
                                                splitter = 'best',
                                                criterion='gini')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

        if Boost_flag: 
            print('Run AdaBoost analysis ...')
            run_Boost_hyper_test(x_train, y_train, leaf_sizes=[3,40], split_method = 'best',boost_method = 'Ada')
            print('Calculating test score: ')
            clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=3,
                    splitter = 'best',
                    criterion='gini'),
                    n_estimators=1)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

        if Bagging_flag: 
            print('Run Bagging analysis ...')
            run_Boost_hyper_test(x_train, y_train, leaf_sizes=[3,40], split_method = 'best', boost_method = 'Bagging')
            print('Calculating test score: ')
            clf = BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=3,
                    splitter = 'best',
                    criterion='gini'),
                    n_estimators=11)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

        if KNN_flag: 
            print('Run KNN analysis ....')
            run_KNN_hyper_test(x_train, y_train)
            print('Calculating test score: ')
            clf = KNeighborsClassifier(n_neighbors=1, weights = 'uniform')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

        if SVM_flag: 
            print('Run SVM analysis ....')
            run_SVM_hyper_test(x_train, y_train)
            print('Calculating test score: ')
            clf = SVC(kernel= 'rbf',C = 10,gamma='auto')
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')


        if NN_flag:
            print('Run NN (1 layer) analysis ...')
            run_1_layer_NN_hyper_test(x_train, y_train)
            print('Calculating test score: ')
            clf = mlrose.NeuralNetwork(
                hidden_nodes = [10], activation = 'relu', \
                algorithm = 'gradient_descent', max_iters = 1000, \
                bias = True, is_classifier = True, learning_rate = 0.0001, \
                early_stopping = True, clip_max = 5, max_attempts = 100, \
                random_state = 3)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

        if NN_multi_flag:
            print('Run NN (multi layer) analysis ...')
            run_multi_layer_NN_hyper_test(x_train, y_train, 6)
            print('Calculating test score: ')
            clf = mlrose.NeuralNetwork(
                hidden_nodes = [6,6], activation = 'relu', \
                algorithm = 'gradient_descent', max_iters = 1000, \
                bias = True, is_classifier = True, learning_rate = 0.0001, \
                early_stopping = True, clip_max = 5, max_attempts = 100, \
                random_state = 30)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_score = f1_score(y_test, y_pred)
            print("Final test score: ", final_score)
            print('\n')

if __name__ == "__main__" :
    import argparse
    print("Running Supervised Learning Experiments")
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='tree')
    parser.add_argument('--LearningCurve', default=0)
    parser.add_argument('--DT', default=0)
    parser.add_argument('--Ada', default=0)
    parser.add_argument('--Bagging', default=0)
    parser.add_argument('--KNN', default=0)
    parser.add_argument('--SVM', default=0)
    parser.add_argument('--NN', default=0)

    args = parser.parse_args()
    problem = args.problem
    learning_curve_flag = args.LearningCurve
    DT_flag = args.DT
    Boost_flag = args.Ada
    Bagging_flag = args.Bagging
    KNN_flag = args.KNN
    SVM_flag = args.SVM
    NN_multi_flag = args.NN
    if args.problem == 'bank':
        print("Running bank problem:...")
    if args.problem == 'tree':
        print("Running tree problem:...")
    main(problem, learning_curve_flag, DT_flag, Boost_flag, Bagging_flag ,KNN_flag, SVM_flag, NN_multi_flag)
  




