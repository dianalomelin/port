from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def random_forest_sklearn(fs, dpt, X, y):
    
    # Initializing a random forest - use same accuracy measurement as our model
    randomForestskl = RandomForestClassifier(n_estimators = fs, max_depth = dpt, bootstrap=True, criterion='entropy', oob_score = True, random_state=0)

    rows = X.shape[0]
    ycol = y.reshape((rows,))
    
    # Building trees in the forest
    print("fitting the forest")
    randomForestskl.fit(X,ycol)

    print("done fitting")
    
    # Prdict labels
    print("predicting")
    y_predicted = randomForestskl.predict(X)

    # Comparing predicted and true labels
    results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))

    sc = randomForestskl.score(X, y, sample_weight=None)
    oob = randomForestskl.oob_score
    
    
    ## compare, decide which one to return
    print("accuracy: %.4f" % accuracy)
    print("score", sc)
    #print("oob_score", oob)
    
    return accuracy


def single_decision_tree_sklearn(cols, dpt, X, y):
    
    treeskl = DecisionTreeClassifier(max_depth = dpt, criterion='entropy', random_state=0)

    ## since there's no voting, we need a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # Building trees in the forest
    print("fitting the forest")
    treeskl.fit(X_train,y_train)

    print("done fitting")

    # Accuracy
    accuracy = treeskl.score(X_test, y_test)
    
    return accuracy


def cart_skl(dpt, X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # initialize
    treeclf = DecisionTreeClassifier(max_depth= dpt, random_state=0, criterion='entropy')
    
    # fit model
    treeclf.fit(X_train, y_train)
    #score = tree.score(X_test, y_test)
    
    # predict
    y_predicted = treeclf.predict(X_test)
    
    # auc
    #auc = roc_auc_score(y_test, y_predicted)
    
    ## accuracy 
    acc = accuracy_score(y_test, y_predicted)
    
    return acc


def feature_importance(fs, dpt, X, y, colnames):
## Feature importance (RF_SKL)  
## Source: https://chrisalbon.com/machine_learning/trees_and_forests/feature_importance/

    # Initializing a random forest - use same accuracy measurement as our model
    randomForestskl = RandomForestClassifier(n_estimators = fs, max_depth = dpt, bootstrap=True, criterion='entropy', oob_score = True, random_state=0) 

    rows = X.shape[0]
    ycol = y.reshape((rows,))
    
    # Building trees in the forest
    model = randomForestskl.fit(X,ycol)

    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

   # Rearrange feature names so they match the sorted feature importances
    names = [colnames[i] for i in indices]
    names = names[:10]

    # Create plot
    #plt.figure()
    fig, ax= plt.subplots(1, 1, figsize=(12, 4))

    # Create plot title
    ax.set_title("Feature Importance", loc='center', size=16)

    xbar = range(X.shape[1])
    xbar = xbar[:10]
    ybar = importances[indices]
    ybar = ybar[:10]
    # Add bars
    #ax.bar(range(X.shape[1]), importances[indices])
    ax.bar(xbar, ybar) 

    # Add feature names as x-axis labels
    plt.xticks(xbar, names, rotation=90, size=14)  #names or indices


    # Show plot
    plt.show()

'''
#Hyper-parameters and Grid Search
params = {'n_estimators': [4, 16, 256],'max_depth': [2, 8, 16]}


#rdFor = RandomForestClassifier(random_state=614)

clf = GridSearchCV(rdFor, params)

clf.fit(x_train, y_train)


# Get the best params, using .best_params_
print(clf.best_params_)

# Get the best score, using .best_score_.
print(clf.best_score_)

'''

