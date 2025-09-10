import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def grid_dt(X_train, y_train):
    model_dt = DecisionTreeClassifier(random_state=1)
    class_weight =  [{0:0.05, 1:0.95},
                     {0:0.1, 1:0.9},
                      {0:0.5, 1:0.5},
                      {0:0.4, 1:0.6}]
    max_depth = [4, 5, 6]
    min_samples_leaf = [13, 14, 15]
    criterion  = ["gini", "entropy", "log_loss"]
    grid_dt = dict(class_weight=class_weight, max_depth=max_depth, min_samples_leaf=min_samples_leaf, criterion=criterion)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search_dt = GridSearchCV(estimator=model_dt, param_grid=grid_dt, n_jobs=-1, cv=cv,
                           scoring='accuracy',error_score='raise')
    grid_result_dt = grid_search_dt.fit(X_train, y_train)
    return grid_result_dt.best_estimator_

def grid_RandomForest(X_train, y_train):
    model_rf = RandomForestClassifier(random_state=1)
    class_weight=[{0: 0.3, 1: 0.7},
                  {0: 0.4, 1: 0.6},
                  {0: 0.5, 1: 0.5}]
    n_estimators = [40, 50, 60]
    criterion = ['gini', 'entropy', 'log_loss']
    min_samples_split = [12, 13, 14]
    max_depth = [9, 10, 11]
    grid_rf = dict(n_estimators = n_estimators, criterion = criterion,
                min_samples_split = min_samples_split, max_depth = max_depth, class_weight=class_weight)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search_rf = GridSearchCV(estimator=model_rf, param_grid=grid_rf, n_jobs=-1, cv=cv,
                                scoring='accuracy',error_score='raise')
    grid_result_rf = grid_search_rf.fit(X_train, y_train)
    return  grid_result_rf.best_estimator_

def grid_Adaboost(X_train, y_train):
    model_adab = AdaBoostClassifier(random_state=1)
    n_estimators = [200, 250, 300, 350, 400]
    learning_rate = np.linspace(0.01, 0.1, 40)
    grid_adab = dict(n_estimators=n_estimators, learning_rate=learning_rate)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    grid_search_adab = GridSearchCV(estimator=model_adab, param_grid=grid_adab, n_jobs=-1, cv=cv,
                               scoring='accuracy', error_score='raise')
    grid_result_adab = grid_search_adab.fit(X_train, y_train)
    return grid_result_adab.best_estimator_

url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
df = pd.read_csv(url)
df.head()

X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    stratify = y,
                                                    random_state = 46)
# Modelo "DecisionTreeClassifier"
modelo_dt = grid_dt(X_train, y_train)
predicciones_dt = modelo_dt.predict(X_test)
print(classification_report(y_test, predicciones_dt))

# Modelo "RandomForestClassifier"
modelo_rf = grid_RandomForest(X_train, y_train)
predicciones_rf = modelo_rf.predict(X_test)
print(classification_report(y_test, predicciones_rf))

# Modelo "AdaboostClassifier"
modelo_adab = grid_Adaboost(X_train, y_train)
predicciones_adab = modelo_adab.predict(X_test)
print(classification_report(y_test, predicciones_adab))

modelo_dt

modelo_rf

modelo_adab





