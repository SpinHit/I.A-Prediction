import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error , make_scorer, mean_absolute_error
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy import stats
from scipy.stats import norm, skew 
import warnings
warnings.filterwarnings('ignore')




dataset = pd.read_csv('audi.csv')
# dataset


dataset.price = np.log1p(dataset.price )
y = dataset.price

x = dataset[[ 'mpg', 'engineSize', 'year', 'mileage']] 
y = dataset[['price']]

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=123413246)

X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


lr = LinearRegression()
lr.fit(X_train,Y_train)
test_pre = lr.predict(X_test)
train_pre = lr.predict(X_train)

#tracé entre les valeurs prédites et les résidus
plt.scatter(train_pre, train_pre - Y_train, c = "blue",  label = "Data d'entrainement")
plt.scatter(test_pre,test_pre - Y_test, c = "black",  label = "Vrai data")
plt.title("Régression linéaire")
plt.xlabel("Valeurs prédites")
plt.ylabel("Résidus")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 6.5, xmax = 13, color = "red")
plt.show()


#Prédictions du tracé - Valeurs réelles
plt.scatter(train_pre, Y_train, c = "blue",  label = "Data d'entrainement")
plt.scatter(test_pre, Y_test, c = "black",  label = "Vrai data")
plt.title("Régression linéaire")
plt.xlabel("Valeurs prédites")
plt.ylabel("Valeurs réelles")
plt.legend(loc = "upper left")
plt.plot([8.5, 12], [8.5, 12], c = "red")
plt.show()


# calcul de la précision en utilisant l'erreur aboslue qui va nous nous permetre de trouver l'eurreur relative 
# 1 - erreur relative = Précision
def accuracy(y_test,y_pred):
    result = 0
    for i in range(len(y_test)):
        error = abs(y_test[i] - y_pred[i])# Erreur aboslue
        relerror = error / y_test[i]# l'erreur relative 
        result += relerror
   
    result = result / len(y_test)
    result = 1 - result
    return result

print('Erreur absolue moyenne :',mean_absolute_error(Y_test, test_pre))
print('Nous avons une précision de' , accuracy(Y_test.to_numpy()[0],test_pre[0]) * 100 ,'%')