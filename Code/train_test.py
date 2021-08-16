import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class splitData():
    
    def __init__(self,df):
        self.df = df
        
    def TrainTest(self, testSize):
        
        X = self.df.drop("Price", axis = 1).copy()
        y = np.log(self.df["Price"])
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=testSize,random_state=2)
        
        return X_train, X_test, y_train, y_test
 