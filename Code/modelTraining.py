#imports
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

class ModelTrainig():
    
    def __init__(self,X_train,X_test,y_train,y_test):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def LinearRegression(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = LinearRegression()

        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}',pipe
    
    def RidgeRegression(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = Ridge(alpha=10)

        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}', pipe
    
    
    def LassoRegression(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = Lasso(alpha=0.001)

        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}',pipe
    
    
    def KNN(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = KNeighborsRegressor(n_neighbors=3)

        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}',pipe
    
    
    def DecisionTree(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = DecisionTreeRegressor(max_depth=8)

        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}', pipe
    
    
    def SVM(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)


        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}', pipe
    
    
    def RandomForest(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)
        
        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}', pipe
    
    
    def ExtraTrees(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = ExtraTreesRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}', pipe
    
    
    def AdaBoostRegressor(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)
        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}', pipe
    
    
    def GradientBoostRegressor(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = GradientBoostingRegressor(n_estimators=500)
        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}',pipe
    
    
    def XgBoostRegressor(self):
        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

        step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)
        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
 
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}', pipe
    
    
    def VotingRegressor(self):
        
        from sklearn.ensemble import VotingRegressor,StackingRegressor

        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')


        rf = RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)
        gbdt = GradientBoostingRegressor(n_estimators=100,max_features=0.5)
        xgb = XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5)
        et = ExtraTreesRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=10)

        step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb',xgb), ('et',et)],weights=[5,1,1,1])

        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
        
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}', pipe

    
    
    def Stacking(self):
        
        from sklearn.ensemble import VotingRegressor,StackingRegressor

        step1 = ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')


        estimators = [('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))]

        step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

        pipe = Pipeline([('step1',step1),('step2',step2)])

        pipe.fit(self.X_train,self.y_train)

        y_pred = pipe.predict(self.X_test)
        
        return f'R2 score:{r2_score(self.y_test,y_pred)}' ,f'MAE:{mean_absolute_error(self.y_test,y_pred)}', pipe




    
    
        