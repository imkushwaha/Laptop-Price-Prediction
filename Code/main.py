#imports

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from preprocessing import data_processing
from train_test import splitData
from modelTraining import ModelTrainig

#path of directory where training data is stored
path = "G:\Deployment\Laptop Price Prediction\Code\laptop_data.csv"

#reading data and storing it to df object
df = pd.read_csv(path)

#Making object of data_processing class
df_preprocess = data_processing(df)

#storing processed data into object processed_data
processed_data = df_preprocess.start_preprocessing()

#final data after removing unnecessary columns from processed data
final_data = df_preprocess.dropColumn(processed_data)

#saving processed data
final_data.to_csv("final_data.csv", index = False)

print("Data Preprocessing Done !!")

#Model development starts here!!

print("Model Training Started !!")
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Model Training Started At:", current_time)

#making object of splitData
split_data = splitData(final_data)

#splitting final_data into train and test set
X_train, X_test, y_train, y_test = split_data.TrainTest(testSize=0.15)

#making object of ModelTraining class from modelTraining.py
modeltraining = ModelTrainig(X_train,X_test,y_train,y_test)

#one by one we will call different models define under ModelTraining class 

#Calling LinearRegression method from ModelTraining class and storing evaluation metrics
R2Score_lin,MAE_lin,Model_lin = modeltraining.LinearRegression()

#Calling RidgeRegression method from ModelTraining class and storing evaluation metrics
R2Score_Ridge,MAE_Ridge,Model_Ridge = modeltraining.RidgeRegression()

#Calling LassoRegression method from ModelTraining class and storing evaluation metrics
R2Score_Lasso,MAE_Lasso,Model_Lasso = modeltraining.LassoRegression()

#Calling KNN method from ModelTraining class and storing evaluation metrics
R2Score_KNN,MAE_KNN,Model_KNN = modeltraining.KNN()

#Calling DecisionTree method from ModelTraining class and storing evaluation metrics
R2Score_DT,MAE_DT,Model_DT = modeltraining.DecisionTree()

#Calling SVM method from ModelTraining class and storing evaluation metrics
R2Score_SVM,MAE_SVM,Model_SVM = modeltraining.SVM()

#Calling RandomForest method from ModelTraining class and storing evaluation metrics
R2Score_RF,MAE_RF,Model_RF = modeltraining.RandomForest()

#Calling ExtraTree method from ModelTraining class and storing evaluation metrics
R2Score_ET,MAE_ET,Model_ET = modeltraining.ExtraTrees()

#Calling AdaBoostRegressor method from ModelTraining class and storing evaluation metrics
R2Score_Ada,MAE_Ada,Model_Ada = modeltraining.AdaBoostRegressor()

#Calling XgBoostRegressor method from ModelTraining class and storing evaluation metrics
R2Score_Xg,MAE_Xg,Model_Xg = modeltraining.XgBoostRegressor()

#Calling GradientBoostRegressor method from ModelTraining class and storing evaluation metrics
R2Score_GB,MAE_GB,Model_GB = modeltraining.GradientBoostRegressor()

#Calling VotingRegressor method from ModelTraining class and storing evaluation metrics
R2Score_Vote,MAE_Vote,Model_Vote = modeltraining.VotingRegressor()

#Calling Stacking method from ModelTraining class and storing evaluation metrics
R2Score_Stack,MAE_Stack,Model_Stack = modeltraining.Stacking()


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Model Training Completed At:", current_time)

# storing all evaluation metrics in a dictionary

metrics_dict = {
    
                "Linear": (R2Score_lin,MAE_lin),
                "Ridge": (R2Score_Ridge, MAE_Ridge),
                "Lasso": (R2Score_Lasso, MAE_Lasso),
                "KNN": (R2Score_KNN, MAE_KNN),
                "DecisionTree": (R2Score_DT, MAE_DT),
                "SVM": (R2Score_SVM, MAE_SVM),
                "RandomForest": (R2Score_RF, MAE_RF),
                "ExtraTree": (R2Score_ET, MAE_ET),
                "AdaBoost": (R2Score_Ada, MAE_Ada),
                "GradientBoost": (R2Score_GB, MAE_GB),
                "XgBoost": (R2Score_Xg, MAE_Xg),
                "Voting": (R2Score_Vote, MAE_Vote),
                "Stacking": (R2Score_Stack, MAE_Stack)
                
                }

models = ['Linear',
          'Ridge',
          'Lasso',
          'KNN',
          'DecisionTree',
          'SVM',
          'RandomForest',
          'ExtraTree',
          'AdaBoost',
          'GradientBoost',
          'XgBoost',
          'Voting',
          'Stacking']

score_dict = {}

for model in models:
    score = round(float(metrics_dict[model][0].split(":")[1]),3)
    score_dict[model]= score
    
print(max(score_dict))



print(f"Code executed successfully, Fianl data saved at: {path}\n")

