#imports

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle
from datetime import datetime
from preprocessing import data_processing
from train_test import splitData
from modelTraining import ModelTrainig

#path of directory where training data is stored
path = open('trainingDataPath.txt', "r")

#reading data and storing it to df object
df = pd.read_csv(path.read())

#Making object of data_processing class
df_preprocess = data_processing(df)

print("Data Preprocessing Started....")

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


bestModel = max(score_dict)

def callingBestModel(bestModel):
    
    """[THIS FUNCTION IS TO RETURN BEST MODEL]
    
    Parameters:
        [bestModel]: [bestmodel object, which contain best model according to R2 score(max)]
    Returns:
        [MODEL]: [BEST MODEL ACCORDING TO R2 SCORE]
    """
    if bestModel=='Linear':
       model = Model_lin
       return model
   
    elif bestModel == 'Ridge':
        model = Model_Ridge
        return model
    
    elif bestModel == 'Lasso':
        model = Model_Lasso
        return model
    
    elif bestModel == 'KNN':
        model = Model_KNN
        return model
    
    elif bestModel == 'DecisionTree':
        model = Model_DT
        return model
    
    elif bestModel == 'SVM':
        model = Model_SVM
        return model
    
    elif bestModel == 'RandomForest':
        model = Model_RF
        return model
        
    elif bestModel == 'ExtraTree':
        model = Model_ET
        return model   
    
    elif bestModel == 'AdaBoost':
        model = Model_Ada
        return model   
    
    elif bestModel == 'GradientBoost':
        model = Model_GB
        return model   
    
    elif bestModel == 'XgBoost':
        model = Model_Xg
        return model   
    
    elif bestModel == 'Voting':
        model = Model_Vote
        return model   
    
    else:
        model = Model_Stack
        return model   
    
#calling best model function to return best model (maximum R2 score)
bestModel = callingBestModel(bestModel=bestModel)      



## Exporting the model and final data

def exportModel():
    pickle.dump(final_data,open('final_data.pkl','wb'))
    pickle.dump(bestModel,open('bestModel.pkl','wb'))

#calling exportModel
exportModel()

print("Model and final data successfully exported and saved in current working directory")


print("\t\t\t*****Complete Pipeline Code Executed Successfully*****")

#This is the end of pipeline.......!!