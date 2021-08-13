import pandas as pd
import numpy as np
from preprocessing import data_processing


path = "G:\Deployment\Laptop Price Prediction\Code\laptop_data.csv"
df = pd.read_csv(path)

df_preprocess = data_processing(df)

processed_data = df_preprocess.start_preprocessing()

final_data = df_preprocess.dropColumn(processed_data)

#saving processed data


final_data.to_csv("final_data.csv", index = False)

print(f"Code executed successfully, Fianl data saved at: {path}\n")

