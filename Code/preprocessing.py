import pandas as pd
import numpy as np




class data_processing():
    
    def __init__(self, df):
        
        self.df = df
        
    
    def start_preprocessing(self):
        self.df["Ram"] = self.df["Ram"].str.replace("GB", "")
        self.df["Weight"] = self.df["Ram"].str.replace("kg", "")   
        
        self.df["Ram"] = self.df["Ram"].astype('int32')
        self.df["Weight"] = self.df["Weight"].astype('float32')
        
        self.df["Touchscreen"] = self.df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in x else 0)

        self.df["Ips"] = self.df["ScreenResolution"].apply(lambda x: 1 if "IPS" in x else 0)
        
        temp_df = self.df["ScreenResolution"].str.split('x', n=1, expand = True)
        
        self.df['X_res'] = temp_df[0]
        self.df['Y_res'] = temp_df[1]     
        
        self.df["X_res"] = self.df["X_res"].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0]) 

        self.df["X_res"] = self.df['X_res'].astype('int')
        self.df["Y_res"] = self.df['Y_res'].astype('int')
        
        self.df["PPi"] = (((self.df["X_res"]**2) + (self.df["Y_res"]**2))**0.5/self.df["Inches"]).astype('float')
        
        self.df["Cpu Name"] = self.df["Cpu"].apply(lambda x: " ".join(x.split()[0:3]))
        
        self.df["Cpu brand"] = self.df["Cpu Name"].apply(self.fetch_processor)
        
        
        ## HDD SDD Hybrid Flash Storage creation from memory column
        
        self.df['Memory'] = self.df['Memory'].astype(str).replace('\.0', '', regex=True)
        self.df["Memory"] = self.df["Memory"].str.replace('GB', '')
        self.df["Memory"] = self.df["Memory"].str.replace('TB', '000')
        new = self.df["Memory"].str.split("+", n = 1, expand = True)

        self.df["first"]= new[0]
        self.df["first"]=self.df["first"].str.strip()

        self.df["second"]= new[1]

        self.df["Layer1HDD"] = self.df["first"].apply(lambda x: 1 if "HDD" in x else 0)
        self.df["Layer1SSD"] = self.df["first"].apply(lambda x: 1 if "SSD" in x else 0)
        self.df["Layer1Hybrid"] = self.df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
        self.df["Layer1Flash_Storage"] = self.df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

        self.df['first'] = self.df['first'].str.replace(r'\D', '')

        self.df["second"].fillna("0", inplace = True)

        self.df["Layer2HDD"] = self.df["second"].apply(lambda x: 1 if "HDD" in x else 0)
        self.df["Layer2SSD"] = self.df["second"].apply(lambda x: 1 if "SSD" in x else 0)
        self.df["Layer2Hybrid"] = self.df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
        self.df["Layer2Flash_Storage"] = self.df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

        self.df['second'] = self.df['second'].str.replace(r'\D', '')

        self.df["first"] = self.df["first"].astype(int)
        self.df["second"] = self.df["second"].astype(int)

        self.df["HDD"]=(self.df["first"]*self.df["Layer1HDD"]+self.df["second"]*self.df["Layer2HDD"])
        self.df["SSD"]=(self.df["first"]*self.df["Layer1SSD"]+self.df["second"]*self.df["Layer2SSD"])
        self.df["Hybrid"]=(self.df["first"]*self.df["Layer1Hybrid"]+self.df["second"]*self.df["Layer2Hybrid"])
        self.df["Flash_Storage"]=(self.df["first"]*self.df["Layer1Flash_Storage"]+self.df["second"]*self.df["Layer2Flash_Storage"])

    
        # GPU Brand column from GPU column
        
        self.df["Gpu brand"] = self.df["Gpu"].apply(lambda x: x.split()[0])
        
        #Selecting only those data which are not equal to Gpu brand ARM
        
        self.df = self.df[self.df["Gpu brand"] != "ARM"]
        
        
        #Operating system 
        
        self.df["Os"] = self.df["OpSys"].apply(self.cat_os)
        
        
        return self.df
        
        
    def cat_os(self,inp):
        if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
            return 'Windows'
        elif inp == 'macOS' or inp == 'Mac OS X':
            return 'Mac'
        else:
            return 'Others/No OS/Linux'
        
          
    def fetch_processor(self,text):
        
        if text == "Intel Core i7" or text == "Intel Core i5" or text == "Intel Core i3":
            return text
   
        else:
            if text.split()[0] == "Intel":
                return "Other Intel Processor"

            else:
                return "AMD Processor"
        
       
        
        
    def dropColumn(self,df):
        
        """This method is for dropping unnecessary columns
        
           Parameter: data frame
        """
        
        #making list of unnecessary columns
        
        columns_ToDrop = ["Unnamed: 0", "ScreenResolution", "Inches", "X_res", "Y_res", "Cpu Name", "Cpu",
                          'first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                          'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
                          'Layer2Flash_Storage', "Memory", "Hybrid","Flash_Storage", "Gpu", "OpSys"]
        
        df.drop(columns = columns_ToDrop, axis = 1, inplace = True)
        
        return df
        
        
        

    
    
        

    
    