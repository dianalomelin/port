#%% Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

COLUMNS_TO_KEEP = [
     'Severity'
    ,'Start_Time'
    ,'End_Time'
    ,'Start_Lat'
    ,'Start_Lng'
    ,'Distance(mi)'
    ,'Side'
    ,'State'
    ,'Temperature(F)'
    ,'Humidity(%)'
    ,'Pressure(in)'
    ,'Visibility(mi)'
    ,'Wind_Direction'
    ,'Wind_Speed(mph)'
    ,'Weather_Condition'
    ,'Amenity'
    ,'Bump'
    ,'Crossing'
    ,'Give_Way'
    ,'Junction'
    ,'No_Exit'
    ,'Railway'
    ,'Roundabout'
    ,'Station'
    ,'Stop'
    ,'Traffic_Calming'
    ,'Traffic_Signal'
    ,'Turning_Loop'
    ,'Sunrise_Sunset'
]

CATEGORICAL_COLUMNS = [
     'Side'
    ,'State'
    ,'Wind_Direction'
    ,'Weather_Condition'
    ,'Sunrise_Sunset'
]

FLOAT_COLUMNS = [
     'Start_Lat'
    ,'Start_Lng'
    ,'Distance(mi)'    
    ,'Temperature(F)'
    ,'Humidity(%)'
    ,'Pressure(in)'
    ,'Visibility(mi)'
    ,'Wind_Speed(mph)'
]


#%% Declare functions
def loadAndProcessData(pathToFile, sampleSize = None, chart = 1):
    data = pd.read_csv(pathToFile)
    print(f'original shape: {data.shape}')


    # Only keep columns we care about
    data = data[COLUMNS_TO_KEEP]
    
    # convert bool types
    data = data.astype({
        'Amenity': 'int8',
        'Bump': 'int8',
        'Crossing': 'int8',
        'Give_Way': 'int8',
        'Junction': 'int8',
        'No_Exit': 'int8',
        'Railway': 'int8',
        'Roundabout': 'int8',
        'Station': 'int8',
        'Stop': 'int8',
        'Traffic_Calming': 'int8',
        'Traffic_Signal': 'int8',
        'Turning_Loop': 'int8',
        })

    # convert date types
    data['Start_Time'] = pd.to_datetime(data['Start_Time'], format=r'%Y-%m-%d %X')
    data['End_Time'] = pd.to_datetime(data['End_Time'], format=r'%Y-%m-%d %X')
    
    # add hour of the start time
    data['Hour'] = pd.DatetimeIndex(data['Start_Time']).hour

    # add duration feature
    data['Duration'] = (data['End_Time'] - data['Start_Time']).dt.seconds.astype('int8')
    data.drop(['Start_Time', 'End_Time'], axis=1, inplace=True)
    
    
    #check for correlation  - before dummies
       
    if chart == 1:
        fig, ax = plt.subplots(figsize=(18,18)) # Sample figsize in inches
        
        Var_Corr = data.corr()  #data.iloc[:,1:].corr()
        print('Correlation plot for sampled dataset')
        print(sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True, ax=ax) )
    
    
    
    # after correlation drop columns with only one value or highly correlated (Bump = Traffic Calming)
    data.drop(['Turning_Loop','Traffic_Calming','Roundabout'], axis=1, inplace=True)

    # break up categorical variables into one hot encoding
    
    for categoricalFeature in CATEGORICAL_COLUMNS:
        data = pd.concat([data, pd.get_dummies(data[categoricalFeature]).astype('int8')], axis=1)
        data = data.drop([categoricalFeature], axis=1)
    
    data.dropna(inplace=True)

    if sampleSize:
        data = data.sample(min(data.shape[0],sampleSize), axis = 0)
        

    data.reset_index(drop=True, inplace=True)
    

    # automatically find the columns that are not binary (0 or 1)
    maxSeries = data.max(axis=0)
    FLOAT_COLS_IDX = [i for i,value in enumerate(maxSeries) if value > 1 or value <0]
    
    FLOAT_COLS_IDX = [c-1 for c in FLOAT_COLS_IDX]
    del FLOAT_COLS_IDX[0]
    
    print(f'final shape: {data.shape}')
    

    return (data,  FLOAT_COLS_IDX)



# %% execute
# data, cols = loadAndProcessData('../data/US_Accidents_Dec19_LITE_10000.csv')
# data.head()


# %%
