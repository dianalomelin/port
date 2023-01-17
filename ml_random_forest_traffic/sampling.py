import pandas as pd

#### upload the full file 
datafull = pd.read_csv("data/US_Accidents_Dec19.csv")
print(f'original shape: {datafull.shape}')

#print(datafull['Severity'].unique())
print(datafull.groupby('Severity').count()['ID'])

## some columns have NA and get rid of all observations.
COLUMNS_TO_KEEP = [
    'ID'
    ,'Severity'
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

#filter on the columns
datafull = datafull[COLUMNS_TO_KEEP]

#datafull.dropna(axis='columns')
datafull.dropna(inplace=True)
print(f'after drop NA shape: {datafull.shape}')

#print(datafull['Severity'].unique())
print(datafull.groupby('Severity').count()['ID'])

### Balanced Sampling script
### source: https://stackoverflow.com/questions/23455728/scikit-learn-balanced-subsampling
### uspl - without replacement

def balanced_spl_by(df, lblcol, uspl=True):
    datas_l = [ df[df[lblcol]==l].copy() for l in list(set(df[lblcol].values)) ]
    lsz = [f.shape[0] for f in datas_l ]
    return pd.concat([f.sample(n = (min(lsz) if uspl else max(lsz)), replace = (not uspl)).copy() for f in datas_l ], axis=0 ).sample(frac=1) 

# do balanced sampling
balanced_data = balanced_spl_by(datafull,'Severity')

data_not1 = datafull[datafull['Severity']!=1]


unbalanced_data = data_not1.sample(n=6784,  replace = False)

print('shapes', balanced_data.shape, unbalanced_data.shape)
print(type(balanced_data), type(unbalanced_data))

balunbal_data = pd.concat([balanced_data, unbalanced_data])


# reset index
balunbal_data.reset_index(drop=True, inplace=True)

print(f'new shape: {balunbal_data.shape}')
# Write to excel file
balunbal_data.to_csv('../data/US_Accidents_Dec19_LITE_10000.csv', index=False)
print('excel file created')
