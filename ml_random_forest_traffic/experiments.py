from data_process import loadAndProcessData
from sklearn.model_selection import train_test_split
from datetime import datetime
import models as md
import pandas as pd

def compareModels(pathfile, sampleSize, forest_sizes, tree_depths):

    start_main = datetime.now()
    
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels
            
    # Loading data set - Pandas version
    print("reading the data")
    data, cols  = loadAndProcessData(pathfile, sampleSize) 
    colnames = data.columns[1:]
    
    XX = data.to_numpy()
    
    ## label on first column 
    y = XX[:,:1]  
    X = XX[:,1:] 
    
    
    ## Hyper-parameters
    models = ['RF_SKL', 'DT_SKL']
   
    
    model_accuracy = []
    
    ## Run models with different options
    for m in models[:1]: 
        #if m != 'DT' and m != 'DT_SKL':
        for fs in forest_sizes:
            for dpt in tree_depths:

                start = datetime.now()
                print('############################')
                print(f"Model: {m}, forest size: {fs}, depth:{dpt}" )  ## remove                
                if  m == 'RF_SKL':
                    ### using sklearn
                    acc = md.random_forest_sklearn(fs, dpt, X, y)


                else:
                    break

                exec_time = str(datetime.now() - start)
                model_accuracy.append([m, fs, dpt, acc, exec_time])
                print("Model Accuracy:", acc, "Execution time: ", exec_time)
    
    
    for m in models[1:]:    
        #else:
            for dpt in tree_depths:
                
                start = datetime.now()
                print('############################')
                print(f"Model: {m}, depth:{dpt}" )  ## remove                  
                                
                if m == 'DT_SKL':
                    acc = md.cart_skl(dpt, X, y) 
                    
                    
                exec_time = str(datetime.now() - start)
                model_accuracy.append([m, 0, dpt, acc, exec_time])
                print("Model Accuracy:", acc, "Execution time: ", exec_time)
    
    ## format results
    results = pd.DataFrame(model_accuracy)
    results.columns = ['Model','Forest Size','Max Depth','Accuracy','ExecTime']
    
    
    #exec time 
    exec_time_main = str(datetime.now() - start_main)
    print('Total execution time', exec_time_main)

    
    ## Feature importance from RF_SKL
    rf_res = results[results['Model']=='RF_SKL' ]
    print('RF_SKL all',rf_res)
    best = rf_res[rf_res['Accuracy']==rf_res['Accuracy'].max()]
    #print(best)
    best = best[(best['Forest Size']==best['Forest Size'].min()) & (best['Max Depth']==best['Max Depth'].min()) ]
    print('best',best)
    fsrf = best.iloc[:,1].values.min()
    dptrf = best.iloc[:,2].values.min()
    print(fsrf,dptrf)
    # Plot feature importance of best RF_SKL
    md.feature_importance(fsrf, dptrf, X, y, colnames)
    
      
       
    return results
    
    
    
    
    