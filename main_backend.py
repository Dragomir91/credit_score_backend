from pyexpat import model
import pandas as pd
import numpy as np
import shap
import pickle
import shap
from fastapi import FastAPI
import sklearn
import os
from typing import Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import imblearn
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from lightgbm import LGBMClassifier, early_stopping
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
import json
import joblib


class Item(BaseModel):

    df : object

class Predict_id(BaseModel):
    id : int
    decision_id: int
    proba: float

class Explain_id(BaseModel):

    shap_values : object
 

class List_id(BaseModel):
    
    id_client: list
    information_client: list


class Info_id(BaseModel):
    
    id_client: int
    infos_id: list


app = FastAPI()

@app.get("/")
async def root():
    
    html_content = """
    <html>
        <head>
            <title>Bienvenue sur score credit</title>
        </head>
        <body>
            <h1>Bienvenue dans le site crédit score</h1>
            <h2>Ce serveur transmet des information sur la décision d'accordé un credit au client, des informations personnelles sur les clients voulants faire un emprunt.</h2>           
        </body>
    </html>
    """
      
    return HTMLResponse(content=html_content, status_code=200)
#######################HTML_RESPONSE#################################################

@app.get("/id_client")  
async def id_client():
    df_kernel = load_df()
    print('dans app get id client ',df_kernel.loc[df_kernel["SK_ID_CURR"] == 100002])

    return {"list_id" : list(df_kernel.loc[:200,'SK_ID_CURR']),
            "infos_id" : list(df_kernel.columns[:200])}
########################ID_CLIENT#######################################################

@app.get("/info_client")
async def info_client(info_id: Info_id):
    
    df_kernel = load_df()
    #print('id_client = ',  info_id.id_client)
    #print('info_client = ',  info_id.infos_id)
    df_id = df_kernel.loc[df_kernel['SK_ID_CURR'] == info_id.id_client,info_id.infos_id]

    #print('df_id = ',df_id)  
    return df_id.to_json()


########################ID_CLIENT#######################################################

@app.get("/info_clients/id_all")
async def info_client2(info_id: Info_id):
    
    df_kernel = load_df()
    print('id_client = ',  info_id.id_client)
    print('info_client = ',  info_id.infos_id)
    df_id = df_kernel.loc[:20,info_id.infos_id]

    print('df_id = ',df_id)  
    return df_id.to_json()

####################################PREDICT_ID#############################################

@app.post("/predict") 
async def predict_decision(pred_id : Predict_id):

    df_app_test = load_df()

    base_dir = "" 
    csv_file_path = os.path.join(base_dir, "model_lightgbm.pkl")

    
    with open(csv_file_path,'rb') as f:
            model_lightgbm = pickle.load(f)
    print(pred_id)       
    
    print(df_app_test)
    


    cols= [
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
        'DAYS_BIRTH',
        'EXT_SOURCE_1',
        'AMT_ANNUITY',
        'APPROVED_CNT_PAYMENT_MEAN',
        'INSTAL_DPD_MEAN',
        'PAYMENT_RATE',
        'CODE_GENDER',
        'INSTAL_AMT_PAYMENT_SUM',
        'DAYS_EMPLOYED',
        'AMT_CREDIT',
        'AMT_GOODS_PRICE']  
    
    df = df_app_test

    print('df shape : ',df.shape)    

    X = df[df.SK_ID_CURR == pred_id.id].loc[:,cols]
    print('x = ',X)

    y_pred = model_lightgbm.predict(X)
    
    y_pred_proba = model_lightgbm.predict_proba(X)[0]
    print('proba id : ', y_pred_proba)
    info_id = Predict_id(id=pred_id.id, decision_id = y_pred, proba=y_pred_proba[0])
    print(info_id)

    return info_id

#############################################info_detail_client########################################

@app.get("/predict/feature_id") 
async def predict_decision(ex_id : Predict_id):

    df_kernel = load_df()

    colonne = []
    
    for idx,i in enumerate(df_kernel.columns):
        colonne.append(i.replace(' ','_').replace(':','_').replace('-','_').replace('/','_').replace(',','_'))
        
    df_kernel.columns = colonne
    df_kernel = df_kernel.fillna(0)
    df_kernel = df_kernel.replace([np.inf, -np.inf], np.nan)

    base_dir = ""  # Vous devrez peut-être adapter ce chemin en fonction de votre configuration

    # Chemin absolu complet du fichier CSV
    csv_file_path = os.path.join(base_dir, "model_lightgbm_shap.pkl")

    print(csv_file_path)
    with open(csv_file_path,'rb') as f:
            shap_values = pickle.load(f)
            
 

    mean_abs_shap_values = shap_values.values.sum(axis = 0)

    # Trier les variables par ordre d'importance décroissante
    sorted_indices = np.argsort(mean_abs_shap_values)

    # Obtenir les noms des variables
    feature_names = df_kernel.iloc[:,1:].columns

    # Obtenir les noms des variables les plus importantes
    top_feature_names = [feature_names[idx] for idx in sorted_indices]

    print(df_kernel.loc[df_kernel.SK_ID_CURR == ex_id.id,feature_names[:15]])
    
  
    cols= ['EXT_SOURCE_2',
           'EXT_SOURCE_3',
           'DAYS_BIRTH',
           'EXT_SOURCE_1',
           'AMT_ANNUITY',
           'APPROVED_CNT_PAYMENT_MEAN',
           'INSTAL_DPD_MEAN',
           'PAYMENT_RATE',
           'CODE_GENDER',
           'INSTAL_AMT_PAYMENT_SUM',
           'DAYS_EMPLOYED',
           'AMT_CREDIT',
           'AMT_GOODS_PRICE']  
    
    print('___',df_kernel.loc[df_kernel.SK_ID_CURR== ex_id.id,cols])
    return df_kernel.loc[df_kernel.SK_ID_CURR==ex_id.id,cols].to_json()



#########################################EXPLAIN_SHAP_VALUES###########################################

@app.get("/predict/explain") 
async def predict_decision(explain_id : Predict_id):
    
    print('explain id = ', explain_id)
    
    df_kernel = load_df()    
    colonne = []

    for idx,i in enumerate(df_kernel.columns):
        colonne.append(i.replace(' ','_').replace(':','_').replace('-','_').replace('/','_').replace(',','_'))
    df_kernel.columns = colonne
   
    df = df_kernel.replace([np.inf, -np.inf], int(0))
    df = df.fillna(0) 
    # Chemin absolu vers le répertoire contenant le fichier CSV
    base_dir = ""  # Vous devrez peut-être adapter ce chemin en fonction de votre configuration

    # Chemin absolu complet du fichier CSV
    csv_file_path = os.path.join(base_dir, "model_lightgbm_shap.pkl")

    print(csv_file_path)
    with open(csv_file_path,'rb') as f:
            shap_values = pickle.load(f)
    
    # prendre le premire élément de la liste
    id = list(df.loc[df.SK_ID_CURR == explain_id.id].index)[0]

    shap_values.data[id] = np.nan_to_num(shap_values.data[id])
    shap_values.values[id] = np.nan_to_num(shap_values.values[id])

    # pd.DataFrame(tab_shap,index=df.SK_ID_CURR[:100] , columns=df_col)
    return {"values" : list(shap_values.values[id]),
            "base_values" : shap_values.base_values[id],
            "data"  : list(shap_values.data[id]),
            "feature_names" :shap_values.feature_names}


#########################################COUT_METIER############################################


def perte(dataframe, y_true, y_pred,col_credit, rate_default:float):
    
    df = {'Index':y_true.index,'y_pred':y_pred,'Target':y_true.values}
    df_pred = pd.DataFrame(df)
    idx = df_pred.loc[(df_pred.y_pred == 1) & (df_pred.Target == 0),'Index'] 
    dataframe['perte'] = -dataframe.loc[idx,col_credit] * rate_default
    return dataframe

def gain(df, y_proba, threshold:float):
        
        
        df['proba'] = y_proba[:,0]
        df['gain'] = 0
        df['y_pred'] = np.nan
        
        #df['gain'] = df.loc[(df.proba >= threshold), 'AMT_CREDIT'] * rate_interest
        df.loc[(df.proba >= threshold), 'y_pred'] = 0
        df.loc[(df.proba < threshold), 'y_pred'] = 1
            
                
        return df['gain']

@app.get("/predict/graph_id") 
async def predict_decision(cout_id : Predict_id):
    
    
    df_test = load_df()    
    df_kernel = load_df_kernel()
    cols= ['SK_ID_CURR',
           'TARGET',
           'EXT_SOURCE_2',
           'EXT_SOURCE_3',
           'DAYS_BIRTH',
           'EXT_SOURCE_1',
           'AMT_ANNUITY',
           'APPROVED_CNT_PAYMENT_MEAN',
           'INSTAL_DPD_MEAN',
           'PAYMENT_RATE',
           'CODE_GENDER',
           'INSTAL_AMT_PAYMENT_SUM',
           'DAYS_EMPLOYED',
           'AMT_CREDIT',
           'AMT_GOODS_PRICE']   
    colonne = []
    top_feature_names = []
    base_dir = "" 
    csv_file_path = os.path.join(base_dir, "model_lightgbm.pkl")

    with open(csv_file_path,'rb') as f:
            model_lightgbm = pickle.load(f)

    for idx,i in enumerate(df_test.columns):
        colonne.append(i.replace(' ','_').replace(':','_').replace('-','_').replace('/','_').replace(',','_'))
    df_test.columns = colonne
   
    df = df_test.replace([np.inf, -np.inf], np.nan)
    df.fillna(0,inplace=True) 
    # Chemin absolu vers le répertoire contenant le fichier CSV
    rus = RandomUnderSampler()
    X = df.loc[:,cols[2:]]
    
    print(df.shape)
    y_pred = model_lightgbm.predict(X)
    print(len(y_pred))
    df = df_test.loc[:,cols]
    df['Y_PRED_PV_ID'] = y_pred
    df['EXT_SOURCE_2_PV_ID'] = X['EXT_SOURCE_2']
    df['PAYMENT_RATE_PV_ID'] = X['PAYMENT_RATE']
    df['EXT_SOURCE_3_PV_ID'] = X['EXT_SOURCE_3']
    df['AMT_ANNUITY_PV_ID'] = X['AMT_ANNUITY']
    df['AMT_CREDIT_PV_ID'] = X['AMT_CREDIT']  

    print(df.iloc[:,10:])
    return  df.to_json()

def load_df_kernel():
    #df_app_test = pd.read_csv('source/application_test.csv',sep=',')
    #df_app_test = pd.read_csv('source/application_test.csv',sep=',')
    # Chemin absolu vers le répertoire contenant le fichier CSV
    base_dir = ""  # Vous devrez peut-être adapter ce chemin en fonction de votre configuratio
 
    csv_file_path = os.path.join(base_dir, "kernel_light.csv")

    # Vérifier si le fichier existe
    if os.path.exists(csv_file_path):
        # Lire le fichier CSV
        df = pd.read_csv(csv_file_path,sep = ',')
        print('dataframe kernel download')
        return  df

    else: 
         return print("Le fichier df_kernel CSV n'existe pas.")  
    
def load_df():
    #df_app_test = pd.read_csv('source/application_test.csv',sep=',')
    #df_app_test = pd.read_csv('source/application_test.csv',sep=',')
    # Chemin absolu vers le répertoire contenant le fichier CSV
    base_dir = ""  # Vous devrez peut-être adapter ce chemin en fonction de votre configuratio
 
    csv_file_path = os.path.join(base_dir, "app_test_light.csv")

    # Vérifier si le fichier existe
    if os.path.exists(csv_file_path):
        # Lire le fichier CSV
        df = pd.read_csv(csv_file_path,sep = ',')
        print('dataframe test download')

        return  df

    else: 
         return print("Le fichier df_kernel CSV n'existe pas.")  

def test_load_df():
    # Chemin absolu vers le répertoire contenant le fichier CSV
    base_dir = ""  
    # Chemin absolu complet du fichier CSV
    csv_file_path = os.path.join(base_dir, "kernel_light.csv")
    # Effectuer des assertions pour vérifier que le DataFrame est correct
    df = pd.read_csv(csv_file_path)
    assert isinstance(df, object)
    assert len(df) > 0  # Vérifier que le DataFrame n'est pas vide
    # Autres assertions selon votre cas d'utilisation
# Exécutez pytest pour exécuter les tests

if __name__ == '__main__':
    df_kernel = load_df()
    print('fin')   

