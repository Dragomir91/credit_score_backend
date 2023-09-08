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
import imblearn
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from lightgbm import LGBMClassifier, early_stopping
from typing import Dict, Any


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
    age : int
    revenu_annuel:float
    days_employed: float
    credit : float
    good_price: float
    annuity: float


app = FastAPI()



@app.get("/")
async def root():
    
    html_content = """
    <html>
        <head>
            <title>information utilisateur</title>
        </head>
        <body>
            <h1>identifiants clients</h1>
        </body>
    </html>
    """
      
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/id_client")
async def info_client(info_id: List_id):
    print('df_ker ______________________________')
    df_kernel = load_df()
    print('df_ker:', df_kernel)
    info = List_id(id_client =df_kernel.SK_ID_CURR,information_client =df_kernel.columns)
    #print(info)
    return info

@app.post("/id_client/id")
async def info_client2(info_id: Info_id):
    
    df_kernel = load_df()
    print('id_client = ',  info_id.id_client)
    df_id = df_kernel.loc[df_kernel['SK_ID_CURR'] == info_id.id_client]
    
    info = Info_id(id_client =info_id.id_client,
                   age = df_id.DAYS_BIRTH,
                   revenu_annuel = df_id.AMT_INCOME_TOTAL,
                   days_employed =df_id.DAYS_EMPLOYED,
                   credit=df_id.AMT_CREDIT,
                   good_price = df_id.AMT_GOODS_PRICE,
                   annuity = df_id.AMT_ANNUITY)
    print(info)  
    return info

####################################PREDICT_ID#############################################

@app.post("/predict") 
async def predict_decision(pred_id : Predict_id):

    df_kernel = load_df()

    base_dir = "/code/app" 
    csv_file_path = os.path.join(base_dir, "model_lightgbm.pkl")

    
    with open(csv_file_path,'rb') as f:
            rdf = pickle.load(f)
    print(pred_id)       
    
    print(df_kernel)
    
    cols= ['TARGET',
           'SK_ID_CURR',
            'EXT_SOURCE_2',
            'PAYMENT_RATE', 
            'EXT_SOURCE_3',
            'DAYS_BIRTH',
            'EXT_SOURCE_1',
            'AMT_ANNUITY',
            'APPROVED_CNT_PAYMENT_MEAN',
            'INSTAL_DPD_MEAN',
            'DAYS_ID_PUBLISH',
            'PREV_CNT_PAYMENT_MEAN',
            'INSTAL_AMT_PAYMENT_SUM',
            'ANNUITY_INCOME_PERC', 
            'DAYS_EMPLOYED',
            'AMT_CREDIT',
            'AMT_GOODS_PRICE']   
    
    df = df_kernel.loc[:,cols]
    print('sk id liste : ',df.SK_ID_CURR)
    #df.dropna(inplace=True)
    #print('bef pred id id = ',pred_id.id)
    print('df shape : ',df.shape)    
    print('sk id liste : ',df.SK_ID_CURR)
    x = df[df.SK_ID_CURR == pred_id.id].iloc[:,2:]
    print('x = ',x)
    y = df.TARGET

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)    
    #rdf.fit(X_train, y_train)

    y_pred = rdf.predict(x)
    
    y_pred_proba = rdf.predict_proba(x)[0]
    print('proba id : ', y_pred_proba)
    info_id = Predict_id(id=pred_id.id, decision_id = y_pred, proba=y_pred_proba[0])
    print(info_id)

    return info_id

#############################################info_detail_client########################################


@app.get("/predict/model_shap") 
async def predict_decision(ex_id : Predict_id):

    df_kernel = load_df()
    
    colonne = []
    
    for idx,i in enumerate(df_kernel.columns):
        colonne.append(i.replace(' ','_').replace(':','_').replace('-','_').replace('/','_').replace(',','_'))
        
    df_kernel.columns = colonne
    df_kernel = df_kernel.fillna(0)
    df_kernel = df_kernel.replace([np.inf, -np.inf], np.nan)

    print(df_kernel.loc[df_kernel.SK_ID_CURR == ex_id.id].iloc[:,:15].values)
    
    cols= ['TARGET',
            'SK_ID_CURR',
            'EXT_SOURCE_2',
            'PAYMENT_RATE', 
            'EXT_SOURCE_3',
            'DAYS_BIRTH',
            'EXT_SOURCE_1',
            'AMT_ANNUITY',
            'APPROVED_CNT_PAYMENT_MEAN',
            'INSTAL_DPD_MEAN',
            'DAYS_ID_PUBLISH',
            'PREV_CNT_PAYMENT_MEAN',
            'INSTAL_AMT_PAYMENT_SUM',
            'ANNUITY_INCOME_PERC', 
            'DAYS_EMPLOYED',
            'AMT_CREDIT',
            'AMT_GOODS_PRICE']  
    
    print(df_kernel.loc[df_kernel.SK_ID_CURR== ex_id.id,cols])
    return df_kernel.loc[df_kernel.SK_ID_CURR==ex_id.id,cols].to_json()


#########################################EXPLAIN_SHAP_VALUES###########################################

@app.get("/predict/explain") 
async def predict_decision(explain_id : Explain_id):
    
    print('explain id = ', explain_id)
    
    df_kernel = load_df()    
    colonne = []
    for idx,i in enumerate(df_kernel.columns):
        colonne.append(i.replace(' ','_').replace(':','_').replace('-','_').replace('/','_').replace(',','_'))
    df_kernel.columns = colonne
   
    df = df_kernel.replace([np.inf, -np.inf], np.nan)
    df.fillna(0,inplace=True) 
    # Chemin absolu vers le répertoire contenant le fichier CSV
    base_dir = "/code/app"  # Vous devrez peut-être adapter ce chemin en fonction de votre configuration

    # Chemin absolu complet du fichier CSV
    csv_file_path = os.path.join(base_dir, "model_lightgbm_shap.pkl")

    print(csv_file_path)
    with open(csv_file_path,'rb') as f:
            shap_values = pickle.load(f)
            
    mean_abs_shap_values = shap_values.values.mean(axis = 0)

    # Trier les variables par ordre d'importance décroissante
    sorted_indices = np.argsort(mean_abs_shap_values)

    # Obtenir les noms des variables
    feature_names = df_kernel.columns
    feature_names = feature_names[3:]

    print('faet name : ', len(feature_names),sorted_indices[:15], len(shap_values.values))


    # Obtenir les noms des variables les plus importantes
    top_feature_names = []
    tab_shap = np.ones((100,15))
    
    [top_feature_names.append(feature_names[idx]) for idx in sorted_indices]
        
    for i in range(0,100):
        for j in range(0,15):
            #for idx in sorted_indices:
            tab_shap[i][j] = shap_values.values[i][sorted_indices[j]]
            
    print('i : ',list(tab_shap[0]))
    
    print('idx : ', sorted_indices[:15])
    print("Variables les plus importantes :", top_feature_names[:15])
    
    print('paquets reçu exoplain id : ',explain_id) 
    #df_as_json = pd.DataFrame(tab_shap).to_json(orient='records')
    #df_as_json = df.to_json(orient='records')
    
    print(tab_shap.shape,len(df.SK_ID_CURR[:1000]))

    
    return pd.DataFrame(tab_shap,index =df.SK_ID_CURR[:100] , columns=top_feature_names[:15])
	
def load_df():
    #df_app_test = pd.read_csv('source/application_test.csv',sep=',')
    # Chemin absolu vers le répertoire contenant le fichier CSV
    base_dir = "/code/app"  # Vous devrez peut-être adapter ce chemin en fonction de votre configuration

    # Chemin absolu complet du fichier CSV

  
    csv_file_path = os.path.join(base_dir, "preprocessing/kernel_light.csv")

    # Vérifier si le fichier existe
    if os.path.exists(csv_file_path):
        # Lire le fichier CSV
        df = pd.read_csv(csv_file_path,sep = ',')
        
        return df
    else:
        return print("Le fichier CSV n'existe pas.")





df_kernel = load_df()

#df_train = laod_df()

#df_kernel = load_df()

print('fin')   

