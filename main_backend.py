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
async def info_client():
    df_kernel = load_df()
    return {"list_id" : list(df_kernel.loc[:101,'SK_ID_CURR']),
            "infos_id" : list(df_kernel.columns[:101])}
########################ID_CLIENT#######################################################

@app.get("/info_client/id")
async def info_client2(info_id: Info_id):
    
    df_kernel = load_df()
    print('id_client = ',  info_id.id_client)
    print('info_client = ',  info_id.infos_id)
    df_id = df_kernel.loc[df_kernel['SK_ID_CURR'] == info_id.id_client,info_id.infos_id]

    print('df_id = ',df_id)  
    return df_id.to_json()

####################################PREDICT_ID#############################################

@app.post("/predict") 
async def predict_decision(pred_id : Predict_id):

    df_kernel = load_df()

    base_dir = "" 
    csv_file_path = os.path.join(base_dir, "model_lightgbm.pkl")

    
    with open(csv_file_path,'rb') as f:
            model_lightgbm = pickle.load(f)
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

    print('df shape : ',df.shape)    

    x = df[df.SK_ID_CURR == pred_id.id].iloc[:,2:]
    print('x = ',x)
    y = df.TARGET

    y_pred = model_lightgbm.predict(x)
    
    y_pred_proba = model_lightgbm.predict_proba(x)[0]
    print('proba id : ', y_pred_proba)
    info_id = Predict_id(id=pred_id.id, decision_id = y_pred, proba=y_pred_proba[0])
    print(info_id)

    return info_id

#############################################info_detail_client########################################


@app.get("/predict/shap_id") 
async def predict_decision(ex_id : Predict_id):

    df_kernel = load_df()

    colonne = []
    
    for idx,i in enumerate(df_kernel.columns):
        colonne.append(i.replace(' ','_').replace(':','_').replace('-','_').replace('/','_').replace(',','_'))
        
    df_kernel.columns = colonne
    df_kernel = df_kernel.fillna(0)
    df_kernel = df_kernel.replace([np.inf, -np.inf], np.nan)

    """clf = GradientBoostingClassifier()

    model_lgbm = clf.fit(df_kernel.iloc[:,2:], df_kernel.TARGET)

    explainer = shap.Explainer(model_lgbm,df_kernel.iloc[:,2:])
    shap_values = explainer(df_kernel.iloc[:,2:])
    print('shap value shape :',shap_values.values.shape)"""

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
    feature_names = df_kernel.iloc[:,2:].columns

    # Obtenir les noms des variables les plus importantes
    top_feature_names = [feature_names[idx] for idx in sorted_indices]

    print(df_kernel.loc[df_kernel.SK_ID_CURR == ex_id.id,feature_names[:15]])
    
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
    return df_kernel.loc[df_kernel.SK_ID_CURR==ex_id.id,top_feature_names[:15]].to_json()



#########################################EXPLAIN_SHAP_VALUES###########################################

@app.get("/predict/explain") 
async def predict_decision(explain_id : Explain_id):
    
    print('explain id = ', explain_id)
    
    df_kernel = load_df()    
    colonne = []
    top_feature_names = []
    tab_shap = np.ones((100,16))


    for idx,i in enumerate(df_kernel.columns):
        colonne.append(i.replace(' ','_').replace(':','_').replace('-','_').replace('/','_').replace(',','_'))
    df_kernel.columns = colonne
   
    df = df_kernel.replace([np.inf, -np.inf], np.nan)
    df.fillna(0,inplace=True) 
    # Chemin absolu vers le répertoire contenant le fichier CSV
    base_dir = ""  # Vous devrez peut-être adapter ce chemin en fonction de votre configuration

    # Chemin absolu complet du fichier CSV
    csv_file_path = os.path.join(base_dir, "model_lightgbm_shap.pkl")

    print(csv_file_path)
    with open(csv_file_path,'rb') as f:
            shap_values = pickle.load(f)
            
    mean_abs_shap_values = shap_values.values.mean(axis = 0)

    # Trier les variables par ordre d'importance décroissante
    sorted_indices = np.argsort(mean_abs_shap_values)

    # Obtenir les noms des variables
    feature_names = df_kernel.iloc[:,2:].columns

    print('faet name : ', len(feature_names),sorted_indices[:15], len(shap_values.values))

    # Obtenir les noms des variables les plus importantes
    
    [top_feature_names.append(feature_names[idx]) for idx in sorted_indices]

    for i in range(0,100):
        for j in range(1,16):
            #for idx in sorted_indices:
            tab_shap[i][0] = df.SK_ID_CURR[i] 
            tab_shap[i][j] = shap_values.values[i][sorted_indices[j]]
        
  
    print('i : ',list(tab_shap[0]))
    
    print('idx : ', sorted_indices[:15])
    print("Variables les plus importantes :", top_feature_names[:15])
    
    print('paquets reçu exoplain id : ',explain_id)     
    print(tab_shap.shape,len(df.SK_ID_CURR[:1000]))

    df_col = []
    df_col.append('ID')
    [df_col.append(col) for col in top_feature_names[:15]]

    return pd.DataFrame(tab_shap,index=df.SK_ID_CURR[:100] , columns=df_col)

#########################################COUT_METIER############################################


def perte(dataframe, y_true, y_pred,col_credit, rate_default:float):
    
    df = {'Index':y_true.index,'y_pred':y_pred,'Target':y_true.values}
    df_pred = pd.DataFrame(df)
    idx = df_pred.loc[(df_pred.y_pred == 1) & (df_pred.Target == 0),'Index'] 
    dataframe['perte'] = -dataframe.loc[idx,col_credit] * rate_default
    return dataframe

def gain(df, y_proba, threshold:float, rate_interest:float):
        
        
        df['proba'] = y_proba[:,0]
        df['gain'] = 0
        df['y_pred'] = np.nan
        
        df['gain'] = df.loc[(df.proba >= threshold), 'AMT_CREDIT'] * rate_interest
        df.loc[(df.proba >= threshold), 'y_pred'] = 0
        df.loc[(df.proba < threshold), 'y_pred'] = 1
            
                
        return df['gain'], threshold

@app.get("/predict/cout_metier") 
async def predict_decision(cout_id : Predict_id):
    
    
    df_kernel = load_df()    
    colonne = []
    top_feature_names = []
    base_dir = "" 
    csv_file_path = os.path.join(base_dir, "model_lightgbm.pkl")

    
    with open(csv_file_path,'rb') as f:
            model_lightgbm = pickle.load(f)

    for idx,i in enumerate(df_kernel.columns):
        colonne.append(i.replace(' ','_').replace(':','_').replace('-','_').replace('/','_').replace(',','_'))
    df_kernel.columns = colonne
   
    df = df_kernel.replace([np.inf, -np.inf], np.nan)
    df.fillna(0,inplace=True) 
    # Chemin absolu vers le répertoire contenant le fichier CSV

    
    cols= [ 'TARGET',
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

    print('df shape : ',df.shape)    

    X = df.iloc[:,1:]
    print('x = ',X)
    y = df.TARGET
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    y_pred = model_lightgbm.predict(X_test)
    
    y_pred_proba = model_lightgbm.predict_proba(X_test)
    print('proba id : ', y_pred_proba)
    print('y_test.index : ', y_test.index)
    df_sample = pd.DataFrame(df.loc[y_test.index])
    df_sample['gain'], th= gain(df_sample,y_pred_proba, 0.8, 0.04)
    df_sample.gain.agg('sum')    
    
    df_sample = perte(df_sample, y_test,y_pred, 'AMT_CREDIT', 0.5)

    df_sample.perte.agg('sum')

    roc_auc0 = roc_auc_score(y_test, model_lightgbm.predict_proba(X_test)[:, 0])

    roc_auc1 = roc_auc_score(y_test, model_lightgbm.predict_proba(X_test)[:, 1])
    return   {"gain" : [df_sample.gain.agg('sum') ],
              "perte" : [df_sample.perte.agg('sum') ],
              "roc_auc_0" : [roc_auc0],
              "roc_auc_1" : [roc_auc1],
              'threshold' : [th]}

def load_df():
    #df_app_test = pd.read_csv('source/application_test.csv',sep=',')
    # Chemin absolu vers le répertoire contenant le fichier CSV
    base_dir = ""  # Vous devrez peut-être adapter ce chemin en fonction de votre configuration

    # Chemin absolu complet du fichier CSV

    csv_file_path = os.path.join(base_dir, "preprocessing/kernel_light.csv")

    # Vérifier si le fichier existe
    if os.path.exists(csv_file_path):
        # Lire le fichier CSV
        df = pd.read_csv(csv_file_path,sep = ',')
        df.drop(columns='index', inplace=True)
        print('dataframe download')
        return df
    else:
        return print("Le fichier CSV n'existe pas.")


if __name__ == '__main__':
    df_kernel = load_df()
    print('fin')   

