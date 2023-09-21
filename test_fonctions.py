import numpy as np
import pandas as pd
import pickle
import os


def test_load_df():
    # Chemin absolu vers le répertoire contenant le fichier CSV
    base_dir = ""  
    # Chemin absolu complet du fichier CSV
    csv_file_path = os.path.join(base_dir, "kernel_light.csv")
    # Effectuer des assertions pour vérifier que le DataFrame est correct
    df = pd.read_csv(csv_file_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0  # Vérifier que le DataFrame n'est pas vide
    # Autres assertions selon votre cas d'utilisation

# Exécutez pytest pour exécuter les tests   

def test_predict_decision():

    # Chemin absolu vers le répertoire contenant le fichier CSV
    base_dir = ""  
    # Chemin absolu complet du fichier CSV
    csv_file_path = os.path.join(base_dir, "kernel_light.csv")
    # Effectuer des assertions pour vérifier que le DataFrame est correct
    df_kernel = pd.read_csv(csv_file_path)
    base_dir = "" 
    csv_file_path = os.path.join(base_dir, "model_lightgbm.pkl")

    
    with open(csv_file_path,'rb') as f:
        model_lightgbm = pickle.load(f)
    
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
    
    df = df_kernel

    print('df shape : ',df.shape)    

    X = df[df.SK_ID_CURR == 100002].loc[:,cols]

    y_pred = model_lightgbm.predict(X)
    
    y_pred_proba = model_lightgbm.predict_proba(X)[0]
    print('proba id : ', y_pred_proba)
    assert (y_pred_proba[0] > 0 and y_pred_proba[0] <= 1) 
    assert (y_pred_proba[1] > 0 and y_pred_proba[1] <= 1) 


