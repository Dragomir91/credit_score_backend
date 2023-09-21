import numpy as np
import pandas as pd
import pickle
import os


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
            rdf = pickle.load(f)
    
    print(df_kernel)
    
import numpy as np
import pandas as pd
import pickle
import os


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
            rdf = pickle.load(f)
    
    print(df_kernel)
    
    cols= ['SK_ID_CURR',
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
    
    df = df_kernel.loc[:,cols]
    print('sk id liste : ',df.SK_ID_CURR)
    #df.dropna(inplace=True)
    #print('bef pred id id = ',pred_id.id)
    print('df shape : ',df.shape)    
    print('sk id liste : ',df.SK_ID_CURR)
    x = df[df.SK_ID_CURR == 100002].iloc[:,2:]
    print('x = ',x)
    y = df.TARGET

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)    
    #rdf.fit(X_train, y_train)

    y_pred = rdf.predict(x)
    
    y_pred_proba = rdf.predict_proba(x)[0]
    print('proba id : ', y_pred_proba)
    assert (y_pred_proba[0] > 0 and y_pred_proba[0] <= 1) 
    assert (y_pred_proba[1] > 0 and y_pred_proba[1] <= 1) 


    #info_id = Predict_id(id=pred_id.id, decision_id = y_pred, proba=y_pred_proba[0])

  
    
    df = df_kernel.loc[:,cols]
    print('sk id liste : ',df.SK_ID_CURR)
    #df.dropna(inplace=True)
    #print('bef pred id id = ',pred_id.id)
    print('df shape : ',df.shape)    
    print('sk id liste : ',df.SK_ID_CURR)
    x = df[df.SK_ID_CURR == 100002].iloc[:,2:]
    print('x = ',x)
    y = df.TARGET

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)    
    #rdf.fit(X_train, y_train)

    y_pred = rdf.predict(x)
    
    y_pred_proba = rdf.predict_proba(x)[0]
    print('proba id : ', y_pred_proba)
    assert (y_pred_proba[0] > 0 and y_pred_proba[0] <= 1) 
    assert (y_pred_proba[1] > 0 and y_pred_proba[1] <= 1) 


    #info_id = Predict_id(id=pred_id.id, decision_id = y_pred, proba=y_pred_proba[0])

