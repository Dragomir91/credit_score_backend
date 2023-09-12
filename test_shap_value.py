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
    colonne = []
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

    assert len(shap_values.values) > 0
        
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
    assert top_feature_names[:15] ==  ['AMT_GOODS_PRICE',
                                    'EXT_SOURCE_2',
                                    'EXT_SOURCE_1',
                                    'EXT_SOURCE_3',
                                    'INSTAL_DPD_MEAN',
                                    'INSTAL_AMT_PAYMENT_SUM',
                                    'PAYMENT_RATE',
                                    'CODE_GENDER',
                                    'NAME_EDUCATION_TYPE_Higher_education',
                                    'DEF_30_CNT_SOCIAL_CIRCLE',
                                    'DAYS_EMPLOYED',
                                    'DAYS_BIRTH',
                                    'PREV_AMT_ANNUITY_MEAN',
                                    'POS_SK_DPD_DEF_MEAN',
                                    'REGION_RATING_CLIENT_W_CITY']    
    
    #df_as_json = pd.DataFrame(tab_shap).to_json(orient='records')
    #df_as_json = df.to_json(orient='records')
    
    print(tab_shap.shape,len(df.SK_ID_CURR[:1000]))

    assert len(tab_shap) > 0 
