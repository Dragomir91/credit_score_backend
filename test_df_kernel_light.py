import pandas as pd
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
