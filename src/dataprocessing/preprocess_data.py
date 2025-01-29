import pandas as pd
import numpy as np

def preprocess_data():
    """
    Script per preprocessare il dataset delle arance, aggiungendo nuove colonne per aspetti visivi.
    """
    try:
        # Percorso del dataset grezzo
        dataset_path = "Cistulli_Domenico/data/raw/Preprocessed_Orange_Data.csv"
        processed_path = "Cistulli_Domenico/data/processed/Processed_Orange_Data.csv"

        # Caricamento del dataset grezzo
        print("Inizio preprocessing dei dati...")
        data = pd.read_csv(dataset_path)
        print(f"Dataset caricato: {data.shape}")

        # Filtrare solo le righe relative alle arance
        data = data[data['name'] == 'orange']
        print(f"Dataset filtrato (solo arance): {data.shape}")

        # Rimuovere la colonna 'name' poiché contiene un valore unico
        data = data.drop(columns=['name'], errors='ignore')

        # Aggiunta di nuove colonne per caratteristiche visive
        data['variety'] = None  # Varietà dell'arancia (es: Valencia, Navel, etc.)
        data['blemishes(Yes/No)'] = None  # Presenza di ammaccature (Yes/No)
        data['cuts(Yes/No)'] = None  # Presenza di tagli o rotture (Yes/No)
        print("Nuove colonne aggiunte: 'variety', 'blemishes', 'cuts'.")

        # Rinominare le colonne per maggiore chiarezza
        data.rename(columns={
            'weight': 'weight(g)',
            'diameter': 'diameter(cm)'
        }, inplace=True)
        print("Colonne rinominate per maggiore chiarezza.")

        # Gestione dei valori mancanti (imputazione con mediana)
        data.fillna(data.median(numeric_only=True), inplace=True)
        print("Valori mancanti imputati.")

        # Salvataggio del dataset preprocessato
        data.to_csv(processed_path, index=False)
        print(f"Dataset preprocessato salvato in: {processed_path}")

    except Exception as e:
        print(f"Errore durante il preprocessamento: {e}")

if __name__ == "__main__":
    preprocess_data()
