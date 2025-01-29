import pandas as pd
import random

def populate_columns():
    """
    Popola le colonne 'variety', 'blemishes(Yes/No)', 'cuts(Yes/No)' nel dataset arricchito.
    """
    try:
        # Percorso del dataset
        enriched_path = "Cistulli_Domenico/data/processed/Enriched_Orange_Data.csv"
        updated_path = "Cistulli_Domenico/data/processed/Final_Orange_Data.csv"

        # Caricamento del dataset arricchito
        data = pd.read_csv(enriched_path)
        print(f"Dataset caricato: {data.shape} righe e colonne.")

        # Lista di varietà di arance
        varieties = ["Valencia", "Navel", "Cara Cara", "Blood Orange", "Hamlin"]

        # Assegnazione diretta di valori casuali per ogni riga
        data['variety'] = [random.choice(varieties) for _ in range(len(data))]
        data['blemishes(Yes/No)'] = ["Yes" if random.random() < 0.3 else "No" for _ in range(len(data))]
        data['cuts(Yes/No)'] = ["Yes" if random.random() < 0.3 else "No" for _ in range(len(data))]

        # Rimuovere colonne aggiunte erroneamente, come 'variety.1'
        if 'variety.1' in data.columns:
            data = data.drop(columns=['variety.1'])
            print("Colonna 'variety.1' rimossa.")

        # Salvataggio del dataset aggiornato
        data.to_csv(updated_path, index=False)
        print(f"Dataset aggiornato salvato in: {updated_path}")

    except Exception as e:
        print(f"Errore durante la popolazione delle colonne: {e}")

if __name__ == "__main__":
    populate_columns()
