import pandas as pd
import time
from SPARQLWrapper import SPARQLWrapper, JSON

def enrich_with_wikidata():
    """
    Questo script arricchisce il dataset locale di arance con dati provenienti da Wikidata.

    Funzioni principali:
    - Caricamento del dataset locale preprocessato.
    - Configurazione ed esecuzione di una query SPARQL per ottenere dati aggiuntivi da Wikidata.
    - Integrazione dei dati recuperati con il dataset locale.
    - Salvataggio del dataset arricchito in un file CSV.
    """
    try:
        # Percorso del dataset locale e del file arricchito
        local_path = "Cistulli_Domenico/data/processed/Processed_Orange_Data.csv"
        enriched_path = "Cistulli_Domenico/data/processed/Enriched_Orange_Data.csv"

        # Caricamento del dataset locale preprocessato
        print("Caricamento del dataset locale...")
        data = pd.read_csv(local_path)
        print(f"Dataset locale caricato con successo: {data.shape} righe e colonne.")

        # Configurazione del client SPARQL per accedere a Wikidata
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        query = """
        SELECT ?orange ?orangeLabel ?variety WHERE {
          ?orange wdt:P31 wd:Q7556. # Oggetti di tipo "arancia"
          OPTIONAL { ?orange wdt:P1056 ?variety. } # Varietà dell'arancia (se disponibile)
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        LIMIT 50
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        

        # Esecuzione della query SPARQL e conversione dei risultati
        print("Esecuzione della query SPARQL su Wikidata...")
        results = sparql.query().convert()
        print("Query completata con successo.")

        # Estrarre i dati utili dai risultati SPARQL
        wikidata = []
        for result in results["results"]["bindings"]:
            wikidata.append({
                "variety": result["variety"]["value"] if "variety" in result else None  # Colonna 'variety'
            })

        # Convertire i dati Wikidata in un DataFrame
        wikidata_df = pd.DataFrame(wikidata)
        print(f"Dati esterni recuperati da Wikidata: {wikidata_df.shape} righe e colonne.")

        # Integrare i dati Wikidata con il dataset locale
        print("Integrazione dei dati esterni con il dataset locale...")
        enriched_data = pd.concat([data, wikidata_df], axis=1)
        print(f"Dataset integrato con successo: {enriched_data.shape} righe e colonne.")

        # Salvataggio del dataset arricchito in un file CSV
        enriched_data.to_csv(enriched_path, index=False)
        print(f"Dataset arricchito salvato con successo in: {enriched_path}")

    except Exception as e:
        # Gestione degli errori
        print(f"Errore durante l'integrazione dei dati: {e}")

if __name__ == "__main__":
    enrich_with_wikidata()
