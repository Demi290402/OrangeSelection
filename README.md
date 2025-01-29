# OrangeSelection: Classificazione della Qualità delle Arance con ML e KB

## Descrizione Generale
Il progetto **OrangeSelection** mira a sviluppare un sistema automatizzato per analizzare la qualità delle arance utilizzando tecniche di Machine Learning (ML) integrate con una Knowledge Base (KB).  
L'obiettivo principale è dimostrare come l'integrazione della conoscenza semantica possa migliorare l'accuratezza delle previsioni e il ragionamento automatizzato.

## Struttura del Progetto

Cistulli_Domenico/
├── config/                                         # file sparql
├── data/
│   ├── raw/
│   │   ├── Preprocessed_Orange_Data.csv            # Dataset originale
│   ├── processed/                             
│   │   ├── Enriched_Orange_Data.csv                # Dataset arricchito
│   │   ├── Processed_Orange_Data.csv               # Dataset processato
│   │   ├── Final_Orange_Data.csv                   # Dataset finale
├── src/
│   ├── dataprocessing/
│   │   ├── preprocess_data.py                      # Script per la preparazione dei dati
│   ├── ml/
│   │   ├── models.py                               # Modelli di Machine Learning
│   │   ├── compare_models.py                       # Comparazione tra i modelli
│   ├── kbs/
│   │   ├── knowledge_base.py                       # Knowledge Base in Prolog
│   ├── neural_network/
│   │   ├── neural_network.py                       # Reti neurali per la classificazione
├── Notebook/                          
│   ├── eda_analysis.ipynb                          # Analisi dei risultati             
├── docs/                   
│   ├── draft/                                      # Bozze varie
│   ├── final/                                      # Documentazione finale
│   ├── images/                                     # Immagini utilizzate
├── scripts/ 
│   ├── generate_report                             # Script per generare un report nella cartella results 
│   ├── run_pipeline                                # Script per eseguire l'intero progetto
├── results/                                        # Output generati (report, modelli salvati)
├── my_env/                                         # Ambiente virtuale python
├── README.md                                       # Descrizione del progetto
└── requirements.txt                                # Dipendenze del progetto

## Dipendenze

pandas==1.5.2
numpy==1.23.5
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.3.2
imbalanced-learn==0.10.0
rdflib==6.3.1
Owlready2==0.38
swiplserver==1.0.2
prolog==1.4.0
rich==13.4.2
SPARQLWrapper==1.9.1
tensorflow==2.12.0

## Per installarle, eseguire:
Lo script principale del progetto è **run_pipeline.py**, che esegue automaticamente tutte le fasi della pipeline:

**Preprocessamento del dataset:**
- Pulizia e trasformazione dei dati.
**Arricchimento con la Knowledge Base:**
- Integrazione delle ontologie semantiche.
**Addestramento dei modelli:**
- Confronto tra Random Forest, SVM e una Rete Neurale.

## Avvio della pipeline
Per eseguire l'intero processo, utilizzare il comando:
python scripts/run_pipeline.py

## Risultati
Lo script genera i seguenti output:

**Report delle metriche dei modelli:**
- Salvato in results/report.txt.

**Metriche dettagliate in formato JSON:**
- Salvato in results/model_metrics.json.

## Interrogazione della Knowledge Base
L'interrogazione del sistema avviene tramite query Prolog utilizzando un interprete come SWI-Prolog. Il file knowledge_base.pl include fatti e regole che permettono di ottenere informazioni sulle varietà di arance, sulla qualità e su altre proprietà.

**Avvio dell'interprete:**
Installare SWI-Prolog da https://www.swi-prolog.org/.
Aprire l'interprete e caricare la Knowledge Base col seguente comando:
- consult('C:/Users/cistu/Desktop/OrangeSelection/Cistulli_Domenico/src/kbs/knowledge_base.pl').

**Esempi di query:**
- Determinare la qualità di una varietà:
    <qualita(valencia, Qualita).>
    Risultato: Qualita = alta.

- Raccomandare varietà di alta qualità:
    <raccomanda(Varieta).>
    Risultato: Varieta = valencia

- Query combinate:
    <dolcezza(Varieta, Dolcezza), Dolcezza > 10, acidita(Varieta, Acidita), Acidita =< 3.5.>
    Risultato: Varieta = valencia, Dolcezza = 12, Acidita = 3.2.
    
## Debug e Analisi
Per esplorare i dati e i risultati visivamente, utilizzare il notebook:

## Crediti
Sviluppato da Domenico Cistulli.

Per ulteriori dettagli, consultare la documentazione in docs/final.