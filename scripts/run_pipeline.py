# OrangeSelection: Pipeline di Preprocessamento e Addestramento

import os
import subprocess

# Definizione delle funzioni principali
def run_preprocess():
    """Esegue il preprocessamento dei dati."""
    print("[Pipeline] Inizio preprocessamento dei dati...")
    subprocess.run(["python", "-m", "Cistulli_Domenico.src.dataprocessing.preprocess_data"])
    print("[Pipeline] Preprocessamento completato con successo.")

def run_knowledge_base():
    """Esegue l'arricchimento della knowledge base."""
    print("[Pipeline] Inizio arricchimento della knowledge base...")
    subprocess.run(["python", "-m", "Cistulli_Domenico.src.kbs.knowledge_base"])
    print("[Pipeline] Arricchimento completato con successo.")

def run_populate_columns():
    """Popola le colonne aggiuntive nel dataset."""
    print("[Pipeline] Inizio popolamento delle colonne aggiuntive...")
    subprocess.run(["python", "-m", "Cistulli_Domenico.src.dataprocessing.populate_columns"])
    print("[Pipeline] Popolamento delle colonne completato con successo.")

def run_neural_network():
    """Esegue l'addestramento della rete neurale."""
    print("[Pipeline] Inizio addestramento della rete neurale...")
    subprocess.run(["python", "-m", "Cistulli_Domenico.src.neural_network.neural_network"])
    print("[Pipeline] Addestramento della rete neurale completato con successo.")

def run_ml_models():
    """Esegue l'addestramento dei modelli di machine learning."""
    print("[Pipeline] Inizio addestramento dei modelli ML...")
    subprocess.run(["python", "-m", "Cistulli_Domenico.src.ml.models"])
    print("[Pipeline] Addestramento dei modelli ML completato con successo.")

def run_compare_models():
    """Esegue il confronto tra i modelli."""
    print("[Pipeline] Inizio confronto tra i modelli...")
    subprocess.run(["python", "-m", "Cistulli_Domenico.src.ml.compare_models"])
    print("[Pipeline] Confronto tra i modelli completato con successo.")

def run_generate_report():
    """Crea un report dei risultati dei modelli."""
    subprocess.run(["python", "-m", "Cistulli_Domenico.scripts.generate_report"])

# Esecuzione della pipeline
if __name__ == "__main__":
    print("========== Inizio esecuzione della pipeline ==========\n")

    # Step 1: Preprocessamento
    run_preprocess()

    # Step 2: Arricchimento della knowledge base
    run_knowledge_base()

    # Step 3: Popolamento delle colonne
    run_populate_columns()

    # Step 4: Addestramento dei modelli ML
    run_ml_models()

    # Step 5: Addestramento della rete neurale
    run_neural_network()

    # Step 6: Confronto tra i modelli
    run_compare_models()

    # Step 7: Generazione report risultati
    run_generate_report()

    print("\n========== Pipeline completata con successo ==========")
