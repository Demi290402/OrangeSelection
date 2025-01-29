import matplotlib.pyplot as plt

def compare_models():
    """
    Questo script confronta le performance di tre modelli (SVM, Random Forest, Neural Network)
    in termini di accuratezza e F1-score medio. I risultati sono visualizzati in due grafici
    a barre con tonalità di arancione per una rappresentazione a tema.

    Funzioni principali:
    - Confronto delle accuratezze dei modelli.
    - Confronto dei F1-score medi dei modelli.
    - Visualizzazione dei risultati tramite grafici.
    """
    # Accuratezze dei modelli (valori predefiniti)
    svm_accuracy = 0.99  # SVM
    rf_accuracy = 1.00   # Random Forest
    nn_accuracy = 0.9934 # Neural Network

    # F1-score medi dei modelli (valori predefiniti)
    svm_f1 = 0.99       # SVM
    rf_f1 = 1.00        # Random Forest
    nn_f1 = 0.993       # Neural Network

    # Etichette dei modelli
    models = ['SVM', 'Random Forest', 'Neural Network']

    # Lista di accuratezze e F1-scores
    accuracies = [svm_accuracy, rf_accuracy, nn_accuracy]
    f1_scores = [svm_f1, rf_f1, nn_f1]

    # Tonalità di arancione per i grafici
    orange_shades = ['#FFA07A', '#FF8C00', '#FF4500']  # Corallo, Arancio scuro, Arancio rosso

    # Grafico delle accuratezze
    print("Creazione del grafico delle accuratezze...")
    plt.figure(figsize=(10, 5))
    plt.bar(models, accuracies, color=orange_shades)
    plt.ylim(0.95, 1.05)  # Limita l'asse Y per una migliore comparazione
    plt.title('Confronto delle Accuratezze dei Modelli')
    plt.ylabel('Accuratezza')
    plt.xlabel('Modelli')
    plt.show()

    # Grafico dei F1-score
    print("Creazione del grafico dei F1-score...")
    plt.figure(figsize=(10, 5))
    plt.bar(models, f1_scores, color=orange_shades)
    plt.ylim(0.95, 1.05)  # Limita l'asse Y per una migliore comparazione
    plt.title('Confronto dei F1-Score dei Modelli')
    plt.ylabel('F1-Score')
    plt.xlabel('Modelli')
    plt.show()

    print("Confronto dei modelli completato.")

if __name__ == "__main__":
    compare_models()
