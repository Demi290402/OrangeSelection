import os
from tkinter import _test
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from pgmpy import models

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Dataset preprocessato caricato con successo.")
        return data
    except Exception as e:
        print(f"Errore durante il caricamento del dataset: {e}")
        return None

def preprocess_data(data):
    print("Colonne presenti nel dataset:", data.columns.tolist())  # Debug

    required_columns = {'diameter(cm)': 'diameter', 'weight(g)': 'weight'}
    missing_columns = [col for col in required_columns.keys() if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Errore: Le seguenti colonne richieste sono mancanti nel dataset: {missing_columns}")
    
    data.rename(columns=required_columns, inplace=True)
    print(f"Colonne rinominate correttamente: {required_columns}")
    
    bins = [0, 7, 8.5, 10, 15]
    labels = ["Small", "Medium", "Large", "Very Large"]
    data['Quality'] = pd.cut(data['diameter'], bins=bins, labels=labels)
    print("Variabile target creata con successo.")
    
    label_encoder = LabelEncoder()
    data['Quality_encoded'] = label_encoder.fit_transform(data['Quality'])
    print(f"Classi target codificate: {label_encoder.classes_}")
    
    X = data[['diameter', 'weight', 'red', 'green', 'blue']]
    y = data['Quality_encoded']
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print("Standardizzazione completata.")

    return X, y, label_encoder

def feature_selection(X, y):
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    selector = SelectFromModel(rf, threshold=0.01, prefit=True)
    X_selected = selector.transform(X.to_numpy())  # Conversione in array numpy per evitare warning
    selected_features = X.columns[selector.get_support()]
    print(f"Feature selezionate: {list(selected_features)}")
    return pd.DataFrame(X_selected, columns=selected_features)

def train_models(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {}
    
    svm_model = SVC(kernel='rbf', C=0.05, random_state=42)
    import numpy as np

    rf_model = RandomForestClassifier(
    n_estimators=20,  # Ridotto ulteriormente per meno overfitting
    max_depth=2,  # Manteniamo la profondità controllata
    min_samples_split=300,  # Più dati per split, meno frammentazione
    min_samples_leaf=250,  # Foglie più grandi per generalizzare meglio
    max_features="sqrt",  # Manteniamo varietà tra alberi
    ccp_alpha=0.02,  # Meno pruning aggressivo
    class_weight="balanced",  # Bilancia le classi
    bootstrap=True,  
    random_state=42
)

    nn_model = MLPClassifier(hidden_layer_sizes=(100), max_iter=1000, learning_rate_init=0.5, random_state=42)
    
    models['SVM'] = svm_model
    models['Random Forest'] = rf_model
    models['Neural Network'] = nn_model
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        print(f"{name} - Accuratezza media con 5-fold CV: {np.mean(scores):.4f}")
        
        # **IMPORTANTE: Addestra il modello sui dati di training dopo la cross-validation**
        model.fit(X, y)

    return models

def save_metrics(results, file_path="Cistulli_Domenico/results/model_metrics.json"):
    """
    Salva le metriche dei modelli in un file JSON.
    """
    try:
        if not os.path.exists("Cistulli_Domenico/results/"):
            os.makedirs("Cistulli_Domenico/results/")
        
        with open(file_path, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Metriche salvate con successo in: {file_path}")
    except Exception as e:
        print(f"Errore durante il salvataggio delle metriche: {e}")
        save_metrics(results)

def plot_confusion_matrix(y_test, y_pred, model_name, label_encoder):
    """
    Genera e salva la matrice di confusione per un modello specifico.

    :param y_test: Array con i valori veri.
    :param y_pred: Array con i valori predetti.
    :param model_name: Nome del modello (es. 'SVM', 'Random Forest', etc.).
    :param label_encoder: Oggetto LabelEncoder per decodificare i target.
    """
    # Creazione della matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Oranges')
    plt.title(f"Matrice di Confusione: {model_name}")
    
    # Salvataggio della matrice di confusione come immagine
    output_path = f"Cistulli_Domenico/results/{model_name}_confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Matrice di confusione salvata in: {output_path}")

def evaluate_models(models, X_test, y_test, label_encoder):
    """
    Valuta i modelli sui dati di test e restituisce i risultati.
    :param models: Dizionario con i modelli addestrati.
    :param X_test: Caratteristiche del test set.
    :param y_test: Target del test set.
    :param label_encoder: Oggetto LabelEncoder per decodificare i target.
    :return: Dizionario con metriche di valutazione.
    """
    results = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Valutazione {model_name}:")
        print(classification_report(label_encoder.inverse_transform(y_test), label_encoder.inverse_transform(y_pred)))
        print(f"Accuratezza: {accuracy:.4f}\n")

        # Generazione e salvataggio della matrice di confusione
        plot_confusion_matrix(y_test, y_pred, model_name, label_encoder)

        # Salvataggio delle metriche
        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        save_metrics(results)
    return results

def main():
    file_path = "Cistulli_Domenico/data/processed/Processed_Orange_Data.csv"
    data = load_data(file_path)
    if data is None:
        return
    
    X, y, label_encoder = preprocess_data(data)
    X_selected = feature_selection(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    models = train_models(X_train, y_train)
    
    # Valutazione modelli
    results = evaluate_models(models, X_test, y_test, label_encoder)

if __name__ == "__main__":
    main()
