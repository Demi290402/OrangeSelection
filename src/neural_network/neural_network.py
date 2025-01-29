import sys
import os
from tensorflow.keras import Input  # type: ignore # Import per Input
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from ..ml.models import save_metrics  # type: ignore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def load_processed_data(file_path):
    """
    Carica il dataset preprocessato dal percorso specificato.
    """
    try:
        data = pd.read_csv(file_path)
        print("Dataset preprocessato caricato con successo.")
        return data
    except Exception as e:
        print(f"Errore durante il caricamento del dataset preprocessato: {e}")
        return None

def preprocess_data(data):
    """
    Preprocessa il dataset:
    - Creazione della variabile target basata sul diametro.
    - Standardizzazione delle caratteristiche numeriche.
    - Codifica della variabile target.
    """
    print("Inizio preprocessamento dei dati...")
    
    bins = [0, 7, 8.5, 10, np.inf]  # Intervalli per il diametro
    labels = ["Small", "Medium", "Large", "Very Large"]
    data['Quality'] = pd.cut(data['diameter'], bins=bins, labels=labels)
    data = data.dropna(subset=['Quality'])
    
    label_encoder = LabelEncoder()
    data['Quality_encoded'] = label_encoder.fit_transform(data['Quality'])
    print(f"Classi target codificate: {label_encoder.classes_}")
    
    X = data[['diameter', 'weight', 'red', 'green', 'blue']]
    y = data['Quality_encoded']
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print("Preprocessamento completato.")
    
    return X, y, label_encoder

def balance_classes(X, y):
    """
    Applica SMOTE per bilanciare le classi.
    """
    print("Inizio bilanciamento delle classi...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Bilanciamento completato.")
    return X_resampled, y_resampled

def build_neural_network(input_dim, num_classes):
    """
    Costruisce una rete neurale feedforward.
    
    :param input_dim: Numero di caratteristiche di input.
    :param num_classes: Numero di classi di output.
    :return: Modello di rete neurale compilato.
    """
    print("Creazione della rete neurale...")
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Rete neurale creata con successo.")
    return model

def train_neural_network(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    """
    Addestra la rete neurale.
    """
    print("Inizio addestramento della rete neurale...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, verbose=1)
    print("Addestramento completato.")
    return history

def evaluate_neural_network(model, X_test, y_test, label_encoder):
    """
    Valuta la rete neurale sui dati di test.
    """
    print("Valutazione della rete neurale...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    precision = precision_score(y_test, y_pred_classes, average='macro')
    recall = recall_score(y_test, y_pred_classes, average='macro')
    f1 = f1_score(y_test, y_pred_classes, average='macro')
    accuracy = accuracy_score(y_test, y_pred_classes)

    print("Classificazione completa:\n")
    print(classification_report(label_encoder.inverse_transform(y_test),
                                label_encoder.inverse_transform(y_pred_classes)))
    print(f"Accuratezza: {accuracy}")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

def main():
    """
    Funzione principale per caricare, preprocessare, bilanciare i dati,
    addestrare e valutare una rete neurale.
    """
    file_path = "Cistulli_Domenico/data/raw/Preprocessed_Orange_Data.csv"

    data = load_processed_data(file_path)
    if data is None:
        return

    X, y, label_encoder = preprocess_data(data)

    X_balanced, y_balanced = balance_classes(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    y_train_onehot = to_categorical(y_train)
    y_val_onehot = to_categorical(y_val)
    y_test_onehot = to_categorical(y_test)

    model = build_neural_network(input_dim=X_train.shape[1], num_classes=len(label_encoder.classes_))
    train_neural_network(model, X_train, y_train_onehot, X_val, y_val_onehot)

if __name__ == "__main__":
    main()
