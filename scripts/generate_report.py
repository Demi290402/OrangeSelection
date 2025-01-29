# OrangeSelection: Generazione del Report Finale

import os
import json

# Percorsi
results_path = "Cistulli_Domenico/results/"
metrics_file = os.path.join(results_path, "model_metrics.json")
report_file = os.path.join(results_path, "report.txt")

def generate_report():
    """
    Genera un report riassuntivo delle metriche dei modelli.
    """
    if not os.path.exists(metrics_file):
        print(f"File delle metriche non trovato: {metrics_file}")
        return

    try:
        # Caricamento delle metriche
        with open(metrics_file, "r") as file:
            metrics = json.load(file)

        # Creazione del report
        with open(report_file, "w") as report:
            report.write("OrangeSelection: Report Risultati modelli\n")
            report.write("=" * 40 + "\n\n")

            for model, model_metrics in metrics.items():
                report.write(f"Modello: {model}\n")
                for metric, value in model_metrics.items():
                    report.write(f"- {metric}: {value:.4f}\n")
                report.write("\n")

        print(f"Report generato con successo: {report_file}")
    except Exception as e:
        print(f"Errore durante la generazione del report: {e}")

if __name__ == "__main__":
    print("Generazione del report iniziata...")
    generate_report()
    print("Generazione del report completata.")