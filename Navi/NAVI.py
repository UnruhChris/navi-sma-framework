# Navi.py: GUI principale del framework Navi
import sys
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton, 
    QFileDialog, QLabel, QCheckBox, QTextEdit, QGridLayout, QComboBox, QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from qt_material import apply_stylesheet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from preprocessing import (
    remove_html_tags, remove_punctuation, remove_hashtags_urls, 
    remove_stopwords, remove_numbers, remove_emojis
)

class NaviGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Navi - Social Media Analysis")
        self.setGeometry(100, 100, 1200, 800)

        # Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Aggiunta dei tab
        self.tabs.addTab(SentimentAnalysisTab(), "Analisi Sentiment")
        self.tabs.addTab(QWidget(), "Topic Modelling")  # Placeholder per futuri tab


class SentimentAnalysisTab(QWidget):
    def __init__(self):
        super().__init__()

        # Layout principale
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Variabili
        self.filepath = None
        self.dataframe = None
        self.preprocessing_stats = {
            "html_tags_removed": 0,
            "punctuation_removed": 0,
            "stopwords_removed": 0,
            "hashtags_urls_removed": 0,
            "numbers_removed": 0,
            "emojis_removed": 0,
        }

        # Sezione 1: Caricamento Dataset e Modello
        self.dataset_section = QWidget()
        self.dataset_layout = QVBoxLayout()

        self.load_dataset_btn = QPushButton("Carica dataset")
        self.load_dataset_btn.clicked.connect(self.load_dataset)

        self.select_column_dropdown = QComboBox()
        self.select_column_dropdown.setEnabled(False)

        self.select_model_dropdown = QComboBox()
        self.select_model_dropdown.addItems(["VADER"])

        self.run_analysis_btn = QPushButton("Avvia analisi")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn.setEnabled(False)

        self.save_results_btn = QPushButton("Salva risultati")
        self.save_results_btn.clicked.connect(self.save_results)
        self.save_results_btn.setEnabled(False)

        self.dataset_layout.addWidget(self.load_dataset_btn)
        self.dataset_layout.addWidget(QLabel("Seleziona colonna:"))
        self.dataset_layout.addWidget(self.select_column_dropdown)
        self.dataset_layout.addWidget(QLabel("Seleziona modello:"))
        self.dataset_layout.addWidget(self.select_model_dropdown)
        self.dataset_layout.addWidget(self.run_analysis_btn)
        self.dataset_layout.addWidget(self.save_results_btn)
        self.dataset_section.setLayout(self.dataset_layout)

        # Sezione 2: Data Preprocessing
        self.preprocessing_section = QWidget()
        self.preprocessing_layout = QVBoxLayout()
        self.preprocessing_options = {}

        for option in [
            "Rimuovi tag HTML", "Rimuovi punteggiatura", "Rimuovi hashtag e URL", 
            "Rimuovi stopword", "Rimuovi numeri", "Rimuovi emoji"
        ]:
            checkbox = QCheckBox(option)
            self.preprocessing_options[option] = checkbox
            self.preprocessing_layout.addWidget(checkbox)

        self.start_preprocessing_btn = QPushButton("Avvia preprocessing")
        self.start_preprocessing_btn.clicked.connect(self.start_preprocessing)
        self.preprocessing_layout.addWidget(self.start_preprocessing_btn)
        self.preprocessing_section.setLayout(self.preprocessing_layout)

        # Sezione 3: Statistiche e Sommario
        self.stats_section = QWidget()
        self.stats_layout = QVBoxLayout()

        self.stats_label = QLabel("Statistiche sul dataset:")
        self.stats_label.setAlignment(Qt.AlignLeft)

        self.stats_view = QTextEdit()
        self.stats_view.setReadOnly(True)
        self.stats_layout.addWidget(self.stats_label)
        self.stats_layout.addWidget(self.stats_view)
        self.stats_section.setLayout(self.stats_layout)

        # Sezione 4: Visualizzazione Grafici
        self.graph_section = QWidget()
        self.graph_layout = QVBoxLayout()

        self.graph_label = QLabel("Grafico dei risultati:")
        self.graph_label.setAlignment(Qt.AlignLeft)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.graph_layout.addWidget(self.graph_label)
        self.graph_layout.addWidget(self.canvas)
        self.graph_section.setLayout(self.graph_layout)

        # Aggiunta sezioni al layout principale
        self.layout.addWidget(self.dataset_section, 0, 0, 1, 1)
        self.layout.addWidget(self.preprocessing_section, 1, 0, 1, 1)
        self.layout.addWidget(self.stats_section, 0, 1, 1, 1)
        self.layout.addWidget(self.graph_section, 1, 1, 1, 1)

    def load_dataset(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Carica Dataset", "", "CSV Files (*.csv)")
        if filepath:
            self.filepath = filepath
            self.dataframe = pd.read_csv(filepath)
            self.stats_view.append(f"Dataset caricato: {filepath}")

            # Statistiche del dataset
            num_rows = self.dataframe.shape[0]
            num_columns = self.dataframe.shape[1]
            missing_values = self.dataframe.isnull().sum().sum()
            column_info = self.dataframe.dtypes

            self.stats_view.append(f"Numero di righe: {num_rows}")
            self.stats_view.append(f"Numero di colonne: {num_columns}")
            self.stats_view.append(f"Valori mancanti: {missing_values}")
            self.stats_view.append("\nTipi di colonne:")
            for col, dtype in column_info.items():
                self.stats_view.append(f"  - {col}: {dtype}")

            self.select_column_dropdown.clear()
            self.select_column_dropdown.addItems(self.dataframe.columns)
            self.select_column_dropdown.setEnabled(True)
            self.run_analysis_btn.setEnabled(True)

    def start_preprocessing(self):
        if self.dataframe is None or self.dataframe.empty:
            QMessageBox.warning(self, "Errore", "Il dataset non è caricato o è vuoto.")
            return

        self.stats_view.append("\n--- Avvio del preprocessing ---")
        for checkbox in self.preprocessing_options.values():
            if checkbox.isChecked():
                option_text = checkbox.text()
                self.stats_view.append(f"Esecuzione: {option_text}")
                self.run_preprocessing_script(option_text)

        self.stats_view.append("--- Preprocessing completato ---\n")
        self.display_stats()
        self.reset_stats()

    def run_preprocessing_script(self, option_text):
        preprocessing_functions = {
            "Rimuovi tag HTML": remove_html_tags,
            "Rimuovi punteggiatura": remove_punctuation,
            "Rimuovi hashtag e URL": remove_hashtags_urls,
            "Rimuovi stopword": remove_stopwords,
            "Rimuovi numeri": remove_numbers,
            "Rimuovi emoji": remove_emojis,
        }

        preprocessing_function = preprocessing_functions.get(option_text)
        if preprocessing_function:
            selected_column = self.select_column_dropdown.currentText()
            processed_column, stats = preprocessing_function(self.dataframe[selected_column])
            self.dataframe[selected_column] = processed_column
            self.update_stats(stats)
            self.save_results_btn.setEnabled(True)
        else:
            self.stats_view.append(f"Errore: Nessuna funzione trovata per '{option_text}'")

    def update_stats(self, new_stats):
        for key, value in new_stats.items():
            self.preprocessing_stats[key] += (value - 1)

    def reset_stats(self):
        for key in self.preprocessing_stats:
            self.preprocessing_stats[key] = 0

    def display_stats(self):
        formatted_stats = "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" 
                                    for key, value in self.preprocessing_stats.items())
        self.stats_view.append("\n--- Statistiche del Preprocessing ---")
        self.stats_view.append(formatted_stats)


    def run_analysis(self):
        selected_model = self.select_model_dropdown.currentText()
        selected_column = self.select_column_dropdown.currentText()

        if selected_model == "VADER" and self.dataframe is not None:
            self.stats_view.append(f"\nEsecuzione analisi sentiment con il modello: {selected_model} sulla colonna: {selected_column}...")

            analyzer = SentimentIntensityAnalyzer()

            def get_sentiment_scores(text):
                if not isinstance(text, str):
                    return None
                return analyzer.polarity_scores(text)

            self.dataframe['sentiment_scores'] = self.dataframe[selected_column].apply(get_sentiment_scores)
            self.dataframe['compound_score'] = self.dataframe['sentiment_scores'].apply(lambda x: x['compound'] if x else None)
            self.dataframe['positive_score'] = self.dataframe['sentiment_scores'].apply(lambda x: x['pos'] if x else None)
            self.dataframe['neutral_score'] = self.dataframe['sentiment_scores'].apply(lambda x: x['neu'] if x else None)
            self.dataframe['negative_score'] = self.dataframe['sentiment_scores'].apply(lambda x: x['neg'] if x else None)

            self.stats_view.append("Analisi completata. Generazione del grafico dei risultati...")
            self.show_results()
            self.save_results_btn.setEnabled(True)

    def show_results(self):
        positive = self.dataframe['positive_score'].mean()
        neutral = self.dataframe['neutral_score'].mean()
        negative = self.dataframe['negative_score'].mean()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        sentiments = ['Positivo', 'Neutrale', 'Negativo']
        values = [positive, neutral, negative]
        ax.pie(values, labels=sentiments, autopct='%1.1f%%', colors=['green', 'blue', 'red'], startangle=90)
        ax.set_title("Distribuzione Analisi Sentiment")
        self.canvas.draw()

    def save_results(self):
        if self.dataframe is not None:
            output_path, _ = QFileDialog.getSaveFileName(self, "Salva risultati", "", "CSV Files (*.csv)")
            if output_path:
                self.dataframe.to_csv(output_path, index=False)
                QMessageBox.information(self, "Salvataggio completato", f"Risultati salvati in: {output_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Define extra options to override the Material theme
    extra = {
        'font_size': '1px'
    }
    
    # Apply the Material theme with extra options
    apply_stylesheet(app, theme='dark_cyan.xml', extra=extra)
    
    main_window = NaviGUI()
    main_window.show()
    sys.exit(app.exec_())
