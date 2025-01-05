# Navi: A Framework for Social Media Analysis

Navi is a user-friendly framework designed for Social Media Analysis (SMA), providing an intuitive graphical dashboard for end-to-end text analysis tasks. It simplifies data import, preprocessing, sentiment analysis, and result visualization for researchers, students, and professionals.

## Features
- **Dataset Import:** Load custom datasets and specify the text column for analysis.
- **Data Preprocessing:** Built-in data cleaning and preparation pipelines.
- **Sentiment Analysis:** Run sentiment analysis models on textual data.
- **Visualization:** Generate summary graphs and statistical metrics for insights.
- **Modular Design:** Designed for easy expansion with future capabilities like:
   - Direct social media data import.
   - Advanced search parameter definition.
   - Topic modeling and semantic analysis support.

---

## Installation

### Using Pip

Clone the repository and navigate to the folder:

```bash
git clone https://github.com/UnruhChris/Navi.git
cd Navi
```

Ensure you have Python 3.8 or later installed. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Run the application with:

```bash
python NAVI.py
```

### Using Anaconda

Clone the repository and navigate to the folder:

```bash
git clone https://github.com/UnruhChris/Navi.git
cd Navi
```

Create a virtual environment using Anaconda:

```bash
conda create --name navi_env python=3.8
```

Activate the environment:

```bash
conda activate navi_env
```

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Run the application with:

```bash
python NAVI.py
```

## Usage

After setting up the environment, you can start using Navi for Social Media Analysis:

- Clone the repository using `git clone https://github.com/UnruhChris/Navi.git` and navigate to the folder with `cd Navi`.
- Activate the Python environment:
  - For pip-based setup: Ensure all dependencies are installed using `pip install -r requirements.txt`.
  - For Anaconda setup: Activate the `navi_env` environment with `conda activate navi_env`.
- Run the application by executing `python NAVI.py` in the terminal.
- Use the graphical interface to:
  - Load your dataset (CSV files).
  - Select text columns for preprocessing and analysis.
  - Apply built-in preprocessing techniques like HTML tag removal, punctuation stripping, and more.
  - Perform sentiment analysis using models like VADER, Flair, or DistilBERT.
  - Visualize results with detailed graphs and export your processed data.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For inquiries, suggestions, or collaboration opportunities, please open an issue on this repository or contact the author via:

- **Email:** gambardella.chriss00@gmail.com
- **GitHub Profile:** [Christian Gambardella](https://github.com/UnruhChris)


