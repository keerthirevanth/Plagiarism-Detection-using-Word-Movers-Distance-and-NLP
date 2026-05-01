# Plagiarism Detection with Word Mover's Distance and Network Visualization

Welcome to the **Text Dissimilarity Analyzer**, a powerful Streamlit-based web application designed to analyze semantic dissimilarity between text documents using **Word Mover's Distance (WMD)**. This tool leverages natural language processing (NLP) and optimization techniques to provide insightful comparisons, visualizations, and plagiarism detection capabilities. Whether you're comparing two texts, analyzing multiple documents, or detecting potential plagiarism, this application offers an intuitive interface and robust functionality.

## Features

- **Single Pair Comparison**: Compare two text inputs to compute their semantic dissimilarity using WMD, with visualizations including word clouds, frequency bar plots, and word flow graphs.
- **Batch File Comparison**: Upload multiple text files to generate a dissimilarity matrix, visualized as a heatmap, along with top pair dissimilarities and downloadable results.
- **Plagiarism Detection**: Identify the closest matching sentence in a text file to a given sample sentence, useful for detecting similarities in large texts.
- **Advanced Visualizations**:
  - Word clouds for visual representation of word frequencies.
  - Word flow graphs to illustrate semantic connections between documents.
  - Heatmaps for dissimilarity matrices.
  - Bar plots for top dissimilarities and word frequencies.
- **Interactive UI**: Built with Streamlit, featuring a modern, user-friendly interface with customizable styling.
- **Robust NLP Pipeline**: Utilizes Word2Vec embeddings, NLTK for preprocessing, and Gurobi for WMD optimization.

## Prerequisites

To run this project, ensure you have the following installed:

- Python 3.8 or higher
- A Gurobi license (free academic licenses available at [Gurobi's website](https://www.gurobi.com/))
- Required Python packages (listed in `requirements.txt`)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/text-dissimilarity-analyzer.git
   cd text-dissimilarity-analyzer
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**:
   The application automatically downloads required NLTK datasets (`stopwords`, `averaged_perceptron_tagger_eng`, `punkt`, `wordnet`) on first run.

5. **Set Up Gurobi**:
   - Install Gurobi by following the instructions at [Gurobi's website](https://www.gurobi.com/downloads/).
   - Ensure the `GUROBI_HOME` environment variable is set and the Gurobi license is configured.

6. **Run the Application**:
   ```bash
   streamlit run txtapp.py
   ```

   This will launch the web application in your default browser.

## Usage

1. **Single Pair Comparison**:
   - Select "Single Pair Comparison" from the sidebar.
   - Enter two text documents in the provided text areas.
   - Click "Compute WMD" to view the dissimilarity score, word clouds, frequency bar plots, word flow graph, and flow details table.

2. **Batch File Comparison**:
   - Select "Batch File Comparison" from the sidebar.
   - Upload multiple `.txt` files.
   - Click "Compute Dissimilarity Matrix" to generate a heatmap, top pair dissimilarities, a dissimilarity score table, and a downloadable CSV.
   - Select a document pair to visualize the word flow between them.

3. **Plagiarism Detection**:
   - Select "Plagiarism Detection" from the sidebar.
   - Enter a sample sentence and upload a text file (e.g., a book or article).
   - Click "Find Closest Match" to view the closest matching sentence, its dissimilarity score, and a table of top matching sentences.

## Project Structure

```
text-dissimilarity-analyzer/
│
├── txtapp.py              # Main Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Dependencies

The project relies on the following Python packages (listed in `requirements.txt`):
- numpy
- pandas
- matplotlib
- seaborn
- gensim
- scipy
- nltk
- wordcloud
- gurobipy
- streamlit
- networkx

## Notes

- **Word2Vec Model**: The application uses the `word2vec-google-news-300` model from `gensim`. It is cached using Streamlit's `@st.cache_resource` to optimize performance.
- **Preprocessing**: The NLP pipeline removes stopwords, lemmatizes words, and excludes proper nouns for robust WMD computation.
- **Gurobi**: WMD is computed using Gurobi's optimization solver. Ensure you have a valid license to avoid errors.
- **Performance**: For large texts or multiple files, computation may take longer due to WMD's complexity. Consider using smaller texts for faster results.

Happy analyzing! 📝✨
