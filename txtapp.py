
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import KeyedVectors
from scipy import spatial
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from wordcloud import WordCloud
import gurobipy as gp
from gurobipy import GRB
import streamlit as st
import networkx as nx
import io
from pathlib import Path
from collections import Counter

# Set page config as the first Streamlit command
st.set_page_config(page_title="Text Dissimilarity Analyzer", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 20px;
    }
    .stTextArea textarea {
        border: 2px solid #1e3a8a;
        border-radius: 8px;
        font-size: 18px;
    }
    .stFileUploader>div {
        border: 2px solid #1e3a8a;
        border-radius: 8px;
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Arial', sans-serif;
        font-size: 38px;
    }
    h2 {
        color: #1e3a8a;
        font-family: 'Arial', sans-serif;
        font-size: 32px;
        margin-top: 20px;
    }
    h3 {
        color: #1e3a8a;
        font-family: 'Arial', sans-serif;
        font-size: 28px;
        margin-top: 15px;
    }
    h4 {
        color: #1e3a8a;
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        margin-top: 10px;
    }
    .stDataFrame {
        border: 2px solid #1e3a8a;
        border-radius: 8px;
        padding: 10px;
    }
    .score-box {
        background-color: #dbeafe;
        border: 2px solid #1e3a8a;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-size: 20px;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
    }
    .highlight {
        color: #f97316;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load word2vec model
@st.cache_resource
def load_word2vec():
    return api.load('word2vec-google-news-300')

model = load_word2vec()

# Preprocessing function with lemmatization
def pre_processing(document):
    tagged_doc = pos_tag(document.split())
    edited_sentence = [word for word, tag in tagged_doc if tag not in ['NNP', 'NNPS']]
    tokenizer = RegexpTokenizer(r'\w+')
    processed_doc = tokenizer.tokenize(' '.join(edited_sentence))
    processed_doc = [lemmatizer.lemmatize(word.lower()) for word in processed_doc 
                    if word.lower() not in stopwords.words('english')]
    return processed_doc

# Handle out-of-vocabulary words
def get_embedding(word):
    if word in model:
        return model[word]
    try:
        similar_words = [w for w, _ in model.most_similar(word, topn=5) if w in model]
        if similar_words:
            return np.mean([model[w] for w in similar_words], axis=0)
        return np.zeros(model.vector_size)
    except:
        return np.zeros(model.vector_size)

# WMD computation function
def score_dissimilarity(D1, D2):
    D1 = set(D1)
    D2 = set(D2)
    D1 = {w for w in D1 if not np.all(get_embedding(w) == 0)}
    D2 = {w for w in D2 if not np.all(get_embedding(w) == 0)}
    
    if not D1 or not D2 or len(D2) < 5:
        return float('inf'), None
    
    freqency_D1 = {i: list(D1).count(i)/len(D1) for i in D1}
    freqency_D2 = {i: list(D2).count(i)/len(D2) for i in D2}
    
    m = gp.Model("Text_similarity")
    m.setParam('OutputFlag', 0)
    distance = {(i, j): spatial.distance.cosine(get_embedding(i), get_embedding(j)) 
                for i in D1 for j in D2}
    f = m.addVars(D1, D2, name="f", lb=0, ub=1)
    m.ModelSense = GRB.MINIMIZE
    m.setObjective(sum(f[w, w_prime] * distance[w, w_prime] for w in D1 for w_prime in D2))
    m.addConstrs(f.sum(w, '*') == freqency_D1[w] for w in D1)
    m.addConstrs(f.sum('*', w_prime) == freqency_D2[w_prime] for w_prime in D2)
    m.optimize()
    flow = {(i, j): f[i, j].X for i in D1 for j in D2 if f[i, j].X > 0}
    return m.ObjVal, flow

# Batch comparison for multiple files (dissimilarity matrix)
def batch_compare_files(files):
    documents = []
    filenames = []
    for file in files:
        content = file.read().decode('utf-8')
        documents.append(content)
        filenames.append(file.name)
    
    processed_docs = [pre_processing(doc) for doc in documents]
    n = len(documents)
    dissimilarity_matrix = np.full((n, n), np.inf)
    flows = {}
    pair_scores = []
    
    for i in range(n):
        dissimilarity_matrix[i, i] = 0  # Same document has zero dissimilarity
        for j in range(i + 1, n):
            score, flow = score_dissimilarity(processed_docs[i], processed_docs[j])
            dissimilarity_matrix[i, j] = score
            dissimilarity_matrix[j, i] = score
            flows[(i, j)] = flow
            pair_scores.append({
                'Pair': f"{filenames[i]} vs {filenames[j]}",
                'Dissimilarity': round(score, 2) if score != float('inf') else 'N/A'
            })
    
    return dissimilarity_matrix, filenames, flows, documents, pair_scores

# Visualize word flow with improved styling
def visualize_flow(flow, title="Word Flow Graph"):
    G = nx.DiGraph()
    for (w1, w2), flow_val in flow.items():
        if flow_val > 0:
            G.add_edge(w1, w2, weight=flow_val)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G, k=0.7, iterations=50)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='#60a5fa', node_size=1000, 
            font_size=14, font_weight='bold', font_color='#1e3a8a', 
            edge_color='#f97316', width=[flow_val * 6 for flow_val in nx.get_edge_attributes(G, 'weight').values()])
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, 
                                font_color='#1e3a8a', font_size=12)
    plt.title(title, fontsize=16, color='#1e3a8a', pad=20)
    return fig

# Visualize dissimilarity matrix as heatmap
def visualize_dissimilarity_matrix(matrix, filenames):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", 
                xticklabels=filenames, yticklabels=filenames, 
                cbar_kws={'label': 'Dissimilarity Score'})
    plt.title("Dissimilarity Matrix", fontsize=16, color='#1e3a8a', pad=20)
    return fig

# Visualize word frequencies as bar plot
def visualize_word_frequencies(doc, title="Word Frequencies"):
    freq = Counter(doc)
    words, counts = zip(*freq.items())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts, y=words, palette="Blues_r")
    plt.title(title, fontsize=16, color='#1e3a8a', pad=20)
    plt.xlabel("Frequency", fontsize=12, color='#1e3a8a')
    plt.ylabel("Words", fontsize=12, color='#1e3a8a')
    return fig

# Visualize top pair dissimilarities
def visualize_pair_scores(pair_scores):
    df = pd.DataFrame(pair_scores)
    df = df[df['Dissimilarity'] != 'N/A'].sort_values(by='Dissimilarity').head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Dissimilarity', y='Pair', data=df, palette="Oranges_r")
    plt.title("Top 10 Lowest Dissimilarity Scores", fontsize=16, color='#1e3a8a', pad=20)
    plt.xlabel("Dissimilarity Score", fontsize=12, color='#1e3a8a')
    plt.ylabel("Document Pair", fontsize=12, color='#1e3a8a')
    return fig

# Streamlit app
def main():
    st.title("Text Dissimilarity Analyzer")
    st.markdown("<p>Analyze text dissimilarity using <span class='highlight'>Word Mover's Distance (WMD)</span> with interactive inputs and enhanced visualizations.</p>", 
                unsafe_allow_html=True)

    # Sidebar for mode selection
    st.sidebar.markdown("<h2 style='color: #1e3a8a;'>Select Mode</h2>", unsafe_allow_html=True)
    mode = st.sidebar.selectbox("", ["Single Pair Comparison", "Batch File Comparison", "Plagiarism Detection"])

    # Single Pair Comparison
    if mode == "Single Pair Comparison":
        st.markdown("<h2 style='color: #1e3a8a;'>Single Pair Comparison</h2>", unsafe_allow_html=True)
        st.markdown("<h4>Enter two documents to compare their semantic dissimilarity.</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Document 1**")
            doc1 = st.text_area("Text",
                               "The little man stood glancing from one to the other of us with half-frightened, half-hopeful eyes, as one who is on the verge of a windfall or of a catastrophe.",
                               height=150)
        with col2:
            st.markdown("**Document 2**")
            doc2 = st.text_area( "Text",
                               "With a gaze that shifted back and forth between us, the diminutive figure appeared to be a mixture of apprehension and anticipation, uncertain if he was on the cusp of a fortune or a disaster.",
                               height=150)
        
        if st.button("Compute WMD"):
            with st.container():
                st.markdown("<h3 style='color: #1e3a8a;'>Results</h3>", unsafe_allow_html=True)
                processed_doc1 = pre_processing(doc1)
                processed_doc2 = pre_processing(doc2)
                
                # Word clouds and frequency bar plots
                st.markdown("<h4 style='color: #1e3a8a;'>Word Visualizations</h4>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h5 style='color: #1e3a8a;'>Document 1: Word Cloud</h5>", unsafe_allow_html=True)
                    wordcloud1 = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(processed_doc1))
                    fig1, ax1 = plt.subplots(figsize=(5, 3))
                    ax1.imshow(wordcloud1, interpolation='bilinear')
                    ax1.axis("off")
                    st.pyplot(fig1)
                    st.markdown("<h5 style='color: #1e3a8a;'>Document 1: Word Frequencies</h5>", unsafe_allow_html=True)
                    fig_freq1 = visualize_word_frequencies(processed_doc1, "Document 1 Word Frequencies")
                    st.pyplot(fig_freq1)
                with col2:
                    st.markdown("<h5 style='color: #1e3a8a;'>Document 2: Word Cloud</h5>", unsafe_allow_html=True)
                    wordcloud2 = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(processed_doc2))
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    ax2.imshow(wordcloud2, interpolation='bilinear')
                    ax2.axis("off")
                    st.pyplot(fig2)
                    st.markdown("<h5 style='color: #1e3a8a;'>Document 2: Word Frequencies</h5>", unsafe_allow_html=True)
                    fig_freq2 = visualize_word_frequencies(processed_doc2, "Document 2 Word Frequencies")
                    st.pyplot(fig_freq2)
                
                # Compute WMD
                score, flow = score_dissimilarity(processed_doc1, processed_doc2)
                if score != float('inf'):
                    st.markdown(f"<div class='score-box'>Dissimilarity Score: {score:.2f}</div>", unsafe_allow_html=True)
                    
                    # Flow visualization
                    st.markdown("<h4 style='color: #1e3a8a;'>Word Flow Visualization</h4>", unsafe_allow_html=True)
                    fig = visualize_flow(flow, "Word Flow Between Documents")
                    st.pyplot(fig)
                    
                    # Flow table
                    st.markdown("<h4 style='color: #1e3a8a;'>Flow Details</h4>", unsafe_allow_html=True)
                    flow_df = pd.DataFrame([
                        {'Word 1': i, 'Word 2': j, 'Flow': flow[i, j], 'Distance': spatial.distance.cosine(get_embedding(i), get_embedding(j))}
                        for (i, j) in flow
                    ]).sort_values(by='Distance')
                    st.dataframe(flow_df, use_container_width=True)
                else:
                    st.error("Unable to compute WMD (e.g., too few valid words).")

    # Batch File Comparison
    elif mode == "Batch File Comparison":
        st.markdown("<h2 style='color: #1e3a8a;'>Batch File Comparison</h2>", unsafe_allow_html=True)
        st.markdown("Upload multiple text files to compute a dissimilarity score matrix.")
        uploaded_files = st.file_uploader("Upload Text Files", type="txt", accept_multiple_files=True)
        
        if uploaded_files and st.button("Compute Dissimilarity Matrix"):
            with st.container():
                st.markdown("<h3 style='color: #1e3a8a;'>Results</h3>", unsafe_allow_html=True)
                matrix, filenames, flows, documents, pair_scores = batch_compare_files(uploaded_files)
                
                # Display heatmap
                st.markdown("<h4 style='color: #1e3a8a;'>Dissimilarity Matrix Heatmap</h4>", unsafe_allow_html=True)
                fig = visualize_dissimilarity_matrix(matrix, filenames)
                st.pyplot(fig)
                
                # Display top pair dissimilarities
                st.markdown("<h4 style='color: #1e3a8a;'>Top Pair Dissimilarities</h4>", unsafe_allow_html=True)
                fig_pairs = visualize_pair_scores(pair_scores)
                st.pyplot(fig_pairs)
                
                # Display matrix as table
                st.markdown("<h4 style='color: #1e3a8a;'>Dissimilarity Scores Table</h4>", unsafe_allow_html=True)
                matrix_df = pd.DataFrame(matrix, index=filenames, columns=filenames)
                st.dataframe(matrix_df, use_container_width=True)
                
                # Download matrix
                csv_buffer = io.StringIO()
                matrix_df.to_csv(csv_buffer)
                st.download_button(
                    label="Download Dissimilarity Matrix as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="dissimilarity_matrix.csv",
                    mime="text/csv"
                )
                
                # Flow visualization for selected pair
                st.markdown("<h4 style='color: #1e3a8a;'>Word Flow Visualization</h4>", unsafe_allow_html=True)
                pair_options = [f"{filenames[i]} vs {filenames[j]}" for i in range(len(filenames)) for j in range(i + 1, len(filenames))]
                pair_indices = [(i, j) for i in range(len(filenames)) for j in range(i + 1, len(filenames))]
                selected_pair = st.selectbox("Select Pair for Flow Visualization", pair_options)
                if selected_pair:
                    pair_idx = pair_options.index(selected_pair)
                    i, j = pair_indices[pair_idx]
                    flow = flows.get((i, j), flows.get((j, i)))
                    if flow:
                        fig = visualize_flow(flow, f"Word Flow: {filenames[i]} vs {filenames[j]}")
                        st.pyplot(fig)

    # Plagiarism Detection
    else:
        st.markdown("<h2 style='color: #1e3a8a;'>Plagiarism Detection</h2>", unsafe_allow_html=True)
        st.markdown("Enter a sample sentence and upload a text to find the closest matching sentence.")
        sample_sentence = st.text_area("Sample Sentence", 
                                     "With a gaze that shifted back and forth between us, the diminutive figure appeared to be a mixture of apprehension and anticipation, uncertain if he was on the cusp of a fortune or a disaster.",
                                     height=150)
        book_file = st.file_uploader("Upload Book Text File", type="txt")
        
        if book_file and st.button("Find Closest Match"):
            with st.container():
                st.markdown("<h3 style='color: #1e3a8a;'>Results</h3>", unsafe_allow_html=True)
                content = book_file.read().decode('utf-8')
                content = content.replace('\n', ' ').replace('_', ' ').replace('\r', '')
                sentences = list(map(str.strip, content.split(".")))
                processed_sentences = [pre_processing(s) for s in sentences]
                
                processed_sample = pre_processing(sample_sentence)
                obj_best = float('inf')
                sentence_best = None
                results = []
                
                for i, proc_sent in enumerate(processed_sentences):
                    obj, flow = score_dissimilarity(processed_sample, proc_sent)
                    if obj < obj_best:
                        obj_best = obj
                        sentence_best = sentences[i]
                        results.append({
                            'Sentence Index': i, 
                            'Dissimilarity': round(obj, 2), 
                            'Sentence': sentences[i][:100] + '...' if len(sentences[i]) > 100 else sentences[i]
                        })
                
                st.markdown(f"<div class='score-box'>Closest Match Dissimilarity Score: {obj_best:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='color: #1e3a8a;'>Closest Matching Sentence</h4><p>{sentence_best}</p>", unsafe_allow_html=True)
                
                st.markdown("<h4 style='color: #1e3a8a;'>Top Matching Sentences</h4>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(results).sort_values(by='Dissimilarity'), use_container_width=True)

if __name__ == "__main__":
    main()
