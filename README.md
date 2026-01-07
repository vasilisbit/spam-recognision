# 📧 Spam Email Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE.md)

A comprehensive machine learning project for spam email detection using multiple classification algorithms and feature representations. This project compares traditional **Bag-of-Words (BoW)** with modern **Sentence Embeddings (SBERT)** approaches, achieving up to **96.22% F1-score** and **99.78% AUC**.

## 🎯 Project Overview

This repository contains a complete implementation and analysis of spam email classification, developed as part of a thesis project. The project explores:

- **Multiple ML Algorithms**: Naive Bayes, Logistic Regression, k-NN, SVM
- **Feature Representations**: TF-IDF (5000 features) vs. Sentence Transformers (384-dim embeddings)
- **Dimensionality Reduction**: PCA with various variance retention ratios
- **Comprehensive Evaluation**: Precision, Recall, F1-Score, AUC-ROC metrics

## 📊 Key Results

| Model                   | Representation   | Precision  | Recall     | F1-Score   | AUC        | Rank |
| ----------------------- | ---------------- | ---------- | ---------- | ---------- | ---------- | ---- |
| **Naive Bayes**         | BoW (5000)       | 0.9439     | **0.9811** | **0.9622** | 0.9959     | 🥇   |
| **SVM**                 | Embeddings (384) | **0.9587** | 0.9496     | 0.9541     | **0.9978** | 🥈   |
| **Logistic Regression** | Embeddings (384) | 0.9391     | 0.9465     | 0.9427     | 0.9949     | 🥉   |
| k-NN (k=1)              | BoW (5000)       | 0.9698     | 0.8583     | 0.9106     | 0.9251     | 4    |
| SVM + PCA (90%)         | Embeddings+PCA   | 0.9887     | 0.8236     | 0.8986     | 0.9969     | 5    |
| LogReg + PCA (10)       | Embeddings+PCA   | 0.8891     | 0.8961     | 0.8925     | 0.9887     | 6    |

### 🏆 Best Performers

- **Best Overall**: Naive Bayes with BoW (F1: 96.22%, balanced performance)
- **Highest Precision**: SVM+PCA (98.87%, excellent at avoiding false positives)
- **Highest Recall**: Naive Bayes (98.11%, catches most spam)
- **Best AUC**: SVM with Polynomial Kernel (99.78%, superior discrimination)

## ✨ Features

- **Text Preprocessing Pipeline**

  - URL and email address removal
  - Special character filtering
  - Lowercase conversion
  - Stopword removal (NLTK)
  - Porter stemming

- **Feature Extraction**

  - Bag-of-Words with CountVectorizer (5000 vocabulary size)
  - Sentence embeddings using `all-MiniLM-L6-v2`
  - PCA dimensionality reduction

- **Classification Models**

  - Multinomial Naive Bayes
  - Logistic Regression (C=1.0)
  - k-Nearest Neighbors (k=1, 3, 5, 11, 21)
  - Support Vector Machines (Linear, Polynomial, RBF kernels)

- **Comprehensive Evaluation**
  - Confusion matrices
  - Precision/Recall/F1-Score
  - ROC curves and AUC
  - Cross-validation
  - Computational performance analysis

## 📁 Project Structure

```
spam-recognision/
├── spam_classification_assignment.ipynb  # Main Jupyter notebook
├── emails.csv                            # Dataset (5572 emails)
├── doc.tex                               # Full LaTeX thesis documentation
├── doc-stock.tex                         # Thesis template
├── references.bib                        # Bibliography
├── hellas.bst                           # Greek BibTeX style
├── media/
│   └── graphs/                          # Generated visualizations
│       ├── confusion_matrix_*.png       # Confusion matrices
│       ├── roc_curves_*.png            # ROC curves
│       ├── *-comparison.png            # Model comparisons
│       └── pca-*.png                   # PCA analysis plots
└── README.md                            # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/IBilba/spam-recognision.git
   cd spam-recognision
   ```

2. **Create a virtual environment**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install numpy pandas scikit-learn
   pip install nltk sentence-transformers
   pip install matplotlib seaborn
   pip install jupyter ipykernel
   pip install tqdm
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```

### Quick Start

1. **Launch Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

2. **Open the notebook**

   - Navigate to `spam_classification_assignment.ipynb`
   - Run cells sequentially (Cell → Run All)

3. **Expected runtime**
   - Full execution: ~10-15 minutes (CPU)
   - With BERT embeddings: ~5-10 minutes (GPU recommended)

## 💻 Usage

### Basic Classification Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('emails.csv')

# Preprocess
from preprocessing import preprocess_text
df['clean_text'] = df['text'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['spam'],
    test_size=0.2, random_state=42, stratify=df['spam']
)

# Create features (Bag-of-Words)
vectorizer = CountVectorizer(max_features=5000)

# Train model
model = MultinomialNB()
model.fit(X_train_bow, y_train)

# Evaluate
y_pred = model.predict(X_test_bow)
print(classification_report(y_test, y_pred))

# Predict on new email
sample = "Congratulations! You won $1000! Click here!"
sample_vec = vectorizer.transform([preprocess_text(sample)])
prediction = model.predict(sample_vec)[0]
print(f"Prediction: {'SPAM' if prediction == 1 else 'HAM'}")
```

### Advanced: Using Sentence Embeddings

```python
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC

# Load SBERT model
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
X_train_emb = encoder.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = encoder.encode(X_test.tolist(), show_progress_bar=True)

# Train SVM with polynomial kernel
svm = SVC(kernel='poly', degree=3, C=1.0, probability=True)
svm.fit(X_train_emb, y_train)

# Evaluate
y_pred = svm.predict(X_test_emb)
print(classification_report(y_test, y_pred))
```

## 📈 Dataset

The `emails.csv` dataset contains:

- **Total**: 5,728 emails
- **Spam**: 1,368 emails (23.88%)
- **Ham**: 4,360 emails (76.12%)
- **Split**: Train / Validation / Test
- **Features**: Raw email text with subject lines

**Data Distribution**:

- Training set: 2,000 emails (513 spam / 1,487 ham)
- Validation set: 1,000 emails (220 spam / 780 ham)
- Test set: 2,728 emails (635 spam / 2,093 ham)

## 🔬 Methodology

### 1. Text Preprocessing

- Remove URLs, email addresses, special characters
- Convert to lowercase
- Tokenization
- Remove English stopwords
- Apply Porter stemming

### 2. Feature Extraction

**Bag-of-Words (BoW)**:

- CountVectorizer with vocabulary size 5000
- Frequency word occurrence representation
- Sparse representation (~1% density)
- Captures presence of spam-indicative words

**Sentence Embeddings**:

- all-MiniLM-L6-v2 model
- 384-dimensional dense vectors
- Semantic meaning preservation
- Transfer learning from 1B+ sentence pairs

### 3. Model Training

- Stratified train/test split
- Hyperparameter tuning (k-values, kernels, PCA components)
- Cross-validation for robustness
- Multiple algorithm comparison

### 4. Evaluation

- Confusion matrices
- Precision, Recall, F1-Score
- ROC curves and AUC
- Statistical significance testing

## 📚 Documentation

The complete thesis documentation is available in the PDF

## 🛠️ Technologies Used

- **Python 3.10**: Core programming language
- **NumPy & Pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
- **NLTK**: Natural language processing
- **Sentence-Transformers**: State-of-the-art embeddings
- **Matplotlib & Seaborn**: Visualization
- **Jupyter**: Interactive development

## 📊 Key Insights

1. **BoW vs Embeddings**: Traditional BoW with Naive Bayes slightly outperforms modern embeddings due to the binary classification nature and distinctive spam vocabulary.

2. **Computational Trade-offs**:

   - Naive Bayes: <1s training, instant inference
   - SVM: 15-30s training, 0.1s per email
   - BERT embeddings: 5-10 min generation (one-time)

3. **PCA Impact**: Dimensionality reduction from 384→112 dims (90% variance) maintains 99.69% AUC but reduces recall by 12%.

4. **Optimal Hyperparameters**:
   - k-NN: k=1 best for spam (high precision)
   - SVM: Polynomial kernel (degree=3) optimal
   - Feature count: 5000 features sufficient for BoW

## 🤝 Contributing

This is a proprietary project. Contributions are not accepted at this time. For questions or collaboration inquiries, please contact the author.

## 📝 License

This project is under a **Proprietary License**. You may use this software for personal, educational, or research purposes only. You may NOT modify, distribute, sell, or use it commercially without explicit permission. See the [LICENSE](LICENSE.md) file for complete terms.

## 👤 Author

**Μπίτζας Βασίλειος (Vasileios Bitzas)**

- GitHub: [@IBilba](https://github.com/IBilba)

## 🙏 Acknowledgments

- Dataset sourced from public spam email collections
- Sentence-Transformers library by Nils Reimers and Iryna Gurevych
- scikit-learn community for excellent ML tools
- NLTK project for NLP resources

---

**⭐ If you found this project helpful, please consider giving it a star!**
