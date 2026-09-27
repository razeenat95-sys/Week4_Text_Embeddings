# Week 4 – Text Embeddings

## Task
Generating and Comparing Text Embeddings using Sentence Transformers.

## Objective
- Generate sentence embeddings using a pre-trained Sentence Transformer.
- Compare semantic similarity using cosine similarity.
- Understand embedding dimensions and prepare data for future semantic search and RAG tasks.

## Model
`all-MiniLM-L6-v2`

## Folder Structure

```text
Week4_Text_Embeddings/
├── dataset/
│   ├── generated_text.csv
│   ├── embeddings.npy
│   ├── similarity_scores.csv
│   └── sentence_pairs.csv
├── notebook/
│   └── Week4_Text_Embeddings.ipynb
├── embeddings.py
├── README.md
└── requirements.txt
```

## Installation

Create/activate your Python virtual environment, then run:

```bash
pip install -r requirements.txt
```

## Run

```bash
python embeddings.py
```

The first run downloads the `all-MiniLM-L6-v2` model. Internet access is required for the first model download.

## Expected Output

The program:
1. Loads at least 20 sentences.
2. Generates embeddings.
3. Displays the embedding vector dimension.
4. Calculates pairwise cosine similarity.
5. Identifies the top five most similar sentence pairs.
6. Saves embeddings and similarity results.

## Week 4 Observation

Text embeddings convert sentences into numerical vectors that capture semantic meaning. Sentences discussing similar concepts generally receive higher cosine similarity scores than unrelated sentences. This is useful for semantic search, document retrieval, recommendation systems, and future RAG applications.

## Note

The supplied `generated_text.csv` contains 20 practice sentences so the task can run immediately. If you already have the `generated_text.csv` created in Week 3, replace the supplied file with your Week 3 file and make sure it contains at least 20 usable text rows.
