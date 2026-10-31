# Tokens & Embeddings Explorer

A Streamlit app that shows, for any text you enter, how it is converted into tokens and embeddings.

## What it shows
- Number of characters, words, tokens, special tokens and sub-word pieces
- How the text is split into tokens (colour-coded) with token IDs and character spans
- The 6-step conversion: tokenization, token ID, embedding lookup, position info, transformer layers, pooling
- Embedding dimensions, number of embeddings and the full values table (CSV download)
- Heatmaps, per-token vector plots, token similarity map and a 2D PCA projection
- For multiple texts: one sentence embedding per text and a similarity matrix

## Models
- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- `bert-base-uncased` (768 dimensions)
- `distilbert-base-uncased` (768 dimensions)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run in Google Colab
```python
!git clone https://github.com/<your-username>/tokens-embeddings-explorer.git
%cd tokens-embeddings-explorer
%run colab_launcher.py
```
Open the printed `trycloudflare.com` link. Keep the cell running.

## Try it
Copy example inputs from `sample_texts.txt`.

## Files
| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `colab_launcher.py` | Starts the app and a public link in Colab |
| `sample_texts.txt` | Example inputs to paste |
