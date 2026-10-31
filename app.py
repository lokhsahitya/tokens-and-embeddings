import html
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

st.set_page_config(page_title="Tokens & Embeddings Explorer", layout="wide")
st.title("Tokens & Embeddings Explorer")
st.caption("Enter text to see how it is split into tokens and how every token becomes a vector.")

MODELS = {
    "all-MiniLM-L6-v2 (384 dims, sentence embeddings)": "sentence-transformers/all-MiniLM-L6-v2",
    "bert-base-uncased (768 dims)": "bert-base-uncased",
    "distilbert-base-uncased (768 dims)": "distilbert-base-uncased",
}

with st.sidebar:
    choice = st.selectbox("Embedding model", list(MODELS))
    multi = st.checkbox("Treat each line as a separate text", value=True)
    st.caption("The first run downloads the model (100-400 MB), so it can take a minute.")


@st.cache_resource(show_spinner="Loading tokenizer and model...")
def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    return tok, model


tok, model = load(MODELS[choice])
cfg = model.config

raw = st.text_area(
    "Enter your text (press Ctrl+Enter to apply)",
    height=150,
    placeholder="Type or paste your text here. Use one line per text if you want several embeddings.",
)
if multi:
    texts = [t.strip() for t in raw.split("\n") if t.strip()]
else:
    texts = [raw.strip()] if raw.strip() else []
if not texts:
    st.info("Enter some text above to see its tokens and embeddings.")
    st.stop()


@st.cache_data(show_spinner=False)
def analyze(model_name, text, _tok, _model):
    full_len = len(_tok(text)["input_ids"])
    enc = _tok(text, return_tensors="pt", truncation=True, max_length=512,
               return_offsets_mapping=True)
    word_ids = enc.word_ids(0)
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        out = _model(**enc, output_hidden_states=True)
        ids = enc["input_ids"]
        static = _model.get_input_embeddings()(ids)[0].numpy()   # lookup table
        layer0 = out.hidden_states[0][0].numpy()                 # lookup + position (+LayerNorm)
        ctx = out.last_hidden_state[0].numpy()                   # after all transformer layers
        m = enc["attention_mask"].unsqueeze(-1).float()
        sent = ((out.last_hidden_state * m).sum(1) / m.sum(1))[0].numpy()  # mean pooling
    id_list = ids[0].tolist()
    return dict(
        tokens=_tok.convert_ids_to_tokens(id_list), ids=id_list, offsets=offsets,
        word_ids=word_ids,
        special=_tok.get_special_tokens_mask(id_list, already_has_special_tokens=True),
        static=static, layer0=layer0, ctx=ctx, sent=sent, full_len=full_len,
    )


def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


results = [analyze(MODELS[choice], t, tok, model) for t in texts]
dim = cfg.hidden_size

# ---------------- Overview ----------------
st.header("1. Overview")
summary = pd.DataFrame({
    "Text": [t if len(t) < 60 else t[:57] + "..." for t in texts],
    "Characters": [len(t) for t in texts],
    "Words": [len(t.split()) for t in texts],
    "Tokens (with special)": [len(r["ids"]) for r in results],
    "Token embeddings": [len(r["ids"]) for r in results],
    "Sentence embeddings": [1] * len(texts),
    "Dimensions": [dim] * len(texts),
})
summary.index = [f"Text {i+1}" for i in range(len(texts))]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Texts", len(texts))
c2.metric("Total tokens", sum(len(r["ids"]) for r in results))
c3.metric("Total sentence embeddings", len(texts))
c4.metric("Embedding dimensions", dim)
st.dataframe(summary)

pick = 0
if len(texts) > 1:
    pick = st.selectbox("Inspect text", range(len(texts)),
                        format_func=lambda i: f"Text {i+1}: {summary['Text'].iloc[i]}")
text, r = texts[pick], results[pick]
toks, ids, special = r["tokens"], r["ids"], r["special"]
n_tok = len(toks)
labels = [f"{i}:{t}" for i, t in enumerate(toks)]
if r["full_len"] > n_tok:
    st.warning(f"This text has {r['full_len']} tokens but the model only reads 512, so it was truncated.")

# ---------------- Tokens ----------------
st.header("2. How the text was divided into tokens")
palette = ["#cfe8ff", "#ffd9c7", "#d5f5d0", "#f6d6ff", "#fff2b3", "#c9f2f0"]
chips = []
for i, (t, s) in enumerate(zip(toks, special)):
    bg = "#dddddd" if s else palette[i % len(palette)]
    chips.append(f'<span style="background:{bg};color:#111;padding:3px 8px;margin:3px;'
                 f'border-radius:6px;display:inline-block;font-family:monospace">{html.escape(t)}</span>')
st.markdown("".join(chips), unsafe_allow_html=True)
st.caption("Grey chips are special tokens added by the tokenizer. A token starting with ## is a "
           "continuation piece of the previous word (WordPiece splits rare words into sub-words).")

n_special = int(sum(special))
n_sub = sum(1 for t, s in zip(toks, special) if t.startswith("##") and not s)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Characters", len(text))
c2.metric("Words", len(text.split()))
c3.metric("Total tokens", n_tok)
c4.metric("Special tokens", n_special)
c5.metric("Sub-word pieces (##)", n_sub)

rows = []
for i, (t, tid, s, off, w) in enumerate(zip(toks, ids, special, r["offsets"], r["word_ids"])):
    rows.append({
        "Position": i, "Token": t, "Token ID": tid,
        "Kind": "special" if s else ("sub-word piece" if t.startswith("##") else "word / start of word"),
        "Word index": "-" if w is None else w,
        "Original text": "" if s else text[off[0]:off[1]],
        "Char span": "" if s else f"{off[0]}-{off[1]}",
    })
st.dataframe(pd.DataFrame(rows))

# ---------------- Conversion ----------------
st.header("3. How tokens were converted into embeddings")
vocab = tok.vocab_size
n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layers", "several"))
st.markdown(f"""
**Step 1: Tokenization.** The text is split into {n_tok} tokens using the model's vocabulary of {vocab:,} entries.

**Step 2: Token to ID.** Each token is replaced by its integer ID (the vocabulary row number), for example `{toks[min(1, n_tok-1)]}` becomes `{ids[min(1, n_tok-1)]}`.

**Step 3: Embedding lookup.** The model stores a learned matrix of shape `({vocab:,} x {dim})`. Row number `ID` is the starting vector of that token. This vector is the same wherever the token appears, so it has no context yet.

**Step 4: Position information.** Position embeddings are added so the model knows token order, and the result is normalised (LayerNorm).

**Step 5: Transformer layers.** {n_layers} layers of self-attention mix information between tokens. After them, each token vector depends on its neighbours, so `bank` in "river bank" and "bank loan" ends up different. This gives {n_tok} contextual vectors of {dim} dimensions each.

**Step 6: Pooling.** The token vectors are averaged (mean pooling, ignoring padding) into one **sentence embedding** of {dim} dimensions.
""")
lookup = pd.DataFrame({
    "Token": toks, "Token ID": ids,
    "Lookup vector (first 6 values)": [np.round(v[:6], 4).tolist() for v in r["static"]],
    "Contextual vector (first 6 values)": [np.round(v[:6], 4).tolist() for v in r["ctx"]],
})
st.dataframe(lookup)

st.subheader("How much did context change each token?")
sim_change = (r["static"] * r["ctx"]).sum(1) / (
    np.linalg.norm(r["static"], axis=1) * np.linalg.norm(r["ctx"], axis=1) + 1e-9)
fig, ax = plt.subplots(figsize=(max(6, n_tok * 0.5), 3.5))
ax.bar(labels, sim_change, color="steelblue")
ax.set_ylabel("Cosine similarity")
ax.set_title("Lookup vector vs final contextual vector (lower = more change from context)")
plt.xticks(rotation=60, ha="right")
st.pyplot(fig)

# ---------------- Embeddings ----------------
st.header("4. Embeddings")
c1, c2, c3 = st.columns(3)
c1.metric("Token embeddings", n_tok)
c2.metric("Dimensions per embedding", dim)
c3.metric("Matrix shape", f"{n_tok} x {dim}")

st.subheader("Embedding values (rows = tokens, columns = dimensions)")
df = pd.DataFrame(r["ctx"], index=labels, columns=[f"d{i}" for i in range(dim)])
st.dataframe(df)
st.download_button("Download token embeddings (CSV)", df.to_csv().encode(), "token_embeddings.csv")

st.subheader("Heatmap of all token embeddings")
fig, ax = plt.subplots(figsize=(12, max(3, n_tok * 0.35)))
sns.heatmap(r["ctx"], cmap="coolwarm", center=0, yticklabels=labels,
            xticklabels=False if dim > 64 else True, ax=ax)
ax.set_xlabel(f"Dimension (0 to {dim-1})")
st.pyplot(fig)

st.subheader("Look at one token's vector")
tsel = st.selectbox("Token", range(n_tok), format_func=lambda i: labels[i])
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
axes[0].bar(range(dim), r["ctx"][tsel], width=1.0, color="darkorange")
axes[0].set_title(f"Values of '{toks[tsel]}' across {dim} dimensions")
axes[0].set_xlabel("Dimension")
axes[1].hist(r["ctx"][tsel], bins=30, color="seagreen")
axes[1].set_title("Distribution of its values")
st.pyplot(fig)
st.write(f"Vector length (L2 norm): **{np.linalg.norm(r['ctx'][tsel]):.4f}**, "
         f"min: **{r['ctx'][tsel].min():.4f}**, max: **{r['ctx'][tsel].max():.4f}**")

st.subheader("Sentence embedding (mean of token vectors)")
fig, ax = plt.subplots(figsize=(13, 3))
ax.bar(range(dim), r["sent"], width=1.0, color="purple")
ax.set_xlabel("Dimension")
ax.set_title(f"Sentence embedding, {dim} values")
st.pyplot(fig)
with st.expander("Show sentence embedding values"):
    st.dataframe(pd.DataFrame(r["sent"], index=[f"d{i}" for i in range(dim)], columns=["value"]))

st.subheader("Token-to-token similarity")
S = cosine(r["ctx"], r["ctx"])
fig, ax = plt.subplots(figsize=(max(5, n_tok * 0.5), max(4, n_tok * 0.45)))
sns.heatmap(S, xticklabels=labels, yticklabels=labels, cmap="viridis",
            annot=n_tok <= 14, fmt=".2f", ax=ax)
st.pyplot(fig)

if n_tok >= 3:
    st.subheader("Tokens projected to 2D (PCA)")
    Z = PCA(n_components=2).fit_transform(r["ctx"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Z[:, 0], Z[:, 1], s=60, c=range(n_tok), cmap="viridis")
    for i, t in enumerate(toks):
        ax.annotate(t, (Z[i, 0], Z[i, 1]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    st.pyplot(fig)

# ---------------- Multiple texts ----------------
if len(texts) > 1:
    st.header("5. Comparing all texts (one sentence embedding each)")
    E = np.stack([x["sent"] for x in results])
    names = [f"Text {i+1}" for i in range(len(texts))]
    st.write(f"Number of sentence embeddings: **{len(texts)}**, each with **{dim}** dimensions "
             f"(matrix shape {len(texts)} x {dim})")
    fig, ax = plt.subplots(figsize=(12, max(2.5, len(texts) * 0.5)))
    sns.heatmap(E, cmap="coolwarm", center=0, yticklabels=names,
                xticklabels=False if dim > 64 else True, ax=ax)
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(max(4, len(texts)), max(3.5, len(texts) * 0.9)))
    sns.heatmap(cosine(E, E), annot=True, fmt=".2f", cmap="viridis",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title("Cosine similarity between texts")
    st.pyplot(fig)
