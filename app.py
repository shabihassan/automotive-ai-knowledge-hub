import os, re, json, time, sqlite3, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import faiss
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from pypdf import PdfReader
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
GROK_MODEL = "grok-2-1212"
GROK_BASE_URL = "https://api.x.ai/v1"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE, CHUNK_OVERLAP = 900, 150
DB_PATH = "automotive_hub.db"

CATEGORIES = [
    "General Automotive", "EV & Battery", "Engine & Powertrain", "ADAS & Autonomous",
    "Manufacturing & Quality", "Aftersales & Service", "Regulations & Safety",
    "Sales, Market & Finance", "Supply Chain & Logistics", "Design & Software (SDV)",
]

if not XAI_API_KEY:
    print("⚠️ Warning: XAI_API_KEY environment variable not set!")

client = OpenAI(api_key=XAI_API_KEY, base_url=GROK_BASE_URL, timeout=120)
embedder = SentenceTransformer(EMBED_MODEL)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.executescript(""""
CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, type TEXT, category TEXT,
    size_kb REAL, n_chunks INTEGER, uploaded_at TEXT, file_hash TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER, chunk_index INTEGER,
    location TEXT, text TEXT, embedding BLOB);
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, question TEXT, answer TEXT,
    sources TEXT, latency REAL);
""")
conn.commit()

def extract_pdf(path):
    reader = PdfReader(path)
    return [(f"Page {i+1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]

def extract_docx(path):
    doc = DocxDocument(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for t in doc.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [(f"Section {i//25 + 1}", "\n".join(parts[i:i+25])) for i in range(0, len(parts), 25)]

def df_to_segments(df, label, rows_per_block=15):
    df = df.fillna("").astype(str)
    records = df.to_dict("records")
    segs = []
    for i in range(0, len(records), rows_per_block):
        block = records[i:i+rows_per_block]
        lines = [" | ".join(f"{k}: {v}" for k, v in r.items() if v.strip()) for r in block]
        segs.append((f"{label}, rows {i+2}-{i+1+len(block)}", "\n".join(lines)))
    return segs

def extract_excel(path):
    segs = []
    for sheet, df in pd.read_excel(path, sheet_name=None).items():
        segs += df_to_segments(df, f"Sheet '{sheet}'")
    return segs

def extract_csv(path):
    for enc in ("utf-8", "latin-1"):
        try:
            return df_to_segments(pd.read_csv(path, encoding=enc), "CSV")
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV file")

EXTRACTORS = {".pdf": extract_pdf, ".docx": extract_docx, ".xlsx": extract_excel,
              ".xls": extract_excel, ".csv": extract_csv}

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", i + size // 2, end)
            if cut != -1:
                end = cut + 1
        out.append(text[i:end].strip())
        if end >= len(text):
            break
        i = max(end - overlap, i + 1)
    return [c for c in out if len(c) > 30]

index, chunk_meta = None, []

def rebuild_index():
    global index, chunk_meta
    rows = conn.execute("""SELECT c.id, c.embedding, d.name, c.location, c.text
                           FROM chunks c JOIN documents d ON d.id = c.doc_id""").fetchall()
    if not rows:
        index, chunk_meta = None, []
        return
    mat = np.vstack([np.frombuffer(r[1], dtype="float32") for r in rows])
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    chunk_meta = [dict(id=r[0], doc=r[2], loc=r[3], text=r[4]) for r in rows]

def ingest_files(files, category, progress=gr.Progress()):
    if not files:
        return "⚠️ Please select at least one file."
    log = []
    for f in progress.tqdm(files, desc="Indexing"):
        path = f if isinstance(f, str) else f.name
        name, ext = os.path.basename(path), os.path.splitext(path)[1].lower()
        if ext not in EXTRACTORS:
            log.append(f"❌ {name}: unsupported type"); continue
        try:
            data = open(path, "rb").read()
            fhash = hashlib.sha256(data).hexdigest()
            if conn.execute("SELECT 1 FROM documents WHERE file_hash=?", (fhash,)).fetchone():
                log.append(f"⏭️ {name}: already indexed"); continue
            pieces = []
            for loc, text in EXTRACTORS[ext](path):
                for ch in chunk_text(text):
                    pieces.append((loc, ch))
            if not pieces:
                log.append(f"⚠️ {name}: no readable text"); continue
            embs = embedder.encode([p[1] for p in pieces], normalize_embeddings=True,
                                   batch_size=64, show_progress_bar=False).astype("float32")
            cur = conn.execute(
                "INSERT INTO documents(name,type,category,size_kb,n_chunks,uploaded_at,file_hash) VALUES(?,?,?,?,?,?,?)",
                (name, ext[1:].upper(), category, round(len(data)/1024, 1), len(pieces),
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"), fhash))
            doc_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO chunks(doc_id,chunk_index,location,text,embedding) VALUES(?,?,?,?,?)",
                [(doc_id, i, p[0], p[1], embs[i].tobytes()) for i, p in enumerate(pieces)])
            conn.commit()
            log.append(f"✅ {name}: {len(pieces)} chunks indexed")
        except Exception as e:
            log.append(f"❌ {name}: {e}")
    rebuild_index()
    return "\n".join(log)

def delete_document(doc_id):
    try:
        doc_id = int(doc_id)
    except Exception:
        return "⚠️ Enter a valid document ID."
    conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    rebuild_index()
    return f"🗑️ Document {doc_id} deleted."

def retrieve(question, k=6, min_score=0.15):
    if index is None:
        return []
    q = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, min(k, len(chunk_meta)))
    return [dict(chunk_meta[i], score=float(s)) for s, i in zip(scores[0], ids[0])
            if i != -1 and s >= min_score]

def build_system_prompt(allow_general):
    base = ("You are an expert assistant covering the entire automotive industry: vehicles, EV & batteries, "
            "powertrain, ADAS/autonomy, manufacturing, quality, aftersales, regulations, supply chain, "
            "market and finance. You are given numbered excerpts from the user's uploaded documents. "
            "Answer using the excerpts and cite them inline like [1], [2]. Never invent citations.")
    if allow_general:
        return base + (" If the excerpts do not fully answer the question, you may add general automotive "
                       "knowledge, but clearly label it as 'General knowledge (not from your documents)'.")
    return base + " If the excerpts do not contain the answer, say so plainly instead of guessing."

def ask_grok(messages):
    r = client.chat.completions.create(model=GROK_MODEL, messages=messages, temperature=0.2)
    return r.choices[0].message.content

def format_sources(hits):
    if not hits:
        return "_No document sources used._"
    lines = ["### 📚 Sources"]
    for n, h in enumerate(hits, 1):
        snippet = h["text"][:220].replace("\n", " ")
        lines.append(f"**[{n}] {h['doc']}** — {h['loc']} · relevance {h['score']:.2f}\n> {snippet}…")
    return "\n\n".join(lines)

def chat_fn(message, history, top_k, allow_general):
    message = (message or "").strip()
    if not message:
         return "", history, gr.update()
    history = history or []
    t0 = time.time()
    hits = retrieve(message, int(top_k))
    if not hits and not allow_general:
        answer = ("I couldn't find relevant content in your documents. Upload related files "
                  "or enable 'Allow general automotive knowledge' in Settings.")
    else:
        context = "\n\n".join(f"[{n}] ({h['doc']} — {h['loc']})\n{h['text']}" for n, h in enumerate(hits, 1))
        msgs = [{"role": "system", "content": build_system_prompt(allow_general)}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
        msgs.append({"role": "user", "content": f"Document excerpts:\n{context or '(none)'}\n\nQuestion: {message}"})
        try:
            answer = ask_grok(msgs)
        except Exception as e:
            answer = f"⚠️ Grok API error: {e}"
    latency = round(time.time() - t0, 2)
    src_text = "; ".join(f"{h['doc']} ({h['loc']})" for h in hits)
    conn.execute("INSERT INTO history(ts,question,answer,sources,latency) VALUES(?,?,?,?,?)",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message, answer, src_text, latency))
    conn.commit()
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": answer}]
    return "", history, format_sources(hits)

def get_history(search=""):
    q = "SELECT id, ts AS time, question, answer, sources, latency AS seconds FROM history"
    params = ()
    if search and search.strip():
        q += " WHERE question LIKE ? OR answer LIKE ?"
        params = (f"%{search.strip()}%",) * 2
    return pd.read_sql_query(q + " ORDER BY id DESC LIMIT 300", conn, params=params)

def clear_history():
    conn.execute("DELETE FROM history"); conn.commit()
    return get_history()

def empty_fig(title):
    f = go.Figure()
    f.update_layout(title=title, height=320,
                    annotations=[dict(text="No data yet", showarrow=False, font=dict(size=16))])
    return f

def dashboard():
    docs = pd.read_sql_query(
        "SELECT id, name, type, category, size_kb, n_chunks AS chunks, uploaded_at FROM documents ORDER BY id DESC", conn)
    hist = pd.read_sql_query("SELECT ts, latency FROM history", conn)
    total_chunks = int(docs["chunks"].sum()) if len(docs) else 0
    avg_lat = f"{hist['latency'].mean():.1f}s" if len(hist) else "–"
    kpi = (f"| 📄 Documents | 🧩 Indexed chunks | 💾 Total size | 💬 Questions asked | ⏱️ Avg response |\n"
           f"|:-:|:-:|:-:|:-:|:-:|\n"
           f"| **{len(docs)}** | **{total_chunks}** | **{docs['size_kb'].sum()/1024:.2f} MB** "
           f"| **{len(hist)}** | **{avg_lat}** |\") if len(docs) or len(hist) else "_Nothing indexed yet._"
    if len(docs):
        t = docs["type"].value_counts().reset_index(); t.columns = ["type", "count"]
        f1 = px.pie(t, names="type", values="count", title="Documents by type", height=320)
        c = docs.groupby("category")["chunks"].sum().reset_index()
        f2 = px.bar(c, x="category", y="chunks", title="Indexed chunks by category", height=320)
    else:
        f1, f2 = empty_fig("Documents by type"), empty_fig("Indexed chunks by category")
    if len(hist):
        h = hist.assign(day=hist["ts"].str[:10]).groupby("day").size().reset_index(name="questions")
        f3 = px.bar(h, x="day", y="questions", title="Questions per day", height=320)
    else:
        f3 = empty_fig("Questions per day")
    return kpi, f1, f2, f3, docs

rebuild_index()

with gr.Blocks(title="Automotive AI Knowledge Hub", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 Automotive AI Knowledge Hub\nRAG assistant for the complete automotive industry · powered by Grok")
    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(type="messages", height=460)
            with gr.Row():
                msg = gr.Textbox(placeholder="e.g. What is the thermal-runaway test procedure for EV battery packs?",
                                 show_label=False, scale=6)
                send = gr.Button("Ask", variant="primary", scale=1)
            with gr.Accordion("⚙️ Settings", open=False):
                top_k = gr.Slider(2, 12, value=6, step=1, label="Excerpts retrieved (top-k)")
                allow_general = gr.Checkbox(value=False,
                                            label="Allow general automotive knowledge when documents lack the answer")
            sources_box = gr.Markdown()
            send.click(chat_fn, [msg, chatbot, top_k, allow_general], [msg, chatbot, sources_box])
            msg.submit(chat_fn, [msg, chatbot, top_k, allow_general], [msg, chatbot, sources_box])
            gr.Button("🧹 New conversation").click(lambda: ([], ""), None, [chatbot, sources_box])

        with gr.Tab("📥 Upload Documents"):
            files = gr.File(label="PDF, Word, Excel, CSV", file_count="multiple",
                            file_types=[".pdf", ".docx", ".xlsx", ".xls", ".csv"], type="filepath")
            category = gr.Dropdown(CATEGORIES, value=CATEGORIES[0], label="Category")
            up_btn = gr.Button("Index documents", variant="primary")
            up_status = gr.Textbox(label="Status", lines=6, interactive=False)

        with gr.Tab("⏱️ Search History"):
            with gr.Row():
                h_search_box = gr.Textbox(placeholder="Search past questions/answers…", show_label=False, scale=5)
                h_btn = gr.Button("Search", scale=1)
                h_clear = gr.Button("Clear all", variant="stop", scale=1)
            h_table = gr.Dataframe(value=get_history, wrap=True, interactive=False)
            h_btn.click(get_history, h_search_box, h_table)
            h_search_box.submit(get_history, h_search_box, h_table)
            h_clear.click(clear_history, None, h_table)

        with gr.Tab("📊 Dashboard") as dash_tab:
            kpi_md = gr.Markdown()
            with gr.Row():
                p1, p2, p3 = gr.Plot(), gr.Plot(), gr.Plot()
            gr.Markdown("### Indexed documents")
            docs_table = gr.Dataframe(wrap=True, interactive=False)
            with gr.Row():
                del_id = gr.Number(label="Document ID to delete", precision=0, scale=2)
                del_btn = gr.Button("🗑️ Delete document", variant="stop", scale=1)
            del_status = gr.Markdown()
            refresh = gr.Button("🔄 Refresh")

    dash_outputs = [kpi_md, p1, p2, p3, docs_table]
    up_btn.click(ingest_files, [files, category], up_status).then(dashboard, None, dash_outputs)
    del_btn.click(delete_document, del_id, del_status).then(dashboard, None, dash_outputs)
    refresh.click(dashboard, None, dash_outputs)
    dash_tab.select(dashboard, None, dash_outputs)
    demo.load(dashboard, None, dash_outputs)

if __name__ == "__main__":
    demo.queue().launch(prevent_thread_lock=True, server_port=7860)
