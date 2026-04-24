import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import base64
import os

# Configure Streamlit page parameters
st.set_page_config(page_title="TF-IDF Engine", layout="wide", initial_sidebar_state="collapsed")

# --- Custom Styling ---
# Use CSS to make the app aesthetic with dark glassmorphic styling, neon text, etc.
custom_css = """
<style>
    /* Global Styling */
    body, .stApp {
        background-color: #131417; 
        color: #e4e6ed;
        font-family: 'Inter', 'Roboto', sans-serif;
    }
    
    /* Clean Header Gradient */
    .title-bar {
        border-left: 6px solid #4da3ff;
        padding-left: 14px;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 2rem;
        color: #ffffff;
        display: flex;
        align-items: center;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #27272a;
        color: #a1a1aa;
        border: 1px solid #3f3f46;
        border-radius: 8px;
        padding: 10px 20px;
        margin-right: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #5CA4FF !important;
        color: #ffffff !important;
        border: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Metric / Search Header text */
    h3 {
        color: #ffffff;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 1.2rem;
    }

    /* Pandas DataFrame styling modification for Streamlit elements */
    [data-testid="stDataFrame"] > div {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #3f3f46;
        background-color: #18181b;
    }

    /* Custom Document Card Formatting */
    .doc-card {
        padding-bottom: 25px;
        margin-bottom: 25px;
        border-bottom: 1px solid #27272a;
    }
    .doc-num {
        color: #5CA4FF;
        font-weight: 800;
        font-size: 1.1rem;
        margin-right: 8px;
    }
    .doc-title {
        color: #d4d4d8;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    .pdf-btn {
        display: inline-block;
        margin-top: 12px;
        padding: 8px 16px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid #3f3f46;
        border-radius: 6px;
        color: #5CA4FF;
        font-weight: 600;
        font-size: 0.9rem;
        text-decoration: none;
    }
    .pdf-btn:hover {
        background-color: rgba(92, 164, 255, 0.1);
        color: #7DB6FF;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Data Loading and Logic ---

@st.cache_data
def load_and_process_data(file_name):
    # 1. Load Data
    df = pd.read_csv(file_name)
    isi_dokumen = df['Isi Dokumen'].tolist()
    
    # 2. Preprocessing Function
    def bersihkan_teks(kalimat):
        # Hapus label [filename.pdf] di awal kalimat agar tidak masuk ke vocabulary
        kalimat = re.sub(r'^\[.*?\]\s*', '', str(kalimat))
        kalimat = kalimat.lower()
        # Hanya simpan karakter alfabet (a-z) agar angka (seperti 0002000) tidak masuk kosakata
        kalimat = re.sub(r'[^a-z\s]', '', kalimat)
        return kalimat

    # Get clean documents
    dokumen_bersih = [bersihkan_teks(d) for d in isi_dokumen]
    
    # Get unique vocabulary in order of appearance (jangan diurutkan abjad)
    kosakata = list(dict.fromkeys([kata for doc in dokumen_bersih for kata in doc.split()]))
    
    # 3. TF Manual Calculation
    def kalkulasi_tf(dokumen):
        term = bersihkan_teks(dokumen).split()
        tf_val = {}
        for t in term:
            tf_val[t] = tf_val.get(t, 0) + 1
        return {t: val / len(term) for t, val in tf_val.items()}
    
    list_tf = [kalkulasi_tf(d) for d in isi_dokumen]
    tabel_tf = pd.DataFrame(list_tf, columns=kosakata).fillna(0)
    
    # 4. IDF Manual Calculation
    N = len(isi_dokumen)
    bobot_idf = {}
    for k in kosakata:
        df_count = sum(1 for d in dokumen_bersih if k in d.split())
        bobot_idf[k] = math.log(N / df_count) + 1
        
    tabel_idf = pd.DataFrame.from_dict(bobot_idf, orient='index', columns=['IDF'])
    tabel_idf.index.name = 'Word'
    tabel_idf = tabel_idf.reset_index()
    tabel_idf['IDF'] = tabel_idf['IDF'].round(3)
    
    # 5. TF-IDF Manual Calculation
    matriks_akhir = []
    for tf in list_tf:
        baris_bobot = [tf.get(k, 0) * bobot_idf[k] for k in kosakata]
        matriks_akhir.append(baris_bobot)
        
    tabel_tfidf = pd.DataFrame(matriks_akhir, columns=kosakata).fillna(0)
    
    # 6. Document-to-Document Cosine Similarity Matrix
    num_docs = len(matriks_akhir)
    cos_sim_matrix = np.zeros((num_docs, num_docs))
    for i in range(num_docs):
        vec_i = np.array(matriks_akhir[i])
        norm_i = np.linalg.norm(vec_i)
        for j in range(num_docs):
            vec_j = np.array(matriks_akhir[j])
            norm_j = np.linalg.norm(vec_j)
            if norm_i > 0 and norm_j > 0:
                cos_sim_matrix[i][j] = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                
    # Kembalikan ke nama asli PDF yang secara default sudah tersorting abjad dari OS
    def extract_label(text, idx):
        match = re.search(r'\[(.*?)\]', text)
        if match:
            return match.group(1)
        return f"Doc {idx+1}"
            
    doc_labels = [extract_label(d, i) for i, d in enumerate(isi_dokumen)]
    # Create Cosine Similarity Matrix. Index rows as 0, 1, 2... to keep visual layout clean (like SS 2)
    tabel_cos_sim = pd.DataFrame(cos_sim_matrix, columns=doc_labels, index=range(num_docs))
    
    return df, dokumen_bersih, kosakata, tabel_tf, bobot_idf, tabel_idf, tabel_tfidf, matriks_akhir, tabel_cos_sim

# --- User Interface Header ---
st.markdown("<div class='title-bar'>Dataset & Metrics</div>", unsafe_allow_html=True)

dataset_choice = st.radio(
    "Pilihan Dataset:",
    ["Dataset Simpel (Contoh)", "Dataset Lengkap (10 PDF Asli)"],
    horizontal=True
)

if dataset_choice == "Dataset Simpel (Contoh)":
    dataset_file = 'dataset_contoh.csv'
else:
    dataset_file = 'dataset_berita_teknologi.csv'

# Load data into app state
df, dokumen_bersih, kosakata, tabel_tf, bobot_idf, tabel_idf, tabel_tfidf, matriks_akhir, tabel_cos_sim = load_and_process_data(dataset_file)

# Logic for Cosine Similarity Search
def cari_relevansi(kueri, top_n=5):
    # Vectorize Query using matching terms
    def bersihkan_teks(kalimat):
        kalimat = str(kalimat).lower()
        return re.sub(r'[^a-z\s]', '', kalimat)

    q_term = bersihkan_teks(kueri).split()
    q_tf = {}
    for t in q_term:
        q_tf[t] = q_tf.get(t, 0) + 1
    if len(q_term) > 0:
        q_tf = {t: val / len(q_term) for t, val in q_tf.items()}
        
    q_vec = np.array([q_tf.get(k, 0) * bobot_idf.get(k, 0) for k in kosakata])
    
    sim_scores = []
    for d_vec in matriks_akhir:
        d_vec = np.array(d_vec)
        # Cosine Similarity Formula
        dot = np.dot(q_vec, d_vec)
        norm_q = np.linalg.norm(q_vec)
        norm_d = np.linalg.norm(d_vec)
        
        skor = dot / (norm_q * norm_d) if (norm_q > 0 and norm_d > 0) else 0
        sim_scores.append(round(skor, 4))
        
    hasil_final = df.copy()
    hasil_final['Skor_Kemiripan'] = sim_scores
    # Sort and take top_n, excluding 0 score if we want (or handle query not found)
    hasil_final = hasil_final[hasil_final['Skor_Kemiripan'] > 0]
    hasil_final = hasil_final.sort_values(by='Skor_Kemiripan', ascending=False).head(top_n)
    hasil_final.insert(0, 'Peringkat', [f"#{i+1}" for i in range(len(hasil_final))])
    
    return hasil_final[['Peringkat', 'ID', 'Isi Dokumen', 'Skor_Kemiripan']]

@st.cache_data
def get_pdf_href(pdf_name):
    # Akses masuk ke dalam folder artikel
    pdf_path = os.path.join("artikel", pdf_name)
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        return f'data:application/pdf;base64,{base64_pdf}'
    return "#"

# Build Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Documents", "TF", "IDF", "TF-IDF (Manual)", "Cosine Similarity"])

with tab1:
    # Layout Setup: Left col for data tables, Right col for search & results
    col_left, col_right = st.columns([2, 1], gap="large")
    
    with col_left:
        # Menampilkan dokumen dengan gaya PDF Card
        st.write("") # Spacer
        for idx, row in df.iterrows():
            text_full = row['Isi Dokumen']
            match = re.search(r'^\[(.*?)\]\s*(.*)', text_full, re.DOTALL)
            href = "#"
            dl_attr = ""
            if match:
                pdf_file_name = match.group(1)
                title = f"[{pdf_file_name}]"
                snippet = match.group(2)
                href = get_pdf_href(pdf_file_name)
                dl_attr = f"download='{pdf_file_name}'"
            else:
                title = f"[Doc {row['ID']}.pdf]"
                snippet = text_full
            
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
                
            st.markdown(f"""
            <div class='doc-card'>
                <span class='doc-num'>#{row['ID']}</span> 
                <span class='doc-title'><b>{title}</b><br>{snippet}</span><br>
                <a href='{href}' {dl_attr} class='pdf-btn'>📄 View PDF Document</a>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        # We use columns to inline the search button next to the input
        st.markdown("<h3>Information Retrieval</h3>", unsafe_allow_html=True)
        
        col_search, col_btn = st.columns([4, 1])
        with col_search:
            query = st.text_input("Search keyword", placeholder="Enter terms like 'sekolah' atau 'gizi'...", label_visibility="collapsed")
        with col_btn:
            search_clicked = st.button("Search", type="primary", use_container_width=True)
        
        st.write("")
        
        if search_clicked:
            if query.strip() != "":
                results = cari_relevansi(query)
                st.markdown(f"**Results for \"{query}\"**")
                
                if not results.empty:
                    # Render custom Search Results Table
                    st.markdown("""
                    <div style='background-color: #202126; padding: 10px; border-radius: 8px 8px 0 0; border: 1px solid #3f3f46; display: flex;'>
                        <div style='width: 15%; font-weight: bold; color: #e4e6ed;'>Rank</div>
                        <div style='width: 55%; font-weight: bold; color: #e4e6ed;'>Document</div>
                        <div style='width: 15%; font-weight: bold; color: #e4e6ed;'>Score</div>
                        <div style='width: 15%; font-weight: bold; color: #e4e6ed;'>Action</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Outer div for all rows
                    st.markdown("<div style='background-color: #18181b; border: 1px solid #3f3f46; border-top: none; border-radius: 0 0 8px 8px; padding: 10px;'>", unsafe_allow_html=True)
                    
                    for idx, row in results.iterrows():
                        full_text = df.loc[df['ID'] == row['ID'], 'Isi Dokumen'].values[0]
                        match = re.search(r'^\[(.*?)\]\s*(.*)', full_text, re.DOTALL)
                        if match:
                            title = f"[{match.group(1)}]"
                            content = match.group(2)
                        else:
                            title = f"[Doc {row['ID']}.pdf]"
                            content = full_text

                        # Highlight Query
                        q_lower = query.lower()
                        c_lower = content.lower()
                        find_idx = c_lower.find(q_lower)
                        if find_idx != -1:
                            start = max(0, find_idx - 40)
                            end = min(len(content), find_idx + len(query) + 80)
                            snip = content[start:end]
                            pattern = re.compile(re.escape(query), re.IGNORECASE)
                            highlighted = pattern.sub(lambda m: f"<span style='color: #5CA4FF; font-weight: bold;'>{m.group(0)}</span>", snip)
                            snip_display = ("..." if start > 0 else "") + highlighted + ("..." if end < len(content) else "")
                        else:
                            snip_display = content[:100] + "..."
                            
                        # Build Row HTML to bypass complex Streamlit column padding behaviors inside loops
                        
                        pdf_href = "#"
                        dl_attr = ""
                        if match:
                            pdf_file_name = match.group(1)
                            pdf_href = get_pdf_href(pdf_file_name)
                            dl_attr = f"download='{pdf_file_name}'"
                            
                        row_html = f"""
                        <div style='display: flex; align-items: flex-start; padding: 12px 0; border-bottom: 1px solid #27272a;'>
                            <div style='width: 15%; color: #e4e6ed;'>{row['Peringkat']}</div>
                            <div style='width: 55%;'>
                                <span style='font-weight: 600; color: #e4e6ed;'>{title}</span><br>
                                <span style='color: #a1a1aa; font-size: 0.85em; line-height: 1.4;'>{snip_display}</span>
                            </div>
                            <div style='width: 15%; color: #e4e6ed;'>{row['Skor_Kemiripan']:.4f}</div>
                            <div style='width: 15%;'>
                                <a href='{pdf_href}' {dl_attr} class='pdf-btn' style='font-size:0.75rem; padding: 6px 12px; margin-top:0;'>📄 View PDF</a>
                            </div>
                        </div>
                        """
                        st.markdown(row_html, unsafe_allow_html=True)
                        
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                else:
                    st.markdown("<span style='color: #888888;'>No documents matched your query.</span>", unsafe_allow_html=True)
                    
with tab2:
    st.markdown("### Term Frequency (TF)")
    # Show custom formatted df taking advantage of styling capabilities in streamlit
    # Highlighting non-zero values optionally
    styled_tf = tabel_tf.style.format("{:.3f}")
    st.dataframe(styled_tf, use_container_width=True)
    
with tab3:
    st.markdown("### Inverse Document Frequency (IDF)")
    # Format similar to the screenshot: column names Word, IDF
    st.dataframe(tabel_idf, use_container_width=True, hide_index=True)
    
with tab4:
    st.markdown("### TF-IDF Manual Weighting")
    styled_tfidf = tabel_tfidf.style.format("{:.3f}")
    st.dataframe(styled_tfidf, use_container_width=True)
    
with tab5:
    st.markdown("### Document-to-Document Cosine Similarity Matrix")
    styled_cos_sim = tabel_cos_sim.style.format("{:.3f}")
    st.dataframe(styled_cos_sim, use_container_width=True)

