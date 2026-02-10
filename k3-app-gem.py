import pandas as pd
import streamlit as st
import google.generativeai as genai
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi API
api_key = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=api_key)

# Page Config
st.set_page_config(
    page_title="K3 Dashboard & AI Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_available_model():
    """Mencari model yang tersedia agar tidak error 404"""
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Urutan prioritas model
        priority = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        for p in priority:
            if p in models:
                return p
        return models[0] if models else None
    except:
        return 'gemini-1.5-flash' # Default fallback

def map_wilayah_indonesia(distrik):
    """Pemetaan Unit ke Wilayah Besar"""
    if pd.isna(distrik): return 'Lainnya'
    d = distrik.upper()
    if any(x in d for x in ['GRESIK', 'PAITON', 'MUARA', 'CIRATA', 'PRIOK', 'INDRAMAYU', 'JAWA', 'PACITAN', 'REMBANG', 'ADIPALA']):
        return 'Jawa'
    elif any(x in d for x in ['BAKARU', 'PUNAGAYA', 'MINAHASA', 'SULAWESI', 'MAMUJU', 'KOLAKA', 'BARRU', 'KENDARI']):
        return 'Sulawesi'
    elif any(x in d for x in ['BARITO', 'ASAM', 'KALIMANTAN', 'BANJAR', 'KETAPANG', 'SAMPIT', 'SINTANG']):
        return 'Kalimantan'
    elif any(x in d for x in ['BELAWAN', 'SEBALANG', 'SUMATERA', 'TELUK SIRIH', 'PANGKALAN SUSU', 'RIAU']):
        return 'Sumatera'
    elif any(x in d for x in ['BOLOK', 'KUPANG', 'SUMBAWA', 'LOMBOK', 'NTB', 'NTT']):
        return 'Nusa Tenggara'
    elif any(x in d for x in ['PAPUA', 'MALUKU', 'AMBON', 'JAYAPURA']):
        return 'Maluku & Papua'
    return 'Lainnya'

def analyze_k3_data(df):
    df['Wilayah'] = df['temuan_nama_distrik'].apply(map_wilayah_indonesia)
    
    # Agregasi untuk AI
    summary_counts = df.groupby(['Wilayah', 'temuan_kategori']).size().reset_index(name='jumlah')
    top_causes = df.groupby('temuan_kategori')['judul'].value_counts().groupby(level=0).head(3).to_string()
    
    model_name = get_available_model()
    
    prompt = f"""
    Sebagai Data Analyst K3, analisis data berikut:
    Ringkasan Wilayah & Kategori:
    {summary_counts.to_string()}
    
    Penyebab Sering Muncul:
    {top_causes}
    
    Tolong berikan ringkasan eksekutif (3-4 paragraf):
    1. Wilayah dengan insiden kritis (Unsafe Action/Condition) tertinggi.
    2. Tren penyebab utama yang harus segera diperbaiki Manajemen.
    3. Rekomendasi langkah preventif untuk wilayah tersebut dengan prioritas.
    """
    
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text

def create_kategori_pie_chart(df):
    """Pie chart untuk distribusi kategori"""
    kategori_dist = df['temuan_kategori'].value_counts()
    fig = px.pie(
        values=kategori_dist.values,
        names=kategori_dist.index,
        title="Distribusi Jenis Temuan",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4  # Donut chart
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_wilayah_bar_chart(df):
    """Bar chart untuk distribusi wilayah"""
    wilayah_dist = df['Wilayah'].value_counts().sort_values(ascending=False)
    fig = px.bar(
        x=wilayah_dist.index,
        y=wilayah_dist.values,
        title="Jumlah Temuan per Wilayah",
        labels={'x': 'Wilayah', 'y': 'Jumlah Insiden'},
        color=wilayah_dist.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(showlegend=False, hovermode='x unified')
    return fig

def create_kategori_wilayah_heatmap(df):
    """Heatmap untuk hubungan kategori & wilayah"""
    pivot_table = df.groupby(['Wilayah', 'temuan_kategori']).size().unstack(fill_value=0)
    fig = px.imshow(
        pivot_table,
        labels=dict(x="Kategori Temuan", y="Wilayah", color="Jumlah"),
        title="Heatmap: Kategori vs Wilayah",
        color_continuous_scale='YlOrRd',
        text_auto=True
    )
    return fig

def create_distrik_histogram(df):
    """Histogram untuk distribusi distrik"""
    distrik_counts = df['temuan_nama_distrik'].value_counts().head(15)
    fig = px.bar(
        x=distrik_counts.values,
        y=distrik_counts.index,
        orientation='h',
        title="Top 15 Unit Pembangkit dengan Temuan Terbanyak",
        labels={'x': 'Jumlah Temuan', 'y': 'Unit Pembangkit'},
        color=distrik_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, hovermode='y unified')
    return fig

def create_kategori_wilayah_stacked_bar(df):
    """Stacked bar chart untuk kombinasi kategori & wilayah"""
    data = df.groupby(['Wilayah', 'temuan_kategori']).size().reset_index(name='count')
    fig = px.bar(
        data,
        x='Wilayah',
        y='count',
        color='temuan_kategori',
        title="Komposisi Kategori Temuan per Wilayah",
        labels={'count': 'Jumlah', 'temuan_kategori': 'Kategori'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(hovermode='x unified', barmode='stack')
    return fig

# Custom CSS
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(to right, #f8f9fa, #ffffff);
    }
    
    /* Title styling */
    h1 {
        color: #1f77b4;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: bold;
    }
    
    /* Info box styling */
    .stInfo {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚡ K3 Corporate Dashboard & AI Analyst")
st.markdown("**Safety & Health (K3) Analytics untuk Unit Pembangkit Listrik**", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Menu Navigasi")
    page = st.radio("Pilih Halaman:", ["📊 Dashboard", "🤖 AI Insights", "📈 Detailed Analytics"])

# UI Streamlit
uploaded_file = st.file_uploader("📂 Upload Data K3", type=["xlsx", "csv"], key="file_uploader")

if uploaded_file:
    # Load data
    try:
        df = pd.read_csv(uploaded_file, header=3) if 'csv' in uploaded_file.name else pd.read_excel(uploaded_file, header=3)
        df['Wilayah'] = df['temuan_nama_distrik'].apply(map_wilayah_indonesia)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Temuan", len(df), delta=None)
        with col2:
            st.metric("🗺️ Jumlah Wilayah", df['Wilayah'].nunique())
        with col3:
            st.metric("🏭 Jumlah Unit", df['temuan_nama_distrik'].nunique())
        with col4:
            st.metric("⚠️ Kategori", df['temuan_kategori'].nunique())
        
        st.markdown("---")
        
        # Pages
        if page == "📊 Dashboard":
            st.header("📊 Dasbor Analytics")
            
            # Row 1: Pie and Bar charts
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_kategori_pie_chart(df), use_container_width=True)
            with col2:
                st.plotly_chart(create_wilayah_bar_chart(df), use_container_width=True)
            
            # Row 2: Heatmap
            st.plotly_chart(create_kategori_wilayah_heatmap(df), use_container_width=True)
            
            # Row 3: Stacked bar
            st.plotly_chart(create_kategori_wilayah_stacked_bar(df), use_container_width=True)
            
            # Row 4: Histogram
            st.plotly_chart(create_distrik_histogram(df), use_container_width=True)
        
        elif page == "🤖 AI Insights":
            st.header("🤖 Analisis AI")
            st.markdown("Gunakan AI untuk mendapatkan insights strategis dari data K3 Anda", unsafe_allow_html=True)
            st.markdown("---")
            
            # Show available model
            model_name = get_available_model()
            st.info(f"✅ Menggunakan model: **{model_name}**")
            
            if st.button("🚀 Jalankan Analisis AI", key="analyze_btn"):
                with st.spinner('🔄 AI sedang memproses dan menganalisis data...'):
                    insight = analyze_k3_data(df)
                    st.markdown("### 📌 Insight Strategis")
                    st.markdown(insight)
        
        elif page == "📈 Detailed Analytics":
            st.header("📈 Analytics Terperinci")
            
            # Tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📋 Summary Table", "📉 Distribution"])
            
            with tab1:
                st.subheader("Preview Data")
                st.dataframe(
                    df[['temuan_nama_distrik', 'Wilayah', 'temuan_kategori', 'judul']].head(20),
                    use_container_width=True
                )
            
            with tab2:
                st.subheader("Ringkasan Data")
                summary = df.groupby(['Wilayah', 'temuan_kategori']).size().reset_index(name='Jumlah')
                st.dataframe(summary, use_container_width=True)
            
            with tab3:
                st.subheader("Distribusi Detail")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Distribusi per Kategori:**")
                    st.dataframe(df['temuan_kategori'].value_counts(), use_container_width=True)
                with col2:
                    st.write("**Distribusi per Wilayah:**")
                    st.dataframe(df['Wilayah'].value_counts(), use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error saat membaca file: {str(e)}")
        st.info("Pastikan file Excel memiliki header pada baris ke-4 dan kolom yang benar")
else:
    st.info("👆 Silakan upload file K3 (Excel/CSV) untuk memulai analisis")