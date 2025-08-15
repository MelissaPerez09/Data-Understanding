import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Dashboard Ejecutivo - Análisis de Ventas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de colores corporativa
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#17becf',
    'light': '#bcbd22',
    'dark': '#7f7f7f',
    'purple': '#9467bd'
}

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .kpi-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Cargar datos limpios desde la carpeta clean"""
    try:
        data_dir = Path("./clean")
        
        categoria = pd.read_csv(data_dir / "categoria_clean.csv")
        cliente = pd.read_csv(data_dir / "cliente_clean.csv")
        events = pd.read_csv(data_dir / "events_clean.csv")
        marca = pd.read_csv(data_dir / "marca_clean.csv")
        producto = pd.read_csv(data_dir / "producto_clean.csv")
        
        # Convertir columnas de fecha que existen
        if 'nacimiento' in cliente.columns:
            cliente['nacimiento'] = pd.to_datetime(cliente['nacimiento'], errors='coerce')
        if 'event_time' in events.columns:
            events['event_time'] = pd.to_datetime(events['event_time'], errors='coerce')
        if 'date' in events.columns:
            events['date'] = pd.to_datetime(events['date'], errors='coerce')
        
        return categoria, cliente, events, marca, producto
        
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, None, None, None, None

def create_sales_data(events, producto, categoria, marca, cliente):
    """Crear dataset de ventas combinando eventos con información de productos"""
    
    # Verificar qué columnas opcionales existen
    has_transactionid = 'transactionid' in events.columns
    has_itemid = 'itemid' in events.columns
    has_precio = 'precio' in producto.columns
    has_marca_id = 'marca_id' in producto.columns
    
    # Filtrar eventos con transactionid si existe, sino usar todos los eventos
    if has_transactionid:
        sales_events = events[events['transactionid'].notna()].copy()
        if sales_events.empty:
            # Si no hay transacciones, usar todos los eventos
            sales_events = events.copy()
    else:
        sales_events = events.copy()
    
    # Solo hacer merge con productos si itemid existe
    if has_itemid:
        # Preparar columnas de producto para merge
        producto_cols = ['id', 'categoria_id', 'nombre']
        if has_marca_id:
            producto_cols.append('marca_id')
        if has_precio:
            producto_cols.append('precio')
            
        sales_data = sales_events.merge(producto[producto_cols], 
                                       left_on='itemid', right_on='id', how='left')
        sales_data.rename(columns={'nombre': 'producto_nombre'}, inplace=True)
    else:
        sales_data = sales_events.copy()
        # Si no hay itemid, crear datos sintéticos para demostración
        sales_data['categoria_id'] = np.random.choice(categoria['id'].tolist(), len(sales_data))
        if has_marca_id and not marca.empty:
            sales_data['marca_id'] = np.random.choice(marca['id'].tolist(), len(sales_data))
    
    # Merge con categorías
    sales_data = sales_data.merge(categoria[['id', 'categoria']], 
                                 left_on='categoria_id', right_on='id', 
                                 how='left', suffixes=('', '_cat'))
    sales_data.rename(columns={'categoria': 'categoria_nombre'}, inplace=True)
    
    # Merge con marcas si existe marca_id
    if has_marca_id and not marca.empty:
        sales_data = sales_data.merge(marca[['id', 'marca']], 
                                     left_on='marca_id', right_on='id', 
                                     how='left', suffixes=('', '_marca'))
        sales_data.rename(columns={'marca': 'marca_nombre'}, inplace=True)
    else:
        sales_data['marca_nombre'] = 'Sin Marca'
    
    # Calcular revenue
    if has_precio and 'precio' in sales_data.columns:
        sales_data['revenue'] = sales_data['precio'].fillna(100)  # Precio por defecto
    else:
        # Simular revenue basado en eventos
        sales_data['revenue'] = np.random.uniform(50, 500, len(sales_data))
    
    # Agregar información temporal si existe event_time
    if 'event_time' in sales_data.columns and sales_data['event_time'].notna().any():
        sales_data['year'] = sales_data['event_time'].dt.year
        sales_data['month'] = sales_data['event_time'].dt.month
        sales_data['day_name'] = sales_data['event_time'].dt.day_name()
        sales_data['hour'] = sales_data['event_time'].dt.hour
    
    return sales_data

def calculate_kpis(sales_data, events, cliente):
    """Calcular KPIs principales"""
    
    # KPI 1: Customer Lifetime Value (CLV) aproximado
    if not sales_data.empty and 'visitorid' in sales_data.columns:
        customer_revenue = sales_data.groupby('visitorid')['revenue'].sum()
        avg_clv = customer_revenue.mean()
        
        # KPI 2: Tasa de conversión (transacciones vs total de eventos)
        total_events = len(events)
        # Contar transacciones reales si existe transactionid, sino usar todos los sales_data
        if 'transactionid' in sales_data.columns:
            transactions = sales_data['transactionid'].notna().sum()
        else:
            transactions = len(sales_data)
        conversion_rate = (transactions / total_events * 100) if total_events > 0 else 0
        
        # KPI 3: Ticket promedio
        avg_order_value = sales_data['revenue'].mean()
        
        # KPI 4: Productos únicos por transacción
        if 'transactionid' in sales_data.columns and 'itemid' in sales_data.columns:
            avg_items_per_transaction = sales_data.groupby('transactionid')['itemid'].nunique().mean()
        else:
            avg_items_per_transaction = 1.0  # Asumir 1 item por evento
        
    else:
        avg_clv = 0
        conversion_rate = 0
        avg_order_value = 0
        avg_items_per_transaction = 0
    
    return {
        'avg_clv': avg_clv,
        'conversion_rate': conversion_rate,
        'avg_order_value': avg_order_value,
        'avg_items_per_transaction': avg_items_per_transaction
    }

def create_charts(sales_data, events):
    """Crear gráficos principales"""
    charts = {}
    
    if not sales_data.empty:
        # Ventas por categoría
        cat_sales = sales_data.groupby('categoria_nombre')['revenue'].sum().reset_index()
        fig_cat = px.pie(cat_sales, values='revenue', names='categoria_nombre',
                        title='Distribución de Ventas por Categoría',
                        color_discrete_sequence=px.colors.qualitative.Set3)
        fig_cat.update_layout(height=400)
        charts['category_sales'] = fig_cat
        
        # Ventas por marca
        marca_sales = sales_data.groupby('marca_nombre')['revenue'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_marca = px.bar(marca_sales, x='marca_nombre', y='revenue',
                          title='Top 10 Marcas por Ventas',
                          color='revenue',
                          color_continuous_scale='viridis')
        fig_marca.update_layout(height=400, xaxis_tickangle=-45)
        charts['brand_sales'] = fig_marca
        
        # Ventas por tiempo (si hay datos temporales)
        if 'event_time' in sales_data.columns:
            daily_sales = sales_data.groupby(sales_data['event_time'].dt.date)['revenue'].sum().reset_index()
            fig_time = px.line(daily_sales, x='event_time', y='revenue',
                              title='Evolución de Ventas Diarias',
                              markers=True)
            fig_time.update_traces(line_color=COLORS['primary'])
            fig_time.update_layout(height=400)
            charts['time_sales'] = fig_time
    
    # Eventos por tipo
    if not events.empty and 'event' in events.columns:
        event_counts = events['event'].value_counts().reset_index()
        fig_events = px.bar(event_counts, x='event', y='count',
                           title='Distribución de Eventos por Tipo',
                           color='count',
                           color_continuous_scale='blues')
        fig_events.update_layout(height=400)
        charts['event_distribution'] = fig_events
    
    return charts

def main():
    # Header principal
    st.markdown('<h1 class="main-header">📊 Dashboard Ejecutivo - Análisis de Ventas</h1>', 
                unsafe_allow_html=True)
    
    # Cargar datos
    with st.spinner('Cargando datos...'):
        categoria, cliente, events, marca, producto = load_data()
    
    if categoria is None:
        st.error("No se pudieron cargar los datos. Asegúrate de que la carpeta './clean' existe y contiene los archivos CSV.")
        return
    
    # Crear datos de ventas
    sales_data = create_sales_data(events, producto, categoria, marca, cliente)
    
    # Calcular KPIs
    kpis = calculate_kpis(sales_data, events, cliente)
    
    # Sidebar con filtros
    st.sidebar.header("🎛️ Filtros")
    
    # Filtro de fecha
    if not sales_data.empty and 'event_time' in sales_data.columns:
        min_date = sales_data['event_time'].min().date()
        max_date = sales_data['event_time'].max().date()
        
        date_range = st.sidebar.date_input(
            "Rango de fechas:",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (sales_data['event_time'].dt.date >= start_date) & (sales_data['event_time'].dt.date <= end_date)
            sales_data = sales_data[mask]
    
    # Filtro de categoría
    if not sales_data.empty and 'categoria_nombre' in sales_data.columns:
        categories = ['Todas'] + list(sales_data['categoria_nombre'].dropna().unique())
        selected_category = st.sidebar.selectbox("Categoría:", categories)
        
        if selected_category != 'Todas':
            sales_data = sales_data[sales_data['categoria_nombre'] == selected_category]
    
    # Métricas principales
    st.header("📈 Métricas Principales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Revenue Total",
            value=f"${sales_data['revenue'].sum():,.2f}" if not sales_data.empty else "$0.00",
            delta=f"{len(sales_data)} transacciones"
        )
    
    with col2:
        visitor_count = sales_data['visitorid'].nunique() if 'visitorid' in sales_data.columns and not sales_data.empty else 0
        st.metric(
            label="👥 Visitantes Únicos",
            value=f"{visitor_count:,}",
            delta=f"{len(cliente):,} clientes registrados"
        )
    
    with col3:
        item_count = sales_data['itemid'].nunique() if 'itemid' in sales_data.columns and not sales_data.empty else 0
        st.metric(
            label="📦 Productos con Actividad",
            value=f"{item_count:,}",
            delta=f"{len(producto):,} en catálogo"
        )
    
    with col4:
        st.metric(
            label="🎯 Eventos Totales",
            value=f"{len(events):,}",
            delta=f"{events['event'].nunique()} tipos diferentes" if 'event' in events.columns else ""
        )
    
    # KPIs Principales
    st.header("🎯 KPIs Clave")
    
    kpi_col1, kpi_col2 = st.columns(2)
    
    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-container">
                <h3>💎 Customer Lifetime Value (CLV)</h3>
                <h2>${kpis['avg_clv']:,.2f}</h2>
                <p>Valor promedio por cliente a lo largo del tiempo</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with kpi_col2:
        st.markdown(
            f"""
            <div class="kpi-container">
                <h3>📊 Tasa de Conversión</h3>
                <h2>{kpis['conversion_rate']:.2f}%</h2>
                <p>Porcentaje de eventos que resultan en transacciones</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Gráficos principales
    charts = create_charts(sales_data, events)
    
    if charts:
        # Primera fila de gráficos
        chart_col1, chart_col2 = st.columns(2)
        
        if 'category_sales' in charts:
            with chart_col1:
                st.plotly_chart(charts['category_sales'], use_container_width=True)
        
        if 'brand_sales' in charts:
            with chart_col2:
                st.plotly_chart(charts['brand_sales'], use_container_width=True)
        
        # Segunda fila de gráficos
        if 'time_sales' in charts:
            st.plotly_chart(charts['time_sales'], use_container_width=True)
        
        if 'event_distribution' in charts:
            st.plotly_chart(charts['event_distribution'], use_container_width=True)
    
    # Tabla de resumen por producto
    st.header("📋 Resumen de Actividad")
    
    if not sales_data.empty:
        # Determinar qué columnas usar para el resumen
        group_cols = []
        if 'itemid' in sales_data.columns:
            group_cols.append('itemid')
        if 'categoria_nombre' in sales_data.columns:
            group_cols.append('categoria_nombre')
        if 'marca_nombre' in sales_data.columns:
            group_cols.append('marca_nombre')
        
        if group_cols:
            agg_dict = {'revenue': ['sum', 'count', 'mean']}
            if 'visitorid' in sales_data.columns:
                agg_dict['visitorid'] = 'nunique'
                
            summary = (sales_data.groupby(group_cols).agg(agg_dict).round(2))
            
            # Renombrar columnas
            new_cols = ['Revenue Total', 'Eventos', 'Revenue Promedio']
            if 'visitorid' in agg_dict:
                new_cols.append('Visitantes Únicos')
            summary.columns = new_cols
            
            summary = summary.sort_values('Revenue Total', ascending=False).head(20)
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No hay suficientes datos para mostrar el resumen detallado.")
    else:
        st.info("No hay datos de actividad para mostrar.")
    
    # Información adicional
    with st.expander("ℹ️ Información del Dashboard"):
        st.markdown("""
        ### Metodología de Cálculo:
        
        **Customer Lifetime Value (CLV):**
        - Suma del revenue total por visitante único
        - Representa el valor promedio que aporta cada visitante
        
        **Tasa de Conversión:**
        - Porcentaje de eventos que generan actividad comercial
        - Calculado basándose en eventos disponibles
        
        **Fuentes de Datos:**
        - **Eventos**: timestamp, visitorid, event, itemid*, transactionid*, event_time*, date*, hour
        - **Productos**: id, categoria_id, nombre, marca_id*, volumen*, precio*
        - **Categorías**: id, categoria
        - **Marcas**: id, marca  
        - **Clientes**: información completa de perfil
        
        *Columnas opcionales que pueden no estar presentes en todos los datasets*
        """)

if __name__ == "__main__":
    main()