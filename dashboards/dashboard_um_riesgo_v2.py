import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# set_page_config debe ser la primera llamada de Streamlit
st.set_page_config(
    page_title="WealthInsights Scoring",
    page_icon="💰",
    layout="wide"
)

# CSS personalizado para diseño profesional
st.markdown("""
<style>
:root {
    --primary-color: #2c3e50;
    --secondary-color: #3498db;
    --success-color: #27ae60;
    --warning-color: #f39c12;
    --danger-color: #e74c3c;
    --light-gray: #f8f9fa;
    --dark-gray: #343a40;
    --border-radius: 12px;
    --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
    margin: 0 auto;
}

.card {
    background: white;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #eaeaea;
    color: #2c3e50;
}

.score-card {
    text-align: center;
    padding: 2rem;
    border-radius: var(--border-radius);
    box-shadow: var(--box-shadow);
    background: white;
    color: #2c3e50;
}

.score-value {
    font-size: 4.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    color: #2c3e50 !important;
}

.risk-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    margin: 0.5rem 0;
    color: white;
}

.factor-positive {
    background: rgba(39, 174, 96, 0.1) !important;
    border-left: 4px solid #27ae60 !important;
    padding: 1rem !important;
    border-radius: 0 8px 8px 0 !important;
    margin: 0.5rem 0 !important;
    color: #27ae60 !important;
}

.factor-negative {
    background: rgba(231, 76, 60, 0.1) !important;
    border-left: 4px solid #e74c3c !important;
    padding: 1rem !important;
    border-radius: 0 8px 8px 0 !important;
    margin: 0.5rem 0 !important;
    color: #e74c3c !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 24px; }
.stTabs [data-baseweb="tab"] {
    height: 50px;
    border-radius: 8px 8px 0 0;
    background-color: #f8f9fa;
    color: #343a40;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #2c3e50 !important;
    color: white !important;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #3498db, transparent);
    margin: 2rem 0;
}

.footer {
    text-align: center;
    padding: 1.5rem;
    margin-top: 2rem;
    font-size: 0.9rem;
    border-top: 1px solid #eaeaea;
    color: #343a40;
}
</style>
""", unsafe_allow_html=True)


# ── Cargar modelo ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        base = Path(__file__).parent
        modelo = joblib.load(base / 'modelo_scoring_v2_1M.joblib')
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

modelo = load_model()

if modelo is None:
    st.error("No se pudo cargar el modelo. Verifica que los archivos .joblib estén en la carpeta dashboards/.")
    st.stop()

FEATURES = [
    'meses_ahorro_positivo',
    'ahorro_medio_mensual',
    'n_domiciliaciones',
    'edad',
    'tipo_interes',
    'tiene_recibos_rechazados',
    'ratio_utilizacion_credito'
]

FEATURE_NAMES_DISPLAY = [
    'Meses ahorro positivo',
    'Ahorro medio mensual',
    'Nº domiciliaciones',
    'Edad',
    'Tipo de interés',
    'Recibos rechazados',
    'Ratio utilización crédito'
]

# Para normalizar y asignar dirección a cada variable
# Dirección: -1 = protectora (valor alto = menos riesgo), +1 = riesgo (valor alto = más riesgo)
FEATURE_DIRECTIONS = np.array([-1, -1, -1, -1, 1, 1, 1], dtype=float)
FEATURE_MIN = np.array([0, -500, 0, 18, 0, 0, 0], dtype=float)
FEATURE_MAX = np.array([12, 5000, 15, 70, 20, 1, 1], dtype=float)


# ── Header ───────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("### 💰")
with col_title:
    st.title("WealthInsights Analytics")
    st.subheader("Scoring Crediticio con Machine Learning Avanzado")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Parámetros del Cliente")
    st.markdown('<hr style="border: 1px solid rgba(255,255,255,0.3);">', unsafe_allow_html=True)

    meses_ahorro = st.slider(
        "Meses con ahorro positivo", 0, 12, 8,
        help="Meses de los últimos 12 con ahorro neto positivo"
    )
    ahorro_mensual = st.number_input(
        "Ahorro medio mensual (€)", -500, 5000, 500,
        help="Promedio mensual de ahorro (ingresos - gastos)"
    )
    n_domiciliaciones = st.slider(
        "Nº de domiciliaciones activas", 0, 15, 5,
        help="Número de recibos domiciliados (luz, agua, gimnasio…)"
    )
    edad = st.slider("Edad", 18, 70, 35, help="Edad del cliente en años")
    tipo_interes = st.slider(
        "Tipo de interés del préstamo (%)", 0.0, 20.0, 5.0, step=0.1,
        help="Tipo de interés anual del préstamo principal"
    )
    tiene_recibos_rechazados = st.selectbox(
        "¿Tiene recibos rechazados?", [0, 1],
        format_func=lambda x: "Sí" if x else "No",
        help="¿Algún recibo devuelto en los últimos 12 meses?"
    )
    ratio_credito = st.slider(
        "Ratio de utilización de crédito", 0.0, 1.0, 0.3, step=0.01,
        help="Porcentaje del crédito disponible utilizado (0=nada, 1=todo)"
    )

    st.markdown('<hr style="border: 1px solid rgba(255,255,255,0.3);">', unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:rgba(255,255,255,0.8);'>"
        "<p style='font-size:0.9rem;'>💡 <strong>Consejo:</strong> Ajusta los parámetros "
        "para ver cómo afectan al score</p>"
        "<p style='font-size:0.8rem; opacity:0.7;'>WealthInsights Analytics v1.0</p>"
        "</div>",
        unsafe_allow_html=True
    )


# ── Cálculo del score ────────────────────────────────────────────────────────
cliente = pd.DataFrame([[
    meses_ahorro, ahorro_mensual, n_domiciliaciones, edad,
    tipo_interes, tiene_recibos_rechazados, ratio_credito
]], columns=FEATURES)

probabilidad = modelo.predict_proba(cliente.values)[0][1]
score = int((1 - probabilidad) * 1000)

# Impacto normalizado y dirigido para el gráfico de factores
valores = cliente.values[0]
normalizado = np.clip((valores - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN), 0, 1)
impactos = modelo.feature_importances_ * normalizado * FEATURE_DIRECTIONS
# Negativo → protector (verde), Positivo → riesgo (rojo)

shap_df = pd.DataFrame({
    'Feature': FEATURE_NAMES_DISPLAY,
    'Impact': impactos
}).sort_values('Impact')


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Score General",
    "🔍 Factores de Riesgo",
    "📝 Análisis Detallado",
    "💡 Recomendaciones"
])


# ── Tab 1: Score General ─────────────────────────────────────────────────────
with tab1:
    st.markdown("### 📊 Resultado del Scoring")
    col_score, col_info = st.columns([2, 1])

    with col_score:
        if probabilidad < 0.20:
            riesgo, badge_color = "EXCELENTE", "#27ae60"
        elif probabilidad < 0.35:
            riesgo, badge_color = "BUENO", "#3498db"
        elif probabilidad < 0.50:
            riesgo, badge_color = "MODERADO", "#f39c12"
        else:
            riesgo, badge_color = "ALTO RIESGO", "#e74c3c"

        st.markdown(f"""
        <div class="score-card">
            <h1 class="score-value">{score}</h1>
            <div style="margin: 1rem 0;">
                <span class="risk-badge" style="background: {badge_color};">{riesgo}</span>
            </div>
            <p style="font-size: 1.2rem; color: #343a40;">
                <strong>Probabilidad de default:</strong> {probabilidad:.1%}
            </p>
            <div style="margin-top: 1.5rem; text-align: left; background: rgba(0,0,0,0.03);
                        padding: 1rem; border-radius: 8px;">
                <p style="margin: 0.5rem 0; font-size: 0.95rem; color: #2c3e50;">
                    <span style="display:inline-block; width:13px; height:13px; background:#27ae60;
                                 border-radius:50%; margin-right:8px;"></span>
                    <strong>0–20 %:</strong> Riesgo excelente
                </p>
                <p style="margin: 0.5rem 0; font-size: 0.95rem; color: #2c3e50;">
                    <span style="display:inline-block; width:13px; height:13px; background:#3498db;
                                 border-radius:50%; margin-right:8px;"></span>
                    <strong>20–35 %:</strong> Riesgo bueno
                </p>
                <p style="margin: 0.5rem 0; font-size: 0.95rem; color: #2c3e50;">
                    <span style="display:inline-block; width:13px; height:13px; background:#f39c12;
                                 border-radius:50%; margin-right:8px;"></span>
                    <strong>35–50 %:</strong> Riesgo moderado
                </p>
                <p style="margin: 0.5rem 0; font-size: 0.95rem; color: #2c3e50;">
                    <span style="display:inline-block; width:13px; height:13px; background:#e74c3c;
                                 border-radius:50%; margin-right:8px;"></span>
                    <strong>&gt;50 %:</strong> Alto riesgo
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown("### ℹ️ Información del Score")
        st.markdown(f"""
        <div class="card">
            <h4 style="color: #2c3e50;">¿Cómo se calcula?</h4>
            <p style="color: #343a40;">El modelo analiza {len(FEATURES)} factores financieros y
            comportamentales para predecir la probabilidad de impago en los próximos 12 meses.</p>
            <h4 style="color: #2c3e50; margin-top: 1rem;">Escala de Score</h4>
            <ul style="color: #343a40;">
                <li><strong>800–1000:</strong> Excelente crédito</li>
                <li><strong>650–799:</strong> Buen crédito</li>
                <li><strong>500–649:</strong> Crédito moderado</li>
                <li><strong>0–499:</strong> Alto riesgo</li>
            </ul>
            <div style="background: linear-gradient(90deg, #27ae60, #3498db);
                        height: 4px; border-radius: 2px; margin: 1rem 0;"></div>
            <p style="font-size: 0.9rem; color: #343a40; margin-top: 0.5rem;">
                <strong>Nota:</strong> Este score es una estimación basada en datos históricos
                y debe usarse como guía complementaria.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ── Tab 2: Factores de Riesgo ────────────────────────────────────────────────
with tab2:
    st.markdown("### 🔍 Análisis de Factores de Riesgo")
    col_chart, col_explanation = st.columns([2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = []
        for val in shap_df['Impact']:
            if val <= -0.05:
                colors.append('#27ae60')   # verde fuerte: muy protector
            elif val < 0:
                colors.append('#2ecc71')   # verde suave: levemente protector
            elif val < 0.05:
                colors.append('#f39c12')   # naranja: impacto neutral/leve
            else:
                colors.append('#e74c3c')   # rojo: factor de riesgo

        bars = ax.barh(shap_df['Feature'], shap_df['Impact'], color=colors, height=0.6)

        for bar in bars:
            width = bar.get_width()
            offset = 0.005 if width >= 0 else -0.005
            ax.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}',
                ha='left' if width >= 0 else 'right',
                va='center',
                fontweight='bold',
                fontsize=9
            )

        ax.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.7)
        ax.set_xlabel('Impacto ponderado (negativo = protector, positivo = riesgo)',
                      fontsize=11, fontweight='bold')
        ax.set_title('Factores que influyen en el score', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with col_explanation:
        st.markdown("### 📋 Interpretación de Factores")

        top_feature = shap_df.iloc[-1] if abs(shap_df.iloc[-1]['Impact']) >= abs(shap_df.iloc[0]['Impact']) else shap_df.iloc[0]
        top_name = top_feature['Feature']
        top_val = abs(top_feature['Impact'])

        st.markdown(f"""
        <div class="card">
            <p style="color: #2c3e50;"><strong>🔍 Cómo leer el gráfico:</strong></p>
            <ul style="color: #343a40;">
                <li><span style="color: #27ae60; font-weight: bold;">Barras verdes:</span>
                    Factores que <strong>reducen</strong> el riesgo</li>
                <li><span style="color: #e74c3c; font-weight: bold;">Barras rojas:</span>
                    Factores que <strong>aumentan</strong> el riesgo</li>
                <li><span style="color: #f39c12; font-weight: bold;">Barras naranjas:</span>
                    Impacto neutral o moderado</li>
            </ul>
            <p style="color: #2c3e50;"><strong>🎯 Factores clave:</strong></p>
            <ul style="color: #343a40;">
                <li><strong>Meses con ahorro positivo:</strong> Muestra consistencia financiera</li>
                <li><strong>Ahorro medio mensual:</strong> Capacidad de ahorro sostenida</li>
                <li><strong>Ratio de crédito:</strong> Nivel de endeudamiento actual</li>
            </ul>
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(52, 152, 219, 0.1);
                        border-radius: 8px;">
                <p style="margin: 0; font-size: 0.95rem; color: #2c3e50;">
                    <strong>💡 Insight:</strong> El factor más influyente es
                    <strong>{top_name}</strong> con un impacto de <strong>{top_val:.3f}</strong>.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Tab 3: Análisis Detallado ────────────────────────────────────────────────
with tab3:
    st.markdown("### 📝 Análisis Detallado del Perfil")

    factores_positivos = []
    factores_negativos = []

    if meses_ahorro >= 10:
        factores_positivos.append(f"✅ <strong>Excelente constancia de ahorro:</strong> {meses_ahorro} meses positivos")
    elif meses_ahorro < 6:
        factores_negativos.append(f"⚠️ <strong>Baja constancia de ahorro:</strong> Solo {meses_ahorro} meses")

    if ahorro_mensual > 800:
        factores_positivos.append(f"✅ <strong>Alto ahorro mensual:</strong> {ahorro_mensual}€ promedio")
    elif ahorro_mensual < 300:
        factores_negativos.append(f"⚠️ <strong>Bajo ahorro mensual:</strong> Solo {ahorro_mensual}€ promedio")

    if tiene_recibos_rechazados:
        factores_negativos.append("⚠️ <strong>Recibos rechazados:</strong> Indica dificultades de pago recientes")
    else:
        factores_positivos.append("✅ <strong>Sin recibos rechazados:</strong> Historial de pagos limpio")

    if ratio_credito > 0.8:
        factores_negativos.append(f"⚠️ <strong>Alto uso de crédito:</strong> {ratio_credito:.0%} del límite utilizado")
    elif ratio_credito < 0.3:
        factores_positivos.append(f"✅ <strong>Bajo uso de crédito:</strong> Solo {ratio_credito:.0%} del límite utilizado")

    if n_domiciliaciones >= 5:
        factores_positivos.append(f"✅ <strong>Domiciliaciones activas:</strong> {n_domiciliaciones} pagos regulares organizados")

    if tipo_interes > 10:
        factores_negativos.append(f"⚠️ <strong>Tipo de interés elevado:</strong> {tipo_interes}% anual")

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("#### ✅ Factores Favorables")
        if factores_positivos:
            for f in factores_positivos:
                st.markdown(f'<div class="factor-positive">{f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="factor-positive">No se identificaron factores favorables destacados</div>', unsafe_allow_html=True)

        if factores_positivos:
            st.markdown(f"""
            <div class="card" style="margin-top: 1rem; background: rgba(39, 174, 96, 0.05);">
                <h4 style="color: #27ae60;">🎯 Puntos Fuertes</h4>
                <p style="color: #343a40;">El perfil muestra <strong>{len(factores_positivos)}</strong> factores
                positivos. Se recomienda mantener estos hábitos para continuar mejorando la solvencia.</p>
            </div>
            """, unsafe_allow_html=True)

    with col_neg:
        st.markdown("#### ⚠️ Factores de Riesgo")
        if factores_negativos:
            for f in factores_negativos:
                st.markdown(f'<div class="factor-negative">{f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="factor-negative">No se identificaron factores de riesgo significativos</div>', unsafe_allow_html=True)

        if factores_negativos:
            st.markdown(f"""
            <div class="card" style="margin-top: 1rem; background: rgba(231, 76, 60, 0.05);">
                <h4 style="color: #e74c3c;">⚠️ Áreas de Mejora</h4>
                <p style="color: #343a40;">Se detectaron <strong>{len(factores_negativos)}</strong> factores
                de riesgo. Trabajar en estas áreas puede mejorar el perfil crediticio.</p>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 4: Recomendaciones ───────────────────────────────────────────────────
with tab4:
    st.markdown("### 💡 Recomendaciones Personalizadas")
    col_rec1, col_rec2 = st.columns(2)

    with col_rec1:
        st.markdown("#### 🚀 Acciones Inmediatas")

        recomendaciones = []
        if probabilidad > 0.35:
            recomendaciones.append({
                "titulo": "📈 Reducir carga financiera",
                "texto": "Considera renegociar préstamos para reducir la cuota mensual y mejorar el ratio de endeudamiento.",
                "prioridad": "alta"
            })
        if meses_ahorro < 8:
            recomendaciones.append({
                "titulo": "💪 Mejorar constancia de ahorro",
                "texto": "Establece ahorro automático para garantizar meses consecutivos con saldo positivo.",
                "prioridad": "media"
            })
        if tiene_recibos_rechazados:
            recomendaciones.append({
                "titulo": "🛡️ Eliminar recibos rechazados",
                "texto": "Revisa tus domiciliaciones y asegura saldo suficiente para evitar devoluciones.",
                "prioridad": "alta"
            })
        if ratio_credito > 0.5:
            recomendaciones.append({
                "titulo": "💳 Reducir uso de crédito",
                "texto": f"Tu ratio de utilización es {ratio_credito:.0%}. Intenta mantenerlo por debajo del 30%.",
                "prioridad": "media"
            })
        if ahorro_mensual < 300:
            recomendaciones.append({
                "titulo": "💰 Incrementar ahorro mensual",
                "texto": "Reduce gastos no esenciales o busca vías de ingreso extra para llegar a 300€/mes de ahorro.",
                "prioridad": "alta"
            })
        if not recomendaciones:
            recomendaciones.append({
                "titulo": "✅ 🎉 Excelente perfil",
                "texto": "Tu perfil crediticio es muy sólido. Mantén estos buenos hábitos financieros y considera diversificar tus inversiones.",
                "prioridad": "baja"
            })

        for rec in recomendaciones:
            color = "#27ae60" if rec["prioridad"] == "baja" else "#f39c12" if rec["prioridad"] == "media" else "#e74c3c"
            st.markdown(f"""
            <div class="card" style="border-left: 4px solid {color}; margin-bottom: 1rem;">
                <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{rec["titulo"]}</h4>
                <p style="margin: 0; color: #343a40;">{rec["texto"]}</p>
                <div style="margin-top: 0.5rem; text-align: right;">
                    <span style="background: {color}; color: white; padding: 0.2rem 0.8rem;
                                 border-radius: 12px; font-size: 0.8rem;">
                        Prioridad: {rec["prioridad"].upper()}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_rec2:
        st.markdown("#### 📊 Proyección de Mejora")

        cliente_mejorado = cliente.copy()
        if meses_ahorro < 10:
            cliente_mejorado['meses_ahorro_positivo'] = 10
        if ahorro_mensual < 500:
            cliente_mejorado['ahorro_medio_mensual'] = 500
        if tiene_recibos_rechazados:
            cliente_mejorado['tiene_recibos_rechazados'] = 0
        if ratio_credito > 0.3:
            cliente_mejorado['ratio_utilizacion_credito'] = 0.3

        prob_mejorada = modelo.predict_proba(cliente_mejorado.values)[0][1]
        score_mejorado = int((1 - prob_mejorada) * 1000)
        mejora = score_mejorado - score

        st.markdown(f"""
        <div class="card" style="text-align: center;
             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h3 style="color: white; margin-bottom: 1rem;">🎯 Score Potencial</h3>
            <div style="display: flex; justify-content: center; align-items: center;
                        gap: 2rem; margin: 1.5rem 0;">
                <div>
                    <p style="opacity: 0.8; margin: 0;">Score Actual</p>
                    <h2 style="color: white; margin: 0.5rem 0; font-size: 2.5rem;">{score}</h2>
                </div>
                <span style="font-size: 2rem;">→</span>
                <div>
                    <p style="opacity: 0.8; margin: 0;">Score Potencial</p>
                    <h2 style="color: white; margin: 0.5rem 0; font-size: 2.5rem;">{score_mejorado}</h2>
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px;">
                <p style="margin: 0; font-size: 1.1rem;">
                    <strong>+{mejora} puntos</strong> potenciales
                </p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">
                    Implementando las recomendaciones anteriores
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📈 Seguimiento")
        st.markdown(f"""
        <div class="card">
            <p style="color: #2c3e50;"><strong>📅 Próximos pasos:</strong></p>
            <ul style="color: #343a40;">
                <li><strong>Semana 1–2:</strong> Analizar gastos y establecer plan de ahorro</li>
                <li><strong>Semana 3–4:</strong> Revisar domiciliaciones y límites de crédito</li>
                <li><strong>Mes 2:</strong> Revisar progreso y ajustar estrategia</li>
            </ul>
            <div style="margin-top: 1rem; padding: 1rem; background: rgba(52, 152, 219, 0.1);
                        border-radius: 8px;">
                <p style="margin: 0; font-size: 0.95rem; color: #2c3e50;">
                    <strong>💡 Consejo:</strong> Pequeños cambios consistentes generan mayor
                    impacto a largo plazo que cambios drásticos no sostenibles. Score actual:
                    <strong>{score}</strong> → Objetivo: <strong>{min(score_mejorado + 50, 1000)}</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="footer">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div style="color: #343a40;">
            <strong>WealthInsights Analytics</strong> | Modelo entrenado con datos de múltiples ciclos económicos
        </div>
        <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
            <span style="background: #3498db; color: white; padding: 0.2rem 0.8rem;
                         border-radius: 12px; font-size: 0.9rem;">AUC-ROC: 0.918</span>
            <span style="background: #2c3e50; color: white; padding: 0.2rem 0.8rem;
                         border-radius: 12px; font-size: 0.9rem;">Recall: 79%</span>
            <span style="background: #27ae60; color: white; padding: 0.2rem 0.8rem;
                         border-radius: 12px; font-size: 0.9rem;">v2.0</span>
        </div>
    </div>
    <div style="margin-top: 0.5rem; text-align: center; font-size: 0.85rem; color: #343a40;">
        © 2025 WealthInsights Analytics | Información confidencial — Úsese exclusivamente con fines internos
    </div>
</div>
""", unsafe_allow_html=True)
