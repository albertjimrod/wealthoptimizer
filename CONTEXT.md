# CONTEXT — WealthOptimizer

Documento de retoma. Destilado del repo a 2026-08-28.

## 1. Qué es

Dashboard interactivo de **scoring de riesgo crediticio** sobre datos de Open Banking.
A partir de 7 variables derivadas de movimientos bancarios, un modelo de ML estima la
probabilidad de impago a 12 meses y la convierte en un score 0–1000 (mayor = menos riesgo).
Añade lectura de factores por variable, análisis del perfil, recomendaciones priorizadas y
una proyección del score potencial. Funciona 100% en local, sin APIs externas. Los datos de
entrenamiento son sintéticos (no hay datos reales de clientes).

## 2. Stack

- **Lenguaje:** Python 3.11.
- **App:** Streamlit 1.52.2 (una sola página, 4 pestañas).
- **ML / datos:** scikit-learn 1.8.0 (GradientBoostingClassifier), pandas 2.3.3, numpy 2.3.5,
  joblib 1.5.3.
- **Gráficas:** matplotlib 3.10.8 (gráfico de factores). plotly 6.5.0 y shap 0.50.0 figuran
  en requisitos del README pero **no se importan** en el código actual.
- **Entorno:** conda, env llamado `wealthinsights` (README menciona `wealthinsights.yaml`,
  **no presente en el repo** — PENDIENTE DE CONFIRMAR).
- **Despliegue:** local. `streamlit run dashboards/dashboard_um_riesgo_v2.py --server.port 8502`.
  No hay Dockerfile, CI ni configuración de hosting.

## 3. Arquitectura

Proyecto de un solo script + artefactos. Piezas:

- **`dashboards/dashboard_um_riesgo_v2.py`** — toda la app. Carga el modelo con
  `@st.cache_resource`, recoge las 7 variables desde el sidebar, construye un DataFrame en el
  orden de `FEATURES`, llama a `predict_proba` y calcula `score = int((1 - prob) * 1000)`.
  Las 4 pestañas: Score General, Factores de Riesgo, Análisis Detallado, Recomendaciones.
- **`dashboards/modelo_scoring_v2_1M.joblib`** — clasificador GradientBoosting entrenado
  (100 árboles, max_depth=3, learning_rate=0.05, class_weight=balanced) sobre 1.050.000
  instancias sintéticas en 3 escenarios (normal ~16% impago, crisis ~40%, boom ~5%).
  AUC-ROC 0.918, recall 79%.
- **`dashboards/features_v2_1M.joblib`** — lista de features en orden. En el script el orden
  está además hardcodeado en la constante `FEATURES`.
- **`docs/scoring_funcionamiento.md`** — doc técnica del scoring (variables, correlaciones,
  3 fases del cálculo, 12 multiplicadores del generador sintético, perfiles de riesgo,
  explicabilidad, uso en producción).
- **`docs/screenshots/`** — 4 capturas usadas en el README.

Variables de entrada (dirección respecto al riesgo): `meses_ahorro_positivo` (−, principal
predictor, corr −0.68), `ahorro_medio_mensual` (−), `n_domiciliaciones` (−), `edad` (−),
`tipo_interes` (+), `tiene_recibos_rechazados` (+), `ratio_utilizacion_credito` (+).

El generador de datos sintéticos, los notebooks de entrenamiento y los datasets están
**fuera del repo** (excluidos por `.gitignore`: `data/`, `notebooks/`, `MOCKDATA/`,
`Generadores_datos_sinteticos/`, `*.csv`, `*.parquet`, modelos v1). No hay pipeline de
reentrenamiento versionado.

## 4. Decisiones clave

- **GradientBoosting frente a modelos más complejos:** árboles poco profundos (max_depth=3)
  y learning_rate bajo para evitar sobreajuste; se prioriza estabilidad y explicabilidad
  sobre exprimir métrica.
- **Entrenar con 3 escenarios económicos** (normal/crisis/boom, 350k c/u): que el modelo
  generalice a distintos ciclos y no solo a condiciones benignas.
- **Datos sintéticos calibrados** en vez de datos reales: evita problemas de privacidad
  (GDPR) y permite controlar la distribución de perfiles; coste asumido = el modelo hereda
  los sesgos del generador (12 multiplicadores de riesgo definidos a mano).
- **Score = (1 − prob) × 1000**: escala lineal simple y directamente interpretable.
- **Explicabilidad prevista con SHAP** por requisitos regulatorios (AI Act / decisiones
  automatizadas). En la práctica, la pestaña "Factores de Riesgo" **no usa SHAP**: aproxima
  el impacto con `feature_importances_` (global) × valor normalizado × dirección fija por
  variable. Es una heurística visual, no una atribución por predicción. PENDIENTE DE
  CONFIRMAR si fue decisión deliberada (rendimiento/dependencias) o deuda técnica.
- **App monolítica en un script:** alcance de demo/PoC; sin capa de servicio ni API.

## 5. Estado actual

- **Funciona:** la app arranca, carga el modelo v2 (1M) y produce score, probabilidad,
  gráfico de factores, análisis del perfil, recomendaciones y proyección de score mejorado.
- **Git:** rama `main`, árbol de trabajo **limpio**, sin cambios sin commitear, al día con
  `origin/main`. Un único commit: `be6c357 Initial commit: credit risk scoring dashboard v2`.
  No hay otras ramas activas.
- **Sin tests, sin CI, sin linter configurado.**
- **Incoherencias detectadas (a resolver):**
  - Nombre del producto: repo `wealthoptimizer` vs. app "WealthInsights Analytics" vs. env
    `wealthinsights`. El footer marca "v2.0" y el sidebar "v1.0"; el copyright dice "© 2025".
  - Umbrales de decisión distintos entre doc y código: `scoring_funcionamiento.md` usa
    bandas por score (900–1000 aprobar / 500–899 revisar / <500 rechazar) y el dashboard usa
    bandas por probabilidad (<20% EXCELENTE, <35% BUENO, <50% MODERADO, ≥50% ALTO RIESGO),
    además de otra escala de score en la tarjeta lateral (800–1000 / 650–799 / 500–649 / <500).
  - README menciona `wealthinsights.yaml` y `pip install ... seaborn`, pero no hay archivo de
    entorno ni `requirements.txt` / `pyproject.toml` en el repo; `seaborn` no se usa.
  - `shap` y `plotly` listados como dependencias pero no usados en el código actual.
- **No hay** DECISIONS.md, TODO.md ni JOURNAL/ en el repo (este CONTEXT.md los sustituye por
  ahora).

## 6. Próximos pasos (priorizados)

1. **Fijar dependencias reproducibles:** añadir `requirements.txt` o `pyproject.toml` (o el
   `wealthinsights.yaml` que cita el README) con las versiones ya conocidas; quitar shap /
   plotly / seaborn si no se van a usar.
2. **Unificar umbrales y nomenclatura:** una sola tabla de bandas score↔probabilidad↔decisión
   compartida entre `dashboard_um_riesgo_v2.py` y `scoring_funcionamiento.md`; decidir el
   nombre del producto y la versión, y corregir footer/sidebar/copyright.
3. **Decidir sobre la explicabilidad:** implementar SHAP real por predicción en la pestaña de
   factores (coherente con el discurso regulatorio del README) o documentar explícitamente
   que es una aproximación con `feature_importances_`.
4. **Versionar el pipeline de datos/entrenamiento:** meter en el repo (o en un repo hermano
   enlazado) el generador sintético y el script de entrenamiento que producen los `.joblib`,
   hoy fuera de control de versiones.
5. **Calidad mínima:** tests de humo (carga de modelo, forma del input, rango del score),
   linter/formatter, y separar la lógica de scoring del render de Streamlit.
6. **Robustez de la app:** validación de entradas, manejo del caso "modelo no carga" ya
   presente pero revisar mensajes; considerar `st.cache_data` para el cálculo.
7. **Despliegue:** si se quiere más allá de local, definir target (Streamlit Community Cloud /
   contenedor) y añadir Dockerfile + configuración. PENDIENTE DE CONFIRMAR si aplica.
