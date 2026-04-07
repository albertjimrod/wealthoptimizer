# Sistema de Scoring — WealthOptimizer

## ¿Qué hace el sistema?

El sistema de scoring es un motor de evaluación de riesgo crediticio que predice la
probabilidad de que un cliente incumpla sus obligaciones de pago en los próximos 12 meses.
Transforma datos bancarios y patrimoniales en un número entre 0 y 1000 que indica el nivel
de riesgo, permitiendo:

- Aprobar o rechazar solicitudes de crédito
- Fijar condiciones (tipo de interés, límites) según el riesgo
- Detectar señales tempranas de dificultad financiera
- Cumplir requisitos regulatorios (GDPR, AI Act) mediante predicciones explicables

---

## Variables de entrada

El modelo utiliza 7 variables seleccionadas por su poder predictivo:

| Variable | Tipo | Rango | Correlación con impago |
|----------|------|-------|------------------------|
| `meses_ahorro_positivo` | Discreto | 0–12 meses | −0.68 (mayor predictor) |
| `ahorro_medio_mensual` | Continuo | −€105 a +€2.084 | −0.48 |
| `n_domiciliaciones` | Discreto | 0–15 | −0.42 |
| `edad` | Discreto | 18–70 años | −0.36 |
| `tipo_interes` | Continuo | 0–20% | +0.42 |
| `tiene_recibos_rechazados` | Booleano | 0 / 1 | +0.34 |
| `ratio_utilizacion_credito` | Continuo | 0–1 | +0.21 |

Correlación negativa = reduce el riesgo. Correlación positiva = aumenta el riesgo.

---

## Cálculo del score — 3 fases

### Fase 1: Estimación de probabilidad base (generador de datos)

El generador sintético asigna una probabilidad de impago inicial según el perfil del cliente,
y luego aplica multiplicadores según los factores de riesgo presentes:

```
prob_base = perfil.prob_default_base + Normal(0, perfil.prob_default_std)
```

**Factores multiplicativos:**

| Factor | Condición | Multiplicador |
|--------|-----------|---------------|
| Recibos rechazados | `tiene_recibos_rechazados = 1` | × 2.5 |
| Empleo inestable | Desempleado | × 1.8 |
| Crédito saturado | Utilización > 80% | × 1.5 |
| Ahorro negativo | Ahorro medio < 0 | × 1.4 |
| Ratio deuda/ingreso alto | DTI > 40% | × 1.3 |
| Empleo estable | Empleado / funcionario | × 0.85 |
| Zona de alta renta | Zona postal A/B | × 0.90 |
| Seguro contratado | `tiene_seguro = 1` | × 0.9 |
| Depósito a plazo | `tiene_deposito_plazo = 1` | × 0.85 |
| Plan de pensiones | `tiene_plan_pensiones = 1` | × 0.75 |
| Cartera de inversiones | Valor > 6× ingreso mensual | × 0.70 |
| Hipoteca avanzada | Edad > 45 + hipoteca | × 0.85 |

La probabilidad final se recorta al rango `[0.001, 0.95]`:

```python
prob_default = np.clip(prob_default, 0.001, 0.95)
default = 1 if random() < prob_default else 0
```

---

### Fase 2: Modelo de machine learning

El modelo de producción es un **Gradient Boosting** (scikit-learn) entrenado sobre
1.050.000 clientes sintéticos distribuidos en 3 escenarios económicos:

| Escenario | Clientes | Tasa de impago media |
|-----------|----------|----------------------|
| Normal | 350.000 | ~16% |
| Crisis | 350.000 | ~40% |
| Boom | 350.000 | ~5% |

**Hiperparámetros del modelo:**
- `n_estimators`: 100 árboles
- `max_depth`: 3 (árboles poco profundos, evitan sobreajuste)
- `learning_rate`: 0.05
- `class_weight`: balanced

**Rendimiento:**

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0.918 |
| Recall | 79% |

AUC de 0.918 significa que el modelo discrimina correctamente el 91.8% de los pares
buen cliente / mal cliente. Recall del 79% indica que detecta 8 de cada 10 impagos reales.

---

### Fase 3: Conversión a score 0–1000

```python
score = int((1 - probabilidad_de_impago) * 1000)
```

Cuanto mayor el score, menor el riesgo:

| Score | Probabilidad de impago | Decisión |
|-------|------------------------|----------|
| 900–1000 | < 10% | Aprobar |
| 500–899 | 10–50% | Revisar |
| 0–499 | > 50% | Rechazar |

**Ejemplos:**

| Score | Prob. impago | Interpretación |
|-------|-------------|----------------|
| 957 | 4.3% | Cliente ideal |
| 715 | 28.5% | Aceptable con condiciones |
| 271 | 72.8% | Rechazar / exigir garantías |
| 34 | 96.6% | Rechazar |

---

## Perfiles de riesgo

El generador crea tres perfiles con distinta distribución en la población:

### Bajo riesgo (30% de la población)
- Ingreso: €4.500–5.500/mes
- Tasa de ahorro: 35–55% del ingreso
- Ratio deuda/ingreso: 15–25%
- Prob. de impago: 1–5%

### Riesgo medio (50% de la población)
- Ingreso: €2.000–3.200/mes
- Tasa de ahorro: 12–30% del ingreso
- Ratio deuda/ingreso: 20–25%
- Prob. de impago: 5–20%

### Alto riesgo (20% de la población)
- Ingreso: €1.200–2.000/mes
- Tasa de ahorro: −15% a +15% (frecuentemente negativa)
- Ratio deuda/ingreso: 30–35%
- Prob. de impago: 22–55%

---

## Explicabilidad (SHAP)

Para cada predicción el sistema calcula los valores SHAP, indicando cuánto contribuye
cada variable al score final:

```
Ejemplo cliente con score 271:
  ↑ +3.2 pts de riesgo — tiene_recibos_rechazados = 1
  ↑ +1.5 pts de riesgo — ratio_utilizacion_credito = 0.87
  ↓ −0.8 pts de riesgo — empleo estable
  ↓ −0.3 pts de riesgo — tiene_plan_pensiones = 1
```

Esto permite cumplir con las obligaciones regulatorias de explicar las decisiones
automatizadas de crédito.

---

## Artefactos del modelo

El modelo entrenado se serializa en dos ficheros:

```
modelo_scoring_v2_1M.joblib   — clasificador GradientBoosting
features_v2_1M.joblib         — lista de features en el orden correcto
```

Uso en producción:

```python
import joblib
import numpy as np

model    = joblib.load('modelo_scoring_v2_1M.joblib')
features = joblib.load('features_v2_1M.joblib')

# X debe ser un DataFrame con las columnas en el orden de features
prob  = model.predict_proba(X)[0][1]   # probabilidad de impago
score = int((1 - prob) * 1000)
```

---

## Resumen

| Dimensión | Detalle |
|-----------|---------|
| Variables de entrada | 7 |
| Datos de entrenamiento | 1.050.000 instancias sintéticas (3 escenarios) |
| Algoritmo | Gradient Boosting (100 árboles, max_depth=3) |
| Salida | Probabilidad 0–1 → score 0–1000 |
| AUC-ROC | 0.918 |
| Recall | 79% |
| Factores de ajuste | 12 multiplicadores en la generación de datos |
| Explicabilidad | Valores SHAP por predicción |
