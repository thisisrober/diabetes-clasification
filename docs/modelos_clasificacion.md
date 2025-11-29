# Modelos de clasificación para predicción de readmisión hospitalaria

## Índice
1. [Introducción](#introducción)
2. [Estrategia de balanceo de clases](#estrategia-de-balanceo-de-clases)
3. [Tarea 2.1: Regresión Logística (Baseline)](#tarea-21-regresión-logística-baseline)
4. [Tarea 2.2: Árboles de decisión](#tarea-22-árboles-de-decisión)
5. [Tarea 2.3: Validación cruzada y comparativa](#tarea-23-validación-cruzada-y-comparativa)
6. [Conclusiones y recomendaciones](#conclusiones-y-recomendaciones)

---

## Introducción

Este documento detalla la implementación y evaluación de modelos clásicos de clasificación para predecir la **readmisión hospitalaria en menos de 30 días** de pacientes diabéticos. El dataset utilizado proviene del repositorio UCI Machine Learning y contiene registros de hospitalizaciones de pacientes con diabetes entre 1999-2008.

### Objetivo
Desarrollar un modelo predictivo que identifique pacientes con alto riesgo de readmisión, permitiendo intervenciones preventivas que mejoren la calidad de atención y reduzcan costos hospitalarios.

### Características del Problema
- **Variable objetivo:** `readmitted_binary` (0 = No readmitido <30 días, 1 = Readmitido <30 días)
- **Desbalance de clases:** ~11% de casos positivos (readmitidos)
- **Número de observaciones:** ~100,000 encuentros hospitalarios
- **Features:** variables demográficas, diagnósticos, medicamentos y estadísticas de hospitalización

---

## Estrategia de balanceo de clases

### El problema del desbalance de clases
El dataset presenta un significativo desbalance de clases:
- **Clase 0 (pacientes no readmitidos):** ~89% de los casos
- **Clase 1 (pacientes readmitidos):** ~11% de los casos

Este desbalance puede causar que los modelos aprendan a predecir siempre la clase mayoritaria, ignorando completamente la clase minoritaria que es precisamente la que queremos detectar.

### Solución: usar `class_weight='balanced'`

En lugar de utilizar técnicas de remuestreo (undersampling, oversampling o SMOTE), optamos por el parámetro `class_weight='balanced'` disponible en scikit-learn.

#### ¿Cómo funciona?
Este parámetro ajusta automáticamente los pesos de las clases de manera inversamente proporcional a su frecuencia:

```python
weight_class_i = n_samples / (n_classes * n_samples_class_i)
```

En nuestro caso:
- **Peso pacientes no readmitidos:** menor (porque tiene más muestras)
- **Peso pacientes readmitidos:** mayor (porque tiene menos muestras)

#### Ventajas
1. **No altera los datos originales**: no se pierden datos (undersampling) ni se crean datos sintéticos (SMOTE)
2. **Computacionalmente eficiente**: no aumenta el tamaño del dataset
3. **Integrado en el algoritmo**: se aplica durante el entrenamiento, no como preprocesamiento
4. **Robusto**: funciona bien con la mayoría de algoritmos

#### Uso en los modelos
```python
# Regresión Logística
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

# Árbol de Decisión
tree = DecisionTreeClassifier(class_weight='balanced', random_state=42)
```

---

## Tarea 2.1: Regresión Logística (Baseline)

### Descripción
La Regresión Logística es un algoritmo lineal que modela la probabilidad de pertenencia a una clase. Se utiliza como **baseline** por su simplicidad, interpretabilidad y robustez.

### Implementación

#### Preparación de datos
```python
# División estratificada (mantiene proporción de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

#### Entrenamiento
```python
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
log_reg.fit(X_train, y_train)
```

### Resultados

#### Métricas de validación cruzada (5-Fold)
| Métrica   | Media  | Desv. Estándar |
|-----------|--------|----------------|
| Accuracy  | 0.66   | ±0.01          |
| F1-Score  | 0.26   | ±0.01          |
| Recall    | 0.53   | ±0.02          |

#### Reporte de clasificación (Test Set, Threshold=0.5)
```
              precision    recall  f1-score   support

No Readmitido     0.92      0.68      0.78     18083
Readmitido        0.17      0.53      0.26      2271

accuracy                              0.66     20354
```

### Matriz de Confusión
|                    | Pred: No Readm. | Pred: Readm. |
|--------------------|-----------------|--------------|
| **Real: No Readm.**| 12,280 (TN)     | 5,803 (FP)   |
| **Real: Readm.**   | 1,073 (FN)      | 1,198 (TP)   |

**Interpretación:**
- El modelo detecta el **53% de los readmitidos** (Recall = 0.53)
- Sin embargo, genera muchos falsos positivos (Precision = 0.17)
- El F1-Score de 0.26 refleja el trade-off entre precisión y recall

### Análisis del Umbral de Decisión

Por defecto, se clasifica como positivo si P(readmitido) > 0.5. En datos desbalanceados, ajustar este umbral puede mejorar las métricas:

| Umbral | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| 0.2    | 0.11      | 0.99   | 0.20     |
| 0.3    | 0.11      | 0.99   | 0.20     |
| 0.4    | 0.13      | 0.87   | 0.23     |
| **0.5**| **0.17**  | **0.53**| **0.26** |
| 0.6    | 0.22      | 0.25   | 0.24     |
| 0.7    | 0.29      | 0.11   | 0.16     |

**Insight:** 
- Umbrales bajos (0.2-0.3) maximizan el Recall (detectan casi todos los readmitidos)
- Umbrales altos (0.6-0.7) maximizan la Precision (menos falsos positivos)
- El umbral 0.5 ofrece un equilibrio razonable

### Top 10 features más influyentes

Las features con mayor impacto en la predicción de readmisión:

| Feature | Coeficiente | Interpretación |
|---------|-------------|----------------|
| number_inpatient | +0.33 | ↑ visitas previas = ↑ riesgo |
| diabetesMed | +0.31 | Uso de medicación = ↑ riesgo |
| age_group_[20-30] | +0.60 | Pacientes jóvenes = ↑ riesgo |
| age_group_[30-40] | +0.45 | Adultos jóvenes = ↑ riesgo |
| age_group_[90-100] | -0.39 | Pacientes muy mayores = ↓ riesgo |
| payer_code_WC | -0.50 | Workers Comp = ↓ riesgo |
| chlorpropamide | -0.42 | Este medicamento = ↓ riesgo |

### Curva ROC y AUC
- **AUC-ROC: 0.65**
- El modelo tiene capacidad discriminativa moderada (superior al azar = 0.5)

---

## Tarea 2.2: Árboles de decisión

### Descripción
Los Árboles de decisión son modelos no lineales que crean reglas de decisión interpretables. Pueden capturar relaciones más complejas que la regresión logística.

### Modelo para Visualización (max_depth=3)
```python
tree_viz = DecisionTreeClassifier(
    max_depth=3,
    class_weight='balanced',
    random_state=42
)
```

#### Primeras reglas de decisión
El árbol aprende reglas como:
1. **Nodo raíz:** `number_inpatient <= 0.685`
   - Si True → rama izquierda (mayoría No Readmitidos)
   - Si False → rama derecha (mayoría Readmitidos)
2. **Segunda división:** `discharge_disposition_id <= 1.5`
3. **Tercera división:** Variables como `total_visits`, `number_inpatient`

### Diagnóstico de overfitting

Entrenamos un árbol sin restricciones para diagnosticar overfitting:

```python
tree_full = DecisionTreeClassifier(
    max_depth=None,
    min_samples_leaf=1,
    class_weight='balanced'
)
```

**Resultados:**
| Métrica | Train | Test | Gap |
|---------|-------|------|-----|
| Accuracy| 1.00  | 0.81 | 0.19|

**⚠️ OVERFITTING SEVERO DETECTADO**
- El modelo alcanza 100% en train pero solo 81% en test
- La diferencia del 19% indica que memoriza los datos de entrenamiento

### Curvas de aprendizaje
Las curvas de aprendizaje muestran que:
- El score de entrenamiento es alto desde el principio
- El score de validación mejora con más datos pero se estanca
- La brecha entre ambos confirma el overfitting

### Experimentación con hiperparámetros

#### Efecto de max_depth

| Profundidad | Train Acc | Test Acc | Train F1 | Test F1 |
|-------------|-----------|----------|----------|---------|
| 3           | 0.51      | 0.47     | 0.25     | 0.24    |
| 5           | 0.59      | 0.56     | 0.27     | 0.26    |
| 7           | 0.64      | 0.59     | 0.29     | 0.27    |
| 10          | 0.74      | 0.62     | 0.32     | 0.26    |
| 15          | 0.89      | 0.64     | 0.38     | 0.24    |
| 20          | 0.97      | 0.67     | 0.43     | 0.22    |
| None        | 1.00      | 0.81     | 0.47     | 0.19    |

**Observaciones:**
- A mayor profundidad, mayor overfitting
- El punto óptimo parece estar entre 5-10 de profundidad
- Después de profundidad 10, el F1 en test empieza a decrecer

---

## Tarea 2.3: Validación cruzada y comparativa

### GridSearch para la optimización del árbol

```python
param_grid = {
    'max_depth': [5, 8, 10, 12, 15],
    'min_samples_leaf': [10, 20, 50, 100],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    param_grid,
    cv=5,
    scoring='f1', # Optimizar para f1, importante en datos desbalanceados
    n_jobs=-1
)
```

#### Mejores parámetros encontrados
- **criterion:** entropy
- **max_depth:** 5
- **min_samples_leaf:** 50
- **Mejor F1-Score (CV):** 0.26

### Validación cruzada (10-Fold) - Regresión Logística

| Métrica   | Media  | Desv. Estándar |
|-----------|--------|----------------|
| Accuracy  | 0.66   | ±0.01          |
| Precision | 0.17   | ±0.01          |
| Recall    | 0.53   | ±0.02          |
| F1-Score  | 0.26   | ±0.01          |
| AUC-ROC   | 0.65   | ±0.01          |

### Comparativa final de modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|--------|----------|---------|
| Regresión Logística | 0.66 | 0.17 | 0.53 | 0.26 | 0.65 |
| Árbol Simple (depth=3) | 0.47 | 0.14 | 0.76 | 0.24 | 0.64 |
| Árbol Completo | 0.81 | 0.31 | 0.19 | 0.24 | 0.53 |
| **Árbol Optimizado** | **0.64** | **0.17** | **0.59** | **0.27** | **0.66** |

### Análisis de resultados

1. **Árbol optimizado** es el ganador por F1-Score (0.27) y AUC-ROC (0.66)
2. **Árbol simple (depth=3)** tiene el mejor Recall (0.76) pero la peor Precision
3. **Regresión logística** ofrece un buen equilibrio y es más interpretable
4. **Árbol completo** sufre de overfitting severo y tiene el peor rendimiento general

---

## Conclusiones y recomendaciones

### Resumen de hallazgos

1. **Desbalance de clases:** el uso de `class_weight='balanced'` es efectivo para compensar el desbalance sin modificar los datos originales.

2. **Rendimiento general:** todos los modelos muestran un rendimiento modesto (F1 ≈ 0.26), lo cual es común en problemas de predicción de readmisión hospitalaria debido a la complejidad del fenómeno.

3. **Trade-off precision/recall:**
   - Mayor Recall → Detectar más readmisiones pero con más falsos positivos
   - Mayor Precision → Menos falsos positivos pero se pierden readmisiones

4. **Overfitting:** los árboles de decisión sin restricciones sobreajustan severamente. La regularización (max_depth, min_samples_leaf) es crucial.

### Recomendación del modelo

**Modelo recomendado:** árbol de decisión optimizado o Regresión Logística

**Justificación:**
- Ambos ofrecen F1-Score similar (~0.26-0.27)
- La Regresión Logística es más interpretable
- El árbol optimizado tiene ligeramente mejor AUC

### Contexto médico

En el contexto hospitalario:
- **Falso Negativo (FN):** un paciente es dado de alta pero es readmitido → costoso y peligroso
- **Falso Positivo (FP):** un paciente es marcado como riesgo pero no es readmitido → recursos adicionales innecesarios

Dependiendo de los recursos disponibles:
- Si hay capacidad para intervenciones preventivas → priorizar **recall** (threshold bajo)
- Si los recursos son limitados → balancear con **F1-Score**

---

## Referencias

- Strack, B., DeShazo, J. P., Gennings, C., et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. *BioMed Research International*.
- UCI Machine Learning Repository - Diabetes 130-US Hospitals Dataset
- Scikit-learn Documentation: https://scikit-learn.org/