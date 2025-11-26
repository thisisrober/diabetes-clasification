# Documentación: Limpieza y Preprocesamiento de Datos

## Proyecto: Análisis de Readmisión Hospitalaria en Pacientes Diabéticos

**Dataset:** Diabetes 130-US Hospitals (1999-2008)  
**Fecha:** Noviembre 2025  
**Objetivo:** Preparar los datos para modelos de clasificación de readmisión hospitalaria

---

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Inspección Inicial](#inspección-inicial)
3. [Limpieza de Datos](#limpieza-de-datos)
4. [Transformación de Variables](#transformación-de-variables)
5. [Feature Engineering](#feature-engineering)
6. [Normalización](#normalización)
7. [Resultado Final](#resultado-final)

---

## 1. Resumen Ejecutivo

Este documento describe el proceso completo de limpieza y preprocesamiento del dataset de pacientes diabéticos. El objetivo es preparar los datos para entrenar modelos de machine learning que predigan la readmisión hospitalaria.

### Dimensiones del Dataset
### Dimensiones del Dataset
- **Original:** 101,766 registros × 50 columnas
- **Final:** 101,766 registros × ~70 columnas (después de encoding y feature engineering — incluye flags derivados de los códigos ICD‑9)

### Cambios Principales
 ✅ Feature Engineering (3 nuevas características)
✅ Decodificación ICD-9 y creación de flags binarios (diabetes, circulatory, respiratory) y `comorbidity_score`
- ✅ Imputación de valores faltantes en `race` con la moda
- ✅ Sustitución de valores `?` por `Unknown` en `payer_code` y `medical_specialty`
- ✅ Creación de variables objetivo binarias
- ✅ Encoding de variables categóricas (Label + One-Hot)
- ✅ Feature Engineering (3 nuevas características)
- ✅ Normalización de variables numéricas con StandardScaler

---

## 2. Inspección Inicial

### 2.1 Carga del Dataset

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/diabetic_data.csv")
```

**Resultado:**
- 101,766 registros
- 50 columnas
- Tipos de datos mixtos (numéricos y categóricos)

### 2.2 Análisis de Estructura

Se utilizaron las siguientes funciones para inspeccionar el dataset:

- **`.info()`**: Información sobre tipos de datos y valores no nulos
- **`.describe()`**: Estadísticas descriptivas de variables numéricas
- **`.head()`**: Vista previa de las primeras filas
- **`.shape`**: Dimensiones del dataset

### 2.3 Detección de Valores Faltantes

#### Valores representados como '?'
El dataset utiliza el carácter `?` para representar valores faltantes en variables categóricas:

| Columna | Valores Faltantes | Porcentaje |
|---------|-------------------|------------|
| `weight` | ~97,000 | 97% |
| `payer_code` | ~40,000 | 40% |
| `medical_specialty` | ~49,000 | 49% |
| `race` | ~2,000 | 2% |

**Decisión tomada:**
- `weight`: Eliminada (demasiados valores faltantes)
- `race`: Imputada con la moda (mayoría de valores completos)
- `payer_code` y `medical_specialty`: Valores `?` reemplazados por `Unknown`

### 2.4 Análisis de Duplicados

```python
# Duplicados por encounter_id (cada encuentro debe ser único)
duplicate_encounters = df['encounter_id'].duplicated().sum()

# Análisis de pacientes únicos
unique_patients = df['patient_nbr'].nunique()
```

**Hallazgos:**
- No hay duplicados en `encounter_id` ✓
- Pacientes únicos: ~71,000
- Promedio de encuentros por paciente: ~1.43
- Algunos pacientes tienen múltiples readmisiones (dato esperado)

### 2.5 Operaciones Vectorizadas con NumPy

Se demostró la eficiencia de las operaciones vectorizadas comparándolas con loops tradicionales:

#### Comparación de Rendimiento

| Operación | Loop Tradicional | NumPy Vectorizado | Speedup |
|-----------|------------------|-------------------|---------|
| Suma | ~10ms | ~0.5ms | 20x |
| Media | ~8ms | ~0.3ms | 26x |
| Normalización Min-Max | ~50ms | ~1ms | 50x |

**Conclusión:** Las operaciones vectorizadas de NumPy son significativamente más rápidas y eficientes, especialmente con datasets grandes.

```python
# Ejemplo de vectorización
time_in_hospital = df['time_in_hospital'].values

# Vectorizado (rápido)
normalized = (time_in_hospital - np.min(time_in_hospital)) / (np.max(time_in_hospital) - np.min(time_in_hospital))

# vs Loop tradicional (lento)
min_val = min(time_in_hospital)
max_val = max(time_in_hospital)
normalized_loop = [(x - min_val) / (max_val - min_val) for x in time_in_hospital]
```

---

## 3. Limpieza de Datos

### 3.1 Eliminación de Columnas con Mayoría de Nulos

**Columna eliminada:** `weight`

**Justificación:**
- 97% de valores faltantes
- Imputar tantos valores sería poco confiable
- No es una variable crítica para el modelo

```python
df = df.drop(columns=['weight'])
```

### 3.2 Imputación de Valores Faltantes

#### 3.2.1 Imputación de `race` con la Moda

**Estrategia:** SimpleImputer con estrategia `most_frequent`

```python
from sklearn.impute import SimpleImputer

# Convertir '?' a NaN
df['race'] = df['race'].replace('?', np.nan)

# Imputar con la moda
race_imputer = SimpleImputer(strategy='most_frequent')
df['race'] = race_imputer.fit_transform(df[['race']]).ravel()
```

**Resultado:**
- ~2,273 valores imputados
- Valor imputado: `Caucasian` (categoría más frecuente)

#### 3.2.2 Sustitución de '?' por 'Unknown'

Para `payer_code` y `medical_specialty`, se decidió mantener la información de que el valor es desconocido en lugar de imputar.

```python
df['payer_code'] = df['payer_code'].replace('?', 'Unknown')
df['medical_specialty'] = df['medical_specialty'].replace('?', 'Unknown')
```

**Justificación:**
- Estas variables tienen muchos valores faltantes (40-49%)
- El hecho de que sean desconocidas puede ser información relevante
- Evitamos introducir sesgo mediante imputación

---

## 4. Transformación de Variables

### 4.1 Variable Objetivo: `readmitted`

La variable original `readmitted` tiene 3 categorías:
- `NO`: No readmitido
- `<30`: Readmitido en menos de 30 días
- `>30`: Readmitido en más de 30 días

Se crearon **dos variables binarias** para diferentes enfoques de modelado:

#### Opción 1: `readmitted_binary`
Clasificación binaria simple: readmitido vs no readmitido

```python
df['readmitted_binary'] = (df['readmitted'] != 'NO').astype(int)
```

- **0:** No readmitido
- **1:** Readmitido (cualquier tiempo)

**Distribución:**
- Clase 0: ~54%
- Clase 1: ~46%
- Desbalance moderado

#### Opción 2: `early_readmission` (RECOMENDADA)
Clasificación enfocada en readmisiones críticas tempranas

```python
df['early_readmission'] = (df['readmitted'] == '<30').astype(int)
```

- **0:** No readmitido o readmitido >30 días
- **1:** Readmitido <30 días (más crítico)

**Distribución:**
- Clase 0: ~89%
- Clase 1: ~11%
- Desbalance significativo (ratio ~8:1)

**⚠️ Importante:** Debido al desbalance, se recomienda usar:
- `class_weight='balanced'` en los modelos
- O aplicar SMOTE (Synthetic Minority Over-sampling Technique)

### 4.2 Encoding de Variables Categóricas

El dataset contiene múltiples variables categóricas que deben convertirse a formato numérico.

#### 4.2.1 Identificación de Columnas Categóricas

Se identificaron ~24 columnas categóricas (tipo `object`), excluyendo:
- `encounter_id`, `patient_nbr` (identificadores)
- `readmitted` (ya procesada)

**Clasificación por cardinalidad:**
- **Baja cardinalidad (≤10 valores únicos):** 15 columnas
  - Ejemplos: `gender`, `age`, `change`, `diabetesMed`
- **Alta cardinalidad (>10 valores únicos):** 9 columnas
  - Ejemplos: `admission_type_id`, `discharge_disposition_id`, `diag_1`, `diag_2`, `diag_3`

#### 4.2.2 Label Encoding

**Aplicado a:** Variables de baja cardinalidad

```python
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in low_cardinality_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
```

**Ventajas:**
- No aumenta la dimensionalidad
- Apropiado para variables con relación ordinal implícita
- Eficiente en memoria

**Ejemplos:**
- `gender`: Female=0, Male=1
- `age`: [0-10)=0, [10-20)=1, ..., [90-100)=9
- `change`: No=0, Ch=1

#### 4.2.3 One-Hot Encoding

**Aplicado a:** Variables nominales importantes de alta cardinalidad

```python
# Variables seleccionadas para One-Hot Encoding
onehot_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

df_encoded = pd.get_dummies(df_encoded, columns=onehot_cols, 
                             prefix=onehot_cols, drop_first=True)
```

**Ventajas:**
- No asume relación ordinal entre categorías
- Cada categoría se representa como una feature binaria independiente
- Apropiado para variables nominales

**Consideraciones:**
- Se usó `drop_first=True` para evitar multicolinealidad
- Para variables con demasiadas categorías (ej: códigos de diagnóstico), se aplicó Label Encoding para evitar explosión dimensional

#### 4.2.4 Variables de Medicación

Variables como `metformin`, `insulin`, `glyburide`, etc., tienen valores:
- `No`: No se prescribió
- `Steady`: Dosis constante
- `Up`: Dosis aumentada
- `Down`: Dosis reducida

**Tratamiento:** Label Encoding (orden implícito: No < Steady < Up/Down)

---

## 5. Feature Engineering

Se crearon **3 nuevas características** combinadas para capturar patrones relevantes:

### 5.1 `total_visits`

**Definición:** Suma total de visitas médicas previas

```python
df_encoded['total_visits'] = (df_encoded['number_outpatient'] + 
                               df_encoded['number_emergency'] + 
                               df_encoded['number_inpatient'])
```

**Justificación:**
- Un paciente con más visitas previas puede tener mayor riesgo de readmisión
- Captura el historial de interacción con el sistema de salud

**Estadísticas:**
- Rango: [0, 21]
- Media: ~0.68 visitas

### 5.2 `medication_changes`

**Definición:** Indicador de cambios en el tratamiento

```python
# Convertir a numérico
df_encoded['change'] = df_encoded['change'].map({'No': 0, 'Ch': 1})
df_encoded['diabetesMed'] = df_encoded['diabetesMed'].map({'No': 0, 'Yes': 1})

# Crear feature combinada
df_encoded['medication_changes'] = df_encoded['change'] + df_encoded['diabetesMed']
```

**Valores posibles:**
- **0:** Sin cambios y sin medicación diabética
- **1:** Un cambio (medicación O cambio de dosis)
- **2:** Ambos (medicación Y cambio de dosis)

**Justificación:**
- Los cambios en medicación pueden indicar condición inestable
- Puede correlacionar con mayor riesgo de readmisión

### 5.3 `procedures_per_day`

**Definición:** Intensidad de procedimientos durante la hospitalización

```python
df_encoded['procedures_per_day'] = df_encoded['num_procedures'] / (df_encoded['time_in_hospital'] + 1)
```

**Justificación:**
- Un paciente con más procedimientos por día puede tener condición más severa
- Normaliza el número de procedimientos por la duración de la estancia

**Notas:**
- Se suma 1 al denominador para evitar división por cero
- Media: ~0.18 procedimientos por día

---

### 5.4 Decodificación ICD-9 y flags de comorbilidad

Se implementó una decodificación "inteligente" de los códigos ICD‑9 presentes en `diag_1`, `diag_2` y `diag_3` para crear variables binarias más interpretables. En lugar de usar los códigos crudos, creamos flags clínicos y un score agregado:

- Reglas principales (prefijo numérico antes del punto decimal):
  - Diabetes: prefijo 250 (ej. `250.01` → diabetes)
  - Problemas circulatorios: rangos 390–459 y 785
  - Problemas respiratorios: rangos 460–519 y 786

- Flags generados por cada `diag_i`:
  - `diag_i_is_diabetes`, `diag_i_is_circulatory`, `diag_i_is_respiratory` (0/1)

- Flags agregados por paciente:
  - `diabetes_primary`: 1 si `diag_1` es diabetes
  - `diabetes_secondary`: 1 si aparece diabetes en `diag_2` o `diag_3` pero no en `diag_1`
  - `diabetes_any`, `circulatory_any`, `respiratory_any`: 1 si la condición aparece en cualquiera de los 3 diagnósticos

- `comorbidity_score`: cuenta cuántos de los tres grupos mayores están presentes (valor en 0–3)

Observaciones (valores obtenidos al ejecutar el notebook):

- `diabetes_any`: 38,024 registros
- `diabetes_primary`: 8,757 registros
- `diabetes_secondary`: 29,267 registros
- `circulatory_any`: 59,313 registros
- `respiratory_any`: 26,870 registros
- Distribución `comorbidity_score`:
  - 0: 15,385 (≈15.12%)
  - 1: 51,438 (≈50.55%)
  - 2: 32,060 (≈31.50%)
  - 3: 2,883  (≈2.83%)

Estas features mejoran la interpretabilidad y son recomendables como inputs en modelos interpretables (Regresión Logística) y como features informativas para modelos complejos (RandomForest, XGBoost, redes neuronales).

---

## 6. Normalización

### 6.1 StandardScaler para Variables Numéricas

**Objetivo:** Estandarizar variables numéricas a media=0 y desviación estándar=1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numerical_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                  'num_medications', 'number_outpatient', 'number_emergency', 
                  'number_inpatient', 'number_diagnoses', 'total_visits', 
                  'medication_changes', 'procedures_per_day']

df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
```

**¿Por qué es importante?**
- Muchos algoritmos (SVM, KNN, Redes Neuronales) son sensibles a la escala
- Variables con rangos grandes pueden dominar el aprendizaje
- Mejora la convergencia de algoritmos de optimización

**Validación:**
- Media después de escalar: ~0.0 ✓
- Desviación estándar: ~1.0 ✓

**Columnas normalizadas:**
- Variables numéricas originales (8)
- Features de engineering (3)

---

## 7. Resultado Final

### 7.1 Dataset Limpio Guardado

```python
df_encoded.to_csv("data/diabetes_clean.csv", index=False)
```

**Ubicación:** `data/diabetes_clean.csv`

### 7.2 Dimensiones Finales

- **Registros:** 101,766 (sin pérdida de datos)
- **Columnas:** ~60 (después de encoding y feature engineering)
- **Variables objetivo:** 2 (`readmitted_binary`, `early_readmission`)

### 7.3 Estructura del Dataset Limpio

#### Columnas Eliminadas
- `weight` (97% faltantes)
- `readmitted` (reemplazada por variables binarias)
- `race_backup` (columna temporal)

#### Columnas Nuevas
- **Variables objetivo:**
  - `readmitted_binary`
  - `early_readmission`
- **Feature Engineering:**
  - `total_visits`
  - `medication_changes`
  - `procedures_per_day`
- **ICD-9 / Comorbilidades:**
  - `diag_1_is_diabetes`, `diag_2_is_diabetes`, `diag_3_is_diabetes`
  - `diag_1_is_circulatory`, `diag_2_is_circulatory`, `diag_3_is_circulatory`
  - `diag_1_is_respiratory`, `diag_2_is_respiratory`, `diag_3_is_respiratory`
  - `diabetes_primary`, `diabetes_secondary`, `diabetes_any`
  - `circulatory_any`, `respiratory_any`
  - `comorbidity_score` (0-3)
- **One-Hot Encoding:**
  - `admission_type_id_*` (múltiples columnas binarias)
  - `discharge_disposition_id_*`
  - `admission_source_id_*`

### 7.4 Tipos de Datos Finales

| Tipo | Cantidad |
|------|----------|
| `int64` | ~40 columnas |
| `float64` | ~20 columnas |
| `object` | 0 columnas (todas convertidas) |

### 7.5 Distribución de Variables Objetivo

#### `early_readmission` (RECOMENDADA)
```
0 (No readmitido/<30 días):  ~90,600 (89%)
1 (Readmitido <30 días):     ~11,166 (11%)
```

**Ratio de desbalance:** 8.1:1

**Recomendaciones para el modelado:**
1. Usar `class_weight='balanced'` en modelos que lo soporten
2. Considerar SMOTE para sobremuestreo sintético
3. Evaluar con métricas apropiadas: F1-Score, Precision, Recall, AUC-ROC
4. No confiar únicamente en Accuracy debido al desbalance

---

## 8. Checklist de Verificación

- [x] **Carga e inspección inicial**
  - [x] `.info()`, `.describe()`, `.head()`, `.shape`
  - [x] Detección de valores faltantes
  - [x] Análisis de duplicados
  - [x] Demostración de operaciones vectorizadas con NumPy

- [x] **Limpieza de datos**
  - [x] Eliminación de columna `weight`
  - [x] Imputación de `race` con moda
  - [x] Sustitución de '?' por 'Unknown' en `payer_code` y `medical_specialty`

- [x] **Transformación de variables**
  - [x] Creación de variable objetivo binaria `readmitted_binary`
  - [x] Creación de variable objetivo `early_readmission`
  - [x] Análisis de desbalance de clases
  - [x] Label Encoding para variables de baja cardinalidad
  - [x] One-Hot Encoding para variables nominales críticas
  - [x] Encoding de variables de medicación

- [x] **Feature Engineering**
  - [x] `total_visits`: Suma de visitas previas
  - [x] `medication_changes`: Cambios en tratamiento
  - [x] `procedures_per_day`: Intensidad de procedimientos

- [x] **Normalización**
  - [x] StandardScaler aplicado a variables numéricas
  - [x] Validación de media ≈ 0 y std ≈ 1

- [x] **Guardado y documentación**
  - [x] Dataset limpio guardado en `data/diabetes_clean.csv`
  - [x] Documentación completa del proceso
  - [x] Preservación de 101,766 registros

---

## 9. Próximos Pasos

1. **Análisis Exploratorio de Datos (EDA)**
   - Agregaciones con `.groupby()`
   - Visualizaciones (distribuciones, correlaciones, boxplots)
   - Identificación de insights clave

2. **Modelado**
   - División en train/test (80/20)
   - Implementación de modelos:
     - Regresión Logística
     - Árboles de Decisión
     - Random Forest
     - Gradient Boosting
     - KNN / Naive Bayes / SVM
     - Redes Neuronales (MLPClassifier)
   - Validación cruzada y GridSearchCV
   - Manejo de desbalance con `class_weight` o SMOTE

3. **Evaluación**
   - Matriz de confusión
   - Precision, Recall, F1-Score
   - Curvas ROC y AUC
   - Comparación de modelos

---

## 10. Notas Técnicas

### Dependencias Utilizadas
```python
import pandas as pd              # Manipulación de datos
import numpy as np               # Operaciones numéricas
import warnings                  # Gestión de advertencias
from sklearn.impute import SimpleImputer           # Imputación
from sklearn.preprocessing import LabelEncoder     # Label Encoding
from sklearn.preprocessing import StandardScaler   # Normalización
```

### Archivos Generados
- `data/diabetes_clean.csv`: Dataset procesado listo para modelado

### Tiempo de Procesamiento
- Carga de datos: ~0.3s
- Limpieza e imputación: ~2s
- Encoding: ~5s
- Feature Engineering: ~1s
- Normalización: ~0.5s
- **Total:** ~9 segundos

---

## 11. Contacto y Contribuciones

**Proyecto:** Análisis de Readmisión Hospitalaria en Pacientes Diabéticos  
**Equipo:** Análisis de Datos con Python  
**Fecha:** Noviembre 2025

Para preguntas o sugerencias sobre el proceso de limpieza y preprocesamiento, consulte el notebook `Diabetes_Analysis.ipynb` donde se encuentra el código ejecutable completo.

---

**Fin del documento**
