# Proyecto: Análisis de Readmisión Hospitalaria en Pacientes Diabéticos

**Dataset:** Diabetes 130-US Hospitals (1999-2008)  
**Objetivo:** Predecir readmisión hospitalaria en <30 días (Clasificación)  

---

## ⚠️ IMPORTANTE: Alcance del Proyecto

### ✅ LO QUE HAREMOS:
- Análisis exploratorio completo (NumPy, Pandas, Seaborn)
- **Regresión Logística** (clasificación binaria)
- **Árboles de Decisión** para clasificación
- **Random Forest y Gradient Boosting**
- **KNN, Naive Bayes o SVM**
- **Redes Neuronales (MLPClassifier)**
- Validación Cruzada, GridSearchCV
- Métricas de clasificación (matriz de confusión, precision, recall, F1, ROC-AUC)

---

## 👤 PERSONA 1: Análisis Exploratorio y Preparación de Datos (Albert)

### **Tiempo de exposición:** 5-7 minutos

### **Tarea 1.1: Carga e Inspección Inicial**
**Qué hacer:**
- Cargar el dataset usando Pandas desde UCI ML Repository o CSV
- Usar `.info()`, `.describe()`, `.head()`, `.shape` para inspeccionar
- Identificar tipos de datos, cantidad de features (50+), instancias (101,766)
- Detectar valores faltantes con `.isnull().sum()`
- Revisar duplicados en `encounter_id` y `patient_nbr`
- **Demostrar operaciones vectorizadas con NumPy:**
  - Convertir alguna columna numérica a array
  - Realizar operaciones (suma, media, normalización) con vectorización
  - Comparar velocidad vs loops tradicionales

**Entregable:**
- Resumen de la estructura del dataset
- Tabla con estadísticas descriptivas clave
- Identificación de problemas (missing values, desbalance de clases)

---

### **Tarea 1.2: Limpieza y Transformación de Datos**
**Qué hacer:**
- **Manejo de valores faltantes:**
  - Decidir estrategia por columna (eliminar, imputar con moda/media, crear categoría "unknown")
  - Documentar decisiones tomadas
- **Variable objetivo:**
  - Convertir `readmitted` a binaria: `<30` → 1 (readmitido), resto → 0
  - Analizar desbalance de clases
- **Encoding de categóricas:**
  - One-Hot Encoding para variables nominales (race, gender, admission_type)
  - Label Encoding para ordinales si las hay
- **Feature Engineering (opcional pero recomendado):**
  - Crear `total_visits = num_outpatient + num_inpatient + num_emergency`
  - Crear `medication_changes = change + diabetesMed`
  - Agrupar edades en rangos más amplios si tiene sentido
- **Normalización:**
  - Estandarizar variables numéricas con StandardScaler (importante para algunos modelos)

**Entregable:**
- Dataset limpio guardado como `diabetes_clean.csv`
- Documento explicando transformaciones realizadas
- Nuevo shape del dataset después de encoding

---

### **Tarea 1.3: Análisis Exploratorio con Pandas y Visualizaciones**
**Qué hacer:**
- **Agregaciones con `.groupby()`:**
  - Tasa de readmisión por grupo de edad
  - Tasa de readmisión por raza y género
  - Promedio de tiempo hospitalizado según readmisión
  - Relación entre número de medicamentos y readmisión
  - ¿Influye el resultado de HbA1c en la readmisión?

- **Visualizaciones clave (máximo 3-4):**
  1. **Distribución de la variable objetivo** (countplot): ¿Cuántos readmitidos vs no?
  2. **Heatmap de correlación** entre variables numéricas principales
  3. **Boxplot o violinplot**: Tiempo hospitalizado vs readmisión
  4. **Barplot**: Tasa de readmisión por grupo de edad o raza

**Entregable:**
- Notebook con análisis exploratorio completo
- 3-4 gráficas guardadas en alta calidad (PNG/PDF)
- Lista de insights clave para presentar (ej: "Pacientes >70 años tienen 15% más readmisión")

---

**Puntos clave para tu parte de la exposición:**
1. Contexto del problema (2 min)
2. Estructura y limpieza del dataset (2 min)
3. Insights principales del EDA (2-3 min)

---

## 👤 PERSONA 2: Modelos Clásicos de Clasificación y Optimización (Robert)

### **Tiempo de exposición:** 5-7 minutos

### **Tarea 2.1: Regresión Logística (Baseline)**
**Qué hacer:**
- **Preparación:**
  - Dividir datos en train/test (80/20 o 70/30)
  - Usar `train_test_split` con `stratify=y` para mantener proporción de clases
- **Entrenamiento:**
  - Implementar `LogisticRegression` de scikit-learn
  - Entrenar modelo básico con parámetros por defecto
- **Evaluación:**
  - Matriz de confusión con `confusion_matrix` y visualizarla con Seaborn
  - Calcular accuracy, precision, recall, F1-score
  - Generar classification report completo
- **Interpretación:**
  - Analizar coeficientes del modelo (`model.coef_`)
  - Identificar las 10 features más importantes (positivas y negativas)
- **Experimentación con umbral:**
  - Probar diferentes thresholds (0.3, 0.5, 0.7)
  - Graficar cómo cambia precision vs recall
  - Curva ROC y calcular AUC

**Entregable:**
- Modelo de regresión logística entrenado
- Matriz de confusión visualizada
- Tabla con métricas baseline
- Gráfica de importancia de features
- Análisis de impacto del umbral de decisión

---

### **Tarea 2.2: Árboles de Decisión**
**Qué hacer:**
- **Modelo básico:**
  - Entrenar `DecisionTreeClassifier` sin restricciones
  - Evaluar con las mismas métricas que regresión logística
- **Visualización del árbol:**
  - Usar `plot_tree` de sklearn o `export_graphviz`
  - Mostrar primeras 3-4 capas del árbol (el completo será gigante)
  - Interpretar las primeras divisiones: ¿qué features usa?
- **Diagnóstico de overfitting:**
  - Calcular accuracy en train y test
  - Si train >> test → overfitting detectado
  - Crear curvas de aprendizaje (learning curves)
- **Experimentación con hiperparámetros:**
  - Probar diferentes `max_depth` (3, 5, 10, 20, None)
  - Probar `min_samples_leaf` (1, 5, 10, 50)
  - Graficar accuracy train vs test según profundidad
  - Identificar el punto óptimo

**Entregable:**
- Árbol de decisión visualizado (primeras capas)
- Comparativa de hiperparámetros (tabla o gráfica)
- Curvas de aprendizaje mostrando overfitting
- Modelo de árbol optimizado

---

### **Tarea 2.3: Validación Cruzada y Comparativa**
**Qué hacer:**
- **Validación Cruzada:**
  - Aplicar `cross_val_score` con k=5 o k=10 folds
  - Calcular media y desviación estándar de las métricas
  - Comparar resultados con simple train/test
- **GridSearchCV para optimización:**
  - Definir grid de hiperparámetros para el mejor modelo hasta ahora
  - Para árbol: `{'max_depth': [5, 10, 15], 'min_samples_leaf': [5, 10, 20]}`
  - Ejecutar búsqueda con scoring='f1' (importante en datos desbalanceados)
  - Obtener mejores parámetros
- **Tabla comparativa:**
  - Comparar Regresión Logística vs Árbol básico vs Árbol optimizado
  - Métricas: Accuracy, Precision, Recall, F1, AUC, tiempo de entrenamiento
  - Añadir columna de interpretabilidad (subjetiva)

**Entregable:**
- Resultados de validación cruzada
- Mejores hiperparámetros encontrados
- Tabla comparativa completa de modelos
- Recomendación preliminar

---

**Puntos clave para tu parte de la exposición:**
1. Baseline con regresión logística y análisis de coeficientes (2 min)
2. Árboles de decisión, overfitting y optimización (2-3 min)
3. Comparativa de modelos clásicos (2 min)

---

## 👤 PERSONA 3: Modelos Avanzados (Ensembles y Comparativa Final) (Linda)

### **Tiempo de exposición:** 6-8 minutos

### **Tarea 3.1: Random Forest y Gradient Boosting**
**Qué hacer:**
- **Random Forest:**
  - Entrenar `RandomForestClassifier` con parámetros base
  - Empezar con n_estimators=100
  - Evaluar con mismas métricas
  - Analizar importancia de features con `feature_importances_`
  - Comparar importancias con las de regresión logística
- **Gradient Boosting:**
  - Entrenar `GradientBoostingClassifier`
  - Empezar con n_estimators=100, learning_rate=0.1
  - Evaluar y comparar
- **Explicación de filosofías:**
  - Preparar explicación visual de Bagging (Random Forest):
    - Múltiples árboles independientes en paralelo
    - Cada uno con subconjunto aleatorio de datos y features
    - Votación por mayoría
  - Preparar explicación de Boosting (Gradient Boosting):
    - Árboles secuenciales que corrigen errores previos
    - Cada árbol aprende de los residuos del anterior
- **Optimización:**
  - GridSearchCV para el mejor de los dos
  - Para RF: `{'n_estimators': [100, 200], 'max_depth': [10, 20, None]}`
  - Para GB: `{'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}`
- **Comparativa:**
  - Tiempo de entrenamiento
  - Rendimiento
  - Interpretabilidad (importancia de features)

**Entregable:**
- Modelos Random Forest y Gradient Boosting entrenados
- Gráfica de importancia de features
- Diagrama explicativo de Bagging vs Boosting
- Modelo ensemble optimizado

---

### **Tarea 3.2: Modelos Alternativos (KNN, Naive Bayes, SVM)**
**Qué hacer:**
- **Seleccionar 2 de estos 3 modelos:**
  
  **Opción A - K-Nearest Neighbors:**
  - Entrenar `KNeighborsClassifier`
  - Probar diferentes valores de k (3, 5, 10, 20)
  - Justificación: Simple, no paramétrico, bueno para datos locales
  - Desventaja: Lento con datasets grandes (101k instancias)
  
  **Opción B - Naive Bayes:**
  - Entrenar `GaussianNB` (para features continuas)
  - Justificación: Rápido, funciona bien con muchas features, asume independencia
  - Evaluar si la asunción de independencia se cumple
  
  **Opción C - Support Vector Machine:**
  - Entrenar `SVC` con kernel='rbf'
  - Justificación: Potente para clasificación binaria
  - Desventaja: Muy lento con datasets grandes (considerar SVC con kernel lineal)

- **Análisis crítico:**
  - ¿Cuál es modelo más adecuado para este problema específico?
  - Considerar: tamaño del dataset, tipo de features, interpretabilidad
  - Comparar tiempos de entrenamiento

**Entregable:**
- 2 modelos alternativos entrenados y evaluados
- Justificación de por qué elegiste esos 2
- Comparación de rendimiento y tiempo

---

### **Tarea 3.3: Red Neuronal MLP (opcional pero recomendado)**
**Qué hacer:**
- **Entrenamiento básico:**
  - Implementar `MLPClassifier` de scikit-learn
  - Arquitectura simple: hidden_layers=(100, 50) o similar
  - Usar activation='relu', solver='adam'
  - Establecer max_iter=500 y early_stopping=True
- **Comparativa con modelos tradicionales:**
  - Evaluar con mismas métricas
  - ¿Realmente supera a Random Forest/Gradient Boosting?
  - Considerar tiempo de entrenamiento
- **Reflexión crítica:**
  - "Romper el mito" de que la red neuronal siempre gana
  - Para datos tabulares, los ensembles suelen ser mejores
  - Discutir cuándo SÍ tendría sentido usar redes neuronales

**Entregable:**
- Modelo MLP entrenado
- Comparativa honesta con otros modelos
- Reflexión sobre cuándo usar cada tipo de modelo

---

### **Tarea 3.4: Comparativa Final y Selección del Modelo**
**Qué hacer:**
- **Tabla comparativa completa:**
  - Incluir TODOS los modelos probados:
    1. Regresión Logística
    2. Árbol de Decisión (optimizado)
    3. Random Forest
    4. Gradient Boosting
    5. 2 modelos alternativos
    6. MLP
  - Columnas: Accuracy, Precision, Recall, F1, ROC-AUC, Tiempo entrenamiento, Interpretabilidad
  
- **Análisis de trade-offs:**
  - Interpretabilidad vs Rendimiento
  - Velocidad vs Precisión
  - Simplicidad vs Complejidad
  
- **Selección del modelo final:**
  - Considerar objetivo de negocio: ¿Qué es peor?
    - Falso Negativo: No detectar una readmisión real (paciente vuelve al hospital)
    - Falso Positivo: Predecir readmisión innecesaria (recursos mal asignados)
  - Si FN es peor → priorizar **Recall**
  - Si balance → priorizar **F1-score**
  - Justificar elección del modelo ganador
  
- **Recomendaciones finales:**
  - Modelo recomendado para producción
  - Features más importantes a monitorear
  - Posibles mejoras futuras

**Entregable:**
- Tabla comparativa profesional (visual)
- Análisis de trade-offs con ejemplos reales
- Modelo final seleccionado y justificado
- Recomendaciones para implementación

---

**Puntos clave para tu parte de la exposición:**
1. Random Forest vs Gradient Boosting: filosofías y resultados (2-3 min)
2. Modelos alternativos y MLP: ¿cuándo usar cada uno? (2 min)
3. Comparativa final y selección del modelo ganador con justificación de negocio (2-3 min)

---

## 📊 RESUMEN DE VISUALIZACIONES (máximo 9 en total)

### Persona 1 (3-4 gráficas):
1. Distribución de variable objetivo (desbalance de clases)
2. Heatmap de correlación
3. Tiempo hospitalizado vs readmisión
4. Tasa de readmisión por edad/raza

### Persona 2 (2-3 gráficas):
1. Matriz de confusión (regresión logística)
2. Importancia de features (regresión logística o árbol)
3. Curvas de aprendizaje (overfitting en árbol)

### Persona 3 (2-3 gráficas):
1. Comparación Bagging vs Boosting (diagrama conceptual)
2. Importancia de features (Random Forest)
3. Tabla/gráfica comparativa final de todos los modelos

---

## 🎯 ESTRUCTURA SUGERIDA DE LA PRESENTACIÓN (20 min)

1. **Introducción** (2 min) - Persona 1
   - Contexto del problema
   - Importancia clínica y económica
   - Objetivo del proyecto

2. **Exploración y preparación** (5 min) - Persona 1
   - Estructura del dataset
   - Limpieza y transformaciones
   - Insights principales del EDA

3. **Modelos clásicos** (5 min) - Persona 2
   - Regresión logística (baseline)
   - Árboles de decisión y overfitting
   - Validación cruzada y optimización

4. **Modelos avanzados** (6 min) - Persona 3
   - Random Forest y Gradient Boosting
   - Modelos alternativos
   - Comparativa completa

5. **Conclusiones y recomendaciones** (2 min) - Persona 3
   - Modelo final seleccionado
   - Justificación de negocio
   - Próximos pasos

---

## ✅ CHECKLIST FINAL

### Antes de la presentación:
- [ ] Notebook limpio y bien comentado
- [ ] Todas las gráficas guardadas en alta calidad
- [ ] Tabla comparativa final completa
- [ ] Modelo final guardado (pickle o joblib)
- [ ] Presentación de diapositivas preparada
- [ ] Ensayo de timing (20 min totales)

### Durante la presentación:
- [ ] Explicar decisiones tomadas, no solo resultados
- [ ] Justificar por qué NO usamos regresión lineal
- [ ] Enfatizar la importancia del contexto médico
- [ ] Mostrar trade-offs, no solo "el mejor modelo"
- [ ] Ser honestos con limitaciones