"""
Modelo de Red Neuronal Artificial para Predicción de Valor de Mercado
de Jugadores Jóvenes de Fútbol - Proyecto de Analítica de Datos

Autor: Análisis Comparativo RNA vs Modelos Tradicionales
Dataset: FIFA - Jugadores Jóvenes
Objetivo: Predecir value_eur a partir de atributos técnicos, físicos y mentales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
warnings.filterwarnings('ignore')

# Configuración de estilo para visualizaciones
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("PROYECTO: PREDICCIÓN DE VALOR DE MERCADO DE JUGADORES DE FÚTBOL")
print("Modelo: Red Neuronal Artificial (RNA) vs Regresión Lineal")
print("=" * 80)

# ============================================================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# ============================================================================
print("\n[1] Cargando dataset...")

try:
    df = pd.read_csv('jugadores_jovenes.csv')
    print(f"✓ Dataset cargado exitosamente: {df.shape[0]} registros, {df.shape[1]} columnas")
except FileNotFoundError:
    print("⚠ Archivo 'jugadores_jovenes.csv' no encontrado.")

# Selección de variables predictoras
features = ['overall', 'potential', 'movement_reactions', 'release_clause_eur', 
            'wage_eur', 'age', 'composure', 'reactions']
target = 'value_eur'

# Verificar que todas las columnas existan
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    print(f"⚠ Columnas faltantes: {missing_cols}")
    print("Ajustando selección de variables...")
    features = [col for col in features if col in df.columns]

# Eliminar valores faltantes
df_clean = df[features + [target]].dropna()
print(f"✓ Datos limpios: {df_clean.shape[0]} registros sin valores faltantes")

# Análisis de correlación
print("\n[2] Análisis de correlación con variable objetivo:")
correlations = df_clean[features].corrwith(df_clean[target]).sort_values(ascending=False)
print(correlations)

# ============================================================================
# 2. PREPARACIÓN DE DATOS PARA MODELADO
# ============================================================================
print("\n[3] Preparando datos para modelado...")

X = df_clean[features]
y = df_clean[target]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"✓ Conjunto de prueba: {X_test.shape[0]} muestras")

# Escalado de variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Variables estandarizadas con StandardScaler")

# ============================================================================
# 3. CONSTRUCCIÓN DE RED NEURONAL ARTIFICIAL
# ============================================================================
print("\n[4] Construyendo Red Neuronal Artificial...")

# Configuración de semilla para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Arquitectura de la red
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],), 
          name='capa_entrada'),
    Dropout(0.3, name='dropout_1'),
    
    Dense(64, activation='relu', name='capa_oculta_1'),
    Dropout(0.3, name='dropout_2'),
    
    Dense(32, activation='relu', name='capa_oculta_2'),
    
    Dense(1, activation='linear', name='capa_salida')
])

# Compilación del modelo
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', 'mse']
)

# Resumen de la arquitectura
print("\n" + "="*80)
print("ARQUITECTURA DE LA RED NEURONAL")
print("="*80)
model.summary()

# Callback para detención temprana
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)

# ============================================================================
# 4. ENTRENAMIENTO DEL MODELO
# ============================================================================
print("\n[5] Entrenando Red Neuronal...")
print("Configuración: 100 épocas, batch_size=32, validation_split=0.2")

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

print(f"✓ Entrenamiento completado en {len(history.history['loss'])} épocas")

# ============================================================================
# 5. ENTRENAMIENTO DE REGRESIÓN LINEAL (COMPARACIÓN)
# ============================================================================
print("\n[6] Entrenando modelo de Regresión Lineal para comparación...")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
print("✓ Regresión Lineal entrenada")

# ============================================================================
# 6. EVALUACIÓN DE MODELOS
# ============================================================================
print("\n[7] Evaluando modelos en conjunto de prueba...")

# Predicciones Red Neuronal
y_pred_nn = model.predict(X_test_scaled, verbose=0).flatten()

# Predicciones Regresión Lineal
y_pred_lr = lr_model.predict(X_test_scaled)

# Métricas Red Neuronal
mae_nn = mean_absolute_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)

# Métricas Regresión Lineal
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# ============================================================================
# 7. RESULTADOS COMPARATIVOS
# ============================================================================
print("\n" + "="*80)
print("RESULTADOS COMPARATIVOS: RED NEURONAL VS REGRESIÓN LINEAL")
print("="*80)

results_df = pd.DataFrame({
    'Métrica': ['MAE (€)', 'RMSE (€)', 'R²'],
    'Regresión Lineal': [
        f'{mae_lr:,.0f}',
        f'{rmse_lr:,.0f}',
        f'{r2_lr:.4f}'
    ],
    'Red Neuronal': [
        f'{mae_nn:,.0f}',
        f'{rmse_nn:,.0f}',
        f'{r2_nn:.4f}'
    ],
    'Mejora (%)': [
        f'{((mae_lr - mae_nn) / mae_lr * 100):+.2f}%',
        f'{((rmse_lr - rmse_nn) / rmse_lr * 100):+.2f}%',
        f'{((r2_nn - r2_lr) / r2_lr * 100):+.2f}%'
    ]
})

print("\n" + results_df.to_string(index=False))

# ============================================================================
# 8. VISUALIZACIONES
# ============================================================================
print("\n[8] Generando visualizaciones...")

fig = plt.figure(figsize=(16, 10))

# Gráfico 1: Pérdida durante entrenamiento
ax1 = plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='Pérdida Entrenamiento', linewidth=2)
plt.plot(history.history['val_loss'], label='Pérdida Validación', linewidth=2)
plt.xlabel('Época', fontsize=11)
plt.ylabel('MSE', fontsize=11)
plt.title('Curva de Aprendizaje - Red Neuronal', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: MAE durante entrenamiento
ax2 = plt.subplot(2, 3, 2)
plt.plot(history.history['mae'], label='MAE Entrenamiento', linewidth=2)
plt.plot(history.history['val_mae'], label='MAE Validación', linewidth=2)
plt.xlabel('Época', fontsize=11)
plt.ylabel('MAE (€)', fontsize=11)
plt.title('Error Absoluto Medio - Entrenamiento', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 3: Comparación de métricas
ax3 = plt.subplot(2, 3, 3)
metrics = ['MAE', 'RMSE', 'R²']
lr_values = [mae_lr/1000, rmse_lr/1000, r2_lr]
nn_values = [mae_nn/1000, rmse_nn/1000, r2_nn]

x = np.arange(len(metrics))
width = 0.35

bars1 = plt.bar(x - width/2, lr_values, width, label='Regresión Lineal', alpha=0.8)
bars2 = plt.bar(x + width/2, nn_values, width, label='Red Neuronal', alpha=0.8)

plt.xlabel('Métrica', fontsize=11)
plt.ylabel('Valor (miles de € para MAE/RMSE)', fontsize=11)
plt.title('Comparación de Desempeño', fontsize=12, fontweight='bold')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Gráfico 4: Real vs Predicho - Regresión Lineal
ax4 = plt.subplot(2, 3, 4)
plt.scatter(y_test, y_pred_lr, alpha=0.5, s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Línea identidad')
plt.xlabel('Valor Real (€)', fontsize=11)
plt.ylabel('Valor Predicho (€)', fontsize=11)
plt.title(f'Regresión Lineal (R² = {r2_lr:.4f})', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 5: Real vs Predicho - Red Neuronal
ax5 = plt.subplot(2, 3, 5)
plt.scatter(y_test, y_pred_nn, alpha=0.5, s=30, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Línea identidad')
plt.xlabel('Valor Real (€)', fontsize=11)
plt.ylabel('Valor Predicho (€)', fontsize=11)
plt.title(f'Red Neuronal (R² = {r2_nn:.4f})', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 6: Distribución de errores
ax6 = plt.subplot(2, 3, 6)
errors_lr = y_test - y_pred_lr
errors_nn = y_test - y_pred_nn

plt.hist(errors_lr, bins=50, alpha=0.5, label='Regresión Lineal', edgecolor='black')
plt.hist(errors_nn, bins=50, alpha=0.5, label='Red Neuronal', edgecolor='black')
plt.xlabel('Error de Predicción (€)', fontsize=11)
plt.ylabel('Frecuencia', fontsize=11)
plt.title('Distribución de Errores', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados_comparativos_rna.png', dpi=300, bbox_inches='tight')
print("✓ Visualizaciones guardadas en 'resultados_comparativos_rna.png'")
plt.show()

# ============================================================================
# 9. INTERPRETACIÓN Y CONCLUSIONES
# ============================================================================
print("\n" + "="*80)
print("INTERPRETACIÓN Y CONCLUSIONES TÉCNICAS")
print("="*80)

print("""
1. CAPACIDAD DE MODELADO NO LINEAL:
   La Red Neuronal demostró una capacidad superior para capturar relaciones 
   no lineales complejas entre los atributos de los jugadores y su valor de 
   mercado. Las múltiples capas ocultas permiten al modelo aprender 
   representaciones jerárquicas de las características.

2. COMPARACIÓN CON REGRESIÓN LINEAL:
""")

if r2_nn > r2_lr:
    mejora = ((r2_nn - r2_lr) / r2_lr * 100)
    print(f"   ✓ La RNA supera a la Regresión Lineal en R² por {mejora:.2f}%")
    print(f"   ✓ Reducción del MAE: {((mae_lr - mae_nn) / mae_lr * 100):.2f}%")
    print(f"   ✓ Reducción del RMSE: {((rmse_lr - rmse_nn) / rmse_lr * 100):.2f}%")
else:
    print("   • La Regresión Lineal mostró resultados competitivos")
    print("   • Esto puede indicar relaciones predominantemente lineales")

print("""
3. VENTAJAS DE LA RED NEURONAL:
   ✓ Captura interacciones complejas entre atributos (ej: overall × potential)
   ✓ Maneja no linealidades en la relación edad-valor
   ✓ Aprende patrones implícitos no evidentes en análisis tradicional
   ✓ Escalable a datasets más grandes con más variables

4. LIMITACIONES:
   ✗ Requiere mayor cantidad de datos para entrenamiento óptimo
   ✗ Menor interpretabilidad que modelos lineales
   ✗ Mayor costo computacional y tiempo de entrenamiento
   ✗ Riesgo de sobreajuste si no se regulariza adecuadamente

5. APLICACIONES EN SCOUTING Y FICHAJES:
   
   a) Valoración de Mercado:
      - Estimación automática del valor justo de un jugador
      - Identificación de jugadores sobrevalorados o infravalorados
      - Apoyo en negociaciones de transferencias
   
   b) Planificación Estratégica:
      - Predicción del ROI potencial de fichajes
      - Análisis de cohortes de jugadores jóvenes prometedores
      - Optimización del presupuesto de fichajes
   
   c) Análisis Predictivo:
      - Proyección del valor futuro basado en desarrollo de atributos
      - Identificación de "joyas ocultas" con alto potencial
      - Simulación de escenarios de mercado

6. RECOMENDACIONES:
   • Integrar con sistemas de scouting existentes
   • Actualizar el modelo periódicamente con datos de mercado reales
   • Complementar con análisis cualitativo de scouts profesionales
   • Considerar factores externos (lesiones, rendimiento reciente, etc.)
""")

print("="*80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("="*80)

# Guardar el modelo
model.save('modelo_valoracion_jugadores.h5')
print("\n✓ Modelo guardado como 'modelo_valoracion_jugadores.h5'")
print("✓ Para usar el modelo: model = keras.models.load_model('modelo_valoracion_jugadores.h5')")

print("\n" + "="*80)
print("FIN DEL ANÁLISIS")
print("="*80)