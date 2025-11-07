"""
Script para CONTINUAR el Entrenamiento de un Modelo Ya Guardado
Útil cuando quieres mejorar un modelo existente sin empezar de cero
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

print("="*80)
print("CONTINUAR ENTRENAMIENTO DE MODELO EXISTENTE")
print("="*80)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# ¿Qué modelo quieres seguir entrenando?
MODELO_EXISTENTE = 'modelo_valoracion_jugadores.h5'

# ¿Cuántas épocas más quieres entrenar?
EPOCAS_ADICIONALES = 50

# ¿Quieres ajustar el learning rate? (None = mantener el actual)
NUEVO_LEARNING_RATE = None  # o 0.0001 para hacerlo más fino

print(f"\n📁 Modelo a continuar: {MODELO_EXISTENTE}")
print(f"🔄 Épocas adicionales: {EPOCAS_ADICIONALES}")

# ============================================================================
# VERIFICAR SI EL MODELO EXISTE
# ============================================================================
if not os.path.exists(MODELO_EXISTENTE):
    print(f"\n❌ ERROR: No se encontró el archivo '{MODELO_EXISTENTE}'")
    print("   Primero debes entrenar un modelo inicial.")
    exit()

# ============================================================================
# CARGAR DATOS (MISMOS QUE EL ENTRENAMIENTO ORIGINAL)
# ============================================================================
print("\n[1] Cargando datos...")

try:
    df = pd.read_csv('jugadores_jovenes.csv')
except:
    # Generar datos sintéticos (MISMA SEMILLA que el entrenamiento original)
    np.random.seed(42)
    n_samples = 2000
    df = pd.DataFrame({
        'overall': np.random.randint(60, 90, n_samples),
        'potential': np.random.randint(65, 95, n_samples),
        'age': np.random.randint(18, 24, n_samples),
        'movement_reactions': np.random.randint(60, 90, n_samples),
        'composure': np.random.randint(55, 85, n_samples),
        'reactions': np.random.randint(60, 90, n_samples),
    })
    df['wage_eur'] = (df['overall'] * 1000 + df['potential'] * 800 + 
                      np.random.normal(0, 5000, n_samples)).clip(lower=5000)
    df['release_clause_eur'] = (df['overall'] ** 2 * 10000 + 
                                 df['potential'] ** 2 * 8000 + 
                                 np.random.normal(0, 500000, n_samples)).clip(lower=100000)
    df['value_eur'] = (
        df['overall'] ** 2.2 * 5000 +
        df['potential'] ** 2 * 4000 +
        df['wage_eur'] * 15 +
        df['release_clause_eur'] * 0.3 +
        df['movement_reactions'] * 20000 +
        df['composure'] * 15000 +
        df['reactions'] * 18000 -
        (df['age'] - 20) ** 2 * 100000 +
        np.random.normal(0, 300000, n_samples)
    ).clip(lower=50000)

features = ['overall', 'potential', 'movement_reactions', 'release_clause_eur', 
            'wage_eur', 'age', 'composure', 'reactions']
target = 'value_eur'

X = df[features]
y = df[target]

# División (MISMA SEMILLA)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Datos preparados: {X_train.shape[0]} train, {X_test.shape[0]} test")

# ============================================================================
# CARGAR MODELO EXISTENTE
# ============================================================================
print(f"\n[2] Cargando modelo existente...")

modelo = keras.models.load_model(MODELO_EXISTENTE)
print("✓ Modelo cargado exitosamente")

# Evaluar rendimiento ANTES de continuar entrenamiento
print("\n📊 Evaluación ANTES de continuar entrenamiento:")
y_pred_antes = modelo.predict(X_test_scaled, verbose=0).flatten()
mae_antes = mean_absolute_error(y_test, y_pred_antes)
rmse_antes = np.sqrt(mean_squared_error(y_test, y_pred_antes))
r2_antes = r2_score(y_test, y_pred_antes)

print(f"   MAE:  €{mae_antes:,.0f}")
print(f"   RMSE: €{rmse_antes:,.0f}")
print(f"   R²:   {r2_antes:.4f}")

# ============================================================================
# AJUSTAR LEARNING RATE (OPCIONAL)
# ============================================================================
if NUEVO_LEARNING_RATE is not None:
    print(f"\n[3] Ajustando learning rate a {NUEVO_LEARNING_RATE}...")
    keras.backend.set_value(modelo.optimizer.learning_rate, NUEVO_LEARNING_RATE)
    print("✓ Learning rate ajustado")

# ============================================================================
# CONTINUAR ENTRENAMIENTO
# ============================================================================
print(f"\n[4] Continuando entrenamiento por {EPOCAS_ADICIONALES} épocas más...")

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'modelo_mejorado.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

# Continuar entrenamiento
historia = modelo.fit(
    X_train_scaled, y_train,
    epochs=EPOCAS_ADICIONALES,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1  # Mostrar progreso
)

print(f"✓ Entrenamiento adicional completado")

# ============================================================================
# EVALUACIÓN DESPUÉS DEL ENTRENAMIENTO ADICIONAL
# ============================================================================
print("\n📊 Evaluación DESPUÉS de continuar entrenamiento:")
y_pred_despues = modelo.predict(X_test_scaled, verbose=0).flatten()
mae_despues = mean_absolute_error(y_test, y_pred_despues)
rmse_despues = np.sqrt(mean_squared_error(y_test, y_pred_despues))
r2_despues = r2_score(y_test, y_pred_despues)

print(f"   MAE:  €{mae_despues:,.0f}")
print(f"   RMSE: €{rmse_despues:,.0f}")
print(f"   R²:   {r2_despues:.4f}")

# ============================================================================
# COMPARACIÓN
# ============================================================================
print("\n" + "="*80)
print("COMPARACIÓN: ANTES vs DESPUÉS")
print("="*80)

mejora_mae = ((mae_antes - mae_despues) / mae_antes * 100)
mejora_rmse = ((rmse_antes - rmse_despues) / rmse_antes * 100)
mejora_r2 = ((r2_despues - r2_antes) / r2_antes * 100)

resultados = pd.DataFrame({
    'Métrica': ['MAE (€)', 'RMSE (€)', 'R²'],
    'Antes': [
        f'{mae_antes:,.0f}',
        f'{rmse_antes:,.0f}',
        f'{r2_antes:.4f}'
    ],
    'Después': [
        f'{mae_despues:,.0f}',
        f'{rmse_despues:,.0f}',
        f'{r2_despues:.4f}'
    ],
    'Mejora': [
        f'{mejora_mae:+.2f}%' if mejora_mae > 0 else f'{mejora_mae:.2f}%',
        f'{mejora_rmse:+.2f}%' if mejora_rmse > 0 else f'{mejora_rmse:.2f}%',
        f'{mejora_r2:+.2f}%' if mejora_r2 > 0 else f'{mejora_r2:.2f}%'
    ]
})

print("\n" + resultados.to_string(index=False))

# ============================================================================
# VISUALIZACIÓN
# ============================================================================
print("\n[5] Generando visualización...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Gráfico 1: Curva de entrenamiento adicional
axes[0].plot(historia.history['loss'], label='Loss', linewidth=2)
axes[0].plot(historia.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Época (adicional)')
axes[0].set_ylabel('MSE')
axes[0].set_title('Entrenamiento Adicional', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Comparación de métricas
metricas = ['MAE', 'RMSE', 'R²']
antes_vals = [mae_antes/1000, rmse_antes/1000, r2_antes]
despues_vals = [mae_despues/1000, rmse_despues/1000, r2_despues]

x = np.arange(len(metricas))
width = 0.35

axes[1].bar(x - width/2, antes_vals, width, label='Antes', alpha=0.8)
axes[1].bar(x + width/2, despues_vals, width, label='Después', alpha=0.8)
axes[1].set_ylabel('Valor (miles € para MAE/RMSE)')
axes[1].set_title('Comparación de Métricas', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metricas)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# Gráfico 3: Real vs Predicho (después)
axes[2].scatter(y_test, y_pred_despues, alpha=0.5, s=30)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Línea perfecta')
axes[2].set_xlabel('Valor Real (€)')
axes[2].set_ylabel('Valor Predicho (€)')
axes[2].set_title(f'Predicciones Mejoradas (R²={r2_despues:.4f})', fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mejora_modelo.png', dpi=300, bbox_inches='tight')
print("✓ Visualización guardada en 'mejora_modelo.png'")
plt.show()

# ============================================================================
# DECISIÓN Y GUARDADO
# ============================================================================
print("\n" + "="*80)
print("DECISIÓN FINAL")
print("="*80)

if r2_despues > r2_antes:
    print("\n✅ El modelo MEJORÓ después del entrenamiento adicional")
    print(f"   Mejora en R²: {mejora_r2:+.2f}%")
    print("\n💾 Se recomienda usar el modelo mejorado:")
    print("   - Archivo: modelo_mejorado.h5")
    
    # Sobrescribir el modelo original (opcional)
    respuesta = input("\n¿Deseas reemplazar el modelo original? (s/n): ")
    if respuesta.lower() == 's':
        modelo.save(MODELO_EXISTENTE)
        print(f"✓ Modelo original actualizado: {MODELO_EXISTENTE}")
else:
    print("\n⚠️  El modelo NO mejoró significativamente")
    print(f"   Cambio en R²: {mejora_r2:.2f}%")
    print("\n📋 Recomendaciones:")
    print("   - El modelo original ya estaba bien ajustado")
    print("   - Considera cambiar la arquitectura en vez de más épocas")
    print("   - Prueba con un learning rate más bajo")
    print("   - Verifica si hay overfitting en las curvas")

print("\n" + "="*80)
print("PROCESO COMPLETADO")
print("="*80)