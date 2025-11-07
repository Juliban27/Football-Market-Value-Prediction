"""
Script para Entrenar Múltiples Configuraciones de RNA y Comparar Resultados
Útil para experimentación y optimización de hiperparámetros
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENTRENAMIENTO MÚLTIPLE DE MODELOS - EXPERIMENTACIÓN")
print("="*80)

# ============================================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ============================================================================

# Define diferentes configuraciones para probar
experimentos = {
    'Modelo_Baseline': {
        'arquitectura': [128, 64, 32],
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    },
    'Modelo_Profundo': {
        'arquitectura': [256, 128, 64, 32],
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    },
    'Modelo_Menos_Dropout': {
        'arquitectura': [128, 64, 32],
        'dropout': 0.1,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    },
    'Modelo_LR_Bajo': {
        'arquitectura': [128, 64, 32],
        'dropout': 0.3,
        'learning_rate': 0.0001,
        'epochs': 150,
        'batch_size': 32
    },
    'Modelo_Compacto': {
        'arquitectura': [64, 32],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 64
    }
}

# ============================================================================
# CARGA Y PREPARACIÓN DE DATOS (UNA SOLA VEZ)
# ============================================================================
print("\n[1] Preparando datos...")

# Generar datos sintéticos (o cargar desde CSV)
try:
    df = pd.read_csv('jugadores_jovenes.csv')
    print(f"✓ Dataset cargado: {df.shape[0]} registros")
except:
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
    print(f"✓ Dataset sintético generado: {df.shape[0]} registros")

features = ['overall', 'potential', 'movement_reactions', 'release_clause_eur', 
            'wage_eur', 'age', 'composure', 'reactions']
target = 'value_eur'

X = df[features]
y = df[target]

# División y escalado
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================================
# FUNCIÓN PARA CONSTRUIR MODELOS
# ============================================================================
def construir_modelo(config, input_dim):
    """Construye un modelo con la configuración especificada"""
    tf.random.set_seed(42)
    
    model = Sequential()
    
    # Primera capa
    model.add(Dense(config['arquitectura'][0], activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(config['dropout']))
    
    # Capas ocultas adicionales
    for neuronas in config['arquitectura'][1:]:
        model.add(Dense(neuronas, activation='relu'))
        model.add(Dropout(config['dropout']))
    
    # Capa de salida
    model.add(Dense(1, activation='linear'))
    
    # Compilar
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ============================================================================
# ENTRENAR TODOS LOS EXPERIMENTOS
# ============================================================================
print("\n[2] Entrenando múltiples configuraciones...")
print("="*80)

resultados = {}
historias = {}

for nombre, config in experimentos.items():
    print(f"\n🔄 Entrenando: {nombre}")
    print(f"   Arquitectura: {config['arquitectura']}")
    print(f"   Dropout: {config['dropout']} | LR: {config['learning_rate']}")
    
    # Construir modelo
    modelo = construir_modelo(config, X_train_scaled.shape[1])
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Entrenar
    inicio = datetime.now()
    historia = modelo.fit(
        X_train_scaled, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    tiempo_entrenamiento = (datetime.now() - inicio).total_seconds()
    
    # Predecir y evaluar
    y_pred = modelo.predict(X_test_scaled, verbose=0).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Guardar resultados
    resultados[nombre] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Épocas_usadas': len(historia.history['loss']),
        'Tiempo_seg': tiempo_entrenamiento,
        'Parámetros': modelo.count_params()
    }
    
    historias[nombre] = historia.history
    
    # Guardar modelo
    modelo.save(f'modelos/{nombre}.h5')
    
    print(f"   ✓ MAE: €{mae:,.0f} | RMSE: €{rmse:,.0f} | R²: {r2:.4f}")
    print(f"   ✓ Épocas: {len(historia.history['loss'])} | Tiempo: {tiempo_entrenamiento:.1f}s")

# ============================================================================
# COMPARACIÓN DE RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("TABLA COMPARATIVA DE TODOS LOS MODELOS")
print("="*80)

df_resultados = pd.DataFrame(resultados).T
df_resultados['MAE'] = df_resultados['MAE'].apply(lambda x: f'€{x:,.0f}')
df_resultados['RMSE'] = df_resultados['RMSE'].apply(lambda x: f'€{x:,.0f}')
df_resultados['R2'] = df_resultados['R2'].apply(lambda x: f'{x:.4f}')
df_resultados['Tiempo_seg'] = df_resultados['Tiempo_seg'].apply(lambda x: f'{x:.1f}s')

print("\n" + df_resultados.to_string())

# Encontrar mejor modelo
mejor_modelo = max(resultados.items(), key=lambda x: x[1]['R2'])
print(f"\n🏆 MEJOR MODELO: {mejor_modelo[0]}")
print(f"   R² = {mejor_modelo[1]['R2']:.4f}")
print(f"   MAE = €{mejor_modelo[1]['MAE']:,.0f}")

# ============================================================================
# VISUALIZACIÓN COMPARATIVA
# ============================================================================
print("\n[3] Generando visualizaciones comparativas...")

fig = plt.figure(figsize=(18, 10))

# Gráfico 1: Curvas de aprendizaje de todos los modelos
ax1 = plt.subplot(2, 3, 1)
for nombre, historia in historias.items():
    plt.plot(historia['loss'], label=nombre, linewidth=2, alpha=0.7)
plt.xlabel('Época')
plt.ylabel('MSE')
plt.title('Curvas de Aprendizaje - Training Loss', fontweight='bold')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# Gráfico 2: Curvas de validación
ax2 = plt.subplot(2, 3, 2)
for nombre, historia in historias.items():
    plt.plot(historia['val_loss'], label=nombre, linewidth=2, alpha=0.7)
plt.xlabel('Época')
plt.ylabel('MSE')
plt.title('Curvas de Aprendizaje - Validation Loss', fontweight='bold')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

# Gráfico 3: Comparación R²
ax3 = plt.subplot(2, 3, 3)
modelos = list(resultados.keys())
r2_values = [resultados[m]['R2'] for m in modelos]
colors = ['green' if m == mejor_modelo[0] else 'steelblue' for m in modelos]
bars = plt.barh(modelos, r2_values, color=colors, alpha=0.8)
plt.xlabel('R² Score')
plt.title('Comparación de R²', fontweight='bold')
plt.xlim(0, 1)
for i, v in enumerate(r2_values):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center')
plt.grid(True, alpha=0.3, axis='x')

# Gráfico 4: Comparación MAE
ax4 = plt.subplot(2, 3, 4)
mae_values = [resultados[m]['MAE']/1000 for m in modelos]
plt.barh(modelos, mae_values, color='coral', alpha=0.8)
plt.xlabel('MAE (miles de €)')
plt.title('Comparación de MAE', fontweight='bold')
for i, v in enumerate(mae_values):
    plt.text(v + 5, i, f'€{v:.0f}k', va='center')
plt.grid(True, alpha=0.3, axis='x')

# Gráfico 5: Tiempo de entrenamiento vs R²
ax5 = plt.subplot(2, 3, 5)
tiempos = [resultados[m]['Tiempo_seg'] for m in modelos]
r2_vals = [resultados[m]['R2'] for m in modelos]
plt.scatter(tiempos, r2_vals, s=200, alpha=0.6, c=range(len(modelos)), cmap='viridis')
for i, nombre in enumerate(modelos):
    plt.annotate(nombre.split('_')[1], (tiempos[i], r2_vals[i]), 
                fontsize=8, ha='center')
plt.xlabel('Tiempo de Entrenamiento (s)')
plt.ylabel('R² Score')
plt.title('Eficiencia: Tiempo vs Precisión', fontweight='bold')
plt.grid(True, alpha=0.3)

# Gráfico 6: Parámetros del modelo vs R²
ax6 = plt.subplot(2, 3, 6)
params = [resultados[m]['Parámetros']/1000 for m in modelos]
plt.scatter(params, r2_vals, s=200, alpha=0.6, c=range(len(modelos)), cmap='plasma')
for i, nombre in enumerate(modelos):
    plt.annotate(nombre.split('_')[1], (params[i], r2_vals[i]), 
                fontsize=8, ha='center')
plt.xlabel('Parámetros del Modelo (miles)')
plt.ylabel('R² Score')
plt.title('Complejidad vs Precisión', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparacion_experimentos.png', dpi=300, bbox_inches='tight')
print("✓ Gráficos guardados en 'comparacion_experimentos.png'")
plt.show()

# ============================================================================
# GUARDAR REPORTE
# ============================================================================
reporte = {
    'fecha_experimento': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'configuraciones': experimentos,
    'resultados': {k: {
        'MAE': float(v['MAE']),
        'RMSE': float(v['RMSE']),
        'R2': float(v['R2']),
        'Épocas': int(v['Épocas_usadas']),
        'Tiempo': float(v['Tiempo_seg'])
    } for k, v in resultados.items()},
    'mejor_modelo': mejor_modelo[0]
}

with open('reporte_experimentos.json', 'w') as f:
    json.dump(reporte, f, indent=4)

print("\n✓ Reporte guardado en 'reporte_experimentos.json'")

print("\n" + "="*80)
print("RECOMENDACIONES:")
print("="*80)
print(f"""
1. El mejor modelo es: {mejor_modelo[0]}
   - Usa este modelo para producción
   - Archivo: modelos/{mejor_modelo[0]}.h5

2. Si necesitas más precisión:
   - Aumenta las épocas del mejor modelo
   - Prueba con más datos de entrenamiento
   - Ajusta el learning rate más fino

3. Si necesitas más velocidad:
   - Usa el modelo más compacto con buen R²
   - Reduce el batch_size
   - Implementa entrenamiento en GPU

4. Para evitar overfitting:
   - Aumenta el dropout si val_loss > train_loss
   - Usa más datos de entrenamiento
   - Simplifica la arquitectura
""")

print("="*80)
print("EXPERIMENTACIÓN COMPLETADA")
print("="*80)