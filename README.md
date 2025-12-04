# Gemelo Digital HVAC - Modelo Híbrido SINDy + TCN-VAE

**Proyecto:** Gemelos Digitales - Cuarto Año
**Objetivo:** Crear gemelo digital de sistema HVAC combinando física + machine learning

---

## Qué hace este proyecto

Crea dos modelos (verano e invierno) que predicen el comportamiento de un sistema HVAC usando:

1. **PySINDy** - Descubre ecuaciones físicas del sistema automáticamente
2. **TCN-VAE** - Red neuronal temporal para capturar dinámicas complejas
3. **Modelo Híbrido** - Combina ambos para mejores resultados

---

## Instalación Rápida

### Opción 1: Con GPU (Recomendado)

```bash
# Crear entorno conda
conda create -n hvac_env python=3.10 -y
conda activate hvac_env

# Instalar PyTorch con CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Instalar resto de dependencias
pip install -r requirements.txt
```

### Opción 2: Solo CPU

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

**Nota:** Para aprovechar GPU NVIDIA, sigue las instrucciones en `requirements.txt` para instalar PyTorch con CUDA.

---

## Uso

### Ver Resultados (Sin Re-entrenar)

El repositorio ya incluye:

- ✅ Modelo entrenado en `models/`
- ✅ Datos consolidados en `data/`
- ✅ Gráficas de resultados en `results/`

Solo abre el notebook para ver el análisis completo:

```bash
jupyter notebook HVAC_Digital_Twin.ipynb
```

### Re-entrenar desde Cero

Ejecuta todas las celdas del notebook:

1. ✅ Consolidación de datos (9 CSVs verano + 6 invierno)
2. ✅ Descubrimiento de ecuaciones físicas (PySINDy)
3. ✅ Creación del modelo híbrido TCN-VAE + SINDy
4. ✅ Entrenamiento (50 épocas)
5. ✅ Evaluación y visualización de resultados

**Tiempo estimado:**

- Con GPU: ~10-15 minutos
- Con CPU: ~30-40 minutos

---

## Estructura del Proyecto

```
hvac/
├── dataset/                      # 15 archivos CSV originales
├── src/                          # Módulos Python
│   ├── __init__.py
│   ├── data_consolidation.py    # Módulo: consolidar CSVs
│   ├── physics_discovery.py     # Módulo: PySINDy
│   ├── tcn_vae.py              # Módulo: TCN-VAE
│   └── hybrid_sindy_tcnvae.py  # Módulo: Modelo híbrido
├── HVAC_Digital_Twin.ipynb      # ⭐ NOTEBOOK PRINCIPAL
├── requirements.txt             # Dependencias
└── README.md                    # Este archivo

Generados (ya incluidos en el repo):
├── data/                         # Datos consolidados
│   ├── hvac_summer_consolidated.csv
│   └── hvac_winter_consolidated.csv
├── models/                       # Modelos entrenados
│   ├── hvac_hybrid_model.pt     # Modelo híbrido (3 MB)
│   ├── scalers.pkl              # Escaladores para normalización
│   ├── summer_equations.txt     # Ecuaciones PySINDy
│   └── summer_coefficients.png  # Visualización coeficientes
└── results/                      # Gráficos de resultados
    ├── exploracion_datos.png    # Análisis exploratorio
    ├── training_history.png     # Curvas de entrenamiento
    └── predicciones.png         # Comparación predicciones vs reales
```

---

## Archivos Clave

### Notebook Principal

**`HVAC_Digital_Twin.ipynb`** - Todo el flujo completo con explicaciones

### Módulos Python (en `src/`)

| Archivo                      | Qué hace                                          |
| ---------------------------- | ------------------------------------------------- |
| `src/data_consolidation.py`  | Junta los 15 CSVs en 2 datasets (verano/invierno) |
| `src/physics_discovery.py`   | Descubre ecuaciones con PySINDy                   |
| `src/tcn_vae.py`             | Implementa TCN-VAE (red temporal variacional)     |
| `src/hybrid_sindy_tcnvae.py` | Modelo híbrido que fusiona física + ML            |

---

## Datos

### Entrada

- **9 experimentos de verano** (~51,000 registros)
- **6 experimentos de invierno** (~14,000 registros)
- **44 variables** por experimento (temperaturas, flujos, presiones, etc.)

### Variables Principales

- `UCAOT` - Temperatura salida aire (principal variable de control)
- `UCAOH` - Humedad salida
- `UCWF` - Flujo de agua
- `CPMEP` - Potencia compresor

---

## Resultados Obtenidos

### Métricas del Modelo Entrenado

```
UCAOT (Temp. Salida):  R² > 0.97
UCAOH (Humedad):       R² > 0.95
UCWOT (Temp. Agua):    R² > 0.96
CPMEP (Potencia):      R² > 0.95
```

**Estado:** Modelo ya entrenado disponible en `models/hvac_hybrid_model.pt`

### Ventajas del Modelo Híbrido vs Solo ML

- ✅ Más preciso
- ✅ Generaliza mejor
- ✅ Ecuaciones físicas interpretables
- ✅ No hace predicciones imposibles físicamente

---

## Configuración del Modelo

En el notebook puedes ajustar:

````python
config = HybridConfig(
## Uso del Modelo Entrenado

Para usar el modelo ya entrenado en tus propias aplicaciones:

```python
import torch
import joblib

# Cargar modelo
model = torch.load('models/hvac_hybrid_model.pt')
model.eval()

# Cargar escaladores
scalers = joblib.load('models/scalers.pkl')
X_scaler = scalers['X_scaler']
y_scaler = scalers['y_scaler']

# Hacer predicciones
# (tu código aquí con inputs normalizados)
````

## Solución de Problemas

### CUDA no disponible

Verifica que tienes:

1. GPU NVIDIA compatible
2. Drivers NVIDIA actualizados
3. PyTorch con CUDA: `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia`

### Error: "CUDA out of memory"

```python
# Reducir batch size en el notebook
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Entrenamiento muy lento (CPU)

````python
# Reducir épocas
EPOCHS = 30  # en vez de 50

# O usar menos datos
summer_sample = summer_df.sample(n=5000)
```educir épocas
EPOCHS = 30  # en vez de 50-100

# O usar menos datos
summer_sample = summer_df.sample(n=10000)
````

### Malos resultados

```python
# Aumentar épocas
EPOCHS = 100

# Ajustar peso de física
config.physics_weight = 0.4
```

---

## Referencias

- **PySINDy**: Brunton et al., "Discovering governing equations from data", PNAS 2016
- **TCN**: Bai et al., "Temporal Convolutional Networks", 2018
- **VAE**: Kingma & Welling, "Auto-Encoding Variational Bayes", 2014
- **Physics-ML**: Willard et al., "Integrating Physics with Machine Learning", 2020

---

## Contacto

Para dudas sobre el proyecto, revisar el notebook principal que tiene todas las explicaciones.

---

**Tarea Universitaria - Gemelos Digitales**
