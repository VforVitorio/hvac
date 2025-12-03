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

```bash
# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso - Un Solo Notebook

Todo el flujo está en un notebook único:

```bash
jupyter notebook HVAC_Digital_Twin.ipynb
```

El notebook incluye:
1. ✅ Consolidación de datos (9 CSVs verano + 6 invierno)
2. ✅ Descubrimiento de ecuaciones físicas (PySINDy)
3. ✅ Creación del modelo híbrido TCN-VAE + SINDy
4. ✅ Entrenamiento
5. ✅ Evaluación y visualización de resultados

**Tiempo estimado:** 20-30 minutos de ejecución completa

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

Generados automáticamente al ejecutar:
├── data/                         # Datos consolidados
├── models/                       # Modelos entrenados
└── results/                      # Gráficos y resultados
```

---

## Archivos Clave

### Notebook Principal
**`HVAC_Digital_Twin.ipynb`** - Todo el flujo completo con explicaciones

### Módulos Python (en `src/`)

| Archivo | Qué hace |
|---------|----------|
| `src/data_consolidation.py` | Junta los 15 CSVs en 2 datasets (verano/invierno) |
| `src/physics_discovery.py` | Descubre ecuaciones con PySINDy |
| `src/tcn_vae.py` | Implementa TCN-VAE (red temporal variacional) |
| `src/hybrid_sindy_tcnvae.py` | Modelo híbrido que fusiona física + ML |

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

## Resultados Esperados

### Métricas Típicas
```
UCAOT (Temp. Salida):  R² > 0.90
UCAOH (Humedad):       R² > 0.85
CPMEP (Potencia):      R² > 0.88
```

### Ventajas del Modelo Híbrido vs Solo ML
- ✅ Más preciso
- ✅ Generaliza mejor
- ✅ Ecuaciones físicas interpretables
- ✅ No hace predicciones imposibles físicamente

---

## Configuración del Modelo

En el notebook puedes ajustar:

```python
config = HybridConfig(
    latent_dim=32,                  # Dimensión latente VAE
    encoder_channels=[32, 64, 128], # Capas TCN
    physics_weight=0.3,             # Peso física (0-1)
    learning_rate=1e-3
)
```

**Valores recomendados:**
- `physics_weight=0.3` - Balance equilibrado
- `physics_weight=0.5` - Más física, más robusto
- `physics_weight=0.1` - Más datos, más flexible

---

## Solución de Problemas

### Error: "CUDA out of memory"
```python
# Usar CPU en vez de GPU
device = 'cpu'
```

### Entrenamiento muy lento
```python
# Reducir épocas
EPOCHS = 30  # en vez de 50-100

# O usar menos datos
summer_sample = summer_df.sample(n=10000)
```

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
