# TFM — Imputación y clustering simultáneos mediante optimización

Repositorio reproducible del Trabajo Fin de Máster de Ignacio González Matilla. El proyecto implementa un modelo conjunto de imputación y clustering con distancia L1, selección de un imputador candidato por variable, centros observacionales y formulaciones diferenciadas para TRAIN y TEST.

El repositorio contiene dos modos de ejecución:

1. **Toy confirmatorio**: experimento sintético utilizado para validar la formulación, estudiar la normalización de la función objetivo y analizar el efecto de `rho`, `lambda`, la geometría y la calidad de resolución de Gurobi.
2. **Caso aplicado Wholesale**: segmentación de clientes B2B del conjunto UCI *Wholesale customers*, con 440 observaciones y seis variables de gasto anual.

## Diseño TRAIN–TEST

- No existe un conjunto de validación independiente en la versión cerrada del proyecto.
- `rho` y `lambda` se fijan a partir de la campaña toy antes de ejecutar el caso aplicado.
- TEST no se utiliza para elegir una configuración ganadora.
- Las máscaras de missing eliminan un número exacto de celdas y sus semillas solo cambian la localización de las ausencias.
- La referencia principal es una p-mediana L1 completa; PAM-Manhattan y k-means se conservan como referencias adicionales.

## Función objetivo normalizada

En TRAIN se minimiza:

```math
\frac{1}{nd}\sum_i d_i
+ \rho \frac{1}{|\Omega_{\mathrm{mis}}|}
  \sum_{(i,\ell)\in\Omega_{\mathrm{mis}}} u_{i\ell}
+ \lambda \frac{1}{K}\sum_j q_j y_j,
```

donde `q_j` es la proporción de dimensiones originalmente ausentes del candidato a centro `j`. En TEST los centros y los métodos de imputación llegan fijados desde TRAIN, por lo que no aparece el término `lambda`.

## Caso aplicado: UCI Wholesale Customers

El dataset se descarga mediante el paquete oficial [`ucimlrepo`](https://github.com/uci-ml-repo/ucimlrepo):

```python
from ucimlrepo import fetch_ucirepo

wholesale_customers = fetch_ucirepo(id=292)
X = wholesale_customers.data.features
y = wholesale_customers.data.targets
```

El cargador del proyecto concatena `X` e `y`, valida el esquema y añade un identificador estable `customer_id`.

- Observaciones: 440 clientes.
- Variables de clustering: `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper` y `Delicassen`.
- Variables externas: `Channel` y `Region`.
- Split: 70 % TRAIN y 30 % TEST.
- Semilla del split: `42`.
- Estratificación: combinación `Channel × Region`.
- Preprocesamiento fijado: `RobustScaler`, ajustado únicamente con TRAIN completo antes de introducir missing artificial.
- Número de clústeres fijado: `K=4`.

Cita del dataset:

> Cardoso, M. (2013). *Wholesale customers* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5030X

El dataset tiene licencia CC BY 4.0. Los datos no se almacenan como entrada versionada: se descargan desde UCI y cada ejecución registra metadatos, hash SHA-256, split, configuración y commit de Git.

## Instalación recomendada

Se recomienda Python 3.11 de 64 bits.

### Conda

```bash
conda env create -f environment.yml
conda activate tfm-joint-clustering
pip install -e .
```

### venv y pip

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Gurobi necesita una licencia válida para modelos que superen el límite de la licencia incluida con `pip`:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

## Ejecución

### Toy de comprobación

```bash
jic-run --config configs/toy_smoke.yaml
```

### Toy completo

```bash
jic-run --config configs/toy_full.yaml
```

### Smoke del caso Wholesale

```bash
jic-run --config configs/wholesale_smoke.yaml
```

El smoke reproduce primero el split completo 308/132 y extrae después submuestras estratificadas de 24 clientes TRAIN y 12 TEST. Su finalidad es comprobar todo el recorrido técnico, no obtener resultados científicos.

### Experimento Wholesale final

1. Copiar `configs/wholesale_final.template.yaml` a `configs/wholesale_final.yaml`.
2. Introducir los valores normalizados definitivos de `rho` y `lambda` obtenidos en los toys.
3. Revisar los límites de tiempo de Gurobi.
4. Ejecutar:

```bash
jic-run --config configs/wholesale_final.yaml
```

La plantilla final se entrega deliberadamente con listas vacías de hiperparámetros para impedir una ejecución accidental antes de cerrar su elección.

## Coste computacional

La comparación temporal distingue:

### Método propuesto

```math
T_{\mathrm{propuesto}}
= T_{\mathrm{candidatos}}
+ T_{\mathrm{TRAIN\ completo}}
+ T_{\mathrm{TEST\ completo}}.
```

Los tiempos completos de TRAIN y TEST incluyen construcción del modelo, optimización y extracción de resultados. También se conservan por separado el tiempo de pared interno de Gurobi, nodos, `Work`, número de variables, binarias, restricciones, `ObjVal`, `ObjBound` y `MIPGap`.

### Baselines

```math
T_{\mathrm{baseline}}
= T_{\mathrm{imputación}}
+ T_{\mathrm{clustering\ y\ asignación}}.
```

`runtime_accounting.csv` contiene tiempos parciales, tiempo de pipeline y tiempo end-to-end, incluyendo como columnas separadas los costes comunes de carga, preprocesamiento y generación de la máscara.

## Métricas comerciales

`Channel` y `Region` no intervienen en las distancias ni en la función objetivo. Se utilizan únicamente para estratificar la partición y para interpretar externamente los segmentos.

Para cada modelo se generan:

- número y proporción de clientes por clúster;
- gasto medio, mediano y total por categoría;
- contribución de cada segmento al gasto total;
- mezcla de categorías de producto;
- composición y lift de `Channel`;
- composición y lift de `Region`;
- chi-cuadrado y V de Cramér corregido entre clúster y variables externas;
- asignación individual de los clientes de TRAIN y TEST.

Estas medidas evalúan diferenciación e interpretabilidad comercial. No demuestran incremento de ROI, ventas o conversión, porque el dataset no contiene resultados de campañas.

## Salidas principales

Cada ejecución crea una carpeta con fecha y hora en `results/` y guarda configuración, manifiesto, logs, tablas crudas, agregaciones, modelos y figuras.

### Técnicas

- `proposed_test_results.csv`
- `baseline_test_results.csv`
- `reference_audit.csv`
- `solver_diagnostics.csv`
- `objective_components.csv`
- `center_diagnostics.csv`
- `method_selection.csv`
- `candidate_audit.csv`
- `missing_masks_long.csv`
- `labels_long.csv`
- `centers_long.csv`

### Aplicación Wholesale

- `wholesale_source_snapshot.csv`
- `wholesale_variables.csv`
- `preprocessing_parameters.csv`
- `split_membership.csv`
- `runtime_accounting.csv`
- `marketing_assignments.csv`
- `marketing_cluster_profiles.csv`
- `marketing_association.csv`
- `wholesale_metadata.json`

Véanse `docs/methodology.md` y `docs/output_tables.md`.

## Tests y estilo

```bash
pytest
ruff check src tests scripts
```

Los tests de datos utilizan objetos UCI simulados, por lo que no dependen de conexión a Internet. Los tests que requieren Gurobi se omiten cuando `gurobipy` o una licencia válida no están disponibles.
