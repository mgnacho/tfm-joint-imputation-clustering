# TFM — Imputación y clustering simultáneos mediante optimización

Repositorio reproducible del experimento toy final **TRAIN–TEST**. El método propuesto selecciona, de forma conjunta, valores imputados, un método candidato de imputación por variable, centros observados y asignaciones de clustering con distancia L1.

## Diseño cerrado

- No existe un conjunto de validación independiente.
- `rho` y `lambda` se fijan antes de ejecutar TEST y todas sus combinaciones se muestran como análisis de sensibilidad.
- Los datos completos, el split y las semillas son fijos.
- `missing_seed` cambia la localización de un número exacto de celdas ausentes, no la cantidad.
- TEST no se utiliza para elegir una configuración ganadora.

## Referencias y métodos

Con datos completos se ajustan tres referencias en TRAIN y se asigna TEST:

1. **K-means completo**: centroides y geometría L2.
2. **PAM completo**: k-medoids estándar, medoides observados y distancia Manhattan.
3. **P-mediana L1 completa de Gurobi**: referencia principal del modelo propuesto.

Con datos incompletos se comparan:

- modelo simultáneo de Gurobi;
- cada imputador candidato seguido de k-means;
- cada imputador candidato seguido de PAM.

## Función objetivo normalizada

En TRAIN:

\[
\frac{1}{n d}\sum_i d_i
+ \rho\frac{1}{|\Omega_{mis}|}\sum_{(i,\ell)\in\Omega_{mis}}u_{i\ell}
+ \lambda\frac{1}{K}\sum_j q_j y_j,
\]

con `q_j` igual a la proporción de dimensiones originalmente ausentes de la observación candidata a centro.

En TEST los centros y los métodos de imputación llegan fijados desde TRAIN, por lo que no aparece el término `lambda`.

### Recalibración sin utilizar TEST

La rejilla normalizada se obtuvo por transformación algebraica de la rejilla anterior. Al multiplicar la nueva función objetivo por `n*d`, el peso equivalente de imputación es `rho_new*(n*d/n_missing)` y el de centros es `lambda_new*(n*d/K)`. Con missing del 20–30 %, `n_train=42` y `d=2,3`, se fijaron:

- `rho = [0.0025, 0.025]`;
- `lambda = [0, 0.03, 0.075, 0.15, 0.30]`.

No se escogieron observando el nuevo TEST.

## Instalación recomendada

Se recomienda Python 3.11 de 64 bits. `scikit-learn-extra 0.3.0` dispone de ruedas para Python 3.11 y se fija `numpy<2` para evitar incompatibilidades binarias.

### Con Conda

```bash
conda env create -f environment.yml
conda activate tfm-joint-clustering
pip install -e .
```

### Con venv y pip

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

Gurobi necesita una licencia válida para resolver modelos que superen el límite de la licencia incluida con `pip`. Compruebe la instalación con:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

## Ejecución

Prueba de instalación, no destinada a obtener conclusiones:

```bash
jic-run --config configs/toy_smoke.yaml
```

Experimento completo:

```bash
jic-run --config configs/toy_full.yaml
```

Alternativa sin instalar el punto de entrada:

```bash
python scripts/run_toy_experiment.py --config configs/toy_full.yaml
```

Cada ejecución crea una carpeta con fecha y hora dentro de `results/`, guarda la configuración utilizada, un manifiesto del entorno, logs, tablas crudas, tablas agregadas y figuras.

## Tiempo máximo

La configuración completa ejecuta 120 modelos TRAIN del método propuesto:

`2 dimensiones × 2 tasas missing × 3 máscaras × 2 rho × 5 lambda`.

Cada TRAIN dispone de 600 segundos, por lo que el límite secuencial teórico es de unas 20 horas, aunque muchas ejecuciones pueden finalizar antes. `threads: 0` permite que Gurobi use automáticamente los hilos disponibles. No elimina `TimeLimit`.

## Tablas principales

- `proposed_test_results.csv`: todas las configuraciones preespecificadas.
- `baseline_test_results.csv`: todos los imputadores con k-means y PAM.
- `reference_audit.csv`: concordancia entre las tres referencias completas.
- `solver_diagnostics.csv`: estado, objetivo, cota, gap, tiempo, nodos y work.
- `objective_components.csv`: descomposición normalizada de la función objetivo.
- `center_diagnostics.csv`: observabilidad y desplazamiento de centros.
- `method_selection.csv`: imputador seleccionado por variable.
- `candidate_audit.csv`: advertencias, errores y tiempos de los imputadores.
- `missing_masks_long.csv`: máscara exacta de cada escenario.
- `labels_long.csv` y `centers_long.csv`: trazabilidad completa.

Véanse `docs/methodology.md` y `docs/output_tables.md`.

## Tests y estilo

```bash
pytest
ruff check src tests scripts
```

Los tests que requieren Gurobi se omiten automáticamente si `gurobipy` o una licencia no están disponibles.
