# Diccionario de salidas

## Tablas crudas

### `proposed_test_results.csv`
Una fila por escenario y combinación `(rho, lambda)`. Contiene métricas TEST, calidad del solver TRAIN/TEST, centros incompletos y componentes del objetivo.

### `baseline_test_results.csv`
Una fila por escenario, imputador y algoritmo (`kmeans` o `pam`). Incluye métricas TEST y tiempos separados de imputación y clustering.

### `reference_audit.csv`
ARI cruzados, ARI frente a `y_true`, silhouettes y certificación de la referencia L1.

### `solver_diagnostics.csv`
Diagnóstico técnico de cada modelo Gurobi.

### `objective_components.csv`
Componentes crudos, normalizados y ponderados del objetivo propuesto.

### `center_diagnostics.csv`
Índice local y global, dimensiones ausentes, valores completos e imputados, desplazamientos L1/L2.

### `candidate_audit.csv`
Métodos candidatos construidos, advertencias capturadas, errores y tiempo.

### `missing_masks_long.csv`
Una fila por celda eliminada, suficiente para reproducir exactamente cada máscara.

## Tablas agregadas

### `proposed_test_agg.csv`
Media, desviación, mínimo, máximo y número de repeticiones por `(d, missing_rate, rho, lambda)`.

### `baseline_test_agg.csv`
Resumen por `(d, missing_rate, method, cluster_algo)`.

### `gap_filtered_summary.csv`
Resultados del modelo propuesto para todos los factibles y para `gap<=20%`, `10%`, `5%` y certificados al `2%`.

### `paired_proposed_vs_baselines.csv`
Comparaciones pareadas por escenario entre cada configuración propuesta y cada baseline.

### `oracle_baseline_by_scenario.csv`
Mejor baseline observado en TEST por ARI y por RMSE. Es un límite descriptivo a posteriori y no un método preseleccionado.
