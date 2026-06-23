# Diccionario de salidas

## Tablas técnicas comunes

### `proposed_test_results.csv`

Una fila por escenario y combinación `(rho, lambda)`. Incluye métricas TEST, calidad del solver, componentes normalizados, centros incompletos y tiempos completos del pipeline.

### `baseline_test_results.csv`

Una fila por escenario, imputador y algoritmo (`kmeans` o `pam`). Incluye métricas TEST, tiempos de imputación, clustering y end-to-end.

### `reference_audit.csv`

Concordancia entre k-means, PAM y p-mediana L1 completos, silhouettes y certificación de la referencia L1.

### `solver_diagnostics.csv`

Status, objetivo, cota, gap, tiempo de Gurobi, tiempo total de llamada, nodos, `Work`, número de variables, binarias y restricciones.

### `objective_components.csv`

Componentes crudos, normalizados y ponderados de la función objetivo.

### `center_diagnostics.csv`

Identidad de los centros, dimensiones originalmente ausentes y desplazamientos L1/L2 respecto a la observación completa.

### `candidate_audit.csv`

Métodos candidatos, éxito, exclusiones, advertencias, errores y tiempos.

### `missing_masks_long.csv`

Una fila por celda eliminada, con `customer_id` en el caso aplicado.

### `labels_long.csv` y `centers_long.csv`

Trazabilidad de etiquetas y centros para referencias, baselines y método propuesto.

## Tablas específicas del caso Wholesale

### `wholesale_source_snapshot.csv`

Copia de auditoría de la descarga UCI utilizada en la ejecución. No es una entrada del pipeline.

### `wholesale_variables.csv`

Información de variables suministrada por `ucimlrepo`.

### `preprocessing_parameters.csv`

Mediana e IQR aprendidos por `RobustScaler` en TRAIN y orden fijo de las seis variables.

### `split_membership.csv`

Pertenencia de cada cliente a TRAIN o TEST, canal, región, semilla, método de estratificación y hash del split completo.

### `runtime_accounting.csv`

Contabilidad comparable del coste computacional:

- costes comunes;
- tiempo de imputación;
- generación de todos los candidatos;
- clustering;
- llamadas completas TRAIN/TEST;
- tiempo de pipeline;
- tiempo end-to-end.

### `marketing_assignments.csv`

Una fila por cliente y modelo con split, clúster, variables externas, gastos originales y metadatos del escenario.

### `marketing_cluster_profiles.csv`

Una fila por clúster y modelo. Contiene tamaño, cuota de clientes, cuota de gasto, medias, medianas, sumas, mezcla de producto, composición y lift de canal y región.

### `marketing_association.csv`

Chi-cuadrado, grados de libertad, p-valor y V de Cramér corregido entre etiquetas de clúster y `Channel`/`Region`.

### `wholesale_metadata.json`

ID de UCI, metadatos originales y hash SHA-256 de los datos descargados.

## Tablas agregadas

### `proposed_test_agg.csv`

Resumen por `(d, missing_rate, rho, lambda)`.

### `baseline_test_agg.csv`

Resumen por `(d, missing_rate, method, cluster_algo)`.

### `gap_filtered_summary.csv`

Resultados del propuesto para todos los factibles y subconjuntos con `gap<=20%`, `10%`, `5%` y certificado al 2 %.

### `paired_proposed_vs_baselines.csv`

Comparaciones pareadas entre cada configuración propuesta y cada baseline.

### `oracle_baseline_by_scenario.csv`

Mejor baseline observado en TEST por ARI y por RMSE. Es un resumen descriptivo a posteriori, no un procedimiento de selección.
