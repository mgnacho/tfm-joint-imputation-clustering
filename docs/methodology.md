# Metodología del experimento final

## Pregunta

Se estudia si una formulación simultánea de imputación y clustering L1 recupera la partición que la propia p-mediana L1 encontraría con datos completos y cómo se compara con procedimientos secuenciales.

## TRAIN–TEST

Los centros, los imputadores candidatos y los métodos seleccionados se aprenden en TRAIN. En TEST, el modelo simultáneo mantiene fijos los centros y el método por variable, pero optimiza conjuntamente la imputación de cada celda ausente y la asignación al centro.

## Selección por variable

La variable binaria `t[r, ell]` selecciona un imputador por variable. No es una selección por celda. Esta decisión reduce el tamaño del MILP y mantiene una interpretación clara.

## Referencia L1

La referencia L1 completa se resuelve con una formulación p-mediana independiente, sin variables de imputación ni binarias de selección de métodos. Las distancias completas son constantes y el objetivo es la distancia Manhattan media por coordenada.

## Missingness

Se elimina exactamente `round(rate*n*d)` celdas. La máscara se rechaza si deja una fila sin ninguna coordenada observada o una variable sin observaciones. Todas las máscaras se exportan.

## Métricas

Métrica principal:

- ARI frente a la referencia L1 completa.

Secundarias:

- ARI frente a PAM completo, k-means completo y etiquetas generadoras;
- RMSE y MAE únicamente en celdas eliminadas;
- NRMSE por variable usando escalas de TRAIN completo;
- silhouette Manhattan y euclídea sobre `X_test_complete`, común a todos;
- compactación L1 común.

La silhouette sobre la matriz imputada propia se conserva solo como diagnóstico secundario.

## Optimalidad

`MIPGap=0.02` significa que Gurobi puede finalizar con estado `OPTIMAL` cuando certifica el objetivo dentro de una tolerancia relativa del 2 %. Se reportan además `ObjVal`, `ObjBound`, gap absoluto, tiempo, nodos, número de soluciones y work. Las conclusiones se presentan para todos los incumbentes factibles y para subconjuntos con gaps más pequeños.

## Lambda

`lambda=0` es el modelo original. Los valores positivos son una extensión regularizada que penaliza la proporción de missing en los centros seleccionados. Su efecto estadístico y computacional se estudia por separado.

## Baselines

Los imputadores se ajustan solo con TRAIN y transforman TEST. Después se ajusta k-means o PAM en TRAIN imputado y se asigna TEST imputado. No se elige un único baseline mirando TEST; se reportan todos. Una eventual tabla de “oracle baseline” se etiqueta expresamente como resumen descriptivo a posteriori.
