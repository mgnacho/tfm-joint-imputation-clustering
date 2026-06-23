# Metodología reproducible

## 1. Pregunta de investigación

Se estudia si una formulación simultánea de imputación y clustering L1 puede recuperar la partición que una p-mediana L1 encontraría con datos completos y cómo se compara con procedimientos secuenciales de imputación seguida de clustering.

El caso aplicado traslada esta pregunta a una segmentación de clientes B2B basada en el comportamiento anual de compra.

## 2. Arquitectura TRAIN–TEST

Los centros, los imputadores candidatos y el método seleccionado por variable se aprenden en TRAIN. En TEST, el modelo mantiene fijos los centros y la elección del imputador, pero optimiza conjuntamente la imputación de cada celda ausente y la asignación al centro.

No se utiliza VALIDATION en el diseño cerrado. Los hiperparámetros se fijan a partir de experimentos sintéticos independientes y TEST no participa en su elección.

## 3. Función objetivo

En TRAIN:

```math
\min
\frac{1}{nd}\sum_i d_i
+ \rho\frac{1}{|\Omega_{\mathrm{mis}}|}
\sum_{(i,\ell)\in\Omega_{\mathrm{mis}}}u_{i\ell}
+ \lambda\frac{1}{K}\sum_j q_jy_j.
```

Los términos representan:

1. distancia L1 media por observación y coordenada;
2. desviación media por celda missing respecto al imputador candidato seleccionado;
3. proporción media de dimensiones ausentes en los centros elegidos.

La normalización permite interpretar los pesos con menor dependencia del tamaño muestral, la dimensión, el número de celdas missing y el número de centros. En TEST desaparece `lambda` porque los centros ya llegan fijados.

## 4. Selección por variable

La variable binaria `t[r, ell]` selecciona un imputador candidato por variable, no por celda. Las matrices candidatas se calculan antes del MILP utilizando solo información ajustada en TRAIN. En TEST se aplica el método elegido en TRAIN sin volver a seleccionarlo.

Los candidatos disponibles son:

- media;
- mediana;
- moda redondeada;
- muestreo empírico aleatorio;
- KNN;
- imputación iterativa con Bayesian Ridge;
- imputación iterativa con Random Forest;
- aproximación PMM basada en donantes.

## 5. Referencias y baselines

Con datos completos se ajustan:

1. k-means completo;
2. PAM completo con distancia Manhattan;
3. p-mediana L1 completa de Gurobi, referencia principal.

Con datos incompletos se comparan el método propuesto y cada imputador candidato seguido de k-means o PAM.

## 6. Missingness

Se induce MCAR eliminando exactamente `round(rate*n*d)` celdas. Se rechazan máscaras que dejen una fila completamente ausente o una variable sin observaciones. Las máscaras de TRAIN y TEST se generan con semillas estables y se exportan a formato largo.

## 7. Métricas estadísticas

Métrica externa principal:

- ARI frente a la referencia p-mediana L1 completa.

Métricas secundarias:

- ARI frente a PAM y k-means completos;
- RMSE y MAE solo en celdas eliminadas;
- NRMSE por rango y desviación de TRAIN completo;
- silhouette Manhattan y euclídea sobre `X_test_complete`, común a todos;
- compactación L1 común.

La silhouette sobre la matriz imputada propia se conserva únicamente como diagnóstico secundario.

## 8. Caso UCI Wholesale Customers

### 8.1 Procedencia

El conjunto UCI 292 se descarga con `ucimlrepo==0.0.7`. El cargador concatena `data.features` y `data.targets`, valida las columnas y registra metadatos y un hash SHA-256.

### 8.2 Variables

El clustering utiliza exclusivamente:

- Fresh;
- Milk;
- Grocery;
- Frozen;
- Detergents_Paper;
- Delicassen.

`Channel` y `Region` se reservan para estratificación e interpretación externa. `Total_Spend` se calcula después de agrupar y no entra en la distancia, porque es una combinación determinista de las seis variables de gasto.

### 8.3 Split

Se usa una partición fija 70/30 con semilla 42 y estratificación conjunta `Channel × Region`. Con las 440 observaciones se obtienen 308 clientes TRAIN y 132 TEST. Cada ejecución exporta la pertenencia al split y su hash.

### 8.4 Preprocesamiento

El preprocesamiento se decidió utilizando solo TRAIN antes de ejecutar Gurobi. Se compararon:

- variables originales + RobustScaler;
- `log1p` + RobustScaler;
- `log1p` + StandardScaler.

Para cada alternativa se estudió `K=2,...,8` mediante PAM-Manhattan, curva del coste L1, reducción marginal, silhouette, estabilidad por remuestreo y tamaño de los grupos. Se seleccionó `RobustScaler` sobre las variables originales y `K=4`.

El escalador se ajusta sobre TRAIN completo antes de inducir missing artificial. Esta decisión mantiene fijo el espacio geométrico que fue utilizado para seleccionar K y permite evaluar todas las máscaras y métodos sobre la misma transformación.

### 8.5 Uso de TEST

TEST se transforma con el escalador aprendido en TRAIN y no interviene en la selección del preprocesamiento, de K, de rho ni de lambda.

## 9. Métricas de marketing

La evaluación comercial utiliza la información completa original para interpretar las etiquetas producidas por los modelos, evitando que los perfiles sean un artefacto de la imputación.

Se calculan:

```math
\mathrm{ClientShare}_c=\frac{n_c}{n},
```

```math
\mathrm{SpendShare}_c=
\frac{\sum_{i\in c}\mathrm{TotalSpend}_i}
{\sum_i\mathrm{TotalSpend}_i},
```

```math
\mathrm{Lift}_{c,h}=
\frac{P(H=h\mid C=c)}{P(H=h)}.
```

También se guardan gasto medio, mediano y total, mezcla por categoría, composición por canal y región, chi-cuadrado y V de Cramér corregido.

Estas medidas describen segmentos potencialmente accionables, pero no miden impacto causal en ventas, ROI o respuesta a campaña.

## 10. Coste computacional

La tabla `runtime_accounting.csv` separa:

- carga del dataset;
- preprocesamiento;
- generación de missing;
- construcción de candidatos;
- construcción, optimización y extracción de TRAIN;
- construcción, optimización y extracción de TEST;
- imputación y clustering de cada baseline;
- tiempo de pipeline;
- tiempo end-to-end.

El tiempo del propuesto incluye todos los candidatos porque el MILP necesita disponer de ellos para seleccionar un método por variable. El tiempo de cada baseline incluye únicamente el imputador que utiliza y su clustering posterior.

## 11. Optimalidad

`MIPGap=0.02` certifica cercanía relativa dentro del 2 % cuando Gurobi finaliza con estado compatible con la tolerancia. Se informan además `ObjVal`, `ObjBound`, gap absoluto, tiempo, nodos, soluciones, `Work`, variables, binarias y restricciones.

Una formulación exacta puede terminar por `TIME_LIMIT`. En ese caso se informa la mejor solución factible encontrada y no se la denomina óptima.
