# Política de manejo de datos y publicación

Acuerdo de trabajo entre el Equipo 48 y Bnext para cumplir con los entregables del Tecnológico de
Monterrey, que exigen un repositorio público en GitHub, sin exponer información confidencial de la
organización. Complementa el convenio de confidencialidad (NDA) firmado por los integrantes.

## Reglas

1. **Código en GitHub, ejecución en Bnext.** El código se versiona en este repositorio; su ejecución
   sobre los datos del proyecto se realiza dentro del servidor de Bnext.
2. **Trabajo con datos solo en el servidor.** Acceso a los datasets, entrenamiento, experimentación
   en notebooks, validación sobre train/val/test, comparación de configuraciones e hiperparámetros,
   evaluación de métricas, inferencia sobre video y pruebas de rendimiento se hacen en el servidor.
3. **Material confidencial fuera de GitHub.** Datasets, videos, credenciales, configuraciones
   internas y cualquier otro material confidencial permanecen exclusivamente en la infraestructura
   de Bnext, salvo autorización expresa.
4. **Contenido de GitHub.** Código, scripts, notebooks, documentación y definición de dependencias
   necesarios para los entregables del Tec.
5. **Restringido por defecto.** Todo archivo o contenido entregado directamente por Bnext se
   considera restringido y no se incorpora a GitHub sin su autorización.
6. **Resultados.** Métricas, gráficas, salidas de inferencia y modelos entrenados se conservan en el
   servidor y se incorporan a GitHub o a los entregables solo cuando el Tec lo requiera.
7. **Revisión previa.** Antes de publicar cualquier elemento que pueda contener información
   sensible, propiedad intelectual o información derivada de recursos internos de Bnext, se revisa
   con Bnext.

## Cómo se aplica en este repositorio

| Regla | Mecanismo |
|---|---|
| 1, 2 | Las rutas a datos se leen de la variable de entorno `PLANOGRAMA_DATA_DIR`; ningún notebook contiene rutas, nombres de host ni direcciones del servidor. |
| 3, 5 | `.gitignore` excluye `data/`, imágenes, video, credenciales, `outputs/`, `results/`, `artifacts/` y pesos de modelos. |
| 4 | Solo se versionan fuentes (`.py`, `.ipynb`, `.md`, `.html`) y los PDF de los entregables. |
| 6 | Las salidas tabulares y los modelos se escriben en `PLANOGRAMA_OUT_DIR`, que vive en el servidor. |
| 6, 7 | Los notebooks de trabajo (`notebooks/`) se suben sin salidas; la libreta entregable que el Tec exige ejecutada solo incluye gráficas y tablas agregadas, nunca imágenes o recortes de anaquel. |
| 7 | Los códigos de SKU y descripciones de producto se anonimizan (`SKU_01`, `SKU_02`, ...) en toda salida publicada. |

## Lista de verificación antes de publicar

- [ ] `git diff --cached` no contiene imágenes, videos, CSV/Parquet de datos ni pesos de modelo.
- [ ] Ningún archivo menciona direcciones IP, nombres de host, usuarios del servidor ni repositorios
      internos de Bnext.
- [ ] Ninguna salida de notebook incluye fotografías, recortes de producto, códigos de SKU reales o
      nombres de clientes de Bnext.
- [ ] Las figuras publicadas son agregados (distribuciones, correlaciones, varianza explicada,
      métricas).
- [ ] Si algo deriva de recursos internos de Bnext y no es claramente agregado, se revisó con Bnext
      antes del push.
