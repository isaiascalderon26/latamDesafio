Documentación del Proyecto

Descripción General
Este proyecto se enfoca en la implementación y despliegue de un modelo de predicción de retrasos de vuelos en una API utilizando Google Cloud Platform (GCP). El proyecto consta de tres partes principales:

Parte I - Modelo de Aprendizaje Automático: Esta parte implica la transcripción y refinamiento del trabajo del Data Scientist, donde se entrena un modelo de aprendizaje automático para predecir retrasos en vuelos. Se ha seleccionado el mejor modelo propuesto por el DS y se han aplicado las mejores prácticas de programación.

Parte II - Implementación de la API con FastAPI: En esta etapa, se ha creado una API utilizando el marco FastAPI. La API permite a los usuarios consultar la predicción de retraso de un vuelo utilizando una variedad de parámetros de entrada. La implementación ha sido probada y sigue las mejores prácticas de desarrollo.

Parte III - Implementación de la API en GCP: En esta fase, la API creada en la Parte II se ha implementado en Google Cloud Platform para que sea accesible desde Internet. La implementación en GCP incluye la exposición de la API al público y la configuración de las reglas de firewall.

Modificacion del archivo exploration.ipynb ya que generaba error
1. Data Analysis: First Sight
En esta sección del análisis de datos, se realizó una exploración inicial del conjunto de datos para obtener una comprensión general de su estructura y contenido.


Errores Corregidos:
Error de Importación de Matplotlib:
Se solucionó el error de importación de Matplotlib instalando la biblioteca de Matplotlib.
Error de Importación de Seaborn:
Se solucionó el error de importación de Seaborn instalando la biblioteca de Seaborn.
Errores en los Gráficos de Barras:
Se corrigieron errores en los gráficos de barras al ajustar los argumentos de sns.barplot(), eliminando argumentos incorrectos y utilizando color en lugar de alpha para definir el color de las barras.


Análisis de la Fecha:
Se exploró la distribución de las fechas en el conjunto de datos y se generó un gráfico de barras para visualizar la cantidad de vuelos por aerolínea.
Análisis por Día:
Se examinó la distribución de vuelos por día y se creó un gráfico de barras para mostrar la cantidad de vuelos por día.
Análisis por Mes:
Se analizó la distribución de vuelos por mes y se generó un gráfico de barras para visualizar la cantidad de vuelos por mes.
Análisis por Día de la Semana:
Se exploró la distribución de vuelos por día de la semana y se creó un gráfico de barras que muestra la cantidad de vuelos por día de la semana.
Análisis por Tipo de Vuelo:
Se examinó la distribución de vuelos por tipo y se generó un gráfico de barras que muestra la cantidad de vuelos por tipo de vuelo.
Análisis por Destino:
Se realizó un análisis de la distribución de vuelos por destino y se creó un gráfico de barras para visualizar la cantidad de vuelos por destino.

En la sección "3. Data Analysis: Second Sight" de exploration.ipynb, se intentó analizar la tasa de retraso en función de diferentes columnas del conjunto de datos. Sin embargo, hubo un error en el código que impidió que se generara un gráfico de barras correctamente. A continuación, se presenta un resumen de la sección y se indican los errores a corregir:

3. Data Analysis: Second Sight
En esta sección, se intentó analizar la tasa de retraso en función de diferentes columnas del conjunto de datos.

Errores Detectados:
Error en el Código:
Se encontró un error en el código al intentar generar el gráfico de barras.
El error se debió al uso incorrecto de los argumentos en sns.barplot(), ya que tomó dos argumentos en lugar de uno.

Análisis de la Tasa de Retraso por Destino:
Se creó una función llamada get_rate_from_column para calcular la tasa de retraso en función de una columna específica del conjunto de datos.
Se intentó calcular la tasa de retraso por destino y generar un gráfico de barras que muestre la tasa de retraso por destino.


Los cambios clave son:
	destination_rate_values = destination_rate.index: Estamos tomando los índices del DataFrame destination_rate para los valores del eje X del gráfico.
	sns.barplot(x=destination_rate_values, y=destination_rate['Tasa (%)'], alpha=0.75): Estamos utilizando los argumentos x y y para especificar los datos en el eje X y el eje Y del gráfico.




Guía de Implementación
A continuación, se presenta una guía paso a paso para levantar el proyecto en tu entorno local y posteriormente implementarlo en Google Cloud Platform.

Requisitos Previos
Antes de comenzar, asegúrate de tener instalados los siguientes componentes:

Python 3.x
Git
Google Cloud SDK (gcloud)
Docker (opcional, para entorno de desarrollo)
Cuenta de Google Cloud Platform
Pasos para Implementar el Proyecto
Paso 1: Clonar el Repositorio
bash
Copy code
git clone https://github.com/TU_USUARIO/tu-repositorio.git
cd tu-repositorio
Parte I - Modelo de Aprendizaje Automático
Asegúrate de tener las bibliotecas de Python necesarias instaladas. Esto se puede hacer ejecutando pip install -r requirements.txt.

Transcribe el trabajo del Data Scientist del cuaderno Jupyter a un archivo model.py. Puedes hacer esto manualmente o utilizando herramientas como nbconvert.

Selecciona el mejor modelo propuesto por el DS y asegúrate de que model.py implemente ese modelo.

Ejecuta las pruebas para el modelo:

bash
Copy code
make model-test
Parte II - Implementación de la API con FastAPI
Asegúrate de tener FastAPI y Uvicorn instalados. Si no, instálalos con:

bash
Copy code
pip install fastapi uvicorn
Implementa la API en el archivo api.py.

Asegúrate de que la API se esté ejecutando correctamente en tu entorno local con el siguiente comando:

bash
Copy code
uvicorn api:app --host 0.0.0.0 --port 8000
Ejecuta las pruebas de la API:

bash
Copy code
make api-test
Parte III - Implementación de la API en GCP
Asegúrate de tener la Google Cloud SDK (gcloud) instalada y configurada en tu máquina. Si no, ejecuta:

bash
Copy code
gcloud init
Conéctate a tu instancia de VM en GCP:

bash
Copy code
gcloud compute ssh NOMBRE-DE-TU-INSTANCIA --project=TU-PROYECTO
En la instancia de VM, clona el repositorio y ve al directorio de tu proyecto.

bash
Copy code
cd ~  # O navega al directorio deseado
git clone URL-DE-TU-REPOSITORIO
cd NOMBRE-DE-TU-PROYECTO
Asegúrate de que la API se ejecute en la instancia de VM con el comando Uvicorn. Asegúrate de configurar el puerto y el host correctamente.

bash
Copy code
uvicorn api:app --host 0.0.0.0 --port PUERTO
Configura las reglas de firewall en GCP para abrir el puerto en el que se ejecuta la API. Reemplaza PUERTO con el número de puerto en el que tu API está escuchando.

bash
Copy code
gcloud compute firewall-rules create allow-api --allow=tcp:PUERTO
Asigna una dirección IP estática a tu instancia de VM en GCP para que sea accesible a través de la misma IP en el futuro.

Guarda la dirección IP de tu instancia de VM, ya que la necesitarás en el siguiente paso.

Actualiza el archivo Makefile en tu máquina local con la URL de tu API en GCP:

make
Copy code
API_URL = http://IP-DE-TU-INSTANCIA-GCP:PUERTO
Ejecuta las pruebas de estrés en tu máquina local:

bash
Copy code
make stress-test