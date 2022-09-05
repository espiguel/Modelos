# document-image-classification #

Repositorio de lambda document-image-classification.

### Objetivo ###

Realizar preprocesamiento e inferencia de una lista de imágenes.

### Tarea ###

A partir de una lista de imágenes la lambda se encarga de obtener los tags de rekognition de cada imagen y preparar los datos para finalmente consultar el endpoint de sagemaker.

### Datos de entrada ###

La función lambda se ejecutará cada vez que ingrese una licencia al motor decisorio. Este realizara una peticion con una lista de los certificados de dicha licencia. La estructura de entrada de datos es la siguiente:

```
{
  "body": "{\"images\":    [{\"id_image\":    123456, \"url_image\": \"https://prod-gcba-us-east-1-upload.s3.amazonaws.com/us-east-1%3Aa55bc7d5-0df0-4f0d-a6ea-8657da1ebb1b/us-east-1%3A6e8d509f-8c79-4489-8213-1b30b272c4d4/certificates/1633376045/Screenshot_2021-10-04-15-41-18-182_com.google.android.gm.jpg\"},{\"id_image\":   789012,\"url_image\": \"https://prod-gcba-us-east-1-upload.s3.amazonaws.com/us-east-1%3Aa55bc7d5-0df0-4f0d-a6ea-8657da1ebb1b/us-east-1%3A6e0a01f6-2164-49b7-a6d5-8fa6683a533a/certificates/1629931671044/certificado%2520Gomez%2520Perla%2520%281%29.pdf\"}]}"
}
```

### Datos de salida ###

Los datos generados como respuesta a la petición contienen estado de la resupueta, datos de las imágenes, predicciones realizadas, endpoint y versión del modelo.

```
{
  "statusCode": 200,
  "body": "{\"predictions\": [{\"id_image\": 123456, \"url_image\": \"https://prod-gcba-us-east-1-upload.s3.amazonaws.com/us-east-1%3Aa55bc7d5-0df0-4f0d-a6ea-8657da1ebb1b/us-east-1%3A6e8d509f-8c79-4489-8213-1b30b272c4d4/certificates/1633376045/Screenshot_2021-10-04-15-41-18-182_com.google.android.gm.jpg\", \"score\": 0.2260225044778195, \"prediction_label\": \"Rechazado\", \"model_version_cla\": \"13\", \"end_point\": \"n2-data-certification-model-deploy-dev-endpoint\", \"prediction_status\": \"Ok\"}, {\"id_image\": 789012, \"url_image\": \"https://prod-gcba-us-east-1-upload.s3.amazonaws.com/us-east-1%3Aa55bc7d5-0df0-4f0d-a6ea-8657da1ebb1b/us-east-1%3A6e0a01f6-2164-49b7-a6d5-8fa6683a533a/certificates/1629931671044/certificado%2520Gomez%2520Perla%2520%281%29.pdf\", \"score\": 0.9124288243498654, \"prediction_label\": \"Aprobado\", \"model_version_cla\": \"13\", \"end_point\": \"n2-data-certification-model-deploy-dev-endpoint\", \"prediction_status\": \"Ok\"}]}"
}
```

### Variables de entorno ###

* BUCKET_REKOGNITION_LABELS: nombre del bucket donde se almacenarán los tags de rekognition. Variable opcional, en caso de no ser necesario guardar los tags de rekognition puede no especificarse.
* ENDPOINT_NAME: nombre del endpoint de sagemaker que realiza la inferencia del modelo de certificados.
* FOLDER_REKOGNITION_LABELS: nombre del directorio del bucket donde se almacenarán los tags de rekognition. Variable opcional, en caso de no ser necesario guardar los tags de rekognition puede no especificarse.
* MODEL_VERSION	: versión del modelo de sagemaker desplegado.

### Lenguaje y librerías necesarias ###

* Poppler: Poppler es una biblioteca de software libre para generar documentos pdf que se usa comúnmente en sistemas Linux
* Lenguaje: Python 3.7
* Librerías necesarias: 
    * matplotlib: versión 3.4.3
    * numpy: versión 1.19.1
    * opencv-python: versión 4.5.3.56
    * pandas: versión 1.3.3
    * pdf2image: versión 1.16.0
    * Pillow: versión 8.3.2
    * requests: versión 2.22.0
    * setuptools: versión 58.1.0
    * tqdm: versión 4.39.0
    * urllib3: versión 1.25.11

### Estructura principal del repositorio ###

```
.
|—— app.py                                  # Contiene la definición de la función lambda
|—— Dockerfile                              # Dockerfile de la función lambda
└── requirements.txt                        # Librerias a instalar al momento de levantar un contenedor
```
