import boto3
import datetime
import io
import json
import numpy as np
import logging
import os
import pandas as pd
import pdf2image
import requests
from concurrent import futures


ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
MODEL_VERSION = os.environ['MODEL_VERSION']
MAX_LABELS = 200
MIN_CONFIDENCE = 5
FEATURES = ['Text', 'Handwriting', 'Symbol', 'Number', 'Paper', 'Document', 'Apparel', 'Clothing', 'Signature', 'Autograph', 
            'Animal', 'Letter', 'Alphabet', 'Page', 'Plant', 'Accessories', 'Accessory', 'Food', 'Envelope', 'Beverage', 
            'Drink', 'Electronics', 'Musical Instrument', 'Word', 'File Binder', 'Alcohol', 'Mail', 'Footwear', 'Invertebrate', 'Furniture', 
            'QR Code', 'Gray', 'Calligraphy', 'Jewelry', 'Broom']

client_rekognition = boto3.client("rekognition", 'us-east-1')
client_sagemaker = boto3.client("runtime.sagemaker")
client_s3 = boto3.resource('s3')

def handler(event, context):
    try:
        time_start = datetime.datetime.now()
        print(f'Hora de inicio de ejecucion: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Event:', event)
        images = json.loads(event['body'])["images"]
        response = []
        ex = futures.ThreadPoolExecutor()
        results = ex.map(get_predictions, images)
        response = list(results)
        print(f'Hora de fin de ejecucion: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Tiempo de ejecucion: {(datetime.datetime.now() - time_start).seconds}')
        return {
            'statusCode': 200,
            'body': json.dumps({"predictions": response})
        }
    except Exception as err:
        print('Error handler')
        print(err)
        logging.error("Exception occurred", exc_info=True)
        return {
        'statusCode': 500,
        'body': json.dumps(format(err))
        }


def detect_labels_from_pdf(client, pdf_file, max_labels, min_confidence):
    response = requests.get(pdf_file, timeout=30)
    pages = pdf2image.convert_from_bytes(response.content, dpi=300)
    img = pages[0].convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    return client.detect_labels(Image={'Bytes': byte_im}, MaxLabels=max_labels, MinConfidence=min_confidence)

def detect_labels_from_bytes(client, file, max_labels, min_confidence):
    response = requests.get(file)
    return client.detect_labels(Image={'Bytes': response.content}, MaxLabels=max_labels, MinConfidence=min_confidence)

def detect_labels(client, file, max_labels, min_confidence):
    try:
        file_extension = file.split('.')[-1].lower()
        if file_extension == 'pdf':
            return detect_labels_from_pdf(client, file, max_labels, min_confidence)
        else:
            return detect_labels_from_bytes(client, file, max_labels, min_confidence)
    except:
        print('Error al procesar la imagen en rekognition')
        logging.error("Exception occurred", exc_info=True)
        return {}
    
def query_endpoint(encoded_tabular_data, endpoint_name, content_type='text/csv'):
    response = client_sagemaker.invoke_endpoint(
        EndpointName=endpoint_name, ContentType=content_type, Body=encoded_tabular_data
    )
    return response

def parse_response(query_response):
    model_predictions = json.loads(query_response["Body"].read())
    predicted_probabilities = model_predictions["probabilities"]
    return np.array(predicted_probabilities)

def save_rekognition_labels(id_image, response):
    if (os.getenv('BUCKET_REKOGNITION_LABELS', default = None) is not None) & (os.getenv('FOLDER_REKOGNITION_LABELS', default = None) is not None):
        try:
            s3object = client_s3.Object(os.getenv('BUCKET_REKOGNITION_LABELS'), 
                                                '{}/image_{}.json'.format(os.getenv('FOLDER_REKOGNITION_LABELS'), id_image))
            s3object.put(Body=(bytes(json.dumps(response).encode('UTF-8'))))
        except:
            logging.error("Exception occurred", exc_info=True)

def get_predictions(data_image):
    labels = ["Rechazado", "Aprobado"]
    try:
        logging.info(f'get_predictions id_image: {data_image["id_image"]}')
        print((f'get_predictions id_image: {data_image["id_image"]}'))
        # Call rekognition
        logging.info(f'call rekognition id_image: {data_image["id_image"]}')
        print((f'call rekognition id_image: {data_image["id_image"]}'))
        response = detect_labels(client_rekognition, data_image['url_image'], max_labels=MAX_LABELS, min_confidence=MIN_CONFIDENCE)
        # Save rekognition features
        logging.info(f'save rekognition features id_image: {data_image["id_image"]}')
        print((f'save rekognition features id_image: {data_image["id_image"]}'))
        save_rekognition_labels(id_image=data_image['id_image'], response=response)
        # Create dataframe
        logging.info(f'create dataframe id_image: {data_image["id_image"]}')
        print((f'create dataframe id_image: {data_image["id_image"]}'))
        df = pd.DataFrame([{i['Name']:i['Confidence'] for i in response['Labels']}])
        columns = df.columns.tolist()
        empty_columns = list(set(FEATURES) - set(columns))
        for column in empty_columns:
            df[column] = 0
        df['file_is_pdf'] = int(data_image['url_image'].split('.')[-1].lower() == 'pdf')
        # Endpoint request
        logging.info(f'call sagemaker id_image: {data_image["id_image"]}')
        print((f'call sagemaker id_image: {data_image["id_image"]}'))
        query_response_batch = query_endpoint(
            df[['file_is_pdf'] + FEATURES].to_csv(header=False, index=False).encode("utf-8"),
            endpoint_name=ENDPOINT_NAME
        )
        predict_prob = np.concatenate(parse_response(query_response_batch), axis=0)
        logging.info(f'response id_image: {data_image["id_image"]}')
        print((f'response id_image: {data_image["id_image"]}'))
        print({
                "id_image": data_image["id_image"],
                "url_image": data_image["url_image"],
                "score": predict_prob[1],
                "prediction_label": labels[np.argmax(predict_prob)],
                "model_version_cla": MODEL_VERSION,
                "end_point": ENDPOINT_NAME,
                "prediction_status": "Ok"
            })
        return {
            "id_image": data_image["id_image"],
            "url_image": data_image["url_image"],
            "score": predict_prob[1],
            "prediction_label": labels[np.argmax(predict_prob)],
            "model_version_cla": MODEL_VERSION,
            "end_point": ENDPOINT_NAME,
            "prediction_status": "Ok"
        }
    except Exception as err:
        print(f'Error al procesar imagen {data_image["id_image"]}')
        logging.error("Exception occurred", exc_info=True)
        return {
            "id_image": data_image["id_image"],
            "url_image": data_image["url_image"],
            "score": 0,
            "prediction_label": "None",
            "model_version_cla": MODEL_VERSION,
            "end_point": ENDPOINT_NAME,
            "prediction_status": format(err)
        }