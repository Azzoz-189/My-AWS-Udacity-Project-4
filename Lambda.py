#_________________My Lambda Function 1: SerializeImageData______________________:
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event["body"]["s3_key"]
    bucket = event["body"]["s3_bucket"]
    
    # Download the data from s3 to /tmp/image.png
    
    s3.download_file(bucket, key, "/tmp/image.png")
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


#_________________My Lambda Function 2: ______________________:

import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-09-11-18-14-17-093"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']["image_data"])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT, ContentType='image/png', Body=image) 

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = predictor.predict(image)
    
    # We return the data back to the Step Function    
    event['body']["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event['body'])
    }

#_________________My Lambda Function 3: ______________________:

import json
import numpy as np 
THRESHOLD = .90

def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = event["body"]["inferences"] 
    
    # Check if any values in our inferences are above THRESHOLD
    #meets_threshold = np.array(json.loads(inferences.decode('utf-8')))
    meets_threshold = np.array(inferences)
    meets_threshold = meets_threshold[meets_threshold > THRESHOLD]
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event["body"])
    }
