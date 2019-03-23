from flask import Flask
from flask import request
from PIL import Image
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import numpy as np
from io import BytesIO
import os

def post():
    global request
    server = os.environ['server']
    host, port = server.split(':')

    body_data = BytesIO(request.get_data())
    origin = Image.open(body_data)
    image = np.array(origin)
    height = image.shape[0]
    width = image.shape[1]

    # create the RPC stub
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # create the request object and set the name and signature_name params
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'deeplab'
    request.model_spec.signature_name = 'predict_images'

    # fill in the request object with the necessary data
    request.inputs['images'].CopyFrom(
              tf.contrib.util.make_tensor_proto(image.astype(dtype=np.float32), shape=[1, height, width, 3]))

    request.inputs['height'].CopyFrom(tf.contrib.util.make_tensor_proto(height, shape=[1]))
    request.inputs['width'].CopyFrom(tf.contrib.util.make_tensor_proto(width, shape=[1]))

    # sync requests
    result_future = stub.Predict(request, 30.)

    # For async requests
    # result_future = stub.Predict.future(request, 10.)
    # result_future = result_future.result()

    # get the results
    output = np.array(result_future.outputs['segmentation_map'].int64_val)
    height = result_future.outputs['segmentation_map'].tensor_shape.dim[1].size
    width = result_future.outputs['segmentation_map'].tensor_shape.dim[2].size

    image_mask = np.reshape(output, (height, width))
    img = Image.fromarray(np.uint8(image_mask * 255), 'L')

    output = Image.blend( origin.convert("RGBA"), img.convert("RGBA"), 0.6 )

    response_bytesIO = BytesIO()
    output.save(response_bytesIO, format="PNG")
    return response_bytesIO.getvalue()

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    return post()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
