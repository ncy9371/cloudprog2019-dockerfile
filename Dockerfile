FROM tensorflow/serving:1.12.0

COPY	versions	/models/deeplab

ENV	MODEL_NAME	deeplab

EXPOSE	8500 8501
