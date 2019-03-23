FROM tensorflow/serving:1.12.0

COPY	versions	/models/deeplab

ENV	MODEL_BASE_PATH	/models
ENV	MODEL_NAME	deeplab

EXPOSE	8500/tcp 8501/tcp

ENTRYPOINT	/usr/bin/tf_serving_entrypoint.sh
