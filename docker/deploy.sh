# Build the Docker image
docker build -f Dockerfile_train -t conceptmod_train:v1.0 . && docker tag conceptmod_train:v1.0 ntcai/conceptmod_train:v1.0 && docker push ntcai/conceptmod_train:v1.0 && \
    RUN echo 'echo "If the lora is not found, it will result in a static image. See the container logs for automatic debug logs."' >> /root/.bashrc
docker build -f Dockerfile_lora_animation -t lora_animation:v1.0 . && docker tag lora_animation:v1.0 ntcai/lora_animation:v1.0 && docker push ntcai/lora_animation:v1.0

