# Build the Docker image
docker build -t lora_animation:v1.0 . && docker tag lora_animation:v1.0 ntcai/lora_animation:v1.0 && docker push ntcai/lora_animation:v1.0

