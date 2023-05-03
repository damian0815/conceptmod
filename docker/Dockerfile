# Use the specified base image
FROM runpod/stable-diffusion:web-automatic-6.0.0

RUN \
  pip3 install imageio moviepy opencv-python ftfy datasets scikit-image && \
  pip3 install git+https://github.com/openai/CLIP.git --no-deps

# Clone the required repositories and install dependencies
RUN git clone https://github.com/ntc-ai/conceptmod /conceptmod && \
    cd /conceptmod && \
    git clone https://github.com/THUDM/ImageReward && \
    cd ImageReward && \
    cd ..

# Set the working directory
WORKDIR /conceptmod

RUN echo 'echo "Installing dependencies..."' > ~/.bashrc
# Create the installdeps.sh script and add it to .bashrc
RUN echo 'pip install --upgrade pip' > /conceptmod/installdeps.sh && \
    echo 'pip install torch torchvision numpy scipy scikit-image scikit-learn tqdm lmdb' > /conceptmod/installdeps.sh && \
    echo 'cd /conceptmod/ImageReward' >> /conceptmod/installdeps.sh && \
    echo 'python setup.py develop > /dev/null 2>&1' >> /conceptmod/installdeps.sh && \
    echo 'source /conceptmod/installdeps.sh > /dev/null && 2>&1 cd /conceptmod' >> ~/.bashrc

RUN echo 'echo " - - - - - - "' >> /root/.bashrc

RUN echo 'echo "Welcome to NTC-AI lora animator. You will need to upload the lora+model then run the lora_anim.py script."' >> /root/.bashrc
RUN echo 'echo "Comment on one of the models if you have a problem civit.ai/user/ntc"' >> /root/.bashrc
RUN echo 'echo "Example command: python3 lora_anim.py -s -0.0 -e 2.0 -l \"smile\" -lp \", smile\" -np \"nipples, weird image.\" -url \"http://localhost:3000/sdapi/v1/txt2img\" -n 32 -sd 7 -m 6"' >> /root/.bashrc
RUN echo 'echo ""' >> /root/.bashrc
# Update the relauncher.py script to add --api flag
RUN sed -i 's/launch_string = "\/workspace\/stable-diffusion-webui\/webui.sh -f"/launch_string = "\/workspace\/stable-diffusion-webui\/webui.sh -f --api"/' /stable-diffusion-webui/relauncher.py
# Run the anim.sh script
CMD ["bash", "/start.sh"]
