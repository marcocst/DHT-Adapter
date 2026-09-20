# DHT-Adapter

Official implementation of DHT-Adapter, a structured low-rank adaptation method with dynamic rank scheduling for diffusion personalization.

DHT-Adapter factorizes each personalized update as

$\Delta W(t)=B^{(t)}\Phi^{(t)}A^{(t)},$

where the orthogonal boundary matrices (A) and (B) are frozen and only the subject-specific intermediate matrix $\Phi$ is optimized. During denoising, nested matrix blocks are activated according to the current timestep: a compact rank is used at high noise levels, while additional capacity is progressively introduced as the noise decreases.


### Requirements

Linux

NVIDIA GPU with CUDA and cuDNN

Python 3.11+ or Conda 24.1.0+

Diffusers

FLUX.1-dev

Installation

git clone https://github.com/ControlGenAI/T-LoRA.git
cd T-LoRA/flux
pip install -r requirements.txt

Training

export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export INSTANCE_DIR="dog_example"
export OUTPUT_DIR="trained-flux-dht_dog"

accelerate launch run.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --class_name="dog" \
  --placeholder_token="sks" \
  --max_train_steps=1000 \
  --checkpointing_steps=100 \
  --tlora \
  --adapter_type=bsa \
  --bsa_init=svd \
  --bsa_svd_device=auto \
  --rank=32 \
  --min_rank=1

Use --bsa_svd_device=cpu to reduce temporary GPU memory during SVD initialization. Random frozen boundaries can be selected with --bsa_init=random.

Inference

export EXP_PATH="trained-flux-dht_dog/<experiment-directory>"

python run_inference.py \
  --exp="$EXP_PATH" \
  --checkpoint_idx=800 \
  --prompts="a {0} riding a bike#a {0} dressed as a ballerina"

Prompts are separated by #. Generated images are saved in the corresponding checkpoint directory.

SDXL

Installation

git clone https://github.com/ControlGenAI/T-LoRA.git
cd T-LoRA
conda env create -f tlora_env.yml
conda activate tlora

### Training

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="dog_example"
export OUTPUT_DIR="trained-dht_dog"

accelerate launch train.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --trainer_type="ortho_lora" \
  --trainer_class="sdxl_tlora" \
  --num_train_epochs=800 \
  --checkpointing_steps=100 \
  --resolution=1024 \
  --validation_prompts="a {0} riding a bike#a {0} dressed as a ballerina" \
  --num_val_imgs_per_prompt=3 \
  --placeholder_token="sks" \
  --class_name="dog" \
  --seed=0 \
  --lora_rank=64 \
  --min_rank=32 \
  --sig_type="last"

### Inference

export CONFIG_PATH="trained-dht_dog/<experiment-directory>/logs/hparams.yml"

python inference.py \
  --config_path="$CONFIG_PATH" \
  --checkpoint_idx=800 \
  --guidance_scale=5.0 \
  --num_inference_steps=25 \
  --prompts="a {0} riding a bike#a {0} dressed as a ballerina" \
  --num_images_per_prompt=5 \
  --seed=0

Data Options

Use --one_image=<filename> for single-image personalization.

Remove --one_image to train on all reference images.

For FLUX, --mask_dir=<path> enables mask-guided subject training.

### Acknowledgements

This implementation builds on huggingface, PEFT, T-LoRA and Diffusers. We thank their authors and open-source communities.

### Citation

Citation information will be added upon publication.
