import os
import argparse
from PIL import Image

from utils.exp import setup_exp_name
from train import run_train


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default='sks',
    )
    parser.add_argument(
        "--tlora",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        default="lora",
        choices=["lora", "bsa"],
        help=(
            "Adapter parameterization. 'lora' keeps the original BA adapter; "
            "'bsa' freezes non-zero A/B bases and trains only a rank-by-rank S matrix."
        ),
    )
    parser.add_argument(
        "--bsa_init",
        type=str,
        default="random",
        choices=["random", "svd"],
        help=(
            "BSA initialization. 'random' uses frozen non-zero random A/B and S=0. "
            "'svd' decomposes each pretrained W0 projection as U_r Sigma_r V_r^T, "
            "freezes B=U_r and A=V_r^T, and trains S from Sigma_r with a dynamic reference shift."
        ),
    )
    parser.add_argument(
        "--bsa_svd_oversample",
        type=int,
        default=4,
        help="SVD low-rank workspace q is min(rank * this value, min(W0.shape)).",
    )
    parser.add_argument(
        "--bsa_svd_niter",
        type=int,
        default=4,
        help="Number of subspace iterations used by torch.svd_lowrank for W0 initialization.",
    )
    parser.add_argument(
        "--bsa_svd_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used only while computing the truncated SVD of pretrained W0 projections.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--min_rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--rank_schedule",
        type=str,
        default="decreasing",
        choices=["decreasing", "increasing"],
        help=(
            "Effective-rank direction with respect to diffusion timestep. "
            "'decreasing' preserves the original T-LoRA behavior (larger timestep -> "
            "smaller rank); 'increasing' is the reversed ablation (larger timestep -> "
            "larger rank)."
        ),
    )
    parser.add_argument(
        "--rank_logging_steps",
        type=int,
        default=100,
        help=(
            "Print and append the sampled diffusion timestep and effective "
            "T-LoRA rank every N optimization steps. Set to 0 to disable "
            "periodic logging; the startup timestep/rank table is still written."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat the training data.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--one_image",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=(
            'We default to the "none" weighting scheme for uniform sampling and uniform loss'
        ),
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="prodigy",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default='a {0} in the jungle in a outfit made of green leaves#a {0} riding a bike in a white t-shirt#a {0} in a purple wizzard outfit and a hat in a magic forest',
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=3,
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.min_rank > args.rank:
        parser.error("--min_rank must be less than or equal to --rank.")
    if args.rank_logging_steps < 0:
        parser.error("--rank_logging_steps must be greater than or equal to 0.")
    if args.bsa_svd_oversample < 1:
        parser.error("--bsa_svd_oversample must be at least 1.")
    if args.bsa_svd_niter < 0:
        parser.error("--bsa_svd_niter must be non-negative.")

    return args


def run(args):
    image_name = os.path.basename(os.path.normpath(args.instance_data_dir))
    exp = setup_exp_name(args.output_dir, image_name)
    name = 'lora' if not args.tlora else 'tlora' + f'{args.rank}'
    if args.adapter_type == "bsa":
        name += f'_bsa-{args.bsa_init}'
    exp += '_' + name

    args.output_dir = os.path.join(args.output_dir, exp)
    args.exp_name = exp
    os.makedirs(args.output_dir, exist_ok=True)

    run_train(args)


if __name__ == "__main__":
    args = parse_args()
    run(args)




# 正确的运行命令：
# accelerate launch   run.py   --pretrained_model_name_or_path="$MODEL_NAME"   --instance_data_dir="$INSTANCE_DIR"   --output_dir="$OUTPUT_DIR"    --class_name="dog"   --placeholder_token="sks"   --one_image="02.jpg"    --max_train_steps=1000   --checkpointing_steps=100  --validation_prompt="a {0} in the jungle"    --tlora   --rank=32   --min_rank=1


# 下面是SVD的命令：
# accelerate launch   run.py   --pretrained_model_name_or_path="$MODEL_NAME"   --instance_data_dir="$INSTANCE_DIR"   --output_dir="$OUTPUT_DIR"    --class_name="glasses"   --placeholder_token="sks"   --one_image="02.jpg"    --max_train_steps=1000   --checkpointing_steps=100  --validation_prompt="a photo of {0} in the jungle"    --tlora   --rank=512   --min_rank=128 --adapter_type=bsa --bsa_init=svd --bsa_svd_device=auto  --bsa_svd_niter=8
