import torch
import torch.nn.functional as F
import os
from PIL import Image
from diffusers import DDPMScheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
from config import load_config
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from dataset import get_dataset
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
from accelerate import notebook_launcher
from model import get_model
import argparse
import numpy as np

@torch.no_grad()
def main(args):
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    assert args.sample_num % config.sample_batch_size == 0
    
    pipeline = DDPMPipeline.from_pretrained(config.output_dir)
    pipeline.to("cuda")
    loop = args.sample_num // config.sample_batch_size
    print(loop)
    cnt = 0
    for i in tqdm(range(loop)):
        images = pipeline(
            batch_size=config.sample_batch_size,
            generator=torch.manual_seed(config.seed),
        ).images

        for _, img in enumerate(images):
            img.save(f"{args.output_dir}/{cnt}.png")
            cnt+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="config name, etc cifar10, lsun_church",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="output dir of samples",
        type=str,
    )
    parser.add_argument(
        "--sample_num",
        help="numbe of sample",
        type=int,
    )
    args = parser.parse_args()
    main(args)
