from dataclasses import dataclass


@dataclass
class BaseConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

@dataclass
class Cifar10Config(BaseConfig):
    name = "cifar10"
    image_size = 32
    train_batch_size = 64
    eval_batch_size = 64
    sample_batch_size = 500
    dataset_name = "cifar10"
    dataset_path = "./datasets/"
    output_dir = "outputs/ddpm-cifar10-64"
    num_feat_map = 2

@dataclass
class LSUNChurchConfig(BaseConfig):
    name = "lsun_church"
    image_size = 256
    train_batch_size = 4
    eval_batch_size = 16
    sample_batch_size = 50
    num_epochs = 5
    save_image_epochs = 1
    learning_rate = 2e-5
    dataset_path = "./datasets/"
    output_dir = "outputs/ddpm-lsun-church-256"
    num_feat_map = 4


def load_config(config):
    if config == "cifar10":
        return Cifar10Config()
    elif config == "lsun_church":
        return LSUNChurchConfig()

    return BaseConfig()
