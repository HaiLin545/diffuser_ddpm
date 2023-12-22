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
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = (
        "outputs/ddpm-butterflies-128"  # the model name locally and on the HF Hub
    )

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

    dataset_name = "huggan/smithsonian_butterflies_subset"


@dataclass
class Cifar10Config(BaseConfig):
    name = "cifar10"
    image_size = 32
    train_batch_size = 64
    eval_batch_size = 64
    dataset_name = "cifar10"
    dataset_path = "./datasets/"
    output_dir = "outputs/ddpm-cifar10-64"

    num_feat_map = 2

@dataclass
class LSUNChurchConfig(BaseConfig):
    name = "lsun_church"
    image_size = 256
    train_batch_size = 8
    eval_batch_size = 8
    learning_rate = 2e-5
    dataset_name = "tglcourse/lsun_church_train"
    output_dir = "outputs/ddpm-lsun-bedroom-256"

    num_feat_map = 4


def load_config(config):
    if config == "cifar10":
        return Cifar10Config()
    elif config == "lsun_bedroom":
        return LSUNChurchConfig()

    return BaseConfig()
