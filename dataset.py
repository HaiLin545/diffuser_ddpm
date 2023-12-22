from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
from config import BaseConfig, Cifar10Config, LSUNBedroomConfig


def get_dataset(config):
    if config.name == "cifar10":
        dataset = get_cifar10_dataset(config)
    elif config.name == "lsun_church":
        dataset = get_church_datset(config)
    return dataset


def get_hub_dataset(config):
    dataset = load_dataset(
        config.dataset_name, split="train"
    )  # , cache_dir="./datasets/")
    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    return dataset

def get_lsun_church_datset(config):
    assert config.dataset_name == "lsun_church"
    return datasets.LSUN

def get_cifar10_dataset(config):
    assert config.dataset_name == "cifar10"
    return datasets.CIFAR10(
        root=config.dataset_path,
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        ),
        # download=,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = Cifar10Config()
    # config = LSUNBedroomConfig()
    # config = BaseConfig()
    dataset = get_cifar10_dataset(config)
    # dataset = get_hub_dataset(config)
    fig, axs = plt.subplots(16, 4, figsize=(16, 4))
    for i in range(4*16):
        axs[i].imshow(dataset[i][0].numpy().transpose(1, 2, 0))
        axs[i].set_axis_off()
    fig.savefig(f"./datasets/{config.name}.png")
