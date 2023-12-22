from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from config import BaseConfig, Cifar10Config, LSUNChurchConfig
from diffusers.utils import make_image_grid
from PIL import Image
import torch


def get_dataset(config):
    if config.name == "cifar10":
        dataset = get_cifar10_dataset(config)
    elif config.name == "lsun_church":
        dataset = get_lsun_church_dataset(config)
    return dataset


def transform(config):
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def get_hub_dataset(config):
    dataset = load_dataset(
        config.dataset_name, split="train"
    )  # , cache_dir="./datasets/")
    preprocess = transform(config)

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    return dataset


def get_lsun_church_dataset(config):
    assert config.name == "lsun_church"
    return datasets.LSUN(
        root="./datasets",
        classes=["church_outdoor_train"],
        transform=transform(config),
    )


def get_cifar10_dataset(config):
    assert config.name == "cifar10"
    return datasets.CIFAR10(
        root=config.dataset_path,
        train=True,
        transform=transform(config),
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = LSUNChurchConfig()
    dataset = get_lsun_church_dataset(config)
    # config = Cifar10Config()
    # dataset = get_cifar10_dataset(config)

    dataloader = DataLoader(
        dataset,
        batch_size=16 * 4,
        shuffle=True,
        generator=torch.manual_seed(config.seed),
    )
    print(config.name)
    print(len(dataloader))
    # for i in dataloader:
    #     images, _ = i
    #     break
    # image = (images / 2 + 0.5).clamp(0, 1)
    # image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    # images = (image * 255).round().astype("uint8")
    # pil_images = [Image.fromarray(image) for image in images]
    # image_grid = make_image_grid(pil_images, rows=4, cols=16)
    # image_grid.save(f"./datasets/{config.name}.png")
