from diffusers import UNet2DModel
from config import load_config

def get_model(config):
    assert config.num_feat_map == 4 or config.num_feat_map == 2
    down_black_types = ("DownBlock2D",) * config.num_feat_map + (
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types = ("UpBlock2D", "AttnUpBlock2D") + (
        "UpBlock2D",
    ) * config.num_feat_map

    if config.num_feat_map == 2:
        block_out_channels = (128, 128, 256, 512)
    else:
        block_out_channels = (128, 128, 256, 256, 512, 512)

    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=block_out_channels,  # the number of output channels for each UNet block
        down_block_types=down_black_types,
        up_block_types=up_block_types,
    )
    return model

if __name__ == "__main__":
    
    name = "cifar10"
    name = "lsun_church"
    config = load_config()
    