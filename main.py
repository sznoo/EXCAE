from omegaconf import OmegaConf
from transformers.models.clip.configuration_clip import CLIPConfig

from dataloader import build_dataloader
from train import train
from model import VideoClipMoeCrossAttentionModel

if __name__ == "__main__":
    dataloader_train = build_dataloader(
        annotation_path="/hub_data2/intern/jinwoo/didemo_ret_train.json",
        video_dir="/hub_data2/intern/jinwoo/videos",
        tokenizer_name="openai/clip-vit-base-patch32",
        batch_size=64,
        shuffle=True,
        num_workers=16,
    )

    dataloader_val = build_dataloader(
        annotation_path="/hub_data2/intern/jinwoo/didemo_ret_val.json",
        video_dir="/hub_data2/intern/jinwoo/videos",
        tokenizer_name="openai/clip-vit-base-patch32",
        batch_size=64,
        shuffle=True,
        num_workers=16,
    )

    dataloader_test = build_dataloader(
        annotation_path="/hub_data2/intern/jinwoo/didemo_ret_test.json",
        video_dir="/hub_data2/intern/jinwoo/videos",
        tokenizer_name="openai/clip-vit-base-patch32",
        batch_size=128,
        shuffle=True,
        num_workers=16,
    )

    excae_config_path = "/home/intern/jinwoo/EXCAE/config.yaml"
    excae_config = OmegaConf.load(excae_config_path)

    excae = VideoClipMoeCrossAttentionModel(excae_config)
    excae.build()
    excae.to("cuda:5")
    train(excae, dataloader_train, dataloader_val, dataloader_test, 50, "output/1")
