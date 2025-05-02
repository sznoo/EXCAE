from model import VideoClipMoeCrossAttentionModel
from omegaconf import OmegaConf
import torch

if __name__ == "__main__":

    excae_config_path = "/home/intern/jinwoo/ECA4VTR/paper_ver/config.yaml"
    excae_config = OmegaConf.load(excae_config_path)

    excae = VideoClipMoeCrossAttentionModel(excae_config)
    excae.build()
    excae.to("cuda:5")
    # model = VideoClipMoeCrossAttentionModel()
    excae.load_state_dict(
        torch.load(
            "/home/intern/jinwoo/ECA4VTR/output/best_model_1.pth", weights_only=True
        )
    )
    # model.eval()
