import torch
import torch.nn as nn
from transformers.models.clip.configuration_clip import CLIPConfig

from clip import CLIPModel
from moe_fusion import ECSModule
import open_clip


class VideoClipMoeCrossAttentionModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config
        self.clip_model_name = config.get("clip_model_name")
        self.moe_config = config.get("moe_config")
        self.neck_config = config.get("neck_config")
        self.clip_config = CLIPConfig.from_pretrained(self.clip_model_name)

    def build(self):
        self.clipmodel = CLIPModel.from_pretrained(
            self.clip_model_name, config=self.clip_config
        )
        # self.clipmodel, _, self.tokenizer = open_clip.create_model_and_transforms(
        #     "ViT-g-14", pretrained="laion2b_s34b_b88k"
        # )
        # self.clipmodel = self.clipmodel.to("cuda:5")
        self.clipmodel.to("cuda:5")
        self.moe_layer = ECSModule(
            self.moe_config.embed_dim,
            self.moe_config.num_experts_per_combination,
            self.moe_config.num_experts,
            self.moe_config.nhead,
            self.moe_config.num_layers,
            self.moe_config.num_embs,
        )
        self.norm = nn.LayerNorm(self.moe_config.embed_dim, eps=1e-5)

    def forward(self, sample_list):
        results = {}
        caption = sample_list.get("caption", None)
        video = sample_list.get("video", None)
        generate_captions = sample_list.get("generate_captions", None)

        # b, number_of_captions, sequence_length = generate_captions.input_ids.shape
        b, number_of_captions, sequence_length = generate_captions["input_ids"].shape

        # generate_captions_input_ids = generate_captions.input_ids.reshape(
        generate_captions_input_ids = generate_captions["input_ids"].reshape(
            b * number_of_captions, sequence_length
        )
        # generate_captions_input_mask = generate_captions.input_mask.reshape(
        generate_captions_input_mask = generate_captions["input_mask"].reshape(
            b * number_of_captions, sequence_length
        )
        generate_caption_features = self.clipmodel.get_text_features(
            generate_captions_input_ids, generate_captions_input_mask
        )
        generate_caption_features = generate_caption_features.reshape(
            b, number_of_captions, -1
        )
        b, t, c, h, w = video.shape
        video = video.reshape(-1, c, h, w)
        vis_features = self.clipmodel.get_image_features(video)
        vis_features = vis_features.reshape(b, t, -1)
        text_features = self.clipmodel.get_text_features(
            # caption.input_ids, caption.input_mask
            caption["input_ids"],
            caption["input_mask"],
        )

        video_features_concat = torch.cat(
            [generate_caption_features, vis_features], dim=1
        )
        video_features = self.moe_layer(
            {"caption": generate_caption_features, "video": vis_features}
        )
        video_features = video_features + video_features_concat.mean(dim=1)
        vis_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        results["text_features"] = text_features
        results["vis_features"] = vis_features

        return results
