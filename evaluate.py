import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
from dataloader import build_dataloader
from model import VideoClipMoeCrossAttentionModel


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    all_text_features = []
    all_vis_features = []

    for batch in tqdm(dataloader, desc="Extracting features from train set"):
        batch = {
            "video": batch["video"].to(device),
            "caption": {
                "input_ids": batch["caption"]["input_ids"].to(device),
                "input_mask": batch["caption"]["input_mask"].to(device),
            },
            "generate_captions": {
                "input_ids": batch["generate_captions"]["input_ids"].to(device),
                "input_mask": batch["generate_captions"]["input_mask"].to(device),
            },
        }

        result = model(batch)
        text_features = result["text_features"]
        vis_features = result["vis_features"]

        all_text_features.append(text_features)
        all_vis_features.append(vis_features)

    all_text_features = torch.cat(all_text_features, dim=0)
    all_vis_features = torch.cat(all_vis_features, dim=0)

    return all_text_features, all_vis_features


@torch.no_grad()
def compute_retrieval_metrics(text_features, vis_features, topk=[1, 3, 5, 10]):
    text_features = F.normalize(text_features, dim=-1)
    vis_features = F.normalize(vis_features, dim=-1)

    sims = text_features @ vis_features.t()  # (N, N)
    N = sims.size(0)
    gt_ranks = torch.arange(N, device=sims.device)

    retrieval_metrics = {}
    for k in topk:
        topk_indices = sims.topk(k, dim=-1).indices
        correct = topk_indices.eq(gt_ranks.unsqueeze(-1))
        correct = correct.any(dim=-1).float()
        recall = correct.mean().item() * 100
        retrieval_metrics[f"R@{k}"] = recall

    return retrieval_metrics


def evaluate_train_dataset(model, train_dataloader, device=torch.device("cuda:5")):
    print(">>>evalueate_train_dataset111111")
    text_features, vis_features = extract_features(model, train_dataloader, device)
    print(">>>evalueate_train_dataset222222")
    retrieval_metrics = compute_retrieval_metrics(text_features, vis_features)

    print("Train Dataset Retrieval Metrics:")
    for metric, value in retrieval_metrics.items():
        print(f"{metric}: {value:.2f}%")

    return retrieval_metrics


if __name__ == "__main__":
    excae_config_path = "/home/intern/jinwoo/ECA4VTR/paper_ver/config.yaml"
    excae_config = OmegaConf.load(excae_config_path)
    excae = VideoClipMoeCrossAttentionModel(excae_config)
    state_dict = torch.load(
        "/home/intern/jinwoo/ECA4VTR/output/best_model.pth", map_location="cpu"
    )
    excae.load_state_dict(state_dict)
    excae.to("cuda:5")

    dataloader_test = build_dataloader(
        annotation_path="/hub_data2/intern/jinwoo/didemo_ret_test.json",
        video_dir="/hub_data2/intern/jinwoo/videos",
        tokenizer_name="openai/clip-vit-base-patch32",
        batch_size=64,
        shuffle=True,
        num_workers=16,
    )

    evaluate_train_dataset(excae, dataloader_test, device=torch.device("cuda:5"))
