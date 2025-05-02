import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_similarity_matrix(similarity_matrix, title="Similarity Matrix Heatmap"):
    sims_np = similarity_matrix.detach().cpu().numpy()  # torch tensor → numpy
    plt.figure(figsize=(8, 6))
    sns.heatmap(sims_np, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.xlabel("Video Index")
    plt.ylabel("Text Index")
    plt.tight_layout()
    plt.savefig('sim.png')

def contrastive_loss(text_features, vis_features,labels=None, temperature=0.1):
    """
    NT-Xent (normalized temperature-scaled cross entropy loss)
    - text_features: (B, D)
    - vis_features: (B, D)
    """
    text_features = F.normalize(text_features, dim=-1)
    vis_features = F.normalize(vis_features, dim=-1)
    logits = torch.matmul(text_features, vis_features.T) / temperature
    if labels == None:
        labels = torch.arange(logits.size(0)).to(logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    loss = (loss_i2t + loss_t2i) / 2
    return loss

@torch.no_grad()
def compute_retrieval_metrics(text_features, vis_features, topk=[1, 3, 5, 10]):
    text_features = F.normalize(text_features, dim=-1)
    vis_features = F.normalize(vis_features, dim=-1)

    sims = text_features @ vis_features.t()  # (N, N), cosine similarity
    visualize_similarity_matrix(sims, title="Text-Video Similarity Heatmap")
    N = sims.size(0)
    gt_indices = torch.arange(N, device=text_features.device)  # 정답은 diagonal

    retrieval_metrics = {}
    for k in topk:
        topk_indices = sims.topk(k, dim=-1).indices  # (N, k)
        correct = topk_indices.eq(gt_indices.unsqueeze(1))  # (N, k)
        recall = correct.any(dim=-1).float().mean().item() * 100
        retrieval_metrics[f"R@{k}"] = recall

    return retrieval_metrics
