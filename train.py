import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from metric import contrastive_loss, compute_retrieval_metrics
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adamax
from dataloader import DataLoader
from torch import amp
import torch.nn.functional as F


def setup_ddp():
    torch.distributed.init_process_group(backend="nccl")  #  DDP 초기화
    local_rank = int(os.environ["LOCAL_RANK"])  #  torchrun 환경변수 사용
    torch.cuda.set_device(local_rank)  #  GPU 지정
    return local_rank


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    epoch: int,
    scaler,
    log_interval: int = 10,
):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, batch in progress_bar:
        # 모든 batch 데이터를 device로 이동
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

        # === video만 섞기 ===
        batch_size = batch["video"].size(0)
        indices = torch.randperm(batch_size)
        video_shuffled = batch["video"][indices]

        optimizer.zero_grad()

        with amp.autocast("cuda:5"):
            # text_features, vis_features = model(batch)
            batch["video"] = video_shuffled
            labels = torch.argsort(indices).to(device)

            result = model(batch)
            text_features = result["text_features"]
            vis_features = result["vis_features"]
            loss = contrastive_loss(text_features, vis_features, labels)

        # Mixed Precision: scaled backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % log_interval == 0:
            progress_bar.set_description(
                f"Epoch {epoch} Step {step} Loss {loss.item():.4f}"
            )

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")

    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_text_features = []
    all_vis_features = []

    for batch in tqdm(dataloader, desc="Evaluating"):
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

    all_text_features = torch.cat(all_text_features, dim=0)  # (N, D)
    all_vis_features = torch.cat(all_vis_features, dim=0)  # (N, D)

    return all_text_features, all_vis_features


def train(
    model,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int,
    save_dir: str,
    device: torch.device = torch.device("cuda:5"),
    log_interval: int = 100,
):
    os.makedirs(save_dir, exist_ok=True)

    optimizer = Adamax(model.parameters(), lr=2e-6)
    scaler = amp.GradScaler("cuda:5")
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs"))

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # ===== Train =====
        train_loss = train_one_epoch(
            model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            log_interval=log_interval,
        )

        # ===== Validation =====
        text_features, vis_features = evaluate(
            model,
            dataloader=val_dataloader,
            device=device,
        )

        # Validation loss 계산 (optional)
        val_loss = contrastive_loss(text_features, vis_features).item()

        print(f"Validation Loss after Epoch {epoch}: {val_loss:.4f}")

        # === Retrieval Metrics ===
        text_features, vis_features = evaluate(
            model,
            dataloader=test_dataloader,
            device=device,
        )
        # print(text_features.shape, vis_features.shape)
        cos_sim = F.cosine_similarity(text_features, vis_features, dim=-1)
        print("Mean cosine similarity between text/video pairs:", cos_sim.mean().item())

        retrieval_metrics = compute_retrieval_metrics(
            text_features, vis_features, topk=[1, 3, 5, 10]
        )

        # TensorBoard 기록
        for k, value in retrieval_metrics.items():
            print(f"{k}: {value:.2f}%")
            writer.add_scalar(f"Retrieval/{k}", value, epoch)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        # Best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "best_model_1.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch} with val_loss {val_loss:.4f}")

    writer.close()
    print("Training completed!")
