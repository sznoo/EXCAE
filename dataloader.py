import os
import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import decord
from decord import VideoReader, cpu
from transformers import CLIPTokenizer
import torch.nn.functional as F

# 디코더 초기화
decord.bridge.set_bridge("torch")


class DiDeMoDataset(Dataset):
    def __init__(
        self,
        annotation_path: str,
        video_dir: str,
        tokenizer_name="openai/clip-vit-base-patch32",
        max_length: int = 70,
    ):
        """
        DiDeMo dataset for video-text retrieval.

        Args:
            annotation_path (str): Path to the annotation json file.
            video_dir (str): Directory containing video files.
            tokenizer_name (str): CLIP tokenizer name.
            max_length (int): Max token length for captions.
        """
        super().__init__()
        self.video_dir = video_dir

        # Annotation 리스트 불러오기
        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)

        # CLIP tokenizer 불러오기
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict containing 'video' tensor and 'caption' inputs (input_ids, attention_mask)
        """
        item = self.annotations[idx]
        video_filename = item["video"].rsplit(".", 2)[0] + ".mp4"
        video_path = os.path.join(self.video_dir, video_filename)
        captions = item["caption"]

        # 여러 개의 캡션을 공백으로 이어붙여 하나의 문장으로 사용
        caption = " ".join(captions)

        # 캡션 토크나이즈
        caption_tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 비디오 로드
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        num_sampled_frames = 8
        indices = torch.linspace(0, num_frames - 1, steps=num_sampled_frames).long()
        frames = vr.get_batch(indices)  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)

        # (0~255) 정수형 -> float형 변환
        frames = frames.float()

        # Resize
        frames = F.interpolate(
            frames, size=(224, 224), mode="bilinear", align_corners=False
        )

        # Normalize
        frames = (frames / 255.0 - self.mean) / self.std


        generate_captions = [""] * 10  # 길이 10의 빈 문자열 리스트

        generate_tokenized = self.tokenizer(
            generate_captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "video": frames,
            "caption": {
                "input_ids": caption_tokenized["input_ids"].squeeze(0),
                "input_mask": caption_tokenized["attention_mask"].squeeze(0),
            },
            "generate_captions": {
                "input_ids": generate_tokenized["input_ids"],         # shape: (10, L)
                "input_mask": generate_tokenized["attention_mask"]  # shape: (10, L)
            },
        }


def build_dataloader(
    annotation_path: str,
    video_dir: str,
    tokenizer_name: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Helper function to build the DiDeMo DataLoader.
    """
    dataset = DiDeMoDataset(
        annotation_path=annotation_path,
        video_dir=video_dir,
        tokenizer_name=tokenizer_name,
        max_length=70,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataloader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function to batch samples together.
    """
    videos = torch.stack([item["video"] for item in batch])
    input_ids = torch.stack([item["caption"]["input_ids"] for item in batch])
    input_masks = torch.stack([item["caption"]["input_mask"] for item in batch])

    return {
        "video": videos,  # (B, T, C, H, W)
        "caption": {
            "input_ids": input_ids,  # (B, L)
            "input_mask": input_masks,  # (B, L)
        },
        "generate_captions": {
            "input_ids": input_ids.unsqueeze(1),  # (B,1, L)
            "input_mask": input_masks.unsqueeze(1),  # (B,1, L)
        },
    }
