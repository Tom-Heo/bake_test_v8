"""
BakeDataset: 사용자 업로드 이미지를 로드하여 sRGB 텐서로 반환.
BakeAugment와 연동하여 clean sRGB를 열화시킨 (degraded, clean) 쌍으로 학습에 사용됩니다.
"""

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


class BakeDataset(Dataset):
    """
    범용 이미지 데이터셋 로더.
    지정 폴더 내 모든 이미지를 로드하고 Random Crop하여 단일 sRGB 텐서를 반환합니다.
    """

    def __init__(self, root_dir: str = "dataset", patch_size: int = 512):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size

        self.paths = sorted(
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not self.paths:
            raise FileNotFoundError(
                f"이미지를 찾을 수 없습니다. {self.root_dir.resolve()}에 "
                f"이미지 파일({', '.join(SUPPORTED_EXTENSIONS)})을 배치해 주세요."
            )

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")

        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            scale = self.patch_size / min(w, h)
            img = TF.resize(img, [round(h * scale), round(w * scale)])

        i, j, th, tw = transforms.RandomCrop.get_params(
            img, (self.patch_size, self.patch_size)
        )
        img = TF.crop(img, i, j, th, tw)

        return self.to_tensor(img)
