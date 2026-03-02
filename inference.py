import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from config import Config
from core.net import BakeNet
from core.palette import Palette

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def load_model(checkpoint_path: str, device: torch.device, cfg: Config) -> BakeNet:
    model = BakeNet(bottleneck_dim=cfg.bottleneck_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "ema_state" in ckpt:
        device_map = next(model.parameters()).device
        state = {k: v.to(device_map) for k, v in ckpt["ema_state"].items()}
        model.load_state_dict(state)
    else:
        model.load_state_dict(ckpt["model_state"])

    model.eval()
    return model


def process_image(
    img_path: Path,
    model: BakeNet,
    srgb2oklab: Palette.sRGBtoOklabP,
    oklab2srgb: Palette.OklabPtosRGB,
    device: torch.device,
) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    _, _, H, W = x.shape

    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        oklab = srgb2oklab(x)
        pred = model(oklab)
        result = oklab2srgb(pred)

    return result[:, :, :H, :W].squeeze(0).cpu().clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="BakeNet Inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="체크포인트 파일 경로"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="입력 이미지 파일 또는 폴더 경로"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="출력 이미지 파일 또는 폴더 경로"
    )
    args = parser.parse_args()

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device, cfg)
    srgb2oklab = Palette.sRGBtoOklabP().to(device)
    oklab2srgb = Palette.OklabPtosRGB().to(device)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        result = process_image(input_path, model, srgb2oklab, oklab2srgb, device)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(result, output_path)
        print(f"저장 완료: {output_path}")

    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        images = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(
                f"이미지를 찾을 수 없습니다: {input_path.resolve()}"
            )
        for i, img_path in enumerate(images, 1):
            result = process_image(img_path, model, srgb2oklab, oklab2srgb, device)
            save_path = output_path / f"{img_path.stem}.png"
            save_image(result, save_path)
            print(f"[{i}/{len(images)}] {save_path}")

    else:
        raise FileNotFoundError(f"입력 경로를 찾을 수 없습니다: {args.input}")


if __name__ == "__main__":
    main()
