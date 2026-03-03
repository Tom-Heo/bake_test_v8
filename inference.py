import argparse
import math
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

TILE_SIZE = 2048
TILE_OVERLAP = 512


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


def _make_blend_weight(h, w, overlap):
    vert = torch.ones(h)
    horiz = torch.ones(w)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap + 2)[1:-1]
        vert[:overlap] = ramp
        vert[-overlap:] = ramp.flip(0)
        horiz[:overlap] = ramp
        horiz[-overlap:] = ramp.flip(0)
    return vert.unsqueeze(1) * horiz.unsqueeze(0)


def _tiled_forward(
    x, model, srgb2oklab, oklab2srgb, device, tile_size=TILE_SIZE, overlap=TILE_OVERLAP
):
    _, _, H, W = x.shape
    stride = tile_size - overlap

    nh = max(1, math.ceil((H - overlap) / stride))
    nw = max(1, math.ceil((W - overlap) / stride))

    need_h = (nh - 1) * stride + tile_size
    need_w = (nw - 1) * stride + tile_size

    if need_h > H or need_w > W:
        x = F.pad(x, (0, max(0, need_w - W), 0, max(0, need_h - H)), mode="reflect")

    blend = _make_blend_weight(tile_size, tile_size, overlap)
    output = torch.zeros(1, 3, need_h, need_w)
    weights = torch.zeros(1, 1, need_h, need_w)

    for i in range(nh):
        for j in range(nw):
            top = i * stride
            left = j * stride
            tile = x[:, :, top : top + tile_size, left : left + tile_size].to(device)

            with torch.no_grad():
                result = oklab2srgb(model(srgb2oklab(tile)))

            output[:, :, top : top + tile_size, left : left + tile_size] += (
                result.cpu() * blend
            )
            weights[:, :, top : top + tile_size, left : left + tile_size] += blend

    output /= weights.clamp(min=1e-8)
    return output[:, :, :H, :W]


def process_image(
    img_path: Path,
    model: BakeNet,
    srgb2oklab: Palette.sRGBtoOklabP,
    oklab2srgb: Palette.OklabPtosRGB,
    device: torch.device,
    tile_size: int = TILE_SIZE,
    tile_overlap: int = TILE_OVERLAP,
) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0)

    _, _, H, W = x.shape

    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, pH, pW = x.shape

    if pH <= tile_size and pW <= tile_size:
        with torch.no_grad():
            x_gpu = x.to(device)
            result = oklab2srgb(model(srgb2oklab(x_gpu)))
        result = result.cpu()
    else:
        result = _tiled_forward(
            x, model, srgb2oklab, oklab2srgb, device, tile_size, tile_overlap
        )

    return result[:, :, :H, :W].squeeze(0).clamp(0.0, 1.0)


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
            p
            for p in input_path.rglob("*")
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
