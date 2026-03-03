import argparse
import tempfile
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image
import gradio as gr
from gradio_imageslider import ImageSlider

from config import Config
from core.palette import Palette
from inference import load_model, process_image

CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt"}


def find_latest_checkpoint(checkpoint_dir: str) -> Path:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"체크포인트 디렉토리가 없습니다: {ckpt_dir.resolve()}")

    candidates = sorted(
        (p for p in ckpt_dir.iterdir() if p.suffix.lower() in CHECKPOINT_EXTENSIONS),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_dir.resolve()}")

    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description="BakeNet Web App")
    parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트 파일 경로")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or str(find_latest_checkpoint(cfg.checkpoint_dir))
    print(f"체크포인트: {ckpt_path}")
    model = load_model(ckpt_path, device, cfg)
    srgb2oklab = Palette.sRGBtoOklabP().to(device)
    oklab2srgb = Palette.OklabPtosRGB().to(device)

    def process(image_path: str):
        if image_path is None:
            raise gr.Error("이미지를 먼저 업로드해 주세요.")

        try:
            result_tensor = process_image(
                Path(image_path), model, srgb2oklab, oklab2srgb, device
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

            output_dir = tempfile.mkdtemp()
            output_name = f"{Path(image_path).stem}_baked.png"
            output_path = Path(output_dir) / output_name
            save_image(result_tensor, str(output_path))

            original = Image.open(image_path).convert("RGB")
            result = Image.open(output_path).convert("RGB")

            return (original, result), str(output_path)
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}")

    with gr.Blocks(title="BakeNet") as demo:
        gr.Markdown("# BakeNet")

        input_image = gr.Image(type="filepath", label="업로드")
        run_btn = gr.Button("변환", variant="primary")
        slider = ImageSlider(label="Before / After")
        download = gr.File(label="다운로드")

        run_btn.click(
            fn=process,
            inputs=input_image,
            outputs=[slider, download],
        )

    demo.launch(server_name=args.host, server_port=args.port, share=True)


if __name__ == "__main__":
    main()
