import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from core.net import BakeNet
from core.heo import Heo
from core.palette import Palette
from core.augments import BakeAugment
from data.dataset import BakeDataset
from utils import get_kst_logger, EMA, CheckpointManager, Visualizer


def get_args():
    parser = argparse.ArgumentParser(description="BakeNet Training Pipeline")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--restart", action="store_true", help="처음부터 새롭게 학습을 시작합니다."
    )
    group.add_argument(
        "--resume",
        type=str,
        metavar="PATH",
        help="지정한 체크포인트부터 학습을 재개합니다.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    cfg = Config()

    # 1. 시스템 및 로거 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_kst_logger("BakeNet", log_dir=cfg.log_dir)
    logger.info(
        f"디바이스: {device} | 학습 모드: {'Resume' if args.resume else 'Restart'}"
    )

    # 2. 데이터셋 및 로더 준비
    logger.info("데이터셋 초기화 중...")
    dataset = BakeDataset(root_dir=cfg.data_dir, patch_size=cfg.patch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 3. 모델, 손실 함수, 옵티마이저 초기화
    model = BakeNet(bottleneck_dim=cfg.bottleneck_dim).to(device)
    criterion = Heo.HeoLoss().to(device)
    optimizer = Heo.Heopimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # [Color Space] 시각화 전용 역변환기 (학습 파라미터 없음)
    oklab2srgb = Palette.OklabPtosRGB().to(device)

    # [Augmentation] BakeAugment: sRGB → (degraded_oklab, clean_oklab) 쌍 생성
    augment = BakeAugment(strength=cfg.augment_strength).to(device)

    # 4. 유틸리티 (EMA, 체크포인트, 시각화) 초기화
    ema = EMA(model, decay=cfg.ema_decay)
    ckpt_manager = CheckpointManager(save_dir=cfg.checkpoint_dir, max_keep=cfg.max_keep)
    visualizer = Visualizer(output_dir=cfg.output_dir)

    # 5. 스케줄러 설정 (Step-wise Warm-up + Step-wise Exponential Decay)
    steps_per_epoch = len(dataloader)
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return cfg.scheduler_gamma ** (current_step - warmup_steps)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 6. Restart / Resume 분기 처리
    start_epoch = 0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {args.resume}")
        start_epoch = ckpt_manager.load(args.resume, model, ema, optimizer, scheduler)
        logger.info(
            f"[{args.resume}] 로드 완료. Epoch {start_epoch}부터 학습을 재개합니다."
        )
    else:
        logger.info("새로운 모델 가중치로 학습을 시작합니다. (--restart)")

    # 7. 메인 학습 루프
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, gt_srgb in enumerate(dataloader):
            gt_srgb = gt_srgb.to(device, non_blocking=True)

            # [BakeAugment] 깨끗한 이미지(gt_srgb)를 열화시켜 (degraded, clean) OklabP 쌍 생성
            with torch.no_grad():
                in_oklab, gt_oklab = augment(gt_srgb)

            # [Forward]
            pred_oklab = model(in_oklab)

            # [Loss] FP32 환경 강제 유지 (AMP 사용 안 함)
            loss = criterion(pred_oklab, gt_oklab)

            # [Backward & Optimize] Gradient Clipping 사용 안 함
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # [Step-wise Updates] 매 배치마다 스케줄러와 EMA 업데이트
            scheduler.step()
            ema.update()

            epoch_loss += loss.item()

        # Epoch 통계 로깅
        avg_loss = epoch_loss / steps_per_epoch
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch:04d}/{cfg.epochs:04d}] Loss: {avg_loss:.6f} | LR: {current_lr:.8f}"
        )

        # 8. Epoch 종료 후 평가 및 시각화 (EMA 가중치 적용)
        ema.apply_shadow()
        model.eval()

        with torch.no_grad():
            hr_oklab = model(in_oklab)

            # [Color Space] 시각화를 위해 OklabP -> sRGB로 원상 복구
            hr_srgb = oklab2srgb(hr_oklab)
            in_srgb_viz = oklab2srgb(in_oklab)
            gt_srgb_viz = oklab2srgb(gt_oklab)

            # (degraded | pred | clean) 순서로 병합하여 저장
            visualizer.save_epoch_result(epoch, in_srgb_viz, hr_srgb, gt_srgb_viz)

        # 다음 Epoch 학습을 위해 원본 가중치로 복구
        ema.restore()
        model.train()

        # 9. 체크포인트 저장
        saved_path = ckpt_manager.save(epoch + 1, model, ema, optimizer, scheduler)
        logger.info(f"Epoch {epoch} 체크포인트 저장 완료: {saved_path}")


if __name__ == "__main__":
    main()
