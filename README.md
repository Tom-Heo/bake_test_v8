# BakeNet v8

GPU 가속 색상 열화(Degradation) 파이프라인과 OklabP 색공간 기반의 AI 색상 복원 모델.

## 구조

```
bake_test_v8/
├── config.py            # 학습 하이퍼파라미터 (단일 진실 공급원)
├── train.py             # 학습 파이프라인
├── inference.py         # 추론 (단일 이미지 / 폴더)
├── utils.py             # EMA, CheckpointManager, Visualizer, Logger
├── core/
│   ├── net.py           # BakeNet (U-Net 인코더-디코더)
│   ├── block.py         # Block, Stage, Bottleneck
│   ├── augments.py      # BakeAugment (GPU 색상 열화 파이프라인)
│   ├── heo.py           # HeLU, HeoGate, Heopimizer, HeoLoss
│   └── palette.py       # sRGB <-> OklabP 색공간 변환
└── data/
    └── dataset.py       # BakeDataset (이미지 로더)
```

## 설치

```bash
pip install -r requirements.txt
```

## 데이터 준비

`dataset/` 폴더에 학습용 이미지를 배치합니다. 하위 폴더 포함, PNG/JPG/JPEG/BMP/TIFF/WEBP를 지원합니다.

## 학습

```bash
# 새로 시작
python train.py --restart

# 체크포인트에서 재개
python train.py --resume checkpoints/ckpt_epoch_0100.pt
```

학습 설정은 `config.py`에서 관리합니다.

## 추론

```bash
# 단일 이미지
python inference.py --checkpoint checkpoints/ckpt_epoch_0100.pt --input photo.jpg --output result.png

# 폴더 전체
python inference.py --checkpoint checkpoints/ckpt_epoch_0100.pt --input input_dir/ --output output_dir/
```
