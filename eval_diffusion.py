import argparse
import os
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import datasets
from models import DenoisingDiffusion, DiffusiveRestoration

def dict2namespace(d):
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns

def parse_args_and_config():
    p = argparse.ArgumentParser(description='Restoring Weather - Evaluation')
    p.add_argument("--config", type=str, required=True, help="Config file path or name under ./configs")
    p.add_argument("--resume", type=str, required=True, help="Path to the checkpoint (.pth.tar)")
    p.add_argument("--grid_r", type=int, default=16, help="Patch overlap r (kept for compatibility)")
    p.add_argument("--sampling_timesteps", type=int, default=25, help="Number of sampling steps")
    p.add_argument("--image_folder", type=str, default="results/images/", help="Where to save restored images")
    p.add_argument("--seed", type=int, default=61)

    args = p.parse_args()

    # config 경로 강건화: 전체 경로 or ./configs/파일명 둘 다 허용
    cfg_path = args.config if os.path.isfile(args.config) else os.path.join("configs", args.config)
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    return args, config

def main():
    args, config = parse_args_and_config()

    # device & seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config.device = device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = False

    # Colab 안전: 로더 만들기 전에 num_workers=0
    if hasattr(config, "data"):
        config.data.num_workers = 0

    # data loading — 우리 로더는 train/val 고정
    print(f"=> using dataset '{config.data.dataset}'")
    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders(parse_patches=False)  # ← validation 인자 제거

    # 모델 생성
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)

    os.makedirs(args.image_folder, exist_ok=True)

    # 복원 실행
    with torch.no_grad():
        # 모델 시그니처가 아직 validation=str 를 요구하면 아래 줄을:
        # model.restore(val_loader, validation="val", r=args.grid_r)
        # 로 쓰고, 모델에서 validation 인자를 제거했으면 아래 줄을 쓰세요.
        model.restore(val_loader, r=args.grid_r)

    print(f"=> done. images saved to: {args.image_folder}")

if __name__ == '__main__':
    main()
