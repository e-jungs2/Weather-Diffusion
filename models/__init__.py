# models/__init__.py
try:
    # 있으면 finetune 버전 사용
    from models_finetune import DenoisingDiffusion as _DDM, DiffusiveRestoration as _DR
    print("[models] Using FINETUNE models")
except Exception:
    # 없으면 기본으로 폴백
    from .ddm import DenoisingDiffusion as _DDM
    from .restoration import DiffusiveRestoration as _DR
    print("[models] Using DEFAULT models")

DenoisingDiffusion = _DDM
DiffusiveRestoration = _DR
__all__ = ["DenoisingDiffusion", "DiffusiveRestoration"]
