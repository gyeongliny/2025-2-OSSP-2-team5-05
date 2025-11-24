import os
import torch
import random
import numpy as np


def seed_everything(seed: int = 50, deterministic: bool = True) -> None:
    """
    모든 랜덤 시드를 고정해서 재현성을 높여주는 함수.

    Args:
        seed        : 시드 값
        deterministic : True면 완전 재현성 모드 (cudnn.benchmark=False)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-GPU

    # cudnn 설정
    torch.backends.cudnn.deterministic = deterministic
    # 재현성을 우선할 땐 benchmark=False가 안전함
    torch.backends.cudnn.benchmark = not deterministic
