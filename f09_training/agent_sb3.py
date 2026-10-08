# -*- coding: utf-8 -*-
from pathlib import Path

def load_sb3(alg: str, path: Path):
    import stable_baselines3 as sb3
    if alg.lower() == "auto":
        for _alg in ("PPO","SAC","TD3","A2C","DQN","DDPG"):
            try:
                return getattr(sb3, _alg).load(str(path))
            except Exception:
                pass
        raise RuntimeError("SB3 auto-detect failed")
    return getattr(sb3, alg).load(str(path))
