# -*- coding: utf-8 -*-
# آموزش SB3 روی TradingEnv + Eval/Checkpoint + ذخیرهٔ .zip در staging
# بعلاوه: ثبت و ذخیرهٔ ابعاد observation (برای جلوگیری از mismatch در بک‌تست)
# توضیحات فارسی (غیرقابل‌اجرا)؛ پیام‌های runtime انگلیسی هستند.
# python -m f09_training\agent_sb3_train.py --symbol XAUUSD --alg PPO --total-timesteps 800000 --eval-freq 50000 --eval-steps 5000 --normalize

from __future__ import annotations
import numpy as np
import argparse, logging, time, json
from pathlib import Path
from typing import Dict, Any, Optional
from stable_baselines3.common.monitor import Monitor  # برای eval

try:
    from f02_utils.config_loader import load_config
    from f06_env.OLD_files.OLD_trading_env import TradingEnv, EnvConfig
    from f06_env.OLD_files.OLD_utils import paths_from_cfg
    from f10_evaluation.backtest import transform_obs
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from f02_utils.config_loader import load_config
    from f06_env.OLD_files.OLD_trading_env import TradingEnv, EnvConfig
    from f06_env.OLD_files.OLD_utils import paths_from_cfg
    from f10_evaluation.backtest import transform_obs

from f06_env.OLD_files.OLD_utils import paths_from_cfg, resolve_spread_selection

try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore

import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    # StopTrainingOnNoModelImprovement,
    BaseCallback
)
#from f04_features.feature_store import ensure_published_from_cfg

logger = logging.getLogger("sb3_train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)

# --- USD-aware eval early-stop callback ---
# هدف:
# کال‌بک ارزیابی که بعد از هر ارزیابی، ep_pnl_usd را چک کند
# و با «صبر» (patience) روی معیار دلاری، early-stop کند.
class USDEvalStopCallback(BaseCallback):
    """کال‌بک توقف زودهنگام بر اساس ep_pnl_usd و/یا reward."""
    def __init__(self, patience_evals: int = 5, use_reward: bool = True,
                 min_delta_reward: float = 0.0, min_delta_usd: float = 0.0,
                 verbose: int = 1):
        super().__init__(verbose)
        self.best_mean_usd = -float("inf")
        self.best_mean_reward = -float("inf")
        self.no_improve = 0
        self.patience = int(patience_evals)
        self.use_reward = bool(use_reward)
        self.min_delta_reward = float(min_delta_reward)
        self.min_delta_usd = float(min_delta_usd)


    def _on_step(self) -> bool:
        """بررسی نتیجهٔ آخرین ارزیابی و اعمال توقف زودهنگام بر اساس USD و/یا reward.

        - منبع USD: فیلد خصوصی env بنام `_last_ep_pnl_usd` که در پایان اپیزود ست می‌شود.
        - منبع reward: فیلدهای EvalCallback (last_mean_reward / best_mean_reward).
        - لاگ CSV: هر ارزیابی یک ردیف در evaluations_usd.csv با ستون‌های (timesteps, ep_pnl_usd).
        """
        # 1) دسترسی به env ارزیابی از روی EvalCallback والد
        env = getattr(self.parent, "eval_env", None)
        if env is None:
            return True

        # 2) robust unwrap: VecEnv → wrappers(.env) → adapter._env → TradingEnv
        base = env
        # اگر EvalCallback داخل خودش DummyVecEnv ساخته باشد
        if hasattr(base, "envs") and getattr(base, "envs"):
            base = base.envs[0]

        # بازکردن زنجیره‌ی .env (Monitor/TimeLimit/…)
        for _ in range(12):
            inner = getattr(base, "env", None)
            if inner is None:
                break
            base = inner

        # عبور از آداپتر SB3 به خود TradingEnv
        for _ in range(3):
            inner = getattr(base, "_env", None)
            if inner is None:
                break
            base = inner
            
            
        # 3) خواندن USD آخر اپیزود
        usd = float(getattr(base, "_last_ep_pnl_usd", 0.0))

        # [LOG:TR_STATS] — transitions & entries/exits & lots
        tr   = int(getattr(base, "_last_ep_transitions", getattr(base, "_ep_transitions", 0)))
        ents = int(getattr(base, "_last_ep_entries",     getattr(base, "_ep_entries", 0)))
        exits= int(getattr(base, "_last_ep_exits",       getattr(base, "_ep_exits", 0)))
        lots = float(getattr(base, "_position_lots", 0.0))

        pips_last = float(getattr(base, "_last_ep_pnl_pips", float("nan")))
        logger.info("USD-EVAL | debug pv=%.6f pnl_pips_last=%.6f usd=%.6f trans=%d entries=%d exits=%d lots=%.4f",
            float(getattr(base, "_point_value", 0.0)), pips_last,
            usd, tr, ents, exits, lots)
        
        # [LOG:ACT-HIST]
        a_neg  = int(getattr(base, "_last_ep_act_neg",  getattr(base, "_ep_act_neg", 0)))
        a_zero = int(getattr(base, "_last_ep_act_zero", getattr(base, "_ep_act_zero", 0)))
        a_pos  = int(getattr(base, "_last_ep_act_pos",  getattr(base, "_ep_act_pos", 0)))
        logger.info("USD-EVAL | actions neg=%d zero=%d pos=%d", a_neg, a_zero, a_pos)

        # اگر همهٔ اکشن‌ها صفر بود، برای عیب‌یابی هشدار بده (بدون تغییر رفتار) --- starta2
        if (a_zero > 0) and (a_neg == 0) and (a_pos == 0):
            logger.info("USD-EVAL | warning: policy produced only HOLD actions; consider increasing exploration or revisiting rewards.")
        # --- enda2


        # [USD:EVAL:DBG] print pv/pip for XAUUSD --- start(temp)
        pv = float(getattr(base, "_point_value", 0.0))
        if self.verbose:
            logger.info("USD-EVAL | debug pv=%.6f pip_size=%.6f usd=%.6f", pv, float(getattr(base,"_pip_size",0.0)), usd)
        # --- end(temp)

        # 4) ثبت در CSV (timesteps, ep_pnl_usd)
        try:
            from pathlib import Path  # اطمینان از وجود Path
            log_dir = Path(getattr(self.parent, "log_path", ""))  # type: ignore[attr-defined]
            if str(log_dir):
                log_dir.mkdir(parents=True, exist_ok=True)
                csv_p = log_dir / "evaluations_usd.csv"
                ts_now = int(getattr(self.parent, "num_timesteps", 0))
                hdr_needed = not csv_p.exists()
                with csv_p.open("a", encoding="utf-8") as f:
                    if hdr_needed:
                        f.write("timesteps,ep_pnl_usd\n")
                    f.write(f"{ts_now},{usd}\n")
        except Exception:
            pass  # لاگ CSV نباید آموزش را از کار بیندازد

        # 5) خواندن reward از EvalCallback
        last_r = float(getattr(self.parent, "last_mean_reward", float("nan")))
        best_r_parent = getattr(self.parent, "best_mean_reward", None)
        if best_r_parent is not None:
            self.best_mean_reward = max(self.best_mean_reward, float(best_r_parent))

        # 6) ارزیابی بهبود (ترکیبی)
        improved_usd = (usd > self.best_mean_usd + self.min_delta_usd)
        improved_rwd = (self.use_reward and (not np.isnan(last_r))
                        and (last_r > self.best_mean_reward + self.min_delta_reward))

        if improved_usd or improved_rwd:
            if improved_usd:
                self.best_mean_usd = usd
            if improved_rwd:
                self.best_mean_reward = last_r
            self.no_improve = 0
            if self.verbose:
                logger.info(
                    "USD-EVAL | improve: usd=%.6f best_usd=%.6f reward=%.6f best_r=%.6f",
                    usd, self.best_mean_usd, last_r, self.best_mean_reward
                )
        else:
            self.no_improve += 1
            if self.verbose:
                logger.info(
                    "USD-EVAL | no improve (%d/%d): usd=%.6f best=%.6f reward=%.6f best_r=%.6f",
                    self.no_improve, self.patience, usd, self.best_mean_usd, last_r, self.best_mean_reward
                )
            if self.no_improve >= self.patience:
                logger.info("USD-EVAL | early stopping (hybrid: USD+reward).")
                return False

        return True



class _SB3EnvAdapter(gym.Env):
    """
    آداپتر TradingEnv برای SB3.
    - observation → 1D با transform_obs
    - action_space: Discrete(3) mapped to {-1,0,+1}
    """
    metadata = {"render.modes": []}

    def _sanitize(self, v):
        # --- تضمین اعداد متناهی ---
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if not np.isfinite(v).all():
            if getattr(self, "_nan_warn_n", 0) < 3:
                logger.warning("NaN/Inf found in observation; replacing with 0.0")
            self._nan_warn_n = getattr(self, "_nan_warn_n", 0) + 1
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v


    def _fix_dim(self, v):
        # --- تضمین طول ثابت با پد/برش ---
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.shape[0] < self._obs_dim:
            v = np.pad(v, (0, self._obs_dim - v.shape[0]))
        elif v.shape[0] > self._obs_dim:
            v = v[-self._obs_dim:]
        return v


    def __init__(self, cfg: Dict[str, Any], env_cfg: EnvConfig,
                 obs_mode: str = "mean", last_k: int = 16, split: str = "train"):
        super().__init__()
        self.cfg, self.env_cfg = cfg, env_cfg
        self._mode, self._last_k, self._split = obs_mode, int(last_k), split
        self._nan_warn_n = 0  # rate-limit NaN/Inf warnings
        self._env = TradingEnv(cfg=self.cfg, env_cfg=self.env_cfg)
        obs, _ = self._env.reset(split=self._split)
        vec = transform_obs(obs, mode=self._mode, last_k=self._last_k)

        vec = self._sanitize(vec)
        self._obs_dim = int(np.asarray(vec).size)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
            )

        env_block = (self.cfg.get("env") or {})
        action_cfg = (env_block.get("action_space") or {})
        n_actions = len(action_cfg.get("discrete_actions", []))
        self.action_space = gym.spaces.Discrete(n_actions)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            try:
                self._env.seed(seed)
            except Exception:
                pass
        obs, info = self._env.reset(split=self._split)
        vec = transform_obs(obs, mode=self._mode, last_k=self._last_k)
        vec = self._sanitize(vec)
        vec = self._fix_dim(vec)
        vec = vec.astype(np.float32, copy=False)
        
        return vec, info

    def step(self, action: int):
        # اطمینان از نوع صحیح اکشن (پیشگیری از numpy scalar/array) --- start
        try:
            action = int(action)
        except Exception:
            action = int(np.asarray(action).reshape(-1)[0])
        # --- end

        act = int(action)
        logger.debug("ADAPTER | action(raw)=%s (pass-through)", act)
        sr = self._env.step(act)

        logger.debug("ADAPTER | step reward=%.6f done=%s/%s", float(sr.reward), str(bool(sr.terminated)), str(bool(sr.truncated)))
        terminated, truncated = bool(sr.terminated), bool(sr.truncated)
        vec = transform_obs(sr.observation, mode=self._mode, last_k=self._last_k)
        vec = self._sanitize(vec)
        vec = self._fix_dim(vec)
        # summary log every 100 steps (keeps INFO signal without flooding)
        self._log_i = getattr(self, "_log_i", 0) + 1
        if self._log_i % 200 == 0:
            logger.debug("ADAPTER | summary i=%d action=%d mapped=%d reward=%.6f", self._log_i, int(action), int(act), float(sr.reward))

        vec = vec.astype(np.float32, copy=False)

        return vec, float(sr.reward), terminated, truncated, (sr.info or {})


def _parse_args():
    p = argparse.ArgumentParser(description="Train SB3 on TradingEnv and save best model to staging (.zip)")
    p.add_argument("--symbol", required=True)
    p.add_argument("-c","--config", default="f01_config/config.yaml")
    p.add_argument("--base-tf", default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--alg", default=None, choices=["PPO","SAC","TD3","A2C","DQN","DDPG"])
    p.add_argument("--policy", default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--eval-freq", type=int, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=None)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--obs-agg", default="mean", choices=["mean","last","lastk_mean","flat"])
    p.add_argument("--last-k", type=int, default=16)
    p.add_argument("--eval-steps", type=int, default=3000)
    p.add_argument("--entropy-coef", type=float, default=None, 
                   help="Override PPO ent_coef (exploration). If None, use SB3 default.")
    # هدف: امکان override کردن target_kl (برای الگوریتم‌هایی مانند PPO)
    # توضیح فارسی: اگر مقدار دهی شود و الگوریتم پارامتر target_kl داشته باشد، قبل از ساخت مدل اعمال می‌شود.
    p.add_argument("--target-kl", type=float, default=None,
                   help="Override target_kl if supported by the selected algorithm.")
    return p.parse_args()

def _make_alg(alg_name: str):
    if not hasattr(sb3, alg_name):
        raise ValueError(f"Unknown SB3 algorithm: {alg_name}")
    return getattr(sb3, alg_name)

def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)

    bt = ((cfg.get("evaluation") or {}).get("backtest") or {})
    sel = resolve_spread_selection(cfg, has_broker_series=bool(bt.get("reuse_historical_spread")))
    if sel["source"] != "zero_default":
        logger.info("[COSTS] spread source=%s fixed_pts=%.4f", sel["source"], float(sel["value"]))

    rl = ((cfg.get("training") or {}).get("rl") or {})
    alg_name = (args.alg or rl.get("algorithm") or "PPO")
    policy = (args.policy or rl.get("policy") or "MlpPolicy")
    train_cfg = (cfg.get("training") or {})
    total_ts = int(args.total_timesteps or train_cfg.get("total_timesteps", 200_000))
    eval_freq = int(args.eval_freq or train_cfg.get("eval_freq_steps", 25_000))
    ckpt_int = int(args.checkpoint_interval or train_cfg.get("checkpoint_interval_steps", 50_000))

    early_cfg = (train_cfg.get("early_stop") or {})
    use_early = (args.early_stop or bool(early_cfg.get("enabled", True)))

    env_block = cfg.get("env") or {}
    symbol = args.symbol.upper()
    base_tf = (args.base_tf or env_block.get("base_tf") or "M1")
    window = int(args.window or env_block.get("window_size", 128))
    normalize = bool(args.normalize or env_block.get("normalize", False))

    # [AUTO-PUBLISH] --- start
    # ساخت/تزریق فیچرها به processed بر اساس cfg قبل از ساخت Env
    """
    try:
        added_cols = ensure_published_from_cfg(cfg, symbol, base_tf)
        logger.info("Auto-publish features into processed done: %d columns.", added_cols)
    except Exception as e:
        logger.warning("Auto-publish skipped: %s", e)
    """
    # [AUTO-PUBLISH] --- end

    env_cfg = EnvConfig(
        symbol=symbol, base_tf=base_tf,
        window_size=window, reward_mode=(env_block.get("reward_mode") or "pnl"),
        normalize=normalize
    )

    paths = paths_from_cfg(cfg)
    staging = paths["models_dir"] / "staging"
    (staging / "checkpoints").mkdir(parents=True, exist_ok=True)
    (staging / "eval_logs").mkdir(parents=True, exist_ok=True)

    # ساخت envهای train/val --- start
    train_env = _SB3EnvAdapter(cfg, env_cfg, obs_mode=args.obs_agg, last_k=args.last_k, split="train")
    train_env = Monitor(
        train_env,
        info_keywords=("ep_pnl_usd","transition_cost","transition_cost_usd","reward_usd",
            "ep_transitions","ep_entries","ep_exits","ep_act_neg","ep_act_zero","ep_act_pos",
            "ep_net_pips_trades")
    )

    eval_env  = _SB3EnvAdapter(cfg, env_cfg, obs_mode=args.obs_agg, last_k=args.last_k, split="val")
    '''
    eval_env  = Monitor(
        eval_env,
        info_keywords=("ep_pnl_usd","transition_cost","transition_cost_usd","reward_usd",
            "ep_transitions","ep_entries","ep_exits","ep_act_neg","ep_act_zero","ep_act_pos")
    )
    '''
    eval_env  = Monitor(_SB3EnvAdapter(cfg, env_cfg, obs_mode=args.obs_agg, last_k=args.last_k, split="val"),
                    info_keywords=("ep_pnl_usd","transition_cost","transition_cost_usd","reward_usd",
                                   "ep_net_pips_trades"))
    # --- end


    # ثبت و ذخیره ابعاد مشاهده برای جلوگیری از mismatch
    obs_dim = int(eval_env.observation_space.shape[0])
    meta_obs = {
        "symbol": symbol,
        "base_tf": base_tf,
        "window_size": window,
        "normalize": normalize,
        "obs_agg": args.obs_agg,
        "last_k": int(args.last_k),
        "obs_dim": obs_dim,
        "rl_alg": alg_name,
        "policy": policy,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    logger.info("Observation dim (eval): %d | agg=%s last_k=%d", obs_dim, args.obs_agg, int(args.last_k))
    # یک فایل ثابت برای خوانش بعدی بک‌تست
    (staging / "obsmeta.eval.json").write_text(
        json.dumps(meta_obs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    hp_all = (rl.get("hyperparams") or {})
    hp = (hp_all.get(alg_name.lower()) or {})
    Alg = _make_alg(alg_name)

    # تنظیم امن و الگوریتم‌آگنوستیک هایپرپرامترها (بدون قفل به یک الگوریتم) --- starta1
    import inspect as _i
    _valid = set(getattr(_i.signature(Alg), "parameters", {}).keys())
    if "target_kl" in _valid and float(hp.get("target_kl") or 0.0) < 0.05:
        hp["target_kl"] = 0.10; logger.info("SB3 | default target_kl=0.10 (alg=%s)", alg_name)
        logger.info("SB3 | default target_kl=%.2f (alg=%s)", hp["target_kl"], alg_name)
    if "ent_coef" in _valid and ("ent_coef" not in hp):
        hp["ent_coef"] = 0.01; logger.info("SB3 | default ent_coef=0.01 (alg=%s)", alg_name)
        logger.info("SB3 | default ent_coef=%.3f (alg=%s)", hp["ent_coef"], alg_name)
    # --- enda1

    # silence env logs
    logging.getLogger("f06_env.trading_env").setLevel(logging.WARNING)

    logger.info(
        "Start training: alg=%s policy=%s total_ts=%d eval_freq=%d window=%d normalize=%s",
        alg_name, policy, total_ts, eval_freq, window, str(normalize)
    )
    # اعمال override از CLI برای ضریب آنتروپی (کاوش) — اگر الگوریتم پشتیبانی کند
    # توضیح فارسی: اگر کاربر --entropy-coef بدهد و پارامتر ent_coef در سازندهٔ الگوریتم موجود باشد،
    # آن را در دیکشنری hp ست می‌کنیم تا مستقیماً به SB3 پاس شود.
    if (args.entropy_coef is not None) and ("ent_coef" in _valid):
        hp["ent_coef"] = float(args.entropy_coef)
        logger.info("SB3 | ent_coef overridden from CLI: %.6f", hp["ent_coef"])
 
    # اگر کاربر --target-kl داده و الگوریتم از آن پشتیبانی می‌کند، آن را هم override کن
    # توضیح فارسی: target_kl آستانهٔ انحراف کل است که می‌تواند رفتار آپدیت policy را کنترل کند.
    if (getattr(args, "target_kl", None) is not None) and ("target_kl" in _valid):
        hp["target_kl"] = float(args.target_kl)
        logger.info("SB3 | target_kl overridden from CLI: %.6f", hp["target_kl"])

    model = Alg(policy, train_env, **hp, verbose=1)

    stop_cb = USDEvalStopCallback(
        patience_evals=int(early_cfg.get("patience_evals", 5)),
        use_reward=bool(early_cfg.get("combine_reward", True)),
        min_delta_reward=float(early_cfg.get("min_delta_reward", 0.0)),
        min_delta_usd=float(early_cfg.get("min_delta_usd", 0.0)),
        verbose=1
    ) if use_early else None

    # محدودسازی طول اپیزود val برای جلوگیری از گیر
    try:
        from gymnasium.wrappers import TimeLimit
    except Exception:
        from gym.wrappers import TimeLimit  # type: ignore

    eval_env  = TimeLimit(eval_env, max_episode_steps=int(args.eval_steps))
    eval_env  = Monitor(eval_env, info_keywords=("ep_pnl_usd","transition_cost","transition_cost_usd","reward_usd"))

    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(staging),
        log_path=str(staging / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=False,
        callback_after_eval=stop_cb
    )

    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_int,
        save_path=str(staging / "checkpoints"),
        name_prefix=f"{alg_name.lower()}_{symbol}_{base_tf}"
    )

    logger.info("SB3 | starting learn(total_timesteps=%d)", int(total_ts))
    model.learn(total_timesteps=total_ts, callback=[eval_cb, ckpt_cb])
    logger.info("SB3 | learn() finished")
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = staging / f"{alg_name}_{symbol}_{base_tf}_{ts}.zip"
    model.save(str(out))
    logger.info("Training finished. Saved model: %s", out)

    # [USD:EVAL:SAVE_NPZ] — pack CSV to NPZ (timesteps, ep_pnl_usd) --- start
    # هدف:
    # بعد از learn()، فایل NPZ فشرده از ارزیابی‌های دلاری ساخته شود (برای مصرف پایپلاین‌ها)
    try:
        log_dir = staging / "eval_logs"
        csv_p = log_dir / "evaluations_usd.csv"
        if csv_p.exists():
            _data = np.genfromtxt(str(csv_p), delimiter=",", names=True, dtype=None, encoding="utf-8")
            if _data.size == 0:
                ts_arr = np.array([], dtype=np.int64)
                usd_arr = np.array([], dtype=float)
            else:
                ts_arr = _data["timesteps"]
                usd_arr = _data["ep_pnl_usd"]
            np.savez_compressed(str(log_dir / "evaluations_usd.npz"),
                                timesteps=ts_arr, ep_pnl_usd=usd_arr)
            logger.info("Saved USD evaluations: %s", log_dir / "evaluations_usd.npz")

        try:
            npz_p = log_dir / "evaluations_usd.npz"
            if npz_p.exists():
                _ev = np.load(str(npz_p))
                _usd = _ev["ep_pnl_usd"]
                if _usd.size:
                    best = float(np.max(_usd))
                    last = float(_usd[-1])
                    logger.info("USD eval summary: best=%.6f last=%.6f", best, last)
        except Exception as e:
            logger.warning("USD eval summary failed: %s", e)


    except Exception as e:
        logger.warning("Failed to save evaluations_usd.npz: %s", e)
    # --- end


    # ذخیرهٔ متادیتا کنار مدل نهایی (برای trace)
    (out.with_suffix(out.suffix + ".obsmeta.json")).write_text(
        json.dumps(meta_obs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
