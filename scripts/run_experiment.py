#!/usr/bin/env python3
"""Treina e avalia uma condição do experimento (SAC/PPO × C1–C4 × semente).

Condições:
    C1 = só direção,          sem atraso      --no-accel --train-delay off
    C2 = só direção,          atraso U{0..3}  --no-accel --train-delay random
    C3 = direção+aceleração,  sem atraso      --accel    --train-delay off
    C4 = direção+aceleração,  atraso U{0..3}  --accel    --train-delay random

Exemplos:
    python scripts/run_experiment.py train --algo sac --condition C2 --seed 0 \
        --timesteps 500000 --out-dir results/exp
    python scripts/run_experiment.py eval --run-dir results/exp/sac_C2_seed0

Saída em ``<out-dir>/<algo>_<cond>_seed<s>/``:
    model.zip            modelo SB3 final
    meta.json            configuração, versões, tempo, histograma de k no treino
    train_episodes.csv   um registro por episódio de treino
    learning_curve.csv   avaliações periódicas (k=0)
    eval_episodes.csv    avaliação final: um registro por (k, episódio)
    eval_summary.csv     médias por k
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_platform.envs.delay_experiment import (  # noqa: E402
    DEFAULT_TRAIN_MAX_DELAY,
    MAX_EVAL_DELAY,
    make_delay_experiment_env,
)

CONDITIONS = {
    "C1": dict(accel=False, train_delay="off"),
    "C2": dict(accel=False, train_delay="random"),
    "C3": dict(accel=True, train_delay="off"),
    "C4": dict(accel=True, train_delay="random"),
}
EVAL_SEED_BASE = 1_000_000      # avaliação final: seeds 1_000_000 + episódio
CURVE_SEED_BASE = 500_000       # avaliações periódicas
EVAL_DELAYS = list(range(MAX_EVAL_DELAY + 1))
LAP_SIGN = -1.0                 # no oval, andar para frente diminui o heading (medido: 1 volta ≈ 298 m)


def env_kwargs(meta: dict) -> dict:
    return dict(
        accel=meta["accel"],
        traffic=meta["traffic"],
        straight_length_m=meta["straight_length_m"],
        duration_s=meta["duration_s"],
    )


def run_episode(model, env, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    ret, length, progress, heading, speeds = 0.0, 0, 0.0, 0.0, []
    crashed = off_road = False
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        length += 1
        progress += float(info.get("progress_m", 0.0))
        heading += float(info.get("heading_delta", 0.0))
        speeds.append(float(info.get("speed", np.nan)))
        crashed = crashed or bool(info.get("crashed", False))
        off_road = off_road or bool(info.get("off_road", False))
    laps = LAP_SIGN * heading / (2 * np.pi)
    return {
        "return": round(ret, 4),
        "length": length,
        "survived": int(truncated and not terminated),
        "crashed": int(crashed),
        "off_road": int(off_road and not crashed),
        "progress_m": round(progress, 2),
        "laps": round(laps, 3),
        "lap_completed": int(laps >= 1.0),
        "mean_speed": round(float(np.nanmean(speeds)), 3) if speeds else float("nan"),
    }


def evaluate(model, meta: dict, delays, n_episodes: int, seed_base: int, episodes_writer=None) -> list[dict]:
    summary = []
    for k in delays:
        env = make_delay_experiment_env(delay_mode="fixed", fixed_delay=k, **env_kwargs(meta))
        rows = []
        for ep in range(n_episodes):
            row = {"eval_delay": k, "episode": ep, "seed": seed_base + ep, **run_episode(model, env, seed_base + ep)}
            rows.append(row)
            if episodes_writer is not None:
                episodes_writer.writerow(row)
        env.close()
        agg = {"eval_delay": k, "n_episodes": n_episodes}
        for key in ("return", "length", "survived", "crashed", "off_road", "progress_m", "laps", "lap_completed", "mean_speed"):
            vals = np.array([r[key] for r in rows], dtype=float)
            agg[f"mean_{key}"] = round(float(np.nanmean(vals)), 4)
        agg["std_return"] = round(float(np.std([r["return"] for r in rows])), 4)
        summary.append(agg)
    return summary


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_algo(name: str):
    from stable_baselines3 import PPO, SAC
    return {"sac": SAC, "ppo": PPO}[name]


def cmd_train(args):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(args.threads)
    cond = CONDITIONS[args.condition]
    run_dir = args.out_dir / f"{args.algo}_{args.condition}_seed{args.seed}"
    if (run_dir / "model.zip").exists() and not args.overwrite:
        print(f"{run_dir}/model.zip já existe; use --overwrite para refazer.")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "algo": args.algo,
        "condition": args.condition,
        "seed": args.seed,
        "accel": cond["accel"],
        "train_delay": cond["train_delay"],
        "train_max_delay": DEFAULT_TRAIN_MAX_DELAY if cond["train_delay"] == "random" else 0,
        "traffic": args.traffic,
        "straight_length_m": args.straight_length_m,
        "duration_s": args.duration_s,
        "timesteps": args.timesteps,
        "hyperparameters": "stable-baselines3 defaults",
        "versions": {p: version(p) for p in ("highway-env", "stable-baselines3", "gymnasium", "torch", "numpy")},
        "python": platform.python_version(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }

    def make_train_env():
        env = make_delay_experiment_env(
            delay_mode=cond["train_delay"], max_delay=DEFAULT_TRAIN_MAX_DELAY, **env_kwargs(meta)
        )
        return Monitor(env)

    train_env = make_train_env()
    Algo = load_algo(args.algo)
    model = Algo("MlpPolicy", train_env, seed=args.seed, device=args.device, verbose=0)

    episodes_f = (run_dir / "train_episodes.csv").open("w", newline="", encoding="utf-8")
    ep_writer = csv.DictWriter(episodes_f, fieldnames=["step", "episode", "return", "length", "crashed", "off_road", "progress_m"])
    ep_writer.writeheader()
    curve_rows: list[dict] = []
    t0 = time.time()

    class Logger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.n_ep = 0
            self.progress = 0.0
            self.off_road = False
            self.next_eval = args.eval_every

        def _on_step(self) -> bool:
            info = self.locals["infos"][0]
            self.progress += float(info.get("progress_m", 0.0))
            self.off_road = self.off_road or bool(info.get("off_road", False))
            if self.locals["dones"][0]:
                ep = info.get("episode", {})
                ep_writer.writerow({
                    "step": self.num_timesteps, "episode": self.n_ep,
                    "return": round(float(ep.get("r", np.nan)), 4), "length": int(ep.get("l", 0)),
                    "crashed": int(bool(info.get("crashed", False))),
                    "off_road": int(self.off_road and not info.get("crashed", False)),
                    "progress_m": round(self.progress, 2),
                })
                self.n_ep += 1
                self.progress, self.off_road = 0.0, False
            if args.eval_every > 0 and self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_every
                s = evaluate(self.model, meta, [0], args.curve_episodes, CURVE_SEED_BASE)[0]
                s.update(step=self.num_timesteps, wall_time_s=round(time.time() - t0, 1))
                curve_rows.append(s)
                write_csv(run_dir / "learning_curve.csv", curve_rows)
                episodes_f.flush()
                print(f"[{run_dir.name}] step {self.num_timesteps}: return={s['mean_return']:.1f} "
                      f"progress={s['mean_progress_m']:.0f}m survived={s['mean_survived']:.2f} "
                      f"({s['wall_time_s']:.0f}s)", flush=True)
            return True

    print(f"Treinando {run_dir.name} por {args.timesteps} passos", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=Logger())
    episodes_f.close()
    meta["train_wall_time_s"] = round(time.time() - t0, 1)
    k_counts = train_env.get_wrapper_attr("k_counts")
    meta["train_k_counts"] = {str(k): int(c) for k, c in enumerate(k_counts)}
    model.save(run_dir / "model.zip")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    train_env.close()

    if not args.skip_eval:
        run_final_eval(model, meta, run_dir, args.eval_episodes)


def run_final_eval(model, meta: dict, run_dir: Path, n_episodes: int):
    t0 = time.time()
    with (run_dir / "eval_episodes.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["eval_delay", "episode", "seed", "return", "length", "survived", "crashed", "off_road",
                  "progress_m", "laps", "lap_completed", "mean_speed"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        summary = evaluate(model, meta, EVAL_DELAYS, n_episodes, EVAL_SEED_BASE, w)
    for s in summary:
        s.update(algo=meta["algo"], condition=meta["condition"], seed=meta["seed"])
    write_csv(run_dir / "eval_summary.csv", summary)
    for s in summary:
        print(f"[{run_dir.name}] k={s['eval_delay']}: return={s['mean_return']:.1f} "
              f"progress={s['mean_progress_m']:.0f}m laps={s['mean_laps']:.2f} "
              f"survived={s['mean_survived']:.2f}", flush=True)
    print(f"Avaliação em {time.time() - t0:.0f}s", flush=True)


def cmd_eval(args):
    import torch
    torch.set_num_threads(args.threads)
    meta = json.loads((args.run_dir / "meta.json").read_text(encoding="utf-8"))
    model = load_algo(meta["algo"]).load(args.run_dir / "model.zip", device=args.device)
    run_final_eval(model, meta, args.run_dir, args.eval_episodes)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--eval-episodes", type=int, default=100, help="episódios por atraso na avaliação final")
    common.add_argument("--threads", type=int, default=1, help="threads do torch")
    common.add_argument("--device", default="cpu")

    t = sub.add_parser("train", parents=[common])
    t.add_argument("--algo", choices=["sac", "ppo"], required=True)
    t.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--timesteps", type=int, default=500_000)
    t.add_argument("--out-dir", type=Path, default=Path("results/exp"))
    t.add_argument("--traffic", default="medium", help="preset (none/light/medium/heavy/jam) ou inteiro")
    t.add_argument("--straight-length-m", type=int, default=30)
    t.add_argument("--duration-s", type=int, default=100)
    t.add_argument("--eval-every", type=int, default=50_000, help="0 desliga a curva de aprendizado")
    t.add_argument("--curve-episodes", type=int, default=10)
    t.add_argument("--skip-eval", action="store_true")
    t.add_argument("--overwrite", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", parents=[common])
    e.add_argument("--run-dir", type=Path, required=True)
    e.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    if getattr(args, "traffic", None) is not None:
        try:
            args.traffic = int(args.traffic)
        except ValueError:
            pass
    args.func(args)


if __name__ == "__main__":
    main()
