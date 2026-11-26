#!/usr/bin/env python3
"""Treina e avalia uma condição do experimento (SAC/PPO × C1–C4 × semente).

Condições:
    C1 = só direção,          sem atraso      --no-accel --train-delay off
    C2 = só direção,          atraso U{0..3}  --no-accel --train-delay random
    C3 = direção+aceleração,  sem atraso      --accel    --train-delay off
    C4 = direção+aceleração,  atraso U{0..3}  --accel    --train-delay random

Exemplos:
    python scripts/run_experiment.py run --algo sac --condition C2 --seed 0 \
        --timesteps 300000 --out-dir results/exp      # treina e avalia, retomando
    python scripts/run_experiment.py train --algo sac --condition C2 --seed 0 ...
    python scripts/run_experiment.py eval --run-dir results/exp/sac_C2_seed0

``run`` é o que os jobs usam. Ele confere o que já existe na pasta:
    avaliação completa e com a mesma configuração  -> pula (código 0)
    modelo completo e com a mesma configuração     -> só avalia
    nada, ou só restos de uma execução interrompida -> treina e avalia
    modelo/avaliação de OUTRA configuração          -> recusa (código 3),
                                                       nunca sobrescreve
Qualquer falha termina com código diferente de 0.

Saída em ``<out-dir>/<algo>_<cond>_seed<s>/``:
    model.zip            modelo SB3 final
    meta.json            configuração, versões, tempo, passos executados, histograma de k
    train_episodes.csv   um registro por episódio de treino
    learning_curve.csv   avaliações periódicas (k=0)
    eval_episodes.csv    avaliação final: um registro por (k, episódio)
    eval_summary.csv     médias por k
Arquivos finais são gravados num temporário e renomeados (substituição atômica).
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
    REWARD_RANGE,
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
# Mude quando o ambiente mudar de semântica: resultados de versões antigas
# deixam de ser aceitos na retomada.
ENV_VERSION = "2026-11-26-v2 (recompensa [-1.3,2], voltas por linha de largada)"
# Chaves de meta.json que precisam bater para reaproveitar um modelo.
CONFIG_KEYS = ("env_version", "algo", "condition", "seed", "accel", "train_delay", "train_max_delay",
               "traffic", "straight_length_m", "duration_s", "timesteps", "reward_range")
EVAL_FIELDS = ["eval_delay", "episode", "seed", "return", "length", "survived", "crashed", "off_road",
               "progress_m", "laps", "lap_completed", "mean_speed"]
EXIT_FAIL, EXIT_MISMATCH = 1, 3


def env_kwargs(meta: dict) -> dict:
    return dict(
        accel=meta["accel"],
        traffic=meta["traffic"],
        straight_length_m=meta["straight_length_m"],
        duration_s=meta["duration_s"],
    )


# ------------------------------------------------------------ gravação atômica
def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def write_text_atomic(path: Path, text: str):
    tmp = _atomic_path(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    if not rows:
        return
    tmp = _atomic_path(path)
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


# ------------------------------------------------------------ episódios
def run_episode(model, env, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    ret, length, progress, laps, speeds = 0.0, 0, 0.0, 0, []
    crashed = off_road = False
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        length += 1
        progress += float(info.get("progress_m", 0.0))
        laps += int(info.get("lap_crossing", 0))
        speeds.append(float(info.get("speed", np.nan)))
        crashed = crashed or bool(info.get("crashed", False))
        off_road = off_road or bool(info.get("off_road", False))
    return {
        "return": round(ret, 4),
        "length": length,
        "survived": int(truncated and not terminated),
        "crashed": int(crashed),
        "off_road": int(off_road and not crashed),
        "progress_m": round(progress, 2),
        "laps": laps,                       # cruzamentos da linha de largada (frente − trás)
        "lap_completed": int(laps >= 1),
        "mean_speed": round(float(np.nanmean(speeds)), 3) if speeds else float("nan"),
    }


def evaluate(model, meta: dict, delays, n_episodes: int, seed_base: int, all_rows: list | None = None) -> list[dict]:
    summary = []
    for k in delays:
        env = make_delay_experiment_env(delay_mode="fixed", fixed_delay=k, **env_kwargs(meta))
        rows = []
        for ep in range(n_episodes):
            rows.append({"eval_delay": k, "episode": ep, "seed": seed_base + ep, **run_episode(model, env, seed_base + ep)})
        env.close()
        if all_rows is not None:
            all_rows.extend(rows)
        agg = {"eval_delay": k, "n_episodes": n_episodes}
        for key in EVAL_FIELDS[3:]:
            agg[f"mean_{key}"] = round(float(np.nanmean(np.array([r[key] for r in rows], dtype=float))), 4)
        agg["std_return"] = round(float(np.std([r["return"] for r in rows])), 4)
        summary.append(agg)
    return summary


def load_algo(name: str):
    from stable_baselines3 import PPO, SAC
    return {"sac": SAC, "ppo": PPO}[name]


# ------------------------------------------------------------ configuração e validação
def requested_config(args) -> dict:
    cond = CONDITIONS[args.condition]
    return {
        "env_version": ENV_VERSION,
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
        "reward_range": list(REWARD_RANGE),
    }


def read_meta(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def config_diff(meta: dict, wanted: dict) -> list[str]:
    return [f"{k}: tem {meta.get(k)!r}, pedido {wanted[k]!r}" for k in CONFIG_KEYS if meta.get(k) != wanted[k]]


def model_complete(run_dir: Path) -> bool:
    meta = read_meta(run_dir)
    return (run_dir / "model.zip").is_file() and meta is not None and "timesteps_executed" in meta


def eval_complete(run_dir: Path, n_episodes: int) -> bool:
    """Avaliação conta como pronta só se tiver os 7 atrasos, cada um com
    n_episodes, e o CSV por episódio tiver todas as linhas."""
    try:
        with (run_dir / "eval_summary.csv").open(encoding="utf-8") as f:
            summary = list(csv.DictReader(f))
        with (run_dir / "eval_episodes.csv").open(encoding="utf-8") as f:
            n_rows = sum(1 for _ in csv.DictReader(f))
    except OSError:
        return False
    delays = sorted(int(r["eval_delay"]) for r in summary)
    return (delays == EVAL_DELAYS
            and all(int(r["n_episodes"]) == n_episodes for r in summary)
            and n_rows == len(EVAL_DELAYS) * n_episodes)


# ------------------------------------------------------------ treino
def train(args, run_dir: Path):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(args.threads)
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("model.zip", "meta.json", "eval_summary.csv", "eval_episodes.csv", "learning_curve.csv"):
        (run_dir / stale).unlink(missing_ok=True)

    meta = {
        **requested_config(args),
        "hyperparameters": "stable-baselines3 defaults",
        "eval_seed_base": EVAL_SEED_BASE,
        "curve_seed_base": CURVE_SEED_BASE,
        "versions": {p: version(p) for p in ("highway-env", "stable-baselines3", "gymnasium", "torch", "numpy")},
        "python": platform.python_version(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    cond = CONDITIONS[args.condition]
    train_env = Monitor(make_delay_experiment_env(
        delay_mode=cond["train_delay"], max_delay=DEFAULT_TRAIN_MAX_DELAY, **env_kwargs(meta)))
    model = load_algo(args.algo)("MlpPolicy", train_env, seed=args.seed, device=args.device, verbose=0)

    episodes_f = (run_dir / "train_episodes.csv").open("w", newline="", encoding="utf-8")
    ep_writer = csv.DictWriter(episodes_f, fieldnames=["step", "episode", "return", "length", "crashed", "off_road",
                                                       "progress_m", "laps"])
    ep_writer.writeheader()
    curve_rows: list[dict] = []
    t0 = time.time()

    class Logger(BaseCallback):
        def __init__(self):
            super().__init__()
            self.n_ep = 0
            self.progress, self.laps, self.off_road = 0.0, 0, False
            self.next_eval = args.eval_every

        def _on_step(self) -> bool:
            info = self.locals["infos"][0]
            self.progress += float(info.get("progress_m", 0.0))
            self.laps += int(info.get("lap_crossing", 0))
            self.off_road = self.off_road or bool(info.get("off_road", False))
            if self.locals["dones"][0]:
                ep = info.get("episode", {})
                ep_writer.writerow({
                    "step": self.num_timesteps, "episode": self.n_ep,
                    "return": round(float(ep.get("r", np.nan)), 4), "length": int(ep.get("l", 0)),
                    "crashed": int(bool(info.get("crashed", False))),
                    "off_road": int(self.off_road and not info.get("crashed", False)),
                    "progress_m": round(self.progress, 2), "laps": self.laps,
                })
                self.n_ep += 1
                self.progress, self.laps, self.off_road = 0.0, 0, False
            if args.eval_every > 0 and self.num_timesteps >= self.next_eval:
                self.next_eval += args.eval_every
                s = evaluate(self.model, meta, [0], args.curve_episodes, CURVE_SEED_BASE)[0]
                s.update(step=self.num_timesteps, wall_time_s=round(time.time() - t0, 1))
                curve_rows.append(s)
                write_csv(run_dir / "learning_curve.csv", curve_rows)
                episodes_f.flush()
                print(f"[{run_dir.name}] step {self.num_timesteps}: return={s['mean_return']:.1f} "
                      f"progress={s['mean_progress_m']:.0f}m laps={s['mean_laps']:.2f} "
                      f"survived={s['mean_survived']:.2f} ({s['wall_time_s']:.0f}s)", flush=True)
            return True

    print(f"Treinando {run_dir.name} por {args.timesteps} passos", flush=True)
    model.learn(total_timesteps=args.timesteps, callback=Logger())
    episodes_f.close()
    k_counts = train_env.get_wrapper_attr("k_counts")
    meta["train_wall_time_s"] = round(time.time() - t0, 1)
    meta["timesteps_executed"] = int(model.num_timesteps)   # o PPO arredonda para múltiplos de n_steps
    meta["train_k_counts"] = {str(k): int(c) for k, c in enumerate(k_counts)}
    tmp_model = run_dir / ".model.tmp.zip"
    model.save(tmp_model)
    os.replace(tmp_model, run_dir / "model.zip")
    write_text_atomic(run_dir / "meta.json", json.dumps(meta, indent=2))   # meta por último: marca o fim
    train_env.close()
    return model, meta


# ------------------------------------------------------------ avaliação final
def run_final_eval(model, meta: dict, run_dir: Path, n_episodes: int):
    t0 = time.time()
    rows: list[dict] = []
    summary = evaluate(model, meta, EVAL_DELAYS, n_episodes, EVAL_SEED_BASE, rows)
    for s in summary:
        s.update(algo=meta["algo"], condition=meta["condition"], seed=meta["seed"])
    write_csv(run_dir / "eval_episodes.csv", rows, EVAL_FIELDS)
    write_csv(run_dir / "eval_summary.csv", summary)   # por último: marca a avaliação como completa
    for s in summary:
        print(f"[{run_dir.name}] k={s['eval_delay']}: return={s['mean_return']:.1f} "
              f"progress={s['mean_progress_m']:.0f}m laps={s['mean_laps']:.2f} "
              f"survived={s['mean_survived']:.2f}", flush=True)
    print(f"Avaliação em {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------ comandos
def run_dir_of(args) -> Path:
    return args.out_dir / f"{args.algo}_{args.condition}_seed{args.seed}"


def cmd_run(args) -> int:
    run_dir = run_dir_of(args)
    wanted = requested_config(args)
    meta = read_meta(run_dir)
    if meta is not None and model_complete(run_dir):
        diff = config_diff(meta, wanted)
        if diff:
            print(f"{run_dir.name}: RECUSADO, já existe um modelo de outra configuração "
                  f"(mova ou apague a pasta):\n  " + "\n  ".join(diff), flush=True)
            return EXIT_MISMATCH
        if eval_complete(run_dir, args.eval_episodes):
            print(f"{run_dir.name}: já concluído", flush=True)
            return 0
        print(f"{run_dir.name}: modelo pronto, avaliando", flush=True)
        import torch
        torch.set_num_threads(args.threads)
        model = load_algo(args.algo).load(run_dir / "model.zip", device=args.device)
        run_final_eval(model, meta, run_dir, args.eval_episodes)
    else:
        model, meta = train(args, run_dir)
        run_final_eval(model, meta, run_dir, args.eval_episodes)
    return 0 if eval_complete(run_dir, args.eval_episodes) else EXIT_FAIL


def cmd_train(args) -> int:
    run_dir = run_dir_of(args)
    if (run_dir / "model.zip").exists() and not args.overwrite:
        print(f"{run_dir}/model.zip já existe; use --overwrite para refazer.")
        return EXIT_MISMATCH
    model, meta = train(args, run_dir)
    if not args.skip_eval:
        run_final_eval(model, meta, run_dir, args.eval_episodes)
    return 0


def cmd_eval(args) -> int:
    import torch
    torch.set_num_threads(args.threads)
    meta = read_meta(args.run_dir)
    if meta is None or not (args.run_dir / "model.zip").is_file():
        print(f"{args.run_dir}: falta model.zip ou meta.json")
        return EXIT_FAIL
    model = load_algo(meta["algo"]).load(args.run_dir / "model.zip", device=args.device)
    run_final_eval(model, meta, args.run_dir, args.eval_episodes)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--eval-episodes", type=int, default=100, help="episódios por atraso na avaliação final")
    common.add_argument("--threads", type=int, default=1, help="threads do torch")
    common.add_argument("--device", default="cpu")

    training = argparse.ArgumentParser(add_help=False)
    training.add_argument("--algo", choices=["sac", "ppo"], required=True)
    training.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--timesteps", type=int, default=300_000)
    training.add_argument("--out-dir", type=Path, default=Path("results/exp"))
    training.add_argument("--traffic", default="medium", help="preset (none/light/medium/heavy/jam) ou inteiro")
    training.add_argument("--straight-length-m", type=int, default=30)
    training.add_argument("--duration-s", type=int, default=100)
    training.add_argument("--eval-every", type=int, default=50_000, help="0 desliga a curva de aprendizado")
    training.add_argument("--curve-episodes", type=int, default=10)

    r = sub.add_parser("run", parents=[common, training])
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("train", parents=[common, training])
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
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
