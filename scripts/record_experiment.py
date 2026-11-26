#!/usr/bin/env python3
"""Grava em MP4 episódios de uma política treinada por run_experiment.py.

    python scripts/record_experiment.py --run-dir results/exp/sac_C2_seed0 \
        --delays 0 3 6 --out-dir videos

Gera um vídeo por atraso (k fixo no episódio), com HUD mostrando condição,
k, tempo, velocidade, progresso e retorno acumulado. Usa as mesmas seeds da
avaliação final (1 000 000 + episódio).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio  # noqa: E402

from rl_platform.envs.delay_experiment import make_delay_experiment_env  # noqa: E402
from rl_platform.visualization.overlay import overlay_hud  # noqa: E402
from scripts.run_experiment import EVAL_SEED_BASE, LAP_SIGN, env_kwargs, load_algo  # noqa: E402

FPS = 5  # um quadro por decisão (5 Hz) = tempo real


def record(model, meta: dict, delay: int, episode: int, path: Path) -> dict:
    env = make_delay_experiment_env(delay_mode="fixed", fixed_delay=delay, render_mode="rgb_array", **env_kwargs(meta))
    env.unwrapped.config.update({"screen_width": 600, "screen_height": 600})
    obs, _ = env.reset(seed=EVAL_SEED_BASE + episode)
    label = f"{meta['algo'].upper()} {meta['condition']} seed{meta['seed']}"
    ret = progress = heading = 0.0
    t = 0
    frames = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        t += 1
        ret += float(reward)
        progress += info["progress_m"]
        heading += info["heading_delta"]
        status = ""
        if done:
            status = "colisão" if info.get("crashed") else ("saiu da pista" if info.get("off_road") else "100 s completos")
        lines = [
            label,
            f"atraso k = {delay}",
            f"t = {t / 5:5.1f} s   v = {info['speed']:4.1f} m/s",
            f"progresso = {progress:5.0f} m   voltas = {LAP_SIGN * heading / 6.283185:4.2f}",
            f"retorno = {ret:6.1f}",
        ] + ([status] if status else [])
        frames.append(overlay_hud(env.render(), lines))
    frames += [frames[-1]] * FPS * 2  # segura o último quadro por 2 s
    env.close()
    imageio.mimwrite(path, frames, fps=FPS, macro_block_size=1)
    return {"delay": delay, "steps": t, "return": round(ret, 1), "progress_m": round(progress), "end": status}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--delays", type=int, nargs="+", default=[0])
    p.add_argument("--episode", type=int, default=0, help="índice do episódio de avaliação (seed 1 000 000 + episódio)")
    p.add_argument("--out-dir", type=Path, default=Path("videos"))
    args = p.parse_args(argv)

    meta = json.loads((args.run_dir / "meta.json").read_text(encoding="utf-8"))
    model = load_algo(meta["algo"]).load(args.run_dir / "model.zip", device="cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k in args.delays:
        path = args.out_dir / f"{args.run_dir.name}_k{k}_ep{args.episode}.mp4"
        print(path, record(model, meta, k, args.episode, path), flush=True)


if __name__ == "__main__":
    main()
