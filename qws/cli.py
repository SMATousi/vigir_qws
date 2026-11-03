"""Command-line entry point: ``qws demo`` and ``qws run <config.yaml>``."""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from .experiment import run_experiment


def _summary(report: dict) -> str:
    lines = ["IoU (mean over images):"]
    lines += [f"  {k:<18} {v:.3f}" for k, v in report["iou"].items()]
    lines.append(f"ECE (structured marginals): {report['ece_structured']:.3f}")
    lines.append(f"max |theta - theta_true|: {report['theta_error']:.3f}")
    if report.get("thetaP_error") is not None:
        lines.append(f"max |thetaP - thetaP_true|: {report['thetaP_error']:.3f}")
    lines.append(f"epochs: {report['epochs_run']}, train time: {report['train_time_s']:.1f}s")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qws", description="Quantum weak supervision for segmentation")
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo", help="synthetic end-to-end run")
    demo.add_argument("--solver", default="gibbs", help="sampling backend (gibbs, sa, exact, anneal_sim, anneal_mock, ...)")
    demo.add_argument("--images", type=int, default=8)
    demo.add_argument("--size", type=int, default=16)
    demo.add_argument("--epochs", type=int, default=20)
    demo.add_argument("--calibrate", action="store_true", help="wrap the sampler in temperature calibration")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--out", default=None, help="write the JSON report here")
    run = sub.add_parser("run", help="run an experiment from a YAML config")
    run.add_argument("config")
    run.add_argument("--json", action="store_true", help="print the full JSON report")
    args = parser.parse_args(argv)

    if args.cmd == "demo":
        cfg = {"seed": args.seed, "solver": {"name": args.solver}, "calibrate": args.calibrate,
               "dataset": {"n_images": args.images, "shape": [args.size, args.size]},
               "train": {"epochs": args.epochs}, "output": args.out}
        if args.solver == "gibbs":
            cfg["solver"]["num_sweeps"] = 100
        report = run_experiment(cfg)
        print(_summary(report))
    else:
        with open(args.config) as f:
            report = run_experiment(yaml.safe_load(f))
        print(json.dumps(report, indent=2, default=str) if args.json else _summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
