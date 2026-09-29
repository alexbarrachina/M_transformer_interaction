#!/usr/bin/env python3
"""Start the checkpoint dashboard, or evaluate one checkpoint headlessly."""
from __future__ import annotations

import argparse
from dataclasses import asdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", help="Checkpoint path relative to save_models")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--no-fmd", action="store_true", help="Skip optional embedding stage")
    parser.add_argument("--force", action="store_true", help="Create a fresh analysis")
    parser.add_argument("--smoke", action="store_true", help="Tiny validation run, stored separately from benchmarks")
    parser.add_argument("--resume", help="Resume an existing analysis identifier")
    args = parser.parse_args()
    from metrics.analyzer.store import Store, read_json, evaluator_fingerprint
    store = Store()
    if args.checkpoint or args.resume:
        from metrics.analyzer.types import EvaluationConfig
        from metrics.analyzer.music import input_manifest
        from metrics.analyzer.runner import run_worker
        if args.resume:
            run_id = args.resume
            manifest = read_json(store.run_dir(run_id) / "manifest.json")
            if not manifest or manifest["evaluator"] != evaluator_fingerprint():
                parser.error("Cannot resume an absent run or a different evaluator version")
        else:
            extra = dict(device=args.device, fmd=not args.no_fmd, smoke=args.smoke)
            if args.smoke:
                extra.update(context=16, continuation=2, seeds=[0], collision_contexts=1,
                             style_pairs=0, evolution_length=16, test_pieces=0)
            settings = EvaluationConfig.for_profile(args.profile, **extra)
            run_id = store.prepare(args.checkpoint, settings, input_manifest(settings), args.force)
        directory = store.run_dir(run_id)
        existing = read_json(directory / "results.json", {})
        if existing.get("status") != "complete":
            (directory / "cancel").unlink(missing_ok=True)
            print(f"Analysis {run_id}: {directory}", flush=True)
            run_worker(str(store.output), run_id)
        final = read_json(directory / "results.json", {})
        print(final.get("status"), final.get("message"), flush=True)
        print(f"Results: {directory / 'results.json'}", flush=True)
        return 0 if final.get("status") == "complete" else 1
    from aiohttp import web
    from metrics.analyzer.server import create_app
    print(f"Checkpoint Analyzer: http://127.0.0.1:{args.port}", flush=True)
    web.run_app(create_app(store), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
