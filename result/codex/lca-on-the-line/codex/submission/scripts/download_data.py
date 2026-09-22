#!/usr/bin/env python3
"""Download / stage the ImageNet and OOD datasets used by the benchmark.

ImageNet-1k comes from HuggingFace (``imagenet-1k`` with
``trust_remote_code=True``) exactly as recommended by the paper addendum.  The
OOD datasets are hosted by their authors and are fetched with ``git``/``curl``;
run them one at a time because they are tens of GB in total.

    python scripts/download_data.py --data-root ./data/datasets --dataset imagenet
    python scripts/download_data.py --data-root ./data/datasets --dataset all --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess

SOURCES = {
    "imagenet": {
        "kind": "huggingface",
        "note": "datasets.load_dataset('imagenet-1k', trust_remote_code=True)",
    },
    "imagenet_v2": {
        "kind": "huggingface",
        "note": "vaishaal/ImageNetV2 @ d626240 -- use the MatchedFrequency split only",
        "hf": "vaishaal/ImageNetV2",
        "revision": "d626240",
    },
    "imagenet_s": {
        "kind": "huggingface",
        "note": "ImageNet-Sketch (songweig/imagenet_sketch)",
        "hf": "songweig/imagenet_sketch",
    },
    "imagenet_r": {
        "kind": "git",
        "url": "https://github.com/hendrycks/imagenet-r.git",
        "note": "ImageNet-Rendition",
    },
    "imagenet_a": {
        "kind": "git",
        "url": "https://github.com/hendrycks/natural-adv-examples.git",
        "note": "ImageNet-Adversarial",
    },
    "objectnet": {
        "kind": "manual",
        "url": "https://objectnet.dev",
        "note": "requires filling in a request form; place the images + mappings.csv "
                "under <data-root>/objectnet",
    },
}


def download_imagenet(data_root: str, split: str = "validation") -> None:
    from datasets import load_dataset

    print("[download] ImageNet-1k (%s) via HuggingFace" % split)
    load_dataset(
        "imagenet-1k",
        split=split,
        cache_dir=data_root,
        trust_remote_code=True,
    )


def download_hf_dataset(repo: str, data_root: str, revision: str = "main") -> None:
    from huggingface_hub import snapshot_download

    target = os.path.join(data_root, repo.split("/")[-1])
    print("[download] %s -> %s" % (repo, target))
    snapshot_download(
        repo_id=repo,
        revision=revision,
        repo_type="dataset",
        local_dir=target,
    )


def download_git(url: str, data_root: str) -> None:
    target = os.path.join(data_root, os.path.basename(url).replace(".git", ""))
    if os.path.exists(target):
        print("[download] %s already present" % target)
        return
    print("[download] git clone %s" % url)
    subprocess.check_call(["git", "clone", "--depth", "1", url, target])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data/datasets")
    parser.add_argument("--dataset", default="imagenet",
                        choices=list(SOURCES) + ["all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    names = list(SOURCES) if args.dataset == "all" else [args.dataset]
    os.makedirs(args.data_root, exist_ok=True)
    for name in names:
        source = SOURCES[name]
        print("-" * 70)
        print("%s: %s" % (name, source["note"]))
        if args.dry_run:
            continue
        if name == "imagenet":
            download_imagenet(args.data_root, split="validation")
            download_imagenet(args.data_root, split="train")
        elif source["kind"] == "huggingface":
            download_hf_dataset(
                source["hf"], args.data_root, source.get("revision", "main")
            )
        elif source["kind"] == "git":
            download_git(source["url"], args.data_root)
        else:
            print("[download] manual step required: see", source["url"])


if __name__ == "__main__":
    main()
