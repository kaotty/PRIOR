"""Download the pinned official SFT sources into the configured local data root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from huggingface_hub import snapshot_download


SOURCES = {
    "cof": {"repo_id": "xintongzhang/CoF-SFT-Data-5.4k",
            "revision": "efc7f49fc0d59ede576701a65eb44630d7f5d9ea",
            "files": ["README.md", "cof_sft_data.json", "images.zip"], "license": "Apache-2.0"},
    "mini_o3": {"repo_id": "Mini-o3/Mini-o3-Coldstart-Dataset",
                "revision": "5640d9c8958557a7841e2592ef0329deb0c3e0b2",
                "files": ["README.md", "data/train-*.parquet"], "license": "See upstream dataset terms"},
    "deepeyes_v2": {"repo_id": "honglyhly/DeepEyesV2_SFT",
                    "revision": "cee88d9e9c494729f719ccca5de5aeb940501a35",
                    "files": ["images.zip", "json/sft_part_*.json"], "license": "See upstream dataset terms"},
    "thyme": {"repo_id": "Kwai-Keye/Thyme-SFT",
              "revision": "8a65e8065475f8a1f95f290b0478f1d12ccf8e68",
              "files": ["README.md", "data/computation-*.parquet"], "license": "MIT"},
    "pixel_reasoner": {"repo_id": "TIGER-Lab/PixelReasoner-SFT-Data",
                       "revision": "23a10b7a944ee81d9e4ec91580b2cdf8d0665660",
                       "files": ["README.md", "release_checked_toolpart.json", "images.zip"],
                       "license": "Apache-2.0"},
}


def extract_images(archive_path: Path, destination: Path, revision: str, *, strip_images_prefix: bool = True) -> None:
    marker = destination / ".tir_extracted_revision"
    identity = f"{revision}:strip_images_prefix={int(strip_images_prefix)}"
    if marker.is_file() and marker.read_text().strip() == identity:
        return
    if destination.exists():
        raise RuntimeError(f"unverified or partial image directory: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=".images-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"unsafe archive member: {member.filename}")
                parts = relative.parts[1:] if strip_images_prefix and relative.parts and relative.parts[0] == "images" else relative.parts
                if not parts:
                    continue
                target = temporary.joinpath(*parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=8 << 20)
        (temporary / marker.name).write_text(identity + "\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def download(input_root: Path, cache_dir: Path, *, include_pixel: bool = False,
             only: list[str] | None = None, dry_run: bool = False) -> None:
    names = only or ["cof", "mini_o3", "deepeyes_v2", "thyme"]
    if include_pixel and "pixel_reasoner" not in names:
        names = [*names, "pixel_reasoner"]
    for name in names:
        spec = SOURCES[name]
        destination = input_root / name
        print(f"Source: {spec['repo_id']} @ {spec['revision']} -> {destination}", flush=True)
        if dry_run:
            continue
        snapshot_download(repo_id=spec["repo_id"], repo_type="dataset", revision=spec["revision"],
                          allow_patterns=spec["files"], local_dir=destination, cache_dir=cache_dir,
                          max_workers=4, token=False)
        if name in {"cof", "deepeyes_v2"}:
            extract_images(destination / "images.zip", destination / "images", spec["revision"],
                           strip_images_prefix=(name == "cof"))
        (destination / "tir_source.json").write_text(json.dumps(spec, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--include-pixel", action="store_true")
    parser.add_argument("--only", choices=tuple(SOURCES), action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    download(args.input_root, args.cache_dir, include_pixel=args.include_pixel, only=args.only, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
