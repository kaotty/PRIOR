"""Download and verify the frozen official Qwen2.5-VL base model.

The manifest includes every weight shard, so the visual encoder is retained.
Hugging Face's local download metadata resumes interrupted transfers. Existing
files from an unrelated model are never silently replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
# Official Hub metadata at REVISION: (bytes, hash kind, digest).
FILES = {
    "README.md": (18574, "git", "5f169031e3f61faa8262272000c36d54bdc9b926"),
    "chat_template.json": (1050, "git", "732bd68bc5427d1fb6c06a59b3bf2456b2155d24"),
    "config.json": (1374, "git", "58ee12a9fb3d8708a1e41e49cb9aa5ac5017b6a6"),
    "generation_config.json": (216, "git", "1088c9865dbf2e8152485664bb07495e89a14efb"),
    "merges.txt": (1671839, "git", "20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0"),
    "model-00001-of-00005.safetensors": (3900233256, "sha256", "e97b877e47fde53a6c6e77aafb36e58e91ee9d95c4a3eeac6f1b5c0e6a1c986e"),
    "model-00002-of-00005.safetensors": (3864726320, "sha256", "a9a300a43b4724eee2abe7c18ceb26768d0ab011eb0cad19d9bfd2476a24d024"),
    "model-00003-of-00005.safetensors": (3864726424, "sha256", "111223d173e00bbee81cba1216fad28668df3476706b7fd26f4d5b50f8b3a507"),
    "model-00004-of-00005.safetensors": (3864733680, "sha256", "ef47f634fa57d46ee134edcc09f34085a47da1e16c12a2abe0d67118be6d72ed"),
    "model-00005-of-00005.safetensors": (1089994880, "sha256", "0c859795ad3a627a9b95bcb762e059d5b768a4a36fdd4affeff269d93fdecc67"),
    "model.safetensors.index.json": (57619, "git", "f0c9e9b8e043f5de0ac99fd8b2ea75515d811e7b"),
    "preprocessor_config.json": (350, "git", "7f3b746825e5eef53ed8ed57a91df9e86ee62c0a"),
    "tokenizer.json": (7031645, "git", "443909a61d429dff23010e5bddd28ff530edda00"),
    "tokenizer_config.json": (5702, "git", "91684ea3dc135a7d815e2c932054ce88c1aab41f"),
    "vocab.json": (2776833, "git", "4783fe10ac3adce15ac8f358ef5462739852c569"),
}


def valid_file(path: Path, record: tuple[int, str, str]) -> bool:
    size, kind, expected = record
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha1() if kind == "git" else hashlib.sha256()
    if kind == "git":
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def managed_download(root: Path, filename: str) -> bool:
    """Only replace a corrupt file already recorded as this exact download."""
    marker = root / ".tir_model_source.json"
    if marker.is_file():
        try:
            source = json.loads(marker.read_text())
        except (OSError, ValueError):
            source = {}
        if source.get("repo_id") == REPO_ID and source.get("revision") == REVISION:
            return True
    metadata = root / ".cache/huggingface/download" / f"{filename}.metadata"
    if metadata.is_file():
        lines = metadata.read_text().splitlines()
        return bool(lines) and lines[0] == REVISION
    return False


def verify_structure(root: Path) -> None:
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError("Expected the official Qwen2.5-VL base model")
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    expected_shards = {name for name in FILES if name.endswith(".safetensors")}
    if set(weight_map.values()) != expected_shards or not any("visual." in key for key in weight_map):
        raise ValueError("The model index is incomplete or has no visual encoder")
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.json"):
        json.loads((root / name).read_text())


def download(output: Path, cache_dir: Path, *, check_only: bool = False) -> None:
    missing = [name for name, record in FILES.items() if not valid_file(output / name, record)]
    if missing and check_only:
        raise RuntimeError(f"Incomplete or changed base model: {', '.join(missing)}")
    for name in missing:
        if ((output / name).exists() or (output / name).is_symlink()) and not managed_download(output, name):
            raise FileExistsError(f"Refusing to replace unverified existing model file: {output / name}")
    if missing:
        output.mkdir(parents=True, exist_ok=True)
        marker = output / ".tir_model_source.json"
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"repo_id": REPO_ID, "revision": REVISION}, indent=2) + "\n"
        )
        temporary.replace(marker)
        for name in missing:
            print(f"Downloading {REPO_ID}: {name}", flush=True)
            hf_hub_download(
                REPO_ID, name, revision=REVISION, local_dir=output, cache_dir=cache_dir,
                force_download=(output / name).exists(),
            )
            if not valid_file(output / name, FILES[name]):
                raise RuntimeError(f"Downloaded model checksum mismatch: {name}")
    verify_structure(output)
    print(f"Base model verified: {output} ({REPO_ID}@{REVISION})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    download(args.output, args.cache_dir, check_only=args.check_only)


if __name__ == "__main__":
    main()
