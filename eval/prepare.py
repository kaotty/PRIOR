"""Download and rebuild the exact paired five-benchmark evaluation suite."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from eval.common import _sha256, frozen_suite, row_fingerprint

EXPECTED = {'vstar': 191, 'chartqa': 2500, 'mathvista': 1000, 'wemath': 1740, 'charxiv': 1000}
SOURCES = {
 'vstar': ('lmms-lab/VStar-Bench','b44023b4dca749ed8a76b85eb576627d05a1c174','data/test-00000-of-00001.parquet','6f9a089e93e75931350157544f8e74713d9d108c5f73585609ca262c33528a27'),
 'chartqa': ('lmms-lab/ChartQA','9e63b7df1592a1c2158e735cc1725454aef0d6d9','data/test-00000-of-00001.parquet','165263505f2998aba65d819b44be832edecd92d676fee2c030645f784cd55d06'),
 'mathvista': ('AI4Math/MathVista','2b6ad69445fbb5695c9b165475e8decdbeb97747','data/testmini-00000-of-00001-725687bf7a18d64b.parquet','373f6c0b412a9be2cec36711cee724e03f4c5db6908f3c13db903aa9694d4f2d'),
 'wemath': ('We-Math/We-Math','527c44d4d94c4e3c7c98157460146a8b18c8420a','data/testmini-00000-of-00001-adbfe22dcd3558e6.parquet','bb16b217ec8ca2f4cd8aa5f1fb041896d9a5341735e0ae9be236982ccd0b2a4b'),
 'charxiv': ('princeton-nlp/CharXiv','f441eb632fc62f6f777830a0f47619e6e86459b0','val.parquet','ed0613ac7d5c045ac30a63d78529217bf267218948170a2b26ec37eaeea1605d'),
}


def download(name: str, cache: Path, *, dry_run: bool) -> Path:
    repo, revision, filename, expected = SOURCES[name]
    path = cache / 'eval_sources' / name / filename
    print(f'Source: https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}', flush=True)
    if dry_run:
        return path
    if not path.is_file() or _sha256(path) != expected:
        path = Path(hf_hub_download(repo, repo_type='dataset', revision=revision, filename=filename,
                                   local_dir=cache / 'eval_sources' / name, token=False,
                                   force_download=path.is_file()))
    if _sha256(path) != expected:
        raise ValueError(f'{name} source checksum differs from the frozen release: {path}')
    return path


def run(module: str, options: list[str], *, dry_run: bool) -> None:
    command = [sys.executable, '-m', module, *options]
    print(' '.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def eval_path(root: Path, benchmark: str, mode: str) -> Path:
    if benchmark in ('vstar','chartqa','mathvista'):
        return root / f'core5_{mode.replace("_", "")}_v1' / benchmark / mode / 'shard_0/test.parquet'
    if benchmark == 'wemath':
        return root / 'ocrbench_wemath_v1/wemath/full' / mode / 'shard_0/test.parquet'
    return root / 'charxiv_reasoning_v1/full' / mode / 'shard_0/test.parquet'


def validate(root: Path) -> None:
    golden = frozen_suite()['benchmarks']
    for benchmark, count in EXPECTED.items():
        paired = []
        for mode in ('tool_on','tool_off'):
            path = eval_path(root,benchmark,mode)
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != count:
                raise ValueError(f'{benchmark}/{mode}: wrong row count')
            index = 0
            ids = []
            for batch in parquet.iter_batches(batch_size=8):
                for row in batch.to_pylist():
                    expected = golden[benchmark][mode][index]
                    if row['extra_info']['item_id'] != expected['item_id'] or row_fingerprint(row) != expected['sha256']:
                        raise ValueError(f'{benchmark}/{mode}: frozen content/order mismatch at row {index}')
                    ids.append(row['extra_info']['item_id'])
                    index += 1
            if len(set(ids)) != count:
                raise ValueError(f'{benchmark}/{mode}: duplicate question IDs')
            paired.append(ids)
        if paired[0] != paired[1]:
            raise ValueError(f'{benchmark}: Tool-On/Off question IDs differ')
        print(f'{benchmark}: {count} rows matched frozen content in both modes', flush=True)
    print(f'Paired evaluation PASS: {sum(EXPECTED.values())} questions per mode', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--rl-train', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.data_root / 'eval'
    if args.check_only:
        if not args.dry_run:
            validate(root)
        return
    complete = all(eval_path(root, b, m).is_file() for b in EXPECTED for m in ('tool_on','tool_off'))
    if complete and not args.dry_run:
        validate(root)
        return
    train = args.rl_train or args.data_root / 'formal/rl_16k_v2/train.parquet'
    if not args.dry_run and not train.is_file():
        raise FileNotFoundError('Prepare RL data first: ' + str(train))
    cache = args.cache_dir or Path(os.environ.get('HF_HOME', str(args.data_root.parent / 'cache/huggingface')))
    raw = {name: download(name, cache, dry_run=args.dry_run) for name in EXPECTED}
    if not all((root / f'core5_{m.replace("_", "")}_v1/manifest.json').is_file() for m in ('tool_on','tool_off')):
        run('eval.core5', ['--vstar',str(raw['vstar']),'--chartqa',str(raw['chartqa']),
                          '--mathvista',str(raw['mathvista']),'--output',str(root)], dry_run=args.dry_run)
    for name, folder in [('wemath','ocrbench_wemath_v1'),('charxiv','charxiv_reasoning_v1')]:
        target = root / folder
        if target.exists():
            if not (target / 'manifest.json').is_file():
                raise RuntimeError(f'incomplete prior output requires inspection: {target}')
            continue
        if args.dry_run:
            run('eval.external', ['--benchmark',name,f'--{name}',str(raw[name]),'--output',str(target),
                                  '--overlap-source',str(train)], dry_run=True)
        else:
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f'.{name}-build-', dir=root) as temporary:
                staging = Path(temporary) / 'output'
                run('eval.external', ['--benchmark',name,f'--{name}',str(raw[name]),'--output',str(staging),
                                      '--overlap-source',str(train)], dry_run=False)
                if not (staging / 'manifest.json').is_file():
                    raise RuntimeError(f'{name}: builder did not complete its manifest')
                staging.rename(target)
    if not args.dry_run:
        validate(root)


if __name__ == '__main__':
    main()
