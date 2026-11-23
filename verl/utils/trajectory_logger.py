# Copyright 2025 Agentic-MLLM authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Env-gated, process-safe JSONL logger for tool-call reasoning trajectories.

Turned on by setting the ``TRAJECTORY_LOG_DIR`` environment variable. When unset,
every function here is a no-op, so this module is safe to import and call from the
RL hot path with zero overhead in normal training.

Design:
- One JSONL file **per process** (``traj_pid<pid>.jsonl``) so parallel rollout /
  reward workers never contend on a shared file handle (no cross-process locking).
  A within-process lock guards against multiple asyncio/threads in the same worker.
- ``log_record`` never raises — a logging failure must never crash training.
- The record schema is intentionally flat and JSON-serializable for downstream
  inspection.
"""

import json
import os
import threading

_ENV_VAR = "TRAJECTORY_LOG_DIR"
_LOCK = threading.Lock()
_CACHED_PATH = None
_CACHED_PID = None


def is_enabled() -> bool:
    """True iff TRAJECTORY_LOG_DIR is set (and non-empty)."""
    return bool(os.getenv(_ENV_VAR))


def _current_path():
    """Return this process's JSONL path, creating the dir on first use.

    Cached per-pid so a forked child (different pid) gets its own file rather than
    inheriting the parent's cached handle.
    """
    global _CACHED_PATH, _CACHED_PID
    log_dir = os.getenv(_ENV_VAR)
    if not log_dir:
        return None
    pid = os.getpid()
    if _CACHED_PATH is None or _CACHED_PID != pid:
        os.makedirs(log_dir, exist_ok=True)
        _CACHED_PATH = os.path.join(log_dir, f"traj_pid{pid}.jsonl")
        _CACHED_PID = pid
    return _CACHED_PATH


def log_record(record: dict) -> None:
    """Append one JSON record as a line. No-op if disabled; never raises."""
    if not is_enabled():
        return
    try:
        path = _current_path()
        if path is None:
            return
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Logging must never break the training loop.
        pass
