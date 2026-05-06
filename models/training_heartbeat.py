"""
Lightweight, terminal-independent progress: append lines to run_status.txt
and (optionally) run a background heartbeat. Works when tqdm/wandb hide output.

Also tries to install a HuggingFace TrainerCallback to update step counts for ETA in heartbeat.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


def _gpu_snapshot() -> str:
    """One-line GPU stats for heartbeats when train step is unknown (no HF callbacks)."""
    try:
        import shutil
        import subprocess

        if not shutil.which("nvidia-smi"):
            return "gpu=no_nvidia_smi"
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode != 0:
            return f"gpu=nvidia_smi_rc{r.returncode}"
        line = (r.stdout or "").strip().splitlines()
        if not line:
            return "gpu=empty"
        cells = [c.strip() for c in line[0].split(",")]
        if len(cells) >= 4:
            util, mused, mtot, temp = cells[0], cells[1], cells[2], cells[3]
            return f"gpu_util%={util} gpu_mem_MiB={mused}/{mtot} temp_C={temp}"
        return f"gpu={line[0]!r}"
    except Exception as e:
        return f"gpu=err({type(e).__name__})"


def write_run_status(
    path: str, message: str, also_print: bool = True, flush: bool = True
) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {message}\n"
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    if also_print:
        try:
            print(f"[Step05] {line.strip()}", file=sys.stderr, flush=flush)
        except Exception:
            pass


def start_training_heartbeat(
    path: str,
    interval_sec: int,
    shared: Dict[str, Any],
    total_train_steps: int,
    epochs: int,
    label: str = "model.fit",
) -> Callable[[], None]:
    """
    Every `interval_sec`, append elapsed time, optional step/ETA from `shared` (set by callback).

    `shared` keys: "global_step" (int, optional, updated by HF callback).
    """
    if interval_sec <= 0:
        return lambda: None

    stop = threading.Event()

    def _loop() -> None:
        t0 = time.perf_counter()
        n = 0
        warned_no_sync = False
        while not stop.is_set():
            elapsed = time.perf_counter() - t0
            st = int(shared.get("global_step", 0) or 0)
            if shared.get("no_step_sync") and not warned_no_sync:
                warned_no_sync = True
                chain = (shared.get("fit_fallback_chain") or "").strip()
                write_run_status(
                    path,
                    "NOTE (once): no HF step counter in this run — see earlier lines for each "
                    f"model.fit TypeError / fallback. fit_fallback_chain={chain!r}. "
                    "Heartbeats add gpu_* from nvidia-smi: util~0% long-term ⇒ likely not training "
                    "on GPU; util bouncing ⇒ training may be OK despite missing step.",
                    also_print=True,
                )
            line_parts = [
                f"heartbeat #{n}  {label}",
                f"elapsed~{int(elapsed)}s ({elapsed / 60:.1f}min)",
            ]
            if shared.get("epoch"):
                line_parts.append(f"epoch~{shared['epoch']}")
            if shared.get("no_step_sync"):
                line_parts.append("inside_model.fit no_hf_step_counter")
                line_parts.append(_gpu_snapshot())
            elif total_train_steps > 0:
                line_parts.append(f"global_step~{st}/{total_train_steps}")
            if (
                not shared.get("no_step_sync")
                and st > 0
                and total_train_steps
                and st < total_train_steps
            ):
                rate = st / max(elapsed, 1e-3)
                eta = (total_train_steps - st) / max(rate, 1e-6)
                line_parts.append(f"rough_eta~{int(eta)}s ({eta / 60:.1f}min)")
            line_parts.append(f"epochs_config={epochs}")
            write_run_status(path, "  ".join(line_parts), also_print=True)
            n += 1
            if stop.wait(timeout=float(interval_sec)):
                break

    th = threading.Thread(target=_loop, name="Step05HeartBeat", daemon=True)
    th.start()

    def _stop() -> None:
        stop.set()
        th.join(timeout=2.0)

    return _stop


def make_trainer_step_callbacks(
    path: str, shared: Dict[str, Any]
) -> List[Any]:
    """Return TrainerCallback instances, or []. If transformers missing, []."""
    out: List[Any] = []
    try:
        from transformers import TrainerCallback
    except Exception:
        return out

    class _StepSink(TrainerCallback):
        # Important: do NOT rely on on_log alone — ST/HF may not call it for a long time
        # (or at all), so heartbeats would show global_step 0/80 while training is fine.
        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if not getattr(state, "is_local_process_zero", True):
                return
            write_run_status(
                path,
                "HF TrainerCallback: on_train_begin (step counter will update each batch)",
                also_print=True,
            )

        def on_train_batch_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            if not getattr(state, "is_local_process_zero", True):
                return
            try:
                shared["global_step"] = int(getattr(state, "global_step", 0) or 0)
            except Exception:
                pass
            try:
                ep = getattr(state, "epoch", None)
                if ep is not None:
                    shared["epoch"] = float(ep)
            except Exception:
                pass

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Optional[Dict[str, float]] = None,
            **kwargs: Any,
        ) -> None:
            if not getattr(state, "is_local_process_zero", True):
                return
            try:
                shared["global_step"] = int(getattr(state, "global_step", 0) or 0)
            except Exception:
                return
            loss = (logs or {}).get("loss", "")
            if loss is not None and str(loss) != "":
                write_run_status(
                    path,
                    f"train_log step={shared['global_step']} loss={loss!s}",
                    also_print=True,
                )

    out.append(_StepSink())
    return out


def training_header(path: str, total_steps: int, epochs: int) -> None:
    abs_p = os.path.abspath(path)
    write_run_status(
        path,
        f"==== Step05 (Contrastive) started — watch progress (ETA in heartbeats) ====",
    )
    write_run_status(
        path,
        f"this_file={abs_p}  |  tail in another shell:  tail -f {abs_p!r}",
    )
    write_run_status(
        path,
        f"expected train optimizer steps (approx): {total_steps}  (epochs={epochs})",
    )
