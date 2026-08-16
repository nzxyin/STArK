import time

import lightning as L


class PlainTextProgressCallback(L.Callback):
    """Prints one plain log line per `log_every_n_steps` training steps and one
    per validation epoch, instead of relying on Lightning's default progress
    bar (RichProgressBar if `rich` is installed, else TQDMProgressBar) — both
    redraw in place using terminal control codes, which is unreadable/mangled
    in a SLURM `.out` file or any log that isn't rendered with a real
    terminal attached (e.g. `tail -f`, `less`, `grep`).
    """

    def __init__(self):
        super().__init__()
        self._epoch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        log_every = trainer.log_every_n_steps or 50
        if trainer.global_step == 0 or trainer.global_step % log_every != 0:
            return
        metrics = {k: v for k, v in trainer.callback_metrics.items() if k.startswith("train/")}
        if not metrics:
            return
        if self._epoch_start_time is None:
            # on_train_epoch_start doesn't re-fire when resuming mid-epoch from a checkpoint
            # (the epoch is continued, not restarted) — callback state like this isn't part of
            # the checkpoint either, so a freshly-constructed instance starts with None here on
            # every resume. Initialize lazily rather than crash.
            self._epoch_start_time = time.time()
        elapsed = time.time() - self._epoch_start_time
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        print(
            f"[train step {trainer.global_step}/{trainer.max_steps}] {metrics_str} (epoch elapsed {elapsed:.0f}s)",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        metrics = {k: v for k, v in trainer.callback_metrics.items() if k.startswith("val/")}
        if not metrics:
            return
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        print(f"[val @ step {trainer.global_step}] {metrics_str}", flush=True)
