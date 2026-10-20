"""Progress logging for runs whose tqdm progress bars are not shown (e.g. output redirected to a log file)."""
import logging
import time

logger = logging.getLogger(__name__)


class ProgressLogger:
    """Logs progress at regular percentage intervals as a fallback when a tqdm
    progress bar is not shown (i.e. when stderr is redirected to a file).

    Usage:
        progress = ProgressLogger(total, label, unit="entries", enabled=not sys.stderr.isatty())
        for item in items:
            process(item)
            progress.update()          # advance by 1
            # or: progress.update(n, extra="42/100 docs done")
    """

    def __init__(self,
                 total: int,
                 label: str,
                 unit: str = "items",
                 interval_pct: int = 10,
                 enabled: bool = True,
                 ):
        steps = 100 // interval_pct
        self.enabled = enabled and total >= steps
        self.total = total
        self.label = label
        self.unit = unit
        self.log_interval = max(1, total // steps)
        self.milestone = 0
        self.completed = 0
        self.t0 = time.time()

    def update(self, n: int = 1, extra: str = None) -> None:
        """Advance counter by n and log if a new milestone is crossed.

        extra: optional context string inserted into the log line between the
               percentage and the rate, e.g. '42/100 docs done'.
        """
        self.completed += n
        new_milestone = self.completed // self.log_interval
        if new_milestone <= self.milestone:
            return
        self.milestone = new_milestone
        if not self.enabled:
            return
        elapsed = time.time() - self.t0
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / rate if rate > 0 else float('inf')
        pct = 100 * self.completed / self.total
        extra_str = f" | {extra}" if extra else ""
        eta_str = f" | ETA {format_duration(eta)}" if eta != float('inf') else ""
        logger.info(
            f"{self.label}: {self.completed}/{self.total} ({pct:.0f}%)"
            f"{extra_str} | {rate:.2f} {self.unit}/s"
            f" | elapsed {format_duration(elapsed)}{eta_str}"
        )


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"
