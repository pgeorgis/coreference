import glob
import json
import logging
import os

from models.coref.coreference import State

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint file helper functions
# ---------------------------------------------------------------------------

def get_checkpoint_path(input_file: str, checkpoint_num: int) -> str:
    """Get the path for a checkpoint file with respect to an input file."""
    base, _ = os.path.splitext(input_file)
    return f"{base}.coref_checkpoint_{checkpoint_num}.json"


def get_checkpoint_offset(input_file: str) -> int:
    """Return max existing checkpoint number + 1, or 0 if none exist.
    Used to avoid overwriting existing checkpoints on resume."""
    base, _ = os.path.splitext(input_file)
    existing = glob.glob(f"{base}.coref_checkpoint_*.json")
    if not existing:
        return 0
    nums = []
    for f in existing:
        suffix = os.path.basename(f).replace(os.path.basename(base) + ".coref_checkpoint_", "").replace(".json", "")
        try:
            nums.append(int(suffix))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 0


def load_coref_checkpoints(input_file: str) -> tuple[dict, dict]:
    """Load all coref checkpoints for a given input file (e.g. corpus JSON).

    Returns:
        completed_entries: {entry_idx: coref_result} merged across all checkpoints.
        in_progress_raw:   raw serialised State dicts from the most recent
                           checkpoint only (older in-progress data is stale).
                           Empty dict when no checkpoints exist or the format
                           pre-dates in-progress serialisation.
    """
    base, _ = os.path.splitext(input_file)
    checkpoint_pattern = f"{base}.coref_checkpoint_*.json"
    checkpoint_files = sorted(glob.glob(checkpoint_pattern))

    completed_entries = {}
    latest_in_progress_raw: dict = {}
    latest_checkpoint_num = -1

    for checkpoint_file in checkpoint_files:
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint: dict = json.load(f)
            for entry in checkpoint.get('completed_entries', []):
                completed_entries[entry['idx']] = entry['coref_result']
            num = checkpoint.get('checkpoint_num', -1)
            if num > latest_checkpoint_num:
                latest_checkpoint_num = num
                latest_in_progress_raw = checkpoint.get('in_progress_states', {})
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")

    # Drop any in-progress entries that are already recorded as completed
    # (they may have finished between an earlier in-progress save and a later
    # completed-entries save).
    in_progress_raw = {
        k: v for k, v in latest_in_progress_raw.items()
        if (int(k) if str(k).lstrip("-").isdigit() else k) not in completed_entries
    }

    return completed_entries, in_progress_raw


def save_coref_checkpoint(input_file: str,
                          completed_entries: dict,
                          checkpoint_num: int,
                          in_progress_states: dict = None,
                          ) -> None:
    """Save a coref checkpoint with completed entries and in-progress states.

    completed_entries: {entry_idx: coref_result} for docs that finished in
                       this interval.
    in_progress_states: optional pre-serialised {str_key: state_dict} for
                        docs still being processed. Pass None to omit.
    """
    checkpoint_path = get_checkpoint_path(input_file, checkpoint_num)
    checkpoint_data = {
        'checkpoint_num': checkpoint_num,
        'completed_entries': [
            {'idx': idx, 'coref_result': result}
            for idx, result in completed_entries.items()
        ],
        'in_progress_states': in_progress_states or {},
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    n_done = len(completed_entries)
    n_prog = len(in_progress_states) if in_progress_states else 0
    logger.debug(
        f"Saved coref checkpoint {checkpoint_num} to {checkpoint_path} "
        f"({n_done} completed, {n_prog} in-progress)"
    )


def cleanup_coref_checkpoints(input_file: str) -> None:
    """Delete all coref checkpoint files for a corpus after successful completion."""
    base, _ = os.path.splitext(input_file)
    checkpoint_pattern = f"{base}.coref_checkpoint_*.json"
    checkpoint_files = glob.glob(checkpoint_pattern)
    for checkpoint_file in checkpoint_files:
        try:
            os.remove(checkpoint_file)
            logger.debug(f"Deleted checkpoint {checkpoint_file}")
        except OSError as e:
            logger.warning(f"Failed to delete checkpoint {checkpoint_file}: {e}")

# ---------------------------------------------------------------------------
# State serialisation helpers
# ---------------------------------------------------------------------------

def serialize_states(states_dict: dict) -> dict:
    """Serialise a {doc_key: State} mapping to a JSON-compatible dict."""
    return {str(k): v.to_dict() for k, v in states_dict.items()}


def deserialize_states(raw: dict) -> dict:
    """Restore a {doc_key: State} mapping from a serialised dict.

    JSON only has string keys; original doc keys that were ints are converted
    back via int() when the string is purely numeric.
    """
    result = {}
    for k, v in raw.items():
        key = int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k
        result[key] = State.from_dict(v)
    return result
