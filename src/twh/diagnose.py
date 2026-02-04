#!/usr/bin/env python3
"""
Guided diagnosis for stuck moves.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from . import run_interactive_add, run_task_command
from .review import ReviewTask, build_review_report, load_pending_tasks, started_marker
from .taskwarrior import missing_udas


DIMENSION_UDAS: Dict[str, str] = {
    "energy": "Energy (0-10)",
    "attention": "Attention (0-10)",
    "emotion": "Emotion (0-10)",
    "interruptible": "Interruptible (0-1)",
    "mechanical": "Mechanical (0-1)",
}

DIMENSION_HINTS: Dict[str, str] = {
    "energy": "0-10, higher = more energy required",
    "attention": "0-10, higher = more focus required",
    "emotion": "0-10, higher = more emotional load",
    "interruptible": "0-1, 1 = easy to pause/resume",
    "mechanical": "0-1, 1 = routine/low emotion",
}

MICRO_TYPES = [
    ("activation", "Activation energy: too hard to start"),
    ("representation", "Representation: wrong format (need outline/diagram/etc.)"),
    ("blocker", "External blocker: waiting on someone/system/info"),
    ("cognitive", "Cognitive expense: too much thinking/uncertainty"),
    ("time", "Time chunk: needs a longer uninterrupted block"),
    ("identity", "Identity risk: fear of looking dumb/judged/failing"),
    ("no_first_action", "No clear first action: next physical step unknown"),
    ("other", "Other / mixed"),
]

LACKING_DIMENSIONS = [
    ("energy", "Low energy"),
    ("attention", "Fragmented attention"),
    ("emotion", "Emotional resistance"),
    ("time", "Not enough time block"),
]

HELPER_TEMPLATES = {
    "activation": "Start '{desc}': open file / set up workspace (2 min)",
    "representation": "Reframe '{desc}': outline/diagram/pseudocode version (10 min)",
    "blocker": "Unblock '{desc}': ask for the missing info",
    "cognitive": "De-risk '{desc}': write 5-bullet plan + unknowns list",
    "time": "Slice '{desc}': define a 15-minute sub-move",
    "identity": "Shrink stakes for '{desc}': draft ugly v0 / private notes only",
    "no_first_action": "First action for '{desc}': identify the next physical step",
}


def parse_numeric_value(value: Any) -> Optional[float]:
    """
    Parse numeric UDA values into floats.

    Parameters
    ----------
    value : Any
        Raw value from Taskwarrior.

    Returns
    -------
    Optional[float]
        Parsed numeric value, or None when invalid.

    Examples
    --------
    >>> parse_numeric_value("3.5")
    3.5
    >>> parse_numeric_value("nope") is None
    True
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = text
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_numeric_uda(task: ReviewTask, name: str) -> Optional[float]:
    """
    Read a numeric UDA from a move payload.

    Parameters
    ----------
    task : ReviewTask
        Move to inspect.
    name : str
        UDA field name.

    Returns
    -------
    Optional[float]
        Parsed numeric value, or None when missing.
    """
    return parse_numeric_value(task.raw.get(name))


def is_flagged(task: ReviewTask, name: str, threshold: float = 0.5) -> bool:
    """
    Interpret numeric UDAs as boolean flags.

    Parameters
    ----------
    task : ReviewTask
        Move to inspect.
    name : str
        UDA field name.
    threshold : float, optional
        Minimum value to treat as True (default: 0.5).

    Returns
    -------
    bool
        True when the UDA value meets the threshold.
    """
    value = get_numeric_uda(task, name)
    return value is not None and value >= threshold


def missing_dimension_values(task: ReviewTask) -> List[str]:
    """
    Return dimension UDAs missing from a move.

    Parameters
    ----------
    task : ReviewTask
        Move to inspect.

    Returns
    -------
    List[str]
        Missing dimension field names.
    """
    missing: List[str] = []
    for field in DIMENSION_UDAS:
        if get_numeric_uda(task, field) is None:
            missing.append(field)
    return missing


def select_move_by_selector(
    pending: Sequence[ReviewTask],
    selector: str,
) -> ReviewTask:
    """
    Select a move by ID or UUID prefix.

    Parameters
    ----------
    pending : Sequence[ReviewTask]
        Pending moves to search.
    selector : str
        Move ID or UUID prefix.

    Returns
    -------
    ReviewTask
        Matching move.

    Raises
    ------
    ValueError
        If no matching move is found.
    """
    if not selector:
        raise ValueError("No selector provided.")

    if selector.isdigit():
        for task in pending:
            if task.id is not None and str(task.id) == selector:
                return task
        raise ValueError(f"No pending move found for id:{selector}.")

    needle = selector.lower()
    matches = [
        task
        for task in pending
        if task.uuid.lower().startswith(needle)
    ]
    if not matches:
        raise ValueError(f"No pending move found for uuid prefix:{selector}.")
    matches.sort(
        key=lambda task: (
            task.id is None,
            task.id if task.id is not None else 10**9,
            task.uuid,
        )
    )
    return matches[0]


def order_ondeck_moves(
    pending: Sequence[ReviewTask],
    *,
    current_mode: Optional[str],
    strict_mode: bool,
    include_dominated: bool,
) -> List[ReviewTask]:
    """
    Order moves using the same ranking as twh ondeck.

    Parameters
    ----------
    pending : Sequence[ReviewTask]
        Pending moves.
    current_mode : Optional[str]
        Current mode for scoring.
    strict_mode : bool
        Require strict mode matching.
    include_dominated : bool
        Include dominated moves when True.

    Returns
    -------
    List[ReviewTask]
        Ordered ready moves.
    """
    if not pending:
        return []
    report = build_review_report(
        pending,
        current_mode=current_mode,
        strict_mode=strict_mode,
        include_dominated=include_dominated,
        top=len(pending),
    )
    return [candidate.task for candidate in report.candidates]


def find_move_by_uuid(
    pending: Sequence[ReviewTask],
    uuid: str,
) -> Optional[ReviewTask]:
    """
    Locate a move by UUID.

    Parameters
    ----------
    pending : Sequence[ReviewTask]
        Pending moves to search.
    uuid : str
        UUID to locate.

    Returns
    -------
    Optional[ReviewTask]
        Matching move, if found.
    """
    for task in pending:
        if task.uuid == uuid:
            return task
    return None


def is_easier_in_dimension(
    candidate: ReviewTask,
    stuck: ReviewTask,
    dimension: str,
) -> bool:
    """
    Determine if a move is easier in a given dimension.

    Parameters
    ----------
    candidate : ReviewTask
        Candidate move.
    stuck : ReviewTask
        Stuck move.
    dimension : str
        Dimension name.

    Returns
    -------
    bool
        True when candidate is strictly easier.
    """
    if dimension == "energy":
        stuck_energy = get_numeric_uda(stuck, "energy")
        cand_energy = get_numeric_uda(candidate, "energy")
        return (
            stuck_energy is not None
            and cand_energy is not None
            and cand_energy < stuck_energy
        )

    if dimension == "attention":
        stuck_attention = get_numeric_uda(stuck, "attention")
        cand_attention = get_numeric_uda(candidate, "attention")
        if stuck_attention is not None and cand_attention is not None:
            return cand_attention < stuck_attention
        return is_flagged(candidate, "interruptible") and not is_flagged(
            stuck, "interruptible"
        )

    if dimension == "emotion":
        stuck_emotion = get_numeric_uda(stuck, "emotion")
        cand_emotion = get_numeric_uda(candidate, "emotion")
        if stuck_emotion is not None and cand_emotion is not None:
            return cand_emotion < stuck_emotion
        return is_flagged(candidate, "mechanical") and not is_flagged(
            stuck, "mechanical"
        )

    if dimension == "time":
        stuck_est = get_numeric_uda(stuck, "estimate_hours")
        cand_est = get_numeric_uda(candidate, "estimate_hours")
        if stuck_est is not None and cand_est is not None:
            return cand_est < stuck_est
        stuck_diff = get_numeric_uda(stuck, "diff")
        cand_diff = get_numeric_uda(candidate, "diff")
        if stuck_diff is not None and cand_diff is not None:
            return cand_diff < stuck_diff
        return is_flagged(candidate, "interruptible") and not is_flagged(
            stuck, "interruptible"
        )

    return False


def pick_easier_move(
    ordered: Sequence[ReviewTask],
    stuck: ReviewTask,
    lacking: str,
) -> Optional[ReviewTask]:
    """
    Pick the first move in order that is easier in the lacking dimension.

    Parameters
    ----------
    ordered : Sequence[ReviewTask]
        Ordered moves.
    stuck : ReviewTask
        Stuck move.
    lacking : str
        Lacking dimension.

    Returns
    -------
    Optional[ReviewTask]
        Easier move, if found.
    """
    for candidate in ordered:
        if candidate.uuid == stuck.uuid:
            continue
        if is_easier_in_dimension(candidate, stuck, lacking):
            return candidate
    return None


def format_missing_uda_instructions(missing: Sequence[str]) -> str:
    """
    Build guidance text for missing UDAs.

    Parameters
    ----------
    missing : Sequence[str]
        Missing UDA names.

    Returns
    -------
    str
        Instructional message for ~/.taskrc edits.
    """
    joined = ", ".join(missing)
    lines = [
        f"Missing Taskwarrior UDA(s): {joined}.",
        "Add the following to your active Taskwarrior config and re-run twh diagnose:",
        "Diagnose will not write dimension values until these UDAs exist to avoid modifying move descriptions.",
    ]
    for field in missing:
        label = DIMENSION_UDAS.get(field, field)
        lines.append(f"uda.{field}.type=numeric")
        lines.append(f"uda.{field}.label={label}")
    return "\n".join(lines)


def ensure_udas_present(fields: Iterable[str]) -> bool:
    """
    Verify required UDAs exist before writing dimension values.

    Parameters
    ----------
    fields : Iterable[str]
        UDA field names.

    Returns
    -------
    bool
        True when all UDAs exist.
    """
    missing = missing_udas(fields, allow_taskrc_fallback=False)
    if not missing:
        return True
    message = format_missing_uda_instructions(missing)
    print(f"twh: diagnose stopped.\n{message}", file=sys.stderr)
    return False


def prompt_dimension_updates(
    fields: Sequence[str],
    *,
    input_func: Callable[[str], str] = input,
) -> Dict[str, str]:
    """
    Prompt for numeric dimension updates.

    Parameters
    ----------
    fields : Sequence[str]
        Dimension field names to prompt for.
    input_func : Callable[[str], str], optional
        Input function (default: input).

    Returns
    -------
    Dict[str, str]
        Dimension updates keyed by field name.
    """
    updates: Dict[str, str] = {}
    for field in fields:
        label = DIMENSION_UDAS.get(field, field)
        hint = DIMENSION_HINTS.get(field, "numeric")
        prompt = f"  {label} ({hint}): "
        while True:
            raw = input_func(prompt).strip()
            if not raw:
                break
            if parse_numeric_value(raw) is None:
                print("Please enter a number or leave blank.")
                continue
            updates[field] = raw
            break
    return updates


def apply_dimension_updates(uuid: str, updates: Dict[str, str]) -> int:
    """
    Apply dimension updates to a move.

    Parameters
    ----------
    uuid : str
        Move UUID.
    updates : Dict[str, str]
        Dimension updates.

    Returns
    -------
    int
        Exit code.
    """
    if not updates:
        return 0
    if not ensure_udas_present(updates.keys()):
        return 1
    parts = [f"{key}:{value}" for key, value in updates.items()]
    result = run_task_command([uuid, "modify", *parts], capture_output=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def prompt_choice(
    prompt: str,
    options: Sequence[tuple[str, str]],
    *,
    default_index: int = 1,
    input_func: Callable[[str], str] = input,
) -> str:
    """
    Prompt for a numbered choice.

    Parameters
    ----------
    prompt : str
        Prompt label.
    options : Sequence[tuple[str, str]]
        Sequence of (value, label) pairs.
    default_index : int, optional
        Default choice index (1-based).
    input_func : Callable[[str], str], optional
        Input function (default: input).

    Returns
    -------
    str
        Selected option value.
    """
    while True:
        raw = input_func(f"{prompt} [{default_index}]: ").strip()
        if not raw:
            index = default_index
        else:
            try:
                index = int(raw)
            except ValueError:
                print("Please enter a number.")
                continue
        if 1 <= index <= len(options):
            return options[index - 1][0]
        print("Choice out of range.")


def prompt_confirm(
    prompt: str,
    *,
    default: bool = False,
    input_func: Callable[[str], str] = input,
) -> bool:
    """
    Prompt for a yes/no confirmation.

    Parameters
    ----------
    prompt : str
        Prompt label.
    default : bool, optional
        Default response when empty (default: False).
    input_func : Callable[[str], str], optional
        Input function (default: input).

    Returns
    -------
    bool
        True when confirmed.
    """
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input_func(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def format_move_snapshot(move: ReviewTask, title: str) -> List[str]:
    """
    Format a move summary for display.

    Parameters
    ----------
    move : ReviewTask
        Move to summarize.
    title : str
        Title label for the summary.

    Returns
    -------
    List[str]
        Formatted lines.
    """
    move_id = str(move.id) if move.id is not None else move.uuid[:8]
    marker = started_marker(move)
    lines = [
        f"{title}:",
        f"  id: {move_id}",
        f"  uuid: {move.uuid}",
        f"  description: {move.description}{marker}",
    ]
    if move.project:
        lines.append(f"  project: {move.project}")
    if move.depends:
        lines.append(f"  depends: {', '.join(move.depends)}")
    return lines


def create_helper_move(
    stuck: ReviewTask,
    kind: str,
    *,
    input_func: Callable[[str], str] = input,
) -> int:
    """
    Create a helper move using the full twh add wizard.

    Parameters
    ----------
    stuck : ReviewTask
        Stuck move.
    kind : str
        Helper kind key.
    input_func : Callable[[str], str], optional
        Input function (default: input).

    Returns
    -------
    int
        Exit code.
    """
    template = HELPER_TEMPLATES.get(kind, "Helper for '{desc}'")
    default_desc = template.format(desc=stuck.description)
    stuck_id = str(stuck.id) if stuck.id is not None else stuck.uuid[:8]
    print("\nLaunching the twh add wizard for the helper move.")
    print(
        "Tip: at the Blocks prompt, enter "
        f"{stuck_id} to make the stuck move depend on this helper."
    )
    print("Leave Blocks empty to skip linking.")
    return run_interactive_add([default_desc], input_func=input_func)


def annotate_move(uuid: str, note: str) -> int:
    """
    Add an annotation to a move.

    Parameters
    ----------
    uuid : str
        Move UUID.
    note : str
        Annotation text.

    Returns
    -------
    int
        Exit code.
    """
    result = run_task_command([uuid, "annotate", note], capture_output=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def identity_risk_exercise(
    stuck: ReviewTask,
    *,
    input_func: Callable[[str], str] = input,
) -> int:
    """
    Run a short identity-risk reflection and offer a helper move.

    Parameters
    ----------
    stuck : ReviewTask
        Stuck move.
    input_func : Callable[[str], str], optional
        Input function (default: input).

    Returns
    -------
    int
        Exit code.
    """
    print("\nIdentity risk drill (2 minutes). Keep it blunt and private.")
    input_func("What is the scary thought? (example: I'll look incompetent): ")
    input_func("Evidence FOR that thought (one line): ")
    input_func("Evidence AGAINST (one line): ")
    reframe = input_func("More accurate thought (one line): ").strip()
    if reframe:
        print("\nPinned reframe:")
        print(reframe)

    if prompt_confirm(
        "Add a helper move that bakes in an ugly v0 and privacy?",
        default=True,
        input_func=input_func,
    ):
        return create_helper_move(stuck, "identity", input_func=input_func)
    return 0


def maybe_rate_dimensions(
    task: ReviewTask,
    *,
    input_func: Callable[[str], str] = input,
) -> int:
    """
    Prompt to fill missing dimension values for a move.

    Parameters
    ----------
    task : ReviewTask
        Move to update.
    input_func : Callable[[str], str], optional
        Input function (default: input).

    Returns
    -------
    int
        Exit code.
    """
    missing = missing_dimension_values(task)
    if not missing:
        return 0
    missing_list = ", ".join(missing)
    print(f"\nMissing dimension values for this move: {missing_list}.")
    default_yes = len(missing) == len(DIMENSION_UDAS)
    if not prompt_confirm(
        "Rate these dimensions now?",
        default=default_yes,
        input_func=input_func,
    ):
        return 0
    updates = prompt_dimension_updates(missing, input_func=input_func)
    if not updates:
        return 0
    exit_code = apply_dimension_updates(task.uuid, updates)
    if exit_code == 0:
        print("Updated dimension values.")
    return exit_code


def run_diagnose(
    selector: Optional[str] = None,
    *,
    mode: Optional[str] = None,
    strict_mode: bool = False,
    include_dominated: bool = True,
    input_func: Callable[[str], str] = input,
) -> int:
    """
    Execute the diagnose flow for a stuck move.

    Parameters
    ----------
    selector : Optional[str], optional
        Move ID or UUID prefix to diagnose.
    mode : Optional[str], optional
        Current mode context.
    strict_mode : bool, optional
        Require strict mode matching (default: False).
    include_dominated : bool, optional
        Include dominated moves in ordering (default: True).
    input_func : Callable[[str], str], optional
        Input function for prompts (default: input).

    Returns
    -------
    int
        Exit code.
    """
    try:
        pending = load_pending_tasks()
    except RuntimeError as exc:
        print(f"twh: diagnose failed: {exc}", file=sys.stderr)
        return 1

    if not pending:
        print("No pending moves found.")
        return 0

    if selector:
        try:
            stuck = select_move_by_selector(pending, selector)
        except ValueError as exc:
            print(f"twh: diagnose failed: {exc}", file=sys.stderr)
            return 1
    else:
        ordered = order_ondeck_moves(
            pending,
            current_mode=mode,
            strict_mode=strict_mode,
            include_dominated=include_dominated,
        )
        if not ordered:
            print(
                "No ready moves found. Pass a move id or uuid to diagnose.",
                file=sys.stderr,
            )
            return 1
        stuck = ordered[0]

    for line in format_move_snapshot(stuck, "Stuck move"):
        print(line)

    exit_code = maybe_rate_dimensions(stuck, input_func=input_func)
    if exit_code != 0:
        return exit_code

    print("\nPick what best matches the resistance:")
    for idx, (_, label) in enumerate(MICRO_TYPES, 1):
        print(f"  {idx}. {label}")
    micro = prompt_choice("Choice", MICRO_TYPES, default_index=1, input_func=input_func)

    if micro in {"activation", "representation", "blocker", "cognitive", "time", "no_first_action"}:
        print("\nRecommended: create a tiny helper move that keeps the top move warm.")
        if prompt_confirm("Create helper move now?", default=True, input_func=input_func):
            exit_code = create_helper_move(stuck, micro, input_func=input_func)
            if exit_code != 0:
                return exit_code
    elif micro == "identity":
        exit_code = identity_risk_exercise(stuck, input_func=input_func)
        if exit_code != 0:
            return exit_code
    else:
        print("\nOk. We will use the lacking dimension to pick an easier move.")

    print("\nWhat are you lacking right now?")
    for idx, (_, label) in enumerate(LACKING_DIMENSIONS, 1):
        print(f"  {idx}. {label}")
    lacking = prompt_choice(
        "Choice",
        LACKING_DIMENSIONS,
        default_index=1,
        input_func=input_func,
    )

    try:
        pending = load_pending_tasks()
    except RuntimeError as exc:
        print(f"twh: diagnose failed: {exc}", file=sys.stderr)
        return 1

    refreshed = find_move_by_uuid(pending, stuck.uuid)
    if refreshed:
        stuck = refreshed

    ordered = order_ondeck_moves(
        pending,
        current_mode=mode,
        strict_mode=strict_mode,
        include_dominated=include_dominated,
    )
    easier = pick_easier_move(ordered, stuck, lacking)

    if easier:
        for line in format_move_snapshot(
            easier,
            f"Easier move (strictly easier in: {lacking})",
        ):
            print(line)
        if prompt_confirm(
            "Switch to this move for now (no changes, just a decision)?",
            default=True,
            input_func=input_func,
        ):
            print("Do that move now. You are selecting the best feasible step.")
        if prompt_confirm(
            "Add a note to the stuck move about what you lacked?",
            default=True,
            input_func=input_func,
        ):
            note = f"diagnose: lacked={lacking}; workaround={easier.description}"
            if annotate_move(stuck.uuid, note) == 0:
                print("Annotated stuck move.")
        return 0

    print(
        f"\nNo pending move found that is strictly easier on '{lacking}'."
    )
    print("Two options that keep things honest:")
    print("  1) Create a helper move (tiny first action) for the stuck move.")
    print("  2) Add dimension values so the filter can work next time.")
    if prompt_confirm(
        "Create a helper move anyway?",
        default=True,
        input_func=input_func,
    ):
        fallback_kind = {
            "energy": "activation",
            "attention": "time",
            "emotion": "identity",
            "time": "time",
        }.get(lacking, "activation")
        return create_helper_move(stuck, fallback_kind, input_func=input_func)
    return 0
