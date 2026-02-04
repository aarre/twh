
# Instructions for codex in the twh repo

## Continuous improvement

Update AGENTS.md whenever you learn something new about the project, including requirements, my preferences, design decisions, techiques that you tried and didn't work, and so on.

When the user says "commit", create the git commit using coherent patches (logical hunks) and a conventional commit message.

## Test-driven development

Use a test-driven development approach for all changes. Ensure that each change is accompanied by a corresponding test case to validate its functionality. The test cases should be placed in the `test` directory. They should be written in advance of the implementation, according to the TDD process. They should use the `pytest` framework. Annotate test classes, methods, and functions with `@pytest.mark.parametrize` to avoid repetition and with `unit`, `integration`, `slow` and other tags to help organize tests and enable selective execution.
Always run thorough tests after changes, even when not explicitly requested.

Use doctest tests to help explain the intended behavior of functions and classes as well as supplement the pytest tests.

## Code style

Follow the PEP 8 style guide for Python code and all Ruff/Black rules and conventions. Use meaningful variable and function names, and keep functions short and focused on a single responsibility. Avoid excessive use of global variables and prefer encapsulation through classes and objects. Use type hints for function parameters and return types and wherever else they may be applicable to improve code readability and maintainability.

## Documentation

Document the public interface and the reasons for major design decisions in `README.md` and, if necessary, other Markdown files in the repo. Limit the number of other Markdown files to a minimum. There should be a clear rationale for any new Markdown files distinguishing its purpose from the README and evey other file in the repo. This rationale should be documented in the file itself.

Use my own `sley` tool to create literate programming notebooks for the project. A sley doc file for `file.py` should be named `FILE.md` (note the capitalization). When modifying existing doc and src files with sley block makers, be especially careful not to delete, change or reorder existing sley block markers. Doing so will break the link between the doc and src files, making it impossible to generate the doc or src again.

Document every function, class, and module with clear and concise docstrings. Use docstrings to describe the purpose, parameters, and return values of functions and methods. For classes, include a docstring for the class itself, as well as docstrings for each method. Use type hints in docstrings to specify the expected types of parameters and return values. Use NumPy-style docstrings.

Use line comments to explain non-obvious code.

## Requirements

Already implemented, among other requirements:

* When twh is invoked inside a Taskwarrior context, twh should restrict its output (both lists and graphs, both standard and reverse) according to the Taskwarrior context.
* `twh` supports a synthetic `blocks` field for `add`/`modify`, translating it into Taskwarrior `depends` updates without new UDAs.
* All user-facing output and documentation use "moves" terminology instead of "tasks".
* `twh dominance` collects dominance ordering with pairwise prompts (transitive, dependency-implied dominance) and writes `dominates`/`dominated_by` UDAs; tie decisions are persisted so they are not re-prompted; `twh ondeck` includes this dominance stage whenever metadata or ordering is incomplete.
* `twh criticality` collects time-criticality ordering with pairwise prompts (transitive) and writes `criticality` UDA values; prompts explain the decay framing and ask which move becomes pointless first if ignored; `twh ondeck` requires criticality before showing the report.
* `twh ondeck` tracks `diff` (difficulty hours) alongside `imp`/`urg`/`opt_human` (legacy `opt` accepted)/`mode`/`criticality`, treats missing dominance/criticality/diff as incomplete metadata, and ranks moves by dominance tier before scoring ties.
* `twh ondeck` prompts for missing metadata on all moves in scope (including blocked) whenever metadata or dominance ordering is incomplete.
* `twh ondeck` formats annotation timestamps into a local, human-readable date-time string (US Eastern).
* Suppress Taskwarrior noise like `Modified 0 tasks.` in `twh` outputs.
* `twh graph` uses fixed-width Graphviz boxes and wraps move descriptions across lines.
* Dominance prompts use A/B/C choices with labels like `[A] Move ID 3: ...` to avoid numeric confusion with move IDs.
* Dominance should never prompt for move pairs already related by dependencies (including when dependencies are stored as IDs), and prompts should show approximate progress (comparisons complete/remaining).
* Dominance missing/unknown pair checks use reachability maps to avoid slow ondeck/dominance runs on large move sets.
* `twh ondeck` defers dominance-missing checks until after metadata prompts when any metadata is missing, so the first prompt appears quickly.
* `twh ondeck` outputs its top-move list using the default Taskwarrior report columns/colors, but places the ID column first and relabels urgency to `Rank` for the composite ordering (1 is highest) while adding a numeric `Score` column; it defaults to showing 25 candidates.
* `twh add` suppresses Taskwarrior modify/project completion noise after creating a move (dominance and blocks updates run quietly unless there are errors).
* Taskwarrior project-completion summary lines are filtered from twh output when relaying command results.
* Mode prompts use a persistent known-modes list (stored in `~/.config/twh/modes.json`, override with `TWH_MODES_PATH`) with inline autocompletion; newly entered modes are added immediately, prompt examples are alphabetized, and Taskwarrior `uda.mode.values` is extended when present.
* Mode prompts reject Taskwarrior core attribute/status keywords (for example `wait`) with a helpful retry prompt.
* Taskwarrior modify failures during metadata updates raise errors to avoid silently losing mode entries; mode updates are verified via export and raise if missing.
* Mode updates validate `uda.mode.type` is `string` and, when `uda.mode.values` is set, require the mode to be listed before applying changes; otherwise abort with guidance before modifying moves.
* Taskwarrior 3.4.2 did not persist `mode:wait` (value cleared); `mode:waiting` worked. Treat `wait` as a problematic value for the `mode` UDA.
* UDA checks for write operations now rely on Taskwarrior's active config (no taskrc fallback) and consult `task udas` to avoid false positives when ~/.taskrc is not loaded.
* README now includes WSL-friendly installation and reinstall steps (venv + pipx) for dependencies like prompt_toolkit.
* README now includes uv-based environment/dependency setup and reinstall steps.
* `twh ondeck` marks started moves as in progress in its output, using a green highlight for visibility.
* Enforce LF line endings via `.gitattributes` and keep `core.autocrlf=input` for WSL development to avoid CRLF warnings.
* `twh ondeck` excludes moves whose `start` time is in the future from its report output, but still includes them in the missing-metadata wizard.
* Before any operation that could modify move descriptions (including writing UDAs that might be misconfigured), halt and ask for guidance. If a required UDA is missing and its absence could overwrite descriptions, stop and ask for guidance before proceeding.
* Move descriptions are sacrosanct. Never change a move description without halting all processes and requesting explicit permission.
* `twh add` is an interactive flow that prompts for description, project, tags, due date, blocks, and metadata (imp/urg/opt_human/diff/mode), then runs dominance sorting.
* `twh ondeck` ordering considers `Scheduled` and `Wait until`, prioritizing imminent/past times and otherwise preferring unscheduled moves ahead of future scheduled ones (using the later timestamp when both are set).
* `twh option` computes opt_auto estimates calibrated from manual `opt_human` ratings (falling back to `opt`) and can write them with `--apply`, which also migrates legacy `opt` values into `opt_human` when missing, using dependencies, due/priority, tags like `probe`, and optional `door`/`kind`/`estimate_minutes` UDAs.
* `twh calibrate` runs interactive pairwise precedence calibration and option value calibration, writes weights to `~/.config/twh/calibration.toml` (override with `TWH_CALIBRATION_PATH`), and can apply updated `opt_auto` values; `twh ondeck` and `twh option` use stored calibration weights when present.
* `twh diagnose` uses `twh ondeck` ordering to pick the default stuck move, launches the full `twh add` wizard for helper moves (enter the stuck move ID at the Blocks prompt to link), prompts to rate dimension UDAs (`energy`, `attention`, `emotion`, `interruptible`, `mechanical`) when missing, and prints `~/.taskrc` setup instructions before any dimension writes if UDAs are missing.
* `twh` enforces case-insensitive Taskwarrior searches for all selection-based commands and delegated task invocations.
* `twh ondeck` precedence scoring incorporates `enablement`, `blocker_relief`, `estimate_hours` (fallback to `diff`), and dependency centrality with a strategic/operational/explore mode multiplier.
* `twh start`/`twh stop` mirror Taskwarrior start/stop while logging time records to `~/.task/twh-time.db`, enforcing a single active move by stopping other started moves, and supporting reports/editing via `twh time`.
* `twh` enables readline-style line editing for interactive prompts (via `readline`/`pyreadline3`) so arrow keys and common editing keys work in terminals like Tabby on WSL.
* `twh defer` prints a summary of the top move from the ondeck ordering, prompts for a defer interval (number + m/h/d/w or minutes/hours/days/weeks), sets `start` to now plus the interval, and annotates the move with the deferral timestamps.

## Project notes

- `twh` delegates unknown commands (including no-arg invocation) to Taskwarrior; `list`, `reverse`, `tree`, `graph`, `simple`, `ondeck`, `defer`, `diagnose`, `option`, `calibrate`, `start`, `stop`, `time`, `dominance`, and `criticality` are handled internally.
- `twh add` uses an interactive prompt sequence and still augments new moves with the active Taskwarrior context's `project:` or tag filters (from `context.<name>`), without overriding explicit `project:` or `+tag` inputs.
- Running tests directly from the repo root needs `PYTHONPATH=src` (or an editable install) so `import twh` resolves the package.
- Prefer `.venv/bin/python -m pytest` over `/usr/bin/pytest`; the system pytest can fail with pluggy mismatch (`warn_on_impl_args`) and Apport `/var/crash` permission errors.
- Prefer Python 3.12 for local virtual environments; `uv venv` defaults to newer Python versions (for example 3.14) and may create a venv without pip, so use `python3.12 -m venv .venv` or `uv venv --python=python3.12`, and if pip is missing run `python -m ensurepip --upgrade`.
- If you recreate the venv with `uv venv`, the `uv` CLI installed in the previous venv is removed; reinstall it in the new venv (or use `python -m pip install -e .` directly).
- An editable install inside a venv only exposes `twh` in that venv (`.venv/bin/twh`); for universal access add that bin dir to `PATH` or install outside the venv (for example with `pipx` or `python3.12 -m pip install --user -e .`).
- On Ubuntu with PEP 668, `/usr/bin/python3.12 -m pip install --user pipx` may fail as externally-managed; install `pipx` inside a venv and run `python -m pipx install -e /mnt/d/local/src/py/twh` to create `~/.local/bin/twh`.
- `twh graph` renders a Graphviz SVG and opens it by default when `dot` is available, falling back to ASCII with `--ascii` or when rendering fails; nodes mirror the legacy graph metadata and coloring, and `reverse` flips edge direction.
- When running under Cygwin with a Windows Graphviz binary, graph converts output paths using `cygpath -w` so `dot` can write into the temp directory.
- `twh` implements `blocks` by exporting blocking and blocked moves, merging existing dependencies, and writing `depends:<uuid-list>` to avoid Taskwarrior rejecting `depends:+<uuid>`.
- Dependency values are normalized by stripping leading `+`/`-` prefixes before writing `depends` updates.
- `twh` delegates to Taskwarrior via `exec` when no internal handling is needed, and builds the Typer app lazily to reduce startup overhead.
- `twh simple` creates a Taskwarrior `report.simple` (if missing) by copying the default report and replacing the description column with `description.count`, then runs `task simple` directly; on WSL it disables pager/confirmations/hooks unless `TWH_SIMPLE_PAGER=1` is set, strips `limit:page` from the simple report, and runs with stdin closed to avoid interactive pauses.
- `twh ondeck` inspects pending moves for missing `imp`, `urg`, `opt_human` (legacy `opt` accepted), `diff`, `mode`, `criticality`, and dominance ordering (ties persisted via `dominated_by`); when anything is missing it runs the wizard for all moves in scope (including blocked), collects criticality rankings, auto-runs `twh option --apply` after updates, and recommends the next move using dominance tiers before the scoring model; mode filters (`--mode`, `--strict-mode`) and dominance UDAs influence candidate selection, extra Taskwarrior filter tokens after `twh ondeck` scope the set, and filtered runs still apply the wizard even for blocked moves in scope.
- `twh option` calibrates option value weights from manual `opt_human` ratings (falling back to `opt`), predicts `opt_auto` values using dependency structure and metadata, and `--apply` also copies legacy `opt` values into `opt_human` when missing.
- `twh calibrate` stores precedence/option calibration weights in `~/.config/twh/calibration.toml`, uses pairwise A/B choices to tune precedence weights, and can apply opt_auto updates after calibrating.
- `twh start`/`twh stop` log time entries (uuid/description/project/tags/mode/start/end) to `~/.task/twh-time.db`, auto-stop other active moves on start, and `twh time` reports by task/project/tag/mode with day/week/month/year/range bucketing plus date filtering and record edits.
- On WSL, `open_in_browser` converts paths with `wslpath -w`, copies UNC (Linux filesystem) paths into the Windows TEMP directory, and launches Windows Edge directly via `msedge.exe` (falling back to `cmd.exe /c start microsoft-edge:<file-url>` when Edge cannot be located).
- Do not modify `CHANGELOG.md` directly; it is managed via commitizen and manual edits interfere with the workflow.
- On Windows/PowerShell, Scoop `coreutils` `head` (Cygwin) can fail with `Couldn't reserve space for cygwin's heap`; prefer `Get-Content -TotalCount N <file>`/`Select-Object -First N`, or install `uutils-coreutils` and ensure it precedes `coreutils` in `PATH`.
- Tests that assert UDA-missing behavior should monkeypatch `missing_udas` or `get_defined_udas` to avoid relying on the host Taskwarrior config.
- Tests that exercise mode prompts or `register_mode` should set `TWH_MODES_PATH` to a temp path to avoid writing to `~/.config/twh/modes.json`.

