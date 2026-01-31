
# Instructions for codex in the twh repo

## Continuous improvement

Update AGENTS.md whenever you learn something new about the project, including requirements, my preferences, design decisions, techiques that you tried and didn't work, and so on.

## Test-driven development

Use a test-driven development approach for all changes. Ensure that each change is accompanied by a corresponding test case to validate its functionality. The test cases should be placed in the `test` directory. They should be written in advance of the implementation, according to the TDD process. They should use the `pytest` framework. Annotate test classes, methods, and functions with `@pytest.mark.parametrize` to avoid repetition and with `unit`, `integration`, `slow` and other tags to help organize tests and enable selective execution.

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
* `twh dominance` collects dominance ordering with pairwise prompts (transitive, dependency-implied dominance) and writes `dominates`/`dominated_by` UDAs; tie decisions are persisted so they are not re-prompted; `twh review --wizard` includes this dominance stage for moves in scope.
* `twh review` tracks `diff` (difficulty hours) alongside `imp`/`urg`/`opt_human` (legacy `opt` accepted)/`mode`, treats missing dominance/diff as incomplete metadata, and ranks moves by dominance tier before scoring ties.
* `twh review --wizard` prompts for missing metadata on all moves in scope (including blocked) so a single run can collect everything.
* `twh review` formats annotation timestamps into a local, human-readable date-time string (US Eastern).
* Suppress Taskwarrior noise like `Modified 0 tasks.` in `twh` outputs.
* `twh graph` uses fixed-width Graphviz boxes and wraps move descriptions across lines.
* Dominance prompts use A/B/C choices with labels like `[A] Move ID 3: ...` to avoid numeric confusion with move IDs.
* Dominance should never prompt for move pairs already related by dependencies (including when dependencies are stored as IDs), and prompts should show approximate progress (comparisons complete/remaining).
* `twh review` outputs a top-move list that includes move descriptions, annotations, and a short list of first-order dominance relations.
* Before any operation that could modify move descriptions (including writing UDAs that might be misconfigured), halt and ask for guidance. If a required UDA is missing and its absence could overwrite descriptions, stop and ask for guidance before proceeding.
* `twh add` is an interactive flow that prompts for description, project, tags, due date, blocks, and metadata (imp/urg/opt_human/diff/mode), then runs dominance sorting.
* `twh review` ordering considers `Scheduled` and `Wait until`, prioritizing imminent/past times and otherwise preferring unscheduled moves ahead of future scheduled ones (using the later timestamp when both are set).
* `twh option` computes opt_auto estimates calibrated from manual `opt_human` ratings (falling back to `opt`) and can write them with `--apply`, which also migrates legacy `opt` values into `opt_human` when missing, using dependencies, due/priority, tags like `probe`, and optional `door`/`kind`/`estimate_minutes` UDAs.
* `twh calibrate` runs interactive pairwise precedence calibration and option value calibration, writes weights to `~/.config/twh/calibration.toml` (override with `TWH_CALIBRATION_PATH`), and can apply updated `opt_auto` values; `twh review` and `twh option` use stored calibration weights when present.
* `twh` enforces case-insensitive Taskwarrior searches for all selection-based commands and delegated task invocations.
* `twh review` precedence scoring incorporates `enablement`, `blocker_relief`, `estimate_hours` (fallback to `diff`), and dependency centrality with a strategic/operational/explore mode multiplier.
* `twh` enables readline-style line editing for interactive prompts (via `readline`/`pyreadline3`) so arrow keys and common editing keys work in terminals like Tabby on WSL.

## Project notes

- `twh` delegates unknown commands (including no-arg invocation) to Taskwarrior; `list`, `reverse`, `tree`, `graph`, `simple`, `review`, and `dominance` are handled internally.
- `twh add` uses an interactive prompt sequence and still augments new moves with the active Taskwarrior context's `project:` or tag filters (from `context.<name>`), without overriding explicit `project:` or `+tag` inputs.
- Running tests directly from the repo root needs `PYTHONPATH=src` (or an editable install) so `import twh` resolves the package.
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
- `twh review` inspects pending moves for missing `imp`, `urg`, `opt_human` (legacy `opt` accepted), `diff`, `mode`, and dominance ordering (ties persisted via `dominated_by`), optionally prompts to fill them (including dominance) with `--wizard`, auto-runs `twh option --apply` after wizard updates, and recommends the next move using dominance tiers before the scoring model; mode filters (`--mode`, `--strict-mode`) and dominance UDAs influence candidate selection, extra Taskwarrior filter tokens after `twh review` scope the review set, and filtered runs apply the wizard even for blocked moves in scope.
- `twh option` calibrates option value weights from manual `opt_human` ratings (falling back to `opt`), predicts `opt_auto` values using dependency structure and metadata, and `--apply` also copies legacy `opt` values into `opt_human` when missing.
- `twh calibrate` stores precedence/option calibration weights in `~/.config/twh/calibration.toml`, uses pairwise A/B choices to tune precedence weights, and can apply opt_auto updates after calibrating.
- On WSL, `open_in_browser` converts paths with `wslpath -w`, copies UNC (Linux filesystem) paths into the Windows TEMP directory, and launches Windows Edge directly via `msedge.exe` (falling back to `cmd.exe /c start microsoft-edge:<file-url>` when Edge cannot be located).
- Do not modify `CHANGELOG.md` directly; it is managed via commitizen and manual edits interfere with the workflow.

