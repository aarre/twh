
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

## Project notes

- `twh` delegates unknown commands (including no-arg invocation) to Taskwarrior; only `list`, `reverse`, `tree`, and `graph` are handled internally.
- `twh add` augments new tasks with the active Taskwarrior context's `project:` or tag filters (from `context.<name>`), without overriding explicit `project:` or `+tag` arguments, and inserts additions before `--` when present.
- Running tests directly from the repo root needs `PYTHONPATH=src` (or an editable install) so `import twh` resolves the package.
- Prefer Python 3.12 for local virtual environments; `uv venv` defaults to newer Python versions (for example 3.14) and may create a venv without pip, so use `python3.12 -m venv .venv` or `uv venv --python=python3.12`, and if pip is missing run `python -m ensurepip --upgrade`.
- If you recreate the venv with `uv venv`, the `uv` CLI installed in the previous venv is removed; reinstall it in the new venv (or use `python -m pip install -e .` directly).
- An editable install inside a venv only exposes `twh` in that venv (`.venv/bin/twh`); for universal access add that bin dir to `PATH` or install outside the venv (for example with `pipx` or `python3.12 -m pip install --user -e .`).
- On Ubuntu with PEP 668, `/usr/bin/python3.12 -m pip install --user pipx` may fail as externally-managed; install `pipx` inside a venv and run `python -m pipx install -e /mnt/d/local/src/py/twh` to create `~/.local/bin/twh`.
- `twh graph` renders a Graphviz SVG and opens it by default when `dot` is available, falling back to ASCII with `--ascii` or when rendering fails; nodes mirror the legacy graph metadata and coloring, and `reverse` flips edge direction.
- When running under Cygwin with a Windows Graphviz binary, graph converts output paths using `cygpath -w` so `dot` can write into the temp directory.
- `twh` implements `blocks` by exporting the blocking tasks to resolve UUIDs and then adding `depends:+<uuid>` to each blocked task.
- `twh simple` creates a Taskwarrior `report.simple` (if missing) by copying the default report and replacing the description column with `description.count`, then runs `task` with default-command filters plus user filters; on WSL it disables the pager unless `TWH_SIMPLE_PAGER=1` is set.
- On WSL, `open_in_browser` converts paths with `wslpath -w`, copies UNC (Linux filesystem) paths into the Windows TEMP directory, and launches Windows Edge directly via `msedge.exe` (falling back to `cmd.exe /c start microsoft-edge:<file-url>` when Edge cannot be located).

