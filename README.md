# twh

Hierarchical Taskwarrior move views and Graphviz dependency graphs.

## Installation

Local development (uv, Python 3.12):

```bash
cd /mnt/d/local/src/py/twh
uv venv --python=python3.12
source .venv/bin/activate
uv pip install -e .
```

If uv creates a venv without pip, run:

```bash
python -m ensurepip --upgrade
```

Local development (standard venv, Python 3.12):

```bash
cd /mnt/d/local/src/py/twh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

System install via pipx (WSL-friendly):

```bash
python3.12 -m pipx install -e /mnt/d/local/src/py/twh
```

Reinstall after dependency changes (uv venv):

```bash
uv pip install -e .
```

Reinstall after dependency changes (pipx):

```bash
python3.12 -m pipx reinstall twh
```

```bash
twh list
twh list reverse
twh simple
twh start
twh stop
twh time
twh dominance
twh ondeck
twh defer
twh diagnose
twh option
twh calibrate
twh graph
twh graph reverse
```

twh refers to Taskwarrior tasks as moves in its output and documentation.
Unlike `task`, `twh` forces case-insensitive searches so `twh Task` matches
moves containing `task`, `Task`, or `TASK`.

Commands that `twh` doesn't recognize are forwarded to Taskwarrior, so `twh`
behaves like `task` for most subcommands and delegated commands replace the
`twh` process with `task` to keep overhead low.
`twh add` is interactive: it prompts for the move description, project, tags,
due date, blocks, and ondeck metadata (imp/urg/opt_human/diff/mode), then runs
the dominance sorting step. Use `task add` for non-interactive adds.
When a Taskwarrior context is active and its definition includes `project:` or
tag filters, `twh add` automatically applies those values to new moves unless
you supply them yourself, and prints an informational message. For example,
after `twh context define grin project:work.competitiveness.gloria.grinsector`
and `twh context grin`, running `twh add` and leaving the project blank applies
`project:work.competitiveness.gloria.grinsector`.
This context-driven behavior
persists until `twh context none` clears the context.

`twh` also supports a synthetic `blocks` field on `modify`, and `twh add`
prompts for blocked move IDs. The relationship is stored in Taskwarrior's
`depends` field; no UDAs are required. For example, `twh 31 modify blocks 32`
makes move 32 depend on move 31.

`twh start` and `twh stop` behave like `task start` and `task stop` but also
write time logs under `~/.task/twh-time.db`. Starting a move stops any other
started moves first, so only one move can be active at a time. Each log entry
stores the move UUID, description, project, tags, mode, and start/end times.
Use `twh time` for reports, `twh time entries` to list raw entries, and
`twh time edit <id>` to adjust start/end/durations after the fact. Reports can
group by task/project/tag/mode/total and by day/week/month/year/range, with
optional date filters. For example:

```bash
twh time --by project --period month --from 2024-01-01 --to 2024-03-31
twh time --by tag --period week --from 2024-02-01
twh time entries --from 2024-02-01 --to 2024-02-07
twh time edit 42 --duration 1.5h
```

`twh simple` wraps Taskwarrior reports and shows annotation counts instead of
inline annotation text. On first run it creates `report.simple` by copying the
default Taskwarrior report (from `default.command`) and replacing the
`description` column with `description.count`, then runs the report directly
for speed. On WSL it disables Taskwarrior's pager, confirmations, and hooks for
this command to avoid hangs; set `TWH_SIMPLE_PAGER=1` to keep paging. When
paging is disabled, any `limit:page` filter is removed from the simple report
and `twh simple` runs with stdin closed to prevent blocking.

`twh dominance` walks you through pairwise dominance choices for moves in
scope, taking dependencies into account, and records the resulting hierarchy
in the `dominates` and `dominated_by` UDAs. Use it to establish a dominance
ordering for your moves with minimal comparisons. Tie selections are persisted
so the same pair will not be prompted again.

`twh ondeck` scans pending moves for missing metadata (imp/urg/opt_human/diff/mode)
and missing dominance ordering. If anything is missing, it walks you through
the wizard to fill metadata (including blocked moves) and collect dominance
ordering, then recommends the next move by scoring ready moves. If metadata and
dominance are complete, it emits the report directly. The top-move list is
rendered in the same table layout and color scheme as the default Taskwarrior
report, with the ID column placed first and the urgency column relabeled
`Rank` to show the composite ordering (1 is the highest-ranked move). A
separate `Score` column shows the underlying numeric score used in ranking.
You can show more candidates by default (25; use `--top` to override). Started
moves are labeled `[IN PROGRESS]` with a green highlight. Use `--mode editorial`
(plus `--strict-mode` if desired) to bias recommendations to your current mode.
When the wizard prompts for a mode value, twh keeps a persistent list of known
modes (stored in `~/.config/twh/modes.json`, override with `TWH_MODES_PATH`) and
offers inline autocompletion; new modes are immediately added to the list,
shown in subsequent prompts, and appended to `uda.mode.values` when that
Taskwarrior setting is present. Mode values that match Taskwarrior core
attribute/status keywords (for example `wait`) are rejected; choose an
alternative like `waiting`.
To list reserved mode values in your environment, compare Taskwarrior core
attributes against UDAs and avoid status keywords like `pending`, `completed`,
`deleted`, `recurring`, and `waiting`:

```bash
task _columns
task udas
```

Any core attribute name (from `task _columns`) that is not a UDA (from
`task udas`) should be treated as reserved mode values.
You can pass Taskwarrior filter tokens after the command (for example
`twh ondeck project:work.competitiveness -WAITING`) to limit the scope. The
ondeck flow expects the `imp`, `urg`, `opt_human`, `diff`, `mode`, `dominates`,
and `dominated_by` fields to exist as Taskwarrior UDAs if you want to edit them.
Manual option values are stored in `opt_human`; legacy `opt` values are still
accepted for scoring and calibration. When the wizard runs, `twh ondeck`
automatically runs `twh option --apply` after updates so opt_auto values stay in
sync.
Ondeck ordering also incorporates a precedence score based on `enablement`,
`blocker_relief`, `estimate_hours` (falling back to `diff`), dependency
centrality, and the move's mode (strategic/operational/explore). When a
calibration file is present (`twh calibrate`), ondeck uses those tuned weights.
Ondeck ordering also considers `Scheduled` and `Wait until`: moves without
those fields come before moves scheduled further in the future, except when the
timestamp is within 24 hours (or has already passed). Earlier scheduled or wait
times are ranked ahead of later ones; when both are set, the later timestamp
governs the ordering.
Moves with a `start` time in the future are excluded from the ondeck report
until that time arrives, but the wizard still prompts for missing metadata on
all moves in scope.
If a required UDA is missing, `twh ondeck` and `twh dominance` will stop before
writing updates to avoid modifying move descriptions.

`twh defer` targets the current top move from the same ordering used by
`twh ondeck`. It prints a summary of that move, then prompts `Defer for how long
(m/h/d/w)?`. Enter a number and unit (for example `15 m`, `2 hours`, or `1 day`).
The move's `start` field is set to the current time plus the interval, and an
annotation like `2026-02-02 18:45 -- Deferred for 1 day to 2026-02-03 18:45.` is
added. `twh defer` accepts the same mode flags as ondeck (`--mode`,
`--strict-mode`, `--include-dominated`) plus Taskwarrior filter tokens after the
command to scope the candidate set.

`twh diagnose` runs a short wizard when a move feels stuck. By default it targets
the current top move based on the same ordering used by `twh ondeck`, asks for a
friction type and what you are lacking, and offers a tiny helper move that keeps
the top move warm by launching the full `twh add` wizard. When the add wizard
prompts for Blocks, enter the stuck move ID to make it depend on the helper.
Diagnose can also prompt to rate dimension values (energy/attention/emotion/
interruptible/mechanical), and it suggests an alternative move that is strictly
easier along the lacking dimension (energy/attention/emotion/time) using those
values. If the dimension UDAs are missing, `twh diagnose` stops and prints the
required `~/.taskrc` entries before writing any dimension values to avoid
modifying move descriptions.

`twh option` estimates option value from the dependency graph and metadata,
calibrates weights using your manual `opt_human` ratings (falling back to legacy
`opt` values), and prints predicted `opt_auto` values. If a calibration file
exists (from `twh calibrate`), `twh option` uses those stored weights; otherwise
it fits weights from your manual ratings. Use `--apply` to write `opt_auto` to
moves in scope and `--include-rated` to see predictions alongside manual
ratings. `twh option --apply` also copies legacy `opt` values into `opt_human`
when `opt_human` is missing. Option value uses
dependency depth, project diversity, due-date urgency, priority, and tags such
as `probe`/`explore`/`call`/`prototype`/`test` to model information gain and
flexibility. If you set `door=oneway` or an `estimate_minutes` UDA, those are
also incorporated as reversibility and effort penalties.

`twh calibrate` runs an interactive pairwise loop to tune precedence weights
using your `enablement`, `blocker_relief`, and `estimate_hours`/`diff` values,
then calibrates option value weights from manual `opt_human` ratings. It writes
the resulting weights to `~/.config/twh/calibration.toml` (override with
`TWH_CALIBRATION_PATH`) and, by default, applies updated `opt_auto` values to
moves in scope (`--no-apply` skips writes).

Taskwarrior UDA setup for ondeck and dominance:

```
# Dominance edges (comma-separated UUIDs)
uda.dominates.type=string
uda.dominates.label=Dom>
uda.dominated_by.type=string
uda.dominated_by.label=<Dom

# Difficulty (estimated hours of effort)
uda.diff.type=numeric
uda.diff.label=Diff(h)

# Precedence scoring helpers
uda.enablement.type=numeric
uda.enablement.label=Enablement (0-10)
uda.blocker_relief.type=numeric
uda.blocker_relief.label=Blocker Relief (0-10)
uda.estimate_hours.type=numeric
uda.estimate_hours.label=Estimate (hours)

# Manual option value ratings
uda.opt_human.type=numeric
uda.opt_human.label=OptHuman

# Auto option value estimates
uda.opt_auto.type=numeric
uda.opt_auto.label=OptAuto

# Optional option value fields
uda.door.type=string
uda.door.label=Door
uda.kind.type=string
uda.kind.label=Kind
uda.estimate_minutes.type=numeric
uda.estimate_minutes.label=Est(min)

# Diagnose dimensions (0-10 scales unless noted)
uda.energy.type=numeric
uda.energy.label=Energy (0-10)
uda.attention.type=numeric
uda.attention.label=Attention (0-10)
uda.emotion.type=numeric
uda.emotion.label=Emotion (0-10)
uda.interruptible.type=numeric
uda.interruptible.label=Interruptible (0-1)
uda.mechanical.type=numeric
uda.mechanical.label=Mechanical (0-1)
```

`twh graph` renders a Graphviz-based dependency graph to `/tmp/tasks-graph.svg`
and opens it by default (requires Graphviz `dot`). On WSL, the SVG opens in
Windows Edge by default, copying the file into the Windows TEMP directory when
needed. Node labels include an
urgency bar with rank-based colors plus a status panel that lists ID,
description (wrapped in fixed-width boxes), due date, and status coloring
(started/blocked/normal). It falls
back to an ASCII tree when Graphviz is unavailable or when `--ascii` is set.
Use `--png` or `--svg` to customize output paths, `--rankdir` to change layout
direction, and `--edges` to print the raw edge list.

Recovery helper:

If move descriptions were accidentally overwritten by `diff:` values, you can
restore them from Taskwarrior history with:

```bash
python -m twh.restore_descriptions --apply
```

Add Taskwarrior filter tokens to scope the restore (for example
`python -m twh.restore_descriptions --apply project:work`).
If Taskwarrior cannot find your data, add `--taskrc` or `--data` (or set
`TASKRC` / `TASKDATA` in your environment).
