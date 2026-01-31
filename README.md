# twh

Hierarchical Taskwarrior move views and Graphviz dependency graphs.

```bash
twh list
twh list reverse
twh simple
twh dominance
twh review
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
due date, blocks, and review metadata (imp/urg/opt_human/diff/mode), then runs
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

`twh review` scans pending moves for missing metadata (imp/urg/opt_human/diff/mode)
and missing dominance ordering, prints a short list (ready moves first), and
then recommends the next move by scoring ready moves. The top-move list includes
each move's description, annotations, and a short list of first-order dominance
relations. Use `--wizard` to fill missing fields interactively and to collect
dominance ordering for moves in scope. Use `--mode editorial` (plus
`--strict-mode` if desired) to bias recommendations to your current mode. You
can pass Taskwarrior filter tokens after the command (for example
`twh review project:work.competitiveness -WAITING`) to limit the review scope.
When the wizard is enabled it prompts for missing metadata on all moves in
scope, even if they are blocked. The review flow expects the `imp`, `urg`, `opt_human`,
`diff`, `mode`, `dominates`, and `dominated_by` fields to exist as Taskwarrior
UDAs if you want to edit them. Manual option values are stored in `opt_human`;
legacy `opt` values are still accepted for scoring and calibration. When
`--wizard` is enabled, `twh review` automatically runs `twh option --apply`
after updates so opt_auto values stay in sync.
Review ordering also incorporates a precedence score based on `enablement`,
`blocker_relief`, `estimate_hours` (falling back to `diff`), dependency
centrality, and the move's mode (strategic/operational/explore). When a
calibration file is present (`twh calibrate`), review uses those tuned weights.
Review ordering also considers `Scheduled` and `Wait until`: moves without
those fields come before moves scheduled further in the future, except when the
timestamp is within 24 hours (or has already passed). Earlier scheduled or wait
times are ranked ahead of later ones; when both are set, the later timestamp
governs the ordering.
If a required UDA is missing, `twh review` and `twh dominance` will stop before
writing updates to avoid modifying move descriptions.

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

Taskwarrior UDA setup for review and dominance:

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
