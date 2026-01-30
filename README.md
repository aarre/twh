# twh

Hierarchical Taskwarrior move views and Graphviz dependency graphs.

```bash
twh list
twh list reverse
twh simple
twh dominance
twh review
twh graph
twh graph reverse
```

twh refers to Taskwarrior tasks as moves in its output and documentation.

Commands that `twh` doesn't recognize are forwarded to Taskwarrior, so `twh`
behaves like `task` for most subcommands and delegated commands replace the
`twh` process with `task` to keep overhead low.
`twh add` is interactive: it prompts for the move description, project, tags,
due date, blocks, and review metadata (imp/urg/opt/diff/mode), then runs the
dominance sorting step. Use `task add` for non-interactive adds.
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
ordering for your moves with minimal comparisons.

`twh review` scans pending moves for missing metadata (imp/urg/opt/diff/mode)
and missing dominance ordering, prints a short list (ready moves first), and
then recommends the next move by scoring ready moves. The top-move list includes
each move's description, annotations, and a short list of first-order dominance
relations. Use `--wizard` to fill missing fields interactively and to collect
dominance ordering for moves in scope. Use `--mode editorial` (plus
`--strict-mode` if desired) to bias recommendations to your current mode. You
can pass Taskwarrior filter tokens after the command (for example
`twh review project:work.competitiveness -WAITING`) to limit the review scope.
When the wizard is enabled it prompts for missing metadata on all moves in
scope, even if they are blocked. The review flow expects the `imp`, `urg`, `opt`,
`diff`, `mode`, `dominates`, and `dominated_by` fields to exist as Taskwarrior
UDAs if you want to edit them.
If a required UDA is missing, `twh review` and `twh dominance` will stop before
writing updates to avoid modifying move descriptions.

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
