# twh

Hierarchical Taskwarrior views and Graphviz dependency graphs.

```bash
twh list
twh list reverse
twh simple
twh graph
twh graph reverse
```

Commands that `twh` doesn't recognize are forwarded to Taskwarrior, so `twh`
behaves like `task` and `twh add "Next task"` runs `task add "Next task"`.
When a Taskwarrior context is active and its definition includes `project:` or
tag filters, `twh add` automatically applies those values to new tasks. For
example, after `twh context define grin project:work.competitiveness.gloria.grinsector`
and `twh context grin`, running `twh add "Implement feedback"` appends
`project:work.competitiveness.gloria.grinsector` and prints an informational
message. This context-driven behavior
persists until `twh context none` clears the context.

`twh` also supports a synthetic `blocks` field on delegated `add` and `modify`
commands. `twh add "New task" blocks:32` creates the new task and then updates
task 32 with `depends:+<new-id>`. `twh 31 modify blocks 32` makes task 32 depend
on task 31. The relationship is stored in Taskwarrior's `depends` field; no UDAs
are required.

`twh simple` wraps Taskwarrior reports and shows annotation counts instead of
inline annotation text. On first run it creates `report.simple` by copying the
default Taskwarrior report (from `default.command`) and replacing the
`description` column with `description.count`, then it applies any default
filters from `default.command` plus your own filters before running the report.
On WSL it disables Taskwarrior's pager, confirmations, and hooks for this
command to avoid hangs; set `TWH_SIMPLE_PAGER=1` to keep paging. When paging is
disabled, any `limit:page` filter is removed from the simple report and from
default filters, and `twh simple` runs with stdin closed to prevent blocking.

`twh graph` renders a Graphviz-based dependency graph to `/tmp/tasks-graph.svg`
and opens it by default (requires Graphviz `dot`). On WSL, the SVG opens in
Windows Edge by default, copying the file into the Windows TEMP directory when
needed. Node labels include an
urgency bar with rank-based colors plus a status panel that lists ID,
description, due date, and status coloring (started/blocked/normal). It falls
back to an ASCII tree when Graphviz is unavailable or when `--ascii` is set.
Use `--png` or `--svg` to customize output paths, `--rankdir` to change layout
direction, and `--edges` to print the raw edge list.
