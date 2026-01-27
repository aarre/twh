# twh

Hierarchical Taskwarrior views and Graphviz dependency graphs.

```bash
twh list
twh list reverse
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

`twh graph` renders a Graphviz-based dependency graph to `/tmp/tasks-graph.svg`
and opens it by default (requires Graphviz `dot`). Node labels include an
urgency bar with rank-based colors plus a status panel that lists ID,
description, due date, and status coloring (started/blocked/normal). It falls
back to an ASCII tree when Graphviz is unavailable or when `--ascii` is set.
Use `--png` or `--svg` to customize output paths, `--rankdir` to change layout
direction, and `--edges` to print the raw edge list.
