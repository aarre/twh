# twh

Hierarchical Taskwarrior views and Mermaid dependency graphs.

```bash
twh list
twh list reverse
twh graph
twh graph2
twh graph2 reverse
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

By default, `twh graph` writes `/tmp/tasks.mmd` and `/tmp/tasks.csv`, then renders
`/tmp/tasks.svg` and opens it in your default web browser (use `--png` for PNG).
`twh list` and `twh graph` both parse `task export` JSON using shared helpers (no scraping).
Graph nodes use a two-tier layout: an urgency bar (rank-based colors, rounded to 2 decimals) and a status panel with ID, task name, and due date, colored by status (started/blocked/normal).

`twh graph2` renders a Graphviz-based dependency graph to
`/tmp/tasks-graph2.svg` and opens it by default (requires Graphviz `dot`). Node
labels mirror the Mermaid view: an urgency bar with rank-based colors plus a
status panel that includes ID, description, due date, and status coloring. It
falls back to an ASCII tree when Graphviz is unavailable or when `--ascii` is
set. Use `--png` or `--svg` to customize output paths, `--rankdir` to change
layout direction, and `--edges` to print the raw edge list.
