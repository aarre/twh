# twh

Hierarchical Taskwarrior views and Mermaid dependency graphs.

```bash
twh
twh list reverse
twh graph
```

By default, `twh graph` writes `/tmp/tasks.mmd` and `/tmp/tasks.csv`, then renders
`/tmp/tasks.svg` and opens it in your default web browser (use `--png` for PNG).
`twh list` and `twh graph` both parse `task export` JSON using shared helpers (no scraping).
Graph nodes use a two-tier layout: an urgency bar (rank-based colors, rounded to 2 decimals) and a status panel with ID, task name, and due date, colored by status (started/blocked/normal).
