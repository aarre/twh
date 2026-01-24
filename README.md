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
Graph nodes include task IDs, due dates, and priority-graded colors with started/blocked overrides.
