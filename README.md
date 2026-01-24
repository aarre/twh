# twh

Hierarchical Taskwarrior views and Mermaid dependency graphs.

```bash
twh
twh list reverse
twh graph
```

By default, `twh graph` writes `/tmp/tasks.mmd` and `/tmp/tasks.csv`, then renders
`/tmp/tasks.svg` and opens it in your default web browser (use `--png` for PNG).
`twh list` output is generated from `task export` JSON with custom formatting.
