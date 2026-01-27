## 0.9.1 (2026-01-27)

### Fixes

- **commitizen**: try to ignore passes-all-unit-test-* tags

## 0.9.0 (2026-01-27)

### Features

- **graph**: make `twh graph` and `twh graph reverse` use graphviz instead of mermaid
- **graph2**: color and label task nodes in the graph2/graphviz views
- **graph2**: add task hierarchy graphing using graphviz

## 0.8.0 (2026-01-27)

### Features

- **context/graph**: (1) use default project within context and (2) switch twh graph to Graphviz instead of Mermaid

## 0.7.0 (2026-01-27)

### Features

- **twh**: make unknown commands pass through to taskwarrior
- **codex**: add AGENTS.md
- **graph**: color tasks by urgency

### Fixes

- **twh**: capture current status

## 0.6.0 (2026-01-23)

### Features

- **graph**: render nodes as urgency bar + status panel
- **graph**: show urgency values in the top bar with rank-based gradient colors
- **graph**: show task ID, name, and due date in the status panel
- **graph**: color status panels by state (started/blocked/normal)
- **graph**: map urgency across the full gradient range with fixed opacity
- **graph**: parse string urgency values and round to 2 decimals before ranking

### Fixes

- **graph**: fix orphan tasks in graph view
- **list**: fall back to default columns/labels when report config is empty

## 0.5.0 (2026-01-23)

### Features

- **graph**: show all tasks (with or without dependencies or dependents) in `twh graph` views

### Fixes

- **twh list**: fix hierarchical list formatting

## 0.4.0 (2026-01-23)

### Features

- **list**: use taskwarrior formatting in list output
- **graph**: default to svg output instead of png
- **graph**: enable `twh graph` and `twh graph reverse` commands, including rendering to svg by default
- **graph**: twh graph works for the first time

## 0.3.0 (2026-01-23)

### Features

- **graph**: add graph feature
- **graph**: add new script from chatgpt pulse

### Fixes

- **gitignore**: add sley to .gitignore

### Tests

- **test**: add unit tests

### Build

- **pyproject**: add commitizen configuration

## 0.2.0 (2026-01-22)

### Features

- **twh**: package as script
- **twh**: check in initial version

### Fixes

- **gitignore**: populate .gitignore with typical entries
- **gitignore**: ignore .sley
- **pyproject**: fix commitizen configuration
- **pyproject**: add pytest dependency
- **reverse**: fix bug where task with multiple dependents only shown once in --reverse
