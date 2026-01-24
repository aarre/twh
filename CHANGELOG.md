## 0.5.0 (2026-01-23)

### Features

- **graph**: show all tasks (with or without dependencies or dependents) in `twh graph` views
- **list/graph**: share Taskwarrior JSON export parsing helpers

### Fixes

- **twh list**: fix hierarchical list formatting
- **graph**: preserve edges for branching tasks with a single predecessor

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
