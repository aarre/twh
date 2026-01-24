## 0.4.0 (2026-01-23)

### Features

- **list**: custom JSON-based list formatting with tree connectors and aligned columns
- **graph**: default to svg output instead of png
- **graph**: enable `twh graph` and `twh graph reverse` commands, including rendering to svg by default
- **graph**: default graph outputs to `/tmp` instead of the current working directory
- **graph**: twh graph works for the first time

### Fixes

- **list**: format output directly from Taskwarrior JSON export to avoid concatenated lines

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
