## 0.9.4 (2026-01-29)

### Performance

- **twh**: improve performance, especially for `twh simple`

## 0.9.3 (2026-01-29)

### Fixes

- **pyproject**: restore `ignored_tag_formats` in `pyproject.toml`

## 0.9.2 (2026-01-29)

### Fixes

- **pyproject/changelog**: align version numbers in pyproject and changelog
- **pyproject**: comment out `ignored_tag_formats` in pyproject.toml
- **pyproject**: change pyproject to `changelog_incremental = false`
- **pyproject**: correct version number in pyproject.toml

## 0.9.1 (2026-01-29)

### Features

- **twh**: open edge not inkscape, create simple report
- **blocks**: add a blocks relation inverse to depends

### Fixes

- **pyproject**: force annotated tags to keep commitizen happy
- **simple**: make `twh simple` work at last
- **commitizen**: try to ignore passes-all-unit-test-* tags

## 0.9.0 (2026-01-27)

### Features

- **graph**: make `twh graph` and `twh graph reverse` use graphviz instead of mermaid
- **graph2**: color and label task nodes in the graph2/graphviz views
- **graph2**: add task hierarchy graphing using graphviz

## 0.8.0 (2026-01-27)

### Features

- **context/graph**: (1) use default project within context and (2) implement twh graph2 to use graphviz instead of mermaid

## 0.7.0 (2026-01-27)

### Features

- **twh**: make unknown commands pass through to taskwarrior
- **codex**: add AGENTS.md
- **graph**: color tasks by urgency

### Fixes

- **twh**: capture current status

## 0.6.0 (2026-01-23)

### Features

- **graph**: color graph nodes by status

### Fixes

- **graph**: fix orphan tasks in graph view

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
