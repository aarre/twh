## 0.14.0 (2026-01-31)

### Features

- **ondeck**: rename twh review to twh ondeck and always run wizard when needed

## 0.13.1 (2026-01-31)

### Fixes

- **readline**: enable readline-style line editing at twh startup so interactive prompts (like the add description field) accept arrow keys and common editing shortcuts in Tabby/WSL

## 0.13.0 (2026-01-31)

### Features

- **review**: integreate new precedence ("o") score into `twh review` ordering

### Fixes

- **twh**: make twh matches non-case-sensitive

## 0.12.1 (2026-01-30)

### Fixes

- **dominance**: stop prompting over and over about dominance ties

## 0.12.0 (2026-01-30)

### Features

- **options**: implement auto calculations of options values

## 0.11.0 (2026-01-30)

### Features

- **review**: make review schedule- and wait-aware
- **add**: make `twh add` run an interactive session to collect all metadata

### Fixes

- **blocks**: attempt to fix bug with `twh modify blocks:nn`

## 0.10.0 (2026-01-29)

### Features

- **domination**: add dominance relations to priority calculations
- **review**: add `twh review` command

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
