# Graph Module

The `twh graph` command generates visual representations of Taskwarrior task dependencies using Mermaid flowcharts.

## Overview

This module converts Taskwarrior task exports into:
1. **Mermaid flowchart diagrams** (.mmd files) - showing task dependencies as a directed graph
2. **CSV exports** (.csv files) - for importing into other systems like Tana
3. **PNG renderings** - automatically rendered and opened in your default image viewer

## Usage

### Basic Usage

Generate a Mermaid graph from your current Taskwarrior tasks:

```bash
twh graph
```

This will:
- Export all pending tasks from Taskwarrior
- Generate `tasks.mmd` (Mermaid diagram)
- Generate `tasks.csv` (CSV export)
- Render to `tasks.png` and open it automatically

### Custom Output Paths

Specify custom output paths:

```bash
twh graph --output my-tasks.mmd --csv my-tasks.csv
```

### Skip PNG Rendering

If you just want the Mermaid file without rendering:

```bash
twh graph --no-render
```

## Features

### Chain Collapsing

The module automatically collapses simple dependency chains for cleaner visualization. For example:

- `A → B → C → D` (where B and C have only one predecessor and one successor)
- Will be displayed as a single arrow: `A → B → C → D`

This reduces visual clutter while preserving dependency information.

### Description Cleaning

Task descriptions are automatically cleaned for display:
- Newlines are replaced with spaces
- Descriptions are truncated to 60 characters
- Quotes are removed to avoid Mermaid syntax errors

### UUID Display

Each task node shows:
- The task description (truncated if needed)
- The first 8 characters of the task UUID (for identification)

Format: `"Task description\n(uuid-pre)"`

## Architecture

### Core Functions

#### `get_tasks_from_taskwarrior()`

Executes `task export` and returns a list of pending tasks.

**Returns:** `List[Dict]` - List of task dictionaries

**Raises:**
- `subprocess.CalledProcessError` - If taskwarrior command fails
- `json.JSONDecodeError` - If output cannot be parsed

#### `parse_dependencies(dep_field)`

Parses the `depends` field from a task into a list of UUID strings.

**Parameters:**
- `dep_field` (Optional[str]): Comma-separated string of UUIDs or None

**Returns:** `List[str]` - List of dependency UUIDs

#### `build_dependency_graph(tasks)`

Builds three core data structures from the task list:

**Parameters:**
- `tasks` (List[Dict]): List of task dictionaries

**Returns:** Tuple of:
- `uid_map` (Dict[str, Dict]): UUID → task dictionary
- `succ` (Dict[str, Set[str]]): UUID → set of tasks that depend on it (successors)
- `pred` (Dict[str, Set[str]]): UUID → set of tasks it depends on (predecessors)

#### `collapse_chains(uid_map, succ, pred)`

Collapses linear dependency chains into single edges for cleaner visualization.

**Parameters:**
- `uid_map`: Task UUID mapping
- `succ`: Successors mapping
- `pred`: Predecessors mapping

**Returns:** `List[List[str]]` - List of chains (each chain is a list of UUIDs)

**Algorithm:**
- Identifies tasks that have exactly one predecessor and one successor
- Combines them into chains with their neighbors
- Preserves branching points and convergence points

#### `generate_mermaid(uid_map, chains)`

Generates Mermaid flowchart syntax from dependency chains.

**Parameters:**
- `uid_map`: Task UUID mapping
- `chains`: List of dependency chains

**Returns:** `str` - Mermaid flowchart definition

**Output Format:**
```
flowchart TD
  "Task A\n(aaaa-111)" --> "Task B\n(bbbb-222)" --> "Task C\n(cccc-333)"
  "Task D\n(dddd-444)" --> "Task E\n(eeee-555)"
```

#### `write_csv_export(uid_map, pred, output_path)`

Writes tasks to CSV file for import into other systems.

**Parameters:**
- `uid_map`: Task UUID mapping
- `pred`: Predecessors mapping
- `output_path` (Path): Output CSV file path

**CSV Columns:**
- `Title`: Task description
- `Body`: Additional info (task ID, entry date)
- `Tags`: Semicolon-separated tags
- `UUID`: Full task UUID
- `DependsOn`: Comma-separated list of dependency UUIDs

#### `create_task_graph(tasks, output_mmd, output_csv)`

Main entry point that orchestrates the complete graph generation workflow.

**Parameters:**
- `tasks` (List[Dict]): List of task dictionaries
- `output_mmd` (Optional[Path]): Path to write Mermaid file (None = don't write)
- `output_csv` (Optional[Path]): Path to write CSV file (None = don't write)

**Returns:** `str` - Mermaid flowchart content

**Workflow:**
1. Build dependency graph structures
2. Collapse chains for cleaner visualization
3. Generate Mermaid syntax
4. Write output files (if paths provided)
5. Return Mermaid content

## Rendering

### PNG Rendering with Pyppeteer

The `renderer` module provides PNG rendering using pyppeteer (headless Chrome):

#### `render_mermaid_to_png(mermaid_file, output_file)`

Renders a Mermaid diagram to PNG.

**Parameters:**
- `mermaid_file` (Path): Input .mmd file
- `output_file` (Path): Output .png file

**Process:**
1. Reads Mermaid file content
2. Launches headless Chrome browser
3. Loads HTML page with Mermaid.js from CDN
4. Waits for diagram rendering
5. Calculates optimal viewport size
6. Takes screenshot
7. Saves as PNG

**Requirements:**
- pyppeteer package
- Chromium (automatically downloaded by pyppeteer on first run)

### Cross-Platform File Opening

#### `open_file(file_path)`

Opens a file in the default system viewer across different platforms.

**Parameters:**
- `file_path` (Union[str, Path]): File to open

**Platform Support:**
- **Cygwin**: Uses `cygstart` (detected by TERM environment or /usr/bin/cygstart)
- **Windows**: Uses `os.startfile()` or `start` command
- **macOS**: Uses `open` command
- **Linux/Unix**: Uses `xdg-open` command

**Raises:**
- `FileNotFoundError` - If file doesn't exist
- `RuntimeError` - If opening fails

## CSV Export for Tana

The CSV export is designed for easy import into Tana (or other systems):

### Import Workflow

1. Open `tasks.csv` in Google Sheets
2. Select all cells and copy
3. Paste into Tana

### Dependency Reconstruction

The CSV includes:
- `UUID`: Unique identifier for each task
- `DependsOn`: Comma-separated list of dependency UUIDs

This allows you to:
- Recreate dependency links in Tana
- Convert dependencies to references
- Build custom task relationship views

## Examples

### Example 1: Basic Workflow

```bash
# Generate graph from current tasks
twh graph

# Output:
# Generating Mermaid graph: tasks.mmd
# Generated Mermaid file: tasks.mmd
# Generated CSV file: tasks.csv
# Rendering to PNG: tasks.png
# Successfully rendered to: tasks.png
# [Image opens in default viewer]
```

### Example 2: Custom Output

```bash
# Generate with custom filenames
twh graph --output project-deps.mmd --csv project-tasks.csv

# Output:
# Generating Mermaid graph: project-deps.mmd
# Generated Mermaid file: project-deps.mmd
# Generated CSV file: project-tasks.csv
# Rendering to PNG: project-deps.png
# Successfully rendered to: project-deps.png
```

### Example 3: Mermaid Only

```bash
# Skip PNG rendering (faster, no browser needed)
twh graph --no-render

# Output:
# Generating Mermaid graph: tasks.mmd
# Generated Mermaid file: tasks.mmd
# Generated CSV file: tasks.csv
```

### Example 4: View in Mermaid Live Editor

```bash
# Generate Mermaid file
twh graph --no-render

# Copy content and paste into:
# https://mermaid.live/
```

## Mermaid Diagram Structure

### Flowchart Syntax

The module generates "top-down" flowcharts:

```mermaid
flowchart TD
  "Task description\n(uuid)" --> "Another task\n(uuid)"
```

### Node Format

Each node contains:
- Task description (first line)
- Short UUID in parentheses (second line)

Example: `"Setup dev environment\n(a1b2c3d4)"`

### Edge Format

Simple arrows (`-->`) connect dependent tasks:
- Arrow points from dependency to dependent
- `A --> B` means "B depends on A" (do A first, then B)

### Chains

Linear chains are preserved as multi-step arrows:

```
"Task A\n(uuid1)" --> "Task B\n(uuid2)" --> "Task C\n(uuid3)"
```

### Branching

When multiple tasks depend on the same task, separate arrows are created:

```
"Base Task\n(uuid)" --> "Task 1\n(uuid1)"
"Base Task\n(uuid)" --> "Task 2\n(uuid2)"
```

### Diamond Dependencies

Complex dependencies (e.g., task D depends on both B and C):

```
"A\n(uuid-a)" --> "B\n(uuid-b)" --> "D\n(uuid-d)"
"A\n(uuid-a)" --> "C\n(uuid-c)" --> "D\n(uuid-d)"
```

## Testing

Comprehensive test suite in `test/test_graph.py`:

### Test Coverage

- **Dependency Parsing**: Handles None, empty strings, single/multiple deps, whitespace
- **Graph Building**: Empty tasks, single tasks, chains, diamonds, missing deps
- **Chain Collapsing**: Independent tasks, simple chains, long chains, branching
- **Mermaid Generation**: Empty graphs, single chains, description truncation, escaping
- **CSV Export**: Basic export, dependencies, tags, all fields
- **Complete Workflow**: File writing, content verification

### Running Tests

```bash
# Run all tests
pytest test/test_graph.py

# Run specific test class
pytest test/test_graph.py::TestBuildDependencyGraph

# Run with verbose output
pytest test/test_graph.py -v

# Run with coverage
pytest test/test_graph.py --cov=twh.graph
```

## Error Handling

### Common Errors

**No tasks found:**
```
No pending tasks found.
```
- Check that you have pending tasks: `task list`
- Verify Taskwarrior is installed: `task --version`

**Taskwarrior not installed:**
```
Error executing taskwarrior: [Errno 2] No such file or directory: 'task'
```
- Install Taskwarrior: https://taskwarrior.org/download/

**Rendering failed:**
```
Error rendering to PNG: <error details>
You can view the Mermaid file in a Mermaid-compatible editor.
```
- Mermaid file was still generated successfully
- PNG rendering requires pyppeteer and Chromium
- View .mmd file in VS Code with Mermaid extension, or https://mermaid.live/
- Cygwin uses its own Python environment. If you installed `pyppeteer` in a Windows dev venv, Cygwin will not see it. Install `twh` (and its deps) into the Cygwin Python you run `twh` with, for example:
  - `python -m pip install -e /cygdrive/d/Local/src/py/twh`
  - or `uv pip install -e /cygdrive/d/Local/src/py/twh` from a Cygwin shell
- `pyppeteer` downloads a bundled Chromium on first render to provide a known-compatible headless browser. That download needs network access and disk space.
- If the download fails (e.g., `NoSuchKey` from the Chromium snapshot URL), `twh` will try to auto-detect a local Chrome/Chromium install. If that still fails, set `TWH_CHROMIUM_PATH` to a local browser binary and rerun, for example:
  - `export TWH_CHROMIUM_PATH=/usr/bin/chromium` (Cygwin package)
  - `export TWH_CHROMIUM_PATH=/cygdrive/c/Program\ Files/Google/Chrome/Application/chrome.exe`
- If Cygwin is picking up the Windows `twh` script first, run `python -m twh` or add a shim in `~/bin/twh` that calls `/usr/bin/python3 -m twh` so Cygwin uses the editable install from `/cygdrive/d/Local/src/py/twh`.

**File opening failed:**
```
Failed to open file <path>: <error>
```
- File was created successfully but couldn't be opened
- Manually open the PNG file from the filesystem

## Dependencies

### Python Packages

- `typer>=0.15.0` - CLI framework
- `pyppeteer>=2.0.0` - Headless Chrome for PNG rendering

### External Tools

- `task` (Taskwarrior) - For exporting task data
- Chrome/Chromium - Auto-downloaded by pyppeteer for rendering

### Platform-Specific

- **Cygwin**: `cygstart` (included with Cygwin)
- **Linux**: `xdg-open` (usually pre-installed)
- **macOS**: `open` (built-in)
- **Windows**: `start` (built-in)

## Integration with twh

The graph module integrates with the main `twh` CLI:

```bash
# Show dependency tree (default)
twh
twh tree
twh tree --reverse

# Generate dependency graph
twh graph
twh graph --output custom.mmd
twh graph --no-render

# Get help
twh --help
twh graph --help
```

## Future Enhancements

Potential future features:

- **Filtering**: Filter by project, tag, or status
- **Styling**: Custom colors, themes, or node shapes
- **Format Options**: Support DOT/Graphviz, PlantUML
- **Interactive**: HTML output with interactive diagram
- **Statistics**: Show dependency depth, critical path
- **Validation**: Detect circular dependencies, orphaned tasks

## Related Documentation

- [Taskwarrior Documentation](https://taskwarrior.org/docs/)
- [Mermaid Documentation](https://mermaid.js.org/)
- [Typer Documentation](https://typer.tiangolo.com/)
- [Pyppeteer Documentation](https://pyppeteer.github.io/pyppeteer/)

## License

This module is part of the `twh` package and uses the same license.
