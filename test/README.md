# Task Tree Test Suite

Comprehensive unit tests for `task_tree.py`.

## Running Tests

```bash
# Run all tests
cd test
python -m unittest test_task_tree.py

# Run specific test class
python -m unittest test_task_tree.TestNormalModeHierarchy

# Run specific test
python -m unittest test_task_tree.TestNormalModeHierarchy.test_all_tasks_appear_exactly_once

# Run with verbose output
python -m unittest test_task_tree.py -v
```

## Test Coverage

### `TestBuildDependencyGraph`
Tests the core dependency graph construction:
- Empty task lists
- Single tasks with/without dependencies
- Simple dependency chains (A depends on B)
- Multiple dependencies (A depends on B and C)
- Diamond dependencies (complex graphs)
- Whitespace handling in dependency fields
- Empty dependency fields

### `TestNormalModeHierarchy`
Tests normal mode output (unblocked tasks at top):
- **All tasks appear exactly once** - Critical test ensuring no tasks are missed
- Unblocked tasks appear at top level with no indentation
- Blocked tasks appear indented under their dependents
- Correct indentation levels for dependency chains
- Completed dependencies don't appear in output
- Tasks with only completed dependencies appear at top level
- Multiple independent tasks all appear at top level

### `TestReverseModeHierarchy`
Tests reverse mode output (blocking tasks at top):
- **All tasks appear exactly once** - Critical test ensuring no tasks are missed
- **Same tasks as normal mode** - Critical test ensuring both modes show identical task sets
- Blocking tasks (no dependencies) appear at top level
- Dependent tasks appear indented
- Correct indentation levels for reverse chains
- Tasks with only completed dependencies appear at top level

### `TestEdgeCases`
Tests fragile scenarios that could break:
- **Circular dependencies** - A depends on B, B depends on A (must not hang)
- Self-dependencies - Task depends on itself
- Orphaned dependencies - Task depends on non-existent task
- Multiple tasks depending on same task
- Complex dependency webs with multiple branches
- Urgency-based sorting at same hierarchy level

### `TestFormatTask`
Tests task formatting:
- Basic task with all fields
- Missing ID (shows "?")
- Missing urgency (shows 0.0)
- Missing description (shows empty string)

## Key Test Assertions

### Hierarchy Tests
- **Task visibility**: Every pending task must appear exactly once in output
- **Mode parity**: Normal and reverse modes must show identical sets of tasks
- **Indentation**: Tasks must be indented correctly based on dependency level
- **Top-level criteria**:
  - Normal mode: Tasks with no pending dependents
  - Reverse mode: Tasks with no pending dependencies

### Edge Case Tests
- **No infinite loops**: Circular dependencies must terminate
- **No duplicates**: Tasks appearing multiple times in dependency graph show once
- **No missing tasks**: All pending tasks must appear regardless of dependency complexity

## Common Failure Scenarios

1. **Missing tasks in output**: If a task doesn't appear, check:
   - Is it being filtered out incorrectly?
   - Is the top-level detection logic correct?
   - Are completed dependencies being handled properly?

2. **Duplicate tasks**: If a task appears multiple times:
   - Check the `visited` set logic
   - Verify recursive functions check visited status

3. **Wrong indentation**: If hierarchy is incorrect:
   - Verify `level` parameter is passed correctly
   - Check dependency direction (depends_on vs depended_by)

4. **Mode mismatch**: If normal and reverse show different tasks:
   - Check top-level detection logic in both modes
   - Ensure completed dependency handling is consistent

## Adding New Tests

When adding tests, consider:
1. Does it test a user-visible behavior?
2. Does it test a fragile code path that could break?
3. Does it verify a bug fix?
4. Does it cover an edge case?

Example test template:
```python
def test_your_scenario(self):
    """Brief description of what this tests."""
    tasks = [
        {"uuid": "a", "id": 1, "description": "Task A", "urgency": 5.0, "depends": "b"}
    ]
    task_map, depends_on, depended_by = build_dependency_graph(tasks)
    output = self.capture_output(task_map, depends_on, depended_by)

    # Your assertions here
    self.assertEqual(output.count("[1]"), 1)
```
