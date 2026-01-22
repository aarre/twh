#!/usr/bin/env python3
"""
Comprehensive test suite for twh.py

Tests cover:
- Dependency graph building
- Normal mode hierarchy (unblocked tasks at top)
- Reverse mode hierarchy (blocking tasks at top)
- Edge cases: circular dependencies, completed dependencies, orphaned tasks
- Task visibility: ensuring all pending tasks appear in output
"""

import sys
import unittest
from io import StringIO
from collections import defaultdict
from typing import Dict, List, Set

# Import from parent directory
sys.path.insert(0, '..')
from twh import build_dependency_graph, print_tree_normal, print_tree_reverse, format_task


class TestBuildDependencyGraph(unittest.TestCase):
    """Test dependency graph construction."""

    def test_empty_task_list(self):
        """Empty task list should produce empty graphs."""
        task_map, depends_on, depended_by = build_dependency_graph([])
        self.assertEqual(task_map, {})
        self.assertEqual(dict(depends_on), {})
        self.assertEqual(dict(depended_by), {})

    def test_single_task_no_dependencies(self):
        """Single task with no dependencies."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Task A", "urgency": 5.0}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        self.assertEqual(len(task_map), 1)
        self.assertIn("a", task_map)
        self.assertEqual(len(depends_on.get("a", set())), 0)
        self.assertEqual(len(depended_by.get("a", set())), 0)

    def test_simple_dependency_chain(self):
        """Test A depends on B."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Task A", "urgency": 5.0, "depends": "b"},
            {"uuid": "b", "id": 2, "description": "Task B", "urgency": 3.0}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        self.assertEqual(depends_on["a"], {"b"})
        self.assertEqual(depended_by["b"], {"a"})
        self.assertEqual(len(depends_on.get("b", set())), 0)
        self.assertEqual(len(depended_by.get("a", set())), 0)

    def test_multiple_dependencies(self):
        """Test A depends on B and C."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Task A", "urgency": 5.0, "depends": "b,c"},
            {"uuid": "b", "id": 2, "description": "Task B", "urgency": 3.0},
            {"uuid": "c", "id": 3, "description": "Task C", "urgency": 2.0}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        self.assertEqual(depends_on["a"], {"b", "c"})
        self.assertEqual(depended_by["b"], {"a"})
        self.assertEqual(depended_by["c"], {"a"})

    def test_diamond_dependency(self):
        """Test diamond: D depends on B,C; B,C depend on A."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Task A", "urgency": 1.0},
            {"uuid": "b", "id": 2, "description": "Task B", "urgency": 2.0, "depends": "a"},
            {"uuid": "c", "id": 3, "description": "Task C", "urgency": 3.0, "depends": "a"},
            {"uuid": "d", "id": 4, "description": "Task D", "urgency": 4.0, "depends": "b,c"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        self.assertEqual(depends_on["d"], {"b", "c"})
        self.assertEqual(depends_on["b"], {"a"})
        self.assertEqual(depends_on["c"], {"a"})
        self.assertEqual(depended_by["a"], {"b", "c"})
        self.assertEqual(depended_by["b"], {"d"})
        self.assertEqual(depended_by["c"], {"d"})

    def test_whitespace_in_depends(self):
        """Test that whitespace in depends field is handled."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Task A", "urgency": 5.0, "depends": " b , c "},
            {"uuid": "b", "id": 2, "description": "Task B", "urgency": 3.0},
            {"uuid": "c", "id": 3, "description": "Task C", "urgency": 2.0}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        self.assertEqual(depends_on["a"], {"b", "c"})

    def test_empty_depends_field(self):
        """Test task with empty depends field."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Task A", "urgency": 5.0, "depends": ""}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        self.assertEqual(len(depends_on.get("a", set())), 0)


class TestNormalModeHierarchy(unittest.TestCase):
    """Test normal mode output (unblocked tasks at top level)."""

    def capture_output(self, task_map, depends_on, depended_by):
        """Helper to capture print output."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            print_tree_normal(task_map, depends_on, depended_by)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return output

    def test_all_tasks_appear_exactly_once(self):
        """Every pending task should appear exactly once in output."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Task A", "urgency": 5.0},
            {"uuid": "b", "id": 2, "description": "Task B", "urgency": 3.0, "depends": "a"},
            {"uuid": "c", "id": 3, "description": "Task C", "urgency": 2.0}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        # Each task should appear exactly once
        self.assertEqual(output.count("[1]"), 1)
        self.assertEqual(output.count("[2]"), 1)
        self.assertEqual(output.count("[3]"), 1)

    def test_unblocked_at_top_level(self):
        """Unblocked tasks should have no indentation."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Unblocked", "urgency": 5.0}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        lines = [l for l in output.split('\n') if '[1]' in l]
        self.assertEqual(len(lines), 1)
        self.assertFalse(lines[0].startswith('  '))  # No indentation

    def test_blocked_task_indented(self):
        """Blocked tasks should be indented under their dependents."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Blocking", "urgency": 1.0},
            {"uuid": "b", "id": 2, "description": "Blocked", "urgency": 5.0, "depends": "a"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        lines = output.split('\n')
        task1_line = next(i for i, l in enumerate(lines) if '[1]' in l)
        task2_line = next(i for i, l in enumerate(lines) if '[2]' in l)

        # Task 2 (blocked) should appear first at top level (no other task depends on it)
        # Task 1 (blocking) should appear indented under task 2
        self.assertLess(task2_line, task1_line)
        # Task 1 should be indented under task 2
        self.assertTrue(lines[task1_line].startswith('  '))

    def test_dependency_chain_indentation(self):
        """Test C->B->A shows correct indentation levels."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "A", "urgency": 1.0},
            {"uuid": "b", "id": 2, "description": "B", "urgency": 2.0, "depends": "a"},
            {"uuid": "c", "id": 3, "description": "C", "urgency": 3.0, "depends": "b"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        lines = output.split('\n')
        task1_line = next(l for l in lines if '[1]' in l)
        task2_line = next(l for l in lines if '[2]' in l)
        task3_line = next(l for l in lines if '[3]' in l)

        # C at level 0, B at level 1, A at level 2
        self.assertEqual(len(task3_line) - len(task3_line.lstrip()), 0)
        self.assertEqual(len(task2_line) - len(task2_line.lstrip()), 2)
        self.assertEqual(len(task1_line) - len(task1_line.lstrip()), 4)

    def test_completed_dependency_not_shown(self):
        """Tasks with completed dependencies should appear at top level."""
        tasks = [
            {"uuid": "b", "id": 2, "description": "Blocked by completed", "urgency": 5.0, "depends": "a"}
            # Note: task 'a' is not in the list (completed/deleted)
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        # Task 2 should appear at top level (no indentation)
        lines = [l for l in output.split('\n') if '[2]' in l]
        self.assertEqual(len(lines), 1)
        self.assertFalse(lines[0].startswith('  '))

    def test_multiple_top_level_tasks(self):
        """Multiple unrelated tasks should all appear at top level."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "A", "urgency": 5.0},
            {"uuid": "b", "id": 2, "description": "B", "urgency": 3.0},
            {"uuid": "c", "id": 3, "description": "C", "urgency": 1.0}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        lines = output.split('\n')
        for task_id in ['1', '2', '3']:
            task_lines = [l for l in lines if f'[{task_id}]' in l]
            self.assertEqual(len(task_lines), 1)
            self.assertFalse(task_lines[0].startswith('  '))


class TestReverseModeHierarchy(unittest.TestCase):
    """Test reverse mode output (blocking tasks at top level)."""

    def capture_output(self, task_map, depends_on, depended_by):
        """Helper to capture print output."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            print_tree_reverse(task_map, depends_on, depended_by)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return output

    def test_all_tasks_appear_exactly_once(self):
        """Every pending task should appear exactly once in reverse mode."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Task A", "urgency": 5.0},
            {"uuid": "b", "id": 2, "description": "Task B", "urgency": 3.0, "depends": "a"},
            {"uuid": "c", "id": 3, "description": "Task C", "urgency": 2.0}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        # Each task should appear exactly once
        self.assertEqual(output.count("[1]"), 1)
        self.assertEqual(output.count("[2]"), 1)
        self.assertEqual(output.count("[3]"), 1)

    def test_same_tasks_as_normal_mode(self):
        """Reverse mode should show exactly the same tasks as normal mode."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "A", "urgency": 1.0},
            {"uuid": "b", "id": 2, "description": "B", "urgency": 2.0, "depends": "a"},
            {"uuid": "c", "id": 3, "description": "C", "urgency": 3.0},
            {"uuid": "d", "id": 4, "description": "D", "urgency": 4.0, "depends": "x"}  # completed dep
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        old_stdout = sys.stdout

        sys.stdout = StringIO()
        print_tree_normal(task_map, depends_on, depended_by)
        normal_output = sys.stdout.getvalue()

        sys.stdout = StringIO()
        print_tree_reverse(task_map, depends_on, depended_by)
        reverse_output = sys.stdout.getvalue()

        sys.stdout = old_stdout

        # Same task IDs should appear
        for task_id in ['1', '2', '3', '4']:
            self.assertEqual(
                normal_output.count(f'[{task_id}]'),
                reverse_output.count(f'[{task_id}]'),
                f"Task {task_id} appears different number of times"
            )

    def test_blocking_task_at_top(self):
        """In reverse mode, tasks with no deps should be at top."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Blocking", "urgency": 1.0},
            {"uuid": "b", "id": 2, "description": "Blocked", "urgency": 5.0, "depends": "a"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        lines = output.split('\n')
        task1_lines = [l for l in lines if '[1]' in l]

        # Task 1 should be at top level (no indentation)
        self.assertEqual(len(task1_lines), 1)
        self.assertFalse(task1_lines[0].startswith('  '))

    def test_dependent_task_indented(self):
        """In reverse mode, dependent tasks should be indented."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Blocking", "urgency": 1.0},
            {"uuid": "b", "id": 2, "description": "Blocked", "urgency": 5.0, "depends": "a"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        lines = output.split('\n')
        task2_lines = [l for l in lines if '[2]' in l]

        # Task 2 should be indented
        self.assertEqual(len(task2_lines), 1)
        self.assertTrue(task2_lines[0].startswith('  '))

    def test_reverse_chain_indentation(self):
        """Test A->B->C in reverse shows A at 0, B at 1, C at 2."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "A", "urgency": 1.0},
            {"uuid": "b", "id": 2, "description": "B", "urgency": 2.0, "depends": "a"},
            {"uuid": "c", "id": 3, "description": "C", "urgency": 3.0, "depends": "b"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        lines = output.split('\n')
        task1_line = next(l for l in lines if '[1]' in l)
        task2_line = next(l for l in lines if '[2]' in l)
        task3_line = next(l for l in lines if '[3]' in l)

        # A at level 0, B at level 1, C at level 2
        self.assertEqual(len(task1_line) - len(task1_line.lstrip()), 0)
        self.assertEqual(len(task2_line) - len(task2_line.lstrip()), 2)
        self.assertEqual(len(task3_line) - len(task3_line.lstrip()), 4)

    def test_completed_dependency_at_top(self):
        """Tasks depending only on completed tasks should appear at top."""
        tasks = [
            {"uuid": "b", "id": 2, "description": "Blocked by completed", "urgency": 5.0, "depends": "a"}
            # Note: task 'a' is not in the list (completed/deleted)
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_output(task_map, depends_on, depended_by)

        # Task 2 should appear at top level (no indentation)
        lines = [l for l in output.split('\n') if '[2]' in l]
        self.assertEqual(len(lines), 1)
        self.assertFalse(lines[0].startswith('  '))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and fragile scenarios."""

    def capture_normal_output(self, task_map, depends_on, depended_by):
        """Helper to capture normal mode output."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            print_tree_normal(task_map, depends_on, depended_by)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return output

    def capture_reverse_output(self, task_map, depends_on, depended_by):
        """Helper to capture reverse mode output."""
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            print_tree_reverse(task_map, depends_on, depended_by)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        return output

    def test_circular_dependency(self):
        """Circular dependencies should not cause infinite loops."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "A", "urgency": 5.0, "depends": "b"},
            {"uuid": "b", "id": 2, "description": "B", "urgency": 3.0, "depends": "a"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        # Should not hang
        output = self.capture_normal_output(task_map, depends_on, depended_by)
        # Both tasks should appear exactly once
        self.assertEqual(output.count("[1]"), 1)
        self.assertEqual(output.count("[2]"), 1)

    def test_self_dependency(self):
        """Task depending on itself should not cause issues."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "A", "urgency": 5.0, "depends": "a"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        output = self.capture_normal_output(task_map, depends_on, depended_by)
        # Task should appear exactly once
        self.assertEqual(output.count("[1]"), 1)

    def test_orphaned_dependency(self):
        """Task depending on non-existent task should still appear."""
        tasks = [
            {"uuid": "b", "id": 2, "description": "B", "urgency": 3.0, "depends": "nonexistent"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        output = self.capture_normal_output(task_map, depends_on, depended_by)
        self.assertEqual(output.count("[2]"), 1)

    def test_multiple_dependents_on_one_task(self):
        """Multiple tasks depending on same task should all appear."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "A", "urgency": 1.0},
            {"uuid": "b", "id": 2, "description": "B", "urgency": 2.0, "depends": "a"},
            {"uuid": "c", "id": 3, "description": "C", "urgency": 3.0, "depends": "a"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        output_normal = self.capture_normal_output(task_map, depends_on, depended_by)
        output_reverse = self.capture_reverse_output(task_map, depends_on, depended_by)

        # All tasks appear in both modes
        for task_id in ['1', '2', '3']:
            self.assertEqual(output_normal.count(f'[{task_id}]'), 1)
            self.assertEqual(output_reverse.count(f'[{task_id}]'), 1)

    def test_complex_dependency_web(self):
        """Complex web of dependencies should show all tasks."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "A", "urgency": 1.0},
            {"uuid": "b", "id": 2, "description": "B", "urgency": 2.0, "depends": "a"},
            {"uuid": "c", "id": 3, "description": "C", "urgency": 3.0, "depends": "a"},
            {"uuid": "d", "id": 4, "description": "D", "urgency": 4.0, "depends": "b,c"},
            {"uuid": "e", "id": 5, "description": "E", "urgency": 5.0},
            {"uuid": "f", "id": 6, "description": "F", "urgency": 6.0, "depends": "e"}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)

        output_normal = self.capture_normal_output(task_map, depends_on, depended_by)
        output_reverse = self.capture_reverse_output(task_map, depends_on, depended_by)

        # All 6 tasks should appear exactly once in both modes
        for task_id in range(1, 7):
            self.assertEqual(output_normal.count(f'[{task_id}]'), 1,
                           f"Normal mode: Task {task_id} should appear exactly once")
            self.assertEqual(output_reverse.count(f'[{task_id}]'), 1,
                           f"Reverse mode: Task {task_id} should appear exactly once")

    def test_urgency_sorting(self):
        """Tasks at same level should be sorted by urgency."""
        tasks = [
            {"uuid": "a", "id": 1, "description": "Low urgency", "urgency": 1.0},
            {"uuid": "b", "id": 2, "description": "High urgency", "urgency": 10.0},
            {"uuid": "c", "id": 3, "description": "Medium urgency", "urgency": 5.0}
        ]
        task_map, depends_on, depended_by = build_dependency_graph(tasks)
        output = self.capture_normal_output(task_map, depends_on, depended_by)

        lines = [l for l in output.split('\n') if l.strip() and '[' in l]
        # Should be ordered: 2, 3, 1 (by urgency descending)
        self.assertIn('[2]', lines[0])
        self.assertIn('[3]', lines[1])
        self.assertIn('[1]', lines[2])


class TestFormatTask(unittest.TestCase):
    """Test task formatting."""

    def test_format_basic_task(self):
        """Format basic task with all fields."""
        task = {"id": 1, "description": "Test task", "urgency": 5.5}
        formatted = format_task(task)
        self.assertEqual(formatted, "[1] Test task (urgency: 5.5)")

    def test_format_missing_id(self):
        """Format task with missing ID."""
        task = {"description": "Test task", "urgency": 5.5}
        formatted = format_task(task)
        self.assertEqual(formatted, "[?] Test task (urgency: 5.5)")

    def test_format_missing_urgency(self):
        """Format task with missing urgency."""
        task = {"id": 1, "description": "Test task"}
        formatted = format_task(task)
        self.assertEqual(formatted, "[1] Test task (urgency: 0.0)")

    def test_format_missing_description(self):
        """Format task with missing description."""
        task = {"id": 1, "urgency": 5.5}
        formatted = format_task(task)
        self.assertEqual(formatted, "[1]  (urgency: 5.5)")


if __name__ == '__main__':
    unittest.main()
