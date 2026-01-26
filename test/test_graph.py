#!/usr/bin/env python3
"""
Comprehensive test suite for twh.graph module.

Tests cover:
- Task parsing from JSON
- Dependency graph building
- Chain collapsing for cleaner visualization
- Mermaid generation
- CSV export
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from twh.graph import (
    read_tasks_from_json,
    parse_dependencies,
    build_dependency_graph,
    collapse_chains,
    generate_mermaid,
    write_csv_export,
    create_task_graph,
)


class TestParseDependencies(unittest.TestCase):
    """Test dependency parsing."""

    def test_none_returns_empty(self):
        """None should return empty list."""
        self.assertEqual(parse_dependencies(None), [])

    def test_empty_string_returns_empty(self):
        """Empty string should return empty list."""
        self.assertEqual(parse_dependencies(''), [])

    def test_single_dependency(self):
        """Single UUID should be parsed correctly."""
        result = parse_dependencies('abc-123')
        self.assertEqual(result, ['abc-123'])

    def test_multiple_dependencies(self):
        """Multiple comma-separated UUIDs should be parsed."""
        result = parse_dependencies('abc-123,def-456,ghi-789')
        self.assertEqual(result, ['abc-123', 'def-456', 'ghi-789'])

    def test_whitespace_handling(self):
        """Whitespace should be stripped."""
        result = parse_dependencies(' abc-123 , def-456 , ghi-789 ')
        self.assertEqual(result, ['abc-123', 'def-456', 'ghi-789'])

    def test_list_input(self):
        """List input should be returned as-is."""
        input_list = ['abc-123', 'def-456']
        result = parse_dependencies(input_list)
        self.assertEqual(result, input_list)


class TestBuildDependencyGraph(unittest.TestCase):
    """Test dependency graph building."""

    def test_empty_tasks(self):
        """Empty task list should produce empty graphs."""
        uid_map, succ, pred = build_dependency_graph([])
        self.assertEqual(uid_map, {})
        self.assertEqual(dict(succ), {})
        self.assertEqual(dict(pred), {})

    def test_single_task_no_deps(self):
        """Single task with no dependencies."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)

        self.assertEqual(len(uid_map), 1)
        self.assertIn('a', uid_map)
        self.assertEqual(len(succ.get('a', set())), 0)
        self.assertEqual(len(pred.get('a', set())), 0)

    def test_simple_chain(self):
        """Test A depends on B."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A', 'depends': 'b'},
            {'uuid': 'b', 'description': 'Task B'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)

        self.assertEqual(succ['a'], {'b'})
        self.assertEqual(pred['b'], {'a'})

    def test_multiple_dependencies(self):
        """Test A depends on B and C."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A', 'depends': 'b,c'},
            {'uuid': 'b', 'description': 'Task B'},
            {'uuid': 'c', 'description': 'Task C'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)

        self.assertEqual(succ['a'], {'b', 'c'})
        self.assertEqual(pred['b'], {'a'})
        self.assertEqual(pred['c'], {'a'})

    def test_diamond_dependency(self):
        """Test diamond: D depends on B,C; B,C depend on A."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A'},
            {'uuid': 'b', 'description': 'Task B', 'depends': 'a'},
            {'uuid': 'c', 'description': 'Task C', 'depends': 'a'},
            {'uuid': 'd', 'description': 'Task D', 'depends': 'b,c'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)

        self.assertEqual(succ['d'], {'b', 'c'})
        self.assertEqual(succ['b'], {'a'})
        self.assertEqual(succ['c'], {'a'})
        self.assertEqual(pred['a'], {'b', 'c'})
        self.assertEqual(pred['b'], {'d'})
        self.assertEqual(pred['c'], {'d'})

    def test_missing_dependency_ignored(self):
        """Dependencies on non-existent tasks should be ignored."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A', 'depends': 'nonexistent'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)

        # 'a' should still be in the map
        self.assertIn('a', uid_map)
        # But the dependency shouldn't be recorded
        self.assertEqual(len(succ.get('a', set())), 0)

    def test_task_without_uuid_ignored(self):
        """Tasks without UUID should be ignored."""
        tasks = [
            {'description': 'No UUID'},
            {'uuid': 'a', 'description': 'Task A'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)

        self.assertEqual(len(uid_map), 1)
        self.assertIn('a', uid_map)

    def test_reverse_parameter(self):
        """Reverse graph should flip edge direction."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A', 'depends': 'b'},
            {'uuid': 'b', 'description': 'Task B'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks, reverse=True)

        self.assertEqual(pred['a'], {'b'})
        self.assertEqual(succ['b'], {'a'})


class TestCollapseChains(unittest.TestCase):
    """Test chain collapsing for cleaner visualization."""

    def test_no_chains(self):
        """Independent tasks produce no chains."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A'},
            {'uuid': 'b', 'description': 'Task B'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)
        chains = collapse_chains(uid_map, succ, pred)

        self.assertEqual(len(chains), 0)

    def test_simple_chain(self):
        """B -> A should produce one chain (dependent to dependency)."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A'},
            {'uuid': 'b', 'description': 'Task B', 'depends': 'a'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)
        chains = collapse_chains(uid_map, succ, pred)

        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0], ['b', 'a'])

    def test_three_task_chain(self):
        """C -> B -> A should produce one chain (dependent to dependency)."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A'},
            {'uuid': 'b', 'description': 'Task B', 'depends': 'a'},
            {'uuid': 'c', 'description': 'Task C', 'depends': 'b'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)
        chains = collapse_chains(uid_map, succ, pred)

        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0], ['c', 'b', 'a'])

    def test_branching_not_collapsed(self):
        """Branching should not be collapsed into single chain."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A'},
            {'uuid': 'b', 'description': 'Task B', 'depends': 'a'},
            {'uuid': 'c', 'description': 'Task C', 'depends': 'a'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)
        chains = collapse_chains(uid_map, succ, pred)

        # Should have two separate chains from A
        self.assertEqual(len(chains), 2)
        self.assertIn(['b', 'a'], chains)
        self.assertIn(['c', 'a'], chains)

    def test_branch_with_single_predecessor_keeps_edges(self):
        """Branching node with one predecessor should still emit edges."""
        tasks = [
            {'uuid': 'a', 'description': 'Parent', 'depends': 'b,c'},
            {'uuid': 'b', 'description': 'Child B'},
            {'uuid': 'c', 'description': 'Child C'},
            {'uuid': 'd', 'description': 'Upstream', 'depends': 'a'}
        ]
        uid_map, succ, pred = build_dependency_graph(tasks)
        chains = collapse_chains(uid_map, succ, pred)

        self.assertIn(['d', 'a'], chains)
        self.assertIn(['a', 'b'], chains)
        self.assertIn(['a', 'c'], chains)


class TestGenerateMermaid(unittest.TestCase):
    """Test Mermaid generation."""

    def test_empty_chains(self):
        """Empty chains should produce basic flowchart."""
        uid_map = {}
        chains = []
        mermaid = generate_mermaid(uid_map, chains)

        self.assertIn('flowchart LR', mermaid)

    def test_standalone_tasks(self):
        """Standalone tasks should appear as nodes."""
        uid_map = {
            'a': {'description': 'Task A'},
            'b': {'description': 'Task B'}
        }
        chains = []
        mermaid = generate_mermaid(uid_map, chains)

        self.assertIn('Task A', mermaid)
        self.assertIn('Task B', mermaid)
        self.assertIn('t_a', mermaid)
        self.assertIn('t_b', mermaid)

    def test_ids_in_labels(self):
        """Task IDs should appear in node labels."""
        uid_map = {
            'a': {'description': 'Task A', 'id': 12},
            'b': {'description': 'Task B', 'id': 7}
        }
        chains = [['a', 'b']]
        mermaid = generate_mermaid(uid_map, chains)

        self.assertIn('ID: 12', mermaid)
        self.assertIn('ID: 7', mermaid)

    def test_due_date_in_labels(self):
        """Due dates should appear in node labels."""
        uid_map = {
            'a': {'description': 'Task A', 'id': 1, 'due': '20260201T000000Z'}
        }
        chains = [['a']]
        mermaid = generate_mermaid(uid_map, chains)

        self.assertIn('Due: 2026-02-01', mermaid)

    def test_started_and_blocked_classes(self):
        """Started and blocked tasks should be styled via classes."""
        uid_map = {
            'a': {'description': 'Blocking', 'id': 1},
            'b': {'description': 'Blocked', 'id': 2, 'depends': 'a'},
            'c': {'description': 'Started', 'id': 3, 'start': '20260101T120000Z'}
        }
        chains = [['b', 'a'], ['c']]
        mermaid = generate_mermaid(uid_map, chains)

        self.assertIn('twh-status-blocked', mermaid)
        self.assertIn('twh-status-started', mermaid)

    def test_id_badge_flush_corner(self):
        """Node layout should include urgency and status rectangles."""
        uid_map = {'a': {'description': 'Task A', 'id': 1}}
        mermaid = generate_mermaid(uid_map, [['a']])

        self.assertIn('.twh-urgency{', mermaid)
        self.assertIn('.twh-status{', mermaid)

    def test_priority_gradient_colors(self):
        """Priority colors should grade across the urgency spectrum."""
        uid_map = {
            'a': {'description': 'Low', 'id': 1, 'urgency': 1.0},
            'b': {'description': 'High', 'id': 2, 'urgency': 10.0},
        }
        mermaid = generate_mermaid(uid_map, [['a'], ['b']])

        self.assertIn('background-color: rgba(', mermaid)

    def test_ranked_urgency_colors(self):
        """Distinct urgencies should map to distinct colors."""
        uid_map = {
            'a': {'description': 'U1', 'id': 1, 'urgency': 1.0},
            'b': {'description': 'U2', 'id': 2, 'urgency': 2.0},
            'c': {'description': 'U3', 'id': 3, 'urgency': 3.0},
            'd': {'description': 'U4', 'id': 4, 'urgency': 4.0},
        }
        mermaid = generate_mermaid(uid_map, [['a'], ['b'], ['c'], ['d']])
        colors = []
        for line in mermaid.splitlines():
            if 'background-color: rgba(' in line:
                colors.append(line.split('background-color: ')[1].split(';')[0])

        self.assertGreaterEqual(len(set(colors)), 4)

    def test_same_displayed_urgency_same_color(self):
        """Urgency values that round to the same display value share a color."""
        uid_map = {
            'a': {'description': 'U9.03288', 'id': 1, 'urgency': 9.03288},
            'b': {'description': 'U9.0274', 'id': 2, 'urgency': 9.0274},
        }
        mermaid = generate_mermaid(uid_map, [['a'], ['b']])
        colors = []
        for line in mermaid.splitlines():
            if 'background-color: rgba(' in line:
                colors.append(line.split('background-color: ')[1].split(';')[0])

        self.assertEqual(len(set(colors)), 1)

    def test_priority_gradient_with_string_urgency(self):
        """String urgency values should still map to different colors."""
        uid_map = {
            'a': {'description': 'Low', 'id': 1, 'urgency': '1.0'},
            'b': {'description': 'High', 'id': 2, 'urgency': '10.0'},
        }
        mermaid = generate_mermaid(uid_map, [['a'], ['b']])

        self.assertIn('Urg: 1', mermaid)
        self.assertIn('Urg: 10', mermaid)

    def test_urgency_badge_in_label(self):
        """Urgency should appear in the urgency bar."""
        uid_map = {'a': {'description': 'Task A', 'id': 1, 'urgency': 9.01}}
        mermaid = generate_mermaid(uid_map, [['a']])

        self.assertIn('Urg: 9.01', mermaid)

    def test_single_chain(self):
        """Single chain should produce one arrow."""
        uid_map = {
            'aaaa-1111': {'description': 'Task A'},
            'bbbb-2222': {'description': 'Task B'}
        }
        chains = [['aaaa-1111', 'bbbb-2222']]
        mermaid = generate_mermaid(uid_map, chains)

        self.assertIn('flowchart LR', mermaid)
        self.assertIn('Task A', mermaid)
        self.assertIn('Task B', mermaid)
        self.assertIn('-->', mermaid)
        self.assertIn('t_aaaa1111', mermaid)
        self.assertIn('t_bbbb2222', mermaid)

    def test_description_preserved(self):
        """Long descriptions should be preserved in full."""
        long_desc = 'A' * 100
        uid_map = {
            'aaaa-1111': {'description': long_desc},
            'bbbb-2222': {'description': 'Task B'}
        }
        chains = [['aaaa-1111', 'bbbb-2222']]
        mermaid = generate_mermaid(uid_map, chains)

        # Should contain full description
        self.assertIn(long_desc, mermaid)

    def test_quote_escaping(self):
        """Quotes in descriptions should be escaped."""
        uid_map = {
            'aaaa-1111': {'description': 'Task "with quotes"'},
            'bbbb-2222': {'description': 'Task B'}
        }
        chains = [['aaaa-1111', 'bbbb-2222']]
        mermaid = generate_mermaid(uid_map, chains)

        # Quotes should be escaped
        self.assertIn('Task &quot;with quotes&quot;', mermaid)
        self.assertNotIn('Task "with quotes"', mermaid)

    def test_newline_handling(self):
        """Newlines in descriptions should be replaced with spaces."""
        uid_map = {
            'aaaa-1111': {'description': 'Task\nwith\nnewlines'},
            'bbbb-2222': {'description': 'Task B'}
        }
        chains = [['aaaa-1111', 'bbbb-2222']]
        mermaid = generate_mermaid(uid_map, chains)

        # Newlines should be replaced with spaces
        self.assertIn('Task with newlines', mermaid)


class TestWriteCSVExport(unittest.TestCase):
    """Test CSV export functionality."""

    def test_csv_export_basic(self):
        """Test basic CSV export."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.csv'

            uid_map = {
                'aaaa-1111': {
                    'description': 'Task A',
                    'id': 1,
                    'entry': '2025-01-01',
                    'tags': ['work', 'urgent']
                },
                'bbbb-2222': {
                    'description': 'Task B',
                    'id': 2,
                    'entry': '2025-01-02',
                    'tags': []
                }
            }
            pred = {
                'bbbb-2222': {'aaaa-1111'}
            }

            write_csv_export(uid_map, pred, output_path)

            # Verify file exists
            self.assertTrue(output_path.exists())

            # Read and verify content
            content = output_path.read_text(encoding='utf-8')
            self.assertIn('Title,Body,Tags,UUID,DependsOn', content)
            self.assertIn('Task A', content)
            self.assertIn('Task B', content)
            self.assertIn('work;urgent', content)
            self.assertIn('aaaa-1111', content)
            self.assertIn('bbbb-2222', content)

    def test_csv_export_with_dependencies(self):
        """Test CSV export includes dependencies."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test.csv'

            uid_map = {
                'aaaa-1111': {'description': 'Task A', 'id': 1, 'tags': []},
                'bbbb-2222': {'description': 'Task B', 'id': 2, 'tags': []}
            }
            pred = {
                'bbbb-2222': {'aaaa-1111'}
            }

            write_csv_export(uid_map, pred, output_path)

            content = output_path.read_text(encoding='utf-8')
            # Task B should list dependency on Task A
            lines = content.strip().split('\n')
            # Find the line for Task B
            task_b_line = [l for l in lines if 'Task B' in l][0]
            self.assertIn('aaaa-1111', task_b_line)


class TestCreateTaskGraph(unittest.TestCase):
    """Test complete graph creation workflow."""

    def test_create_graph_returns_mermaid(self):
        """Test that create_task_graph returns Mermaid content."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A'},
            {'uuid': 'b', 'description': 'Task B', 'depends': 'a'}
        ]

        mermaid = create_task_graph(tasks)

        self.assertIsInstance(mermaid, str)
        self.assertIn('flowchart LR', mermaid)
        self.assertIn('Task A', mermaid)
        self.assertIn('Task B', mermaid)

    def test_create_graph_writes_files(self):
        """Test that create_task_graph writes output files."""
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mmd_path = tmpdir_path / 'test.mmd'
            csv_path = tmpdir_path / 'test.csv'

            tasks = [
                {'uuid': 'a', 'description': 'Task A'},
                {'uuid': 'b', 'description': 'Task B', 'depends': 'a'}
            ]

            mermaid = create_task_graph(
                tasks,
                output_mmd=mmd_path,
                output_csv=csv_path
            )

            # Verify files were created
            self.assertTrue(mmd_path.exists())
            self.assertTrue(csv_path.exists())

            # Verify content
            mmd_content = mmd_path.read_text(encoding='utf-8')
            self.assertEqual(mmd_content, mermaid)

            csv_content = csv_path.read_text(encoding='utf-8')
            self.assertIn('Task A', csv_content)
            self.assertIn('Task B', csv_content)

    def test_create_graph_no_files(self):
        """Test that create_task_graph works without writing files."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A'},
            {'uuid': 'b', 'description': 'Task B', 'depends': 'a'}
        ]

        mermaid = create_task_graph(tasks)

        # Should return content even without file output
        self.assertIn('flowchart LR', mermaid)


class TestReadTasksFromJSON(unittest.TestCase):
    """Test JSON task reading."""

    def test_parse_list_of_tasks(self):
        """Test parsing a list of tasks."""
        json_str = json.dumps([
            {'uuid': 'a', 'description': 'Task A'},
            {'uuid': 'b', 'description': 'Task B'}
        ])

        tasks = read_tasks_from_json(json_str)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]['uuid'], 'a')
        self.assertEqual(tasks[1]['uuid'], 'b')

    def test_parse_single_task(self):
        """Test parsing a single task (not in array)."""
        json_str = json.dumps({'uuid': 'a', 'description': 'Task A'})

        tasks = read_tasks_from_json(json_str)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]['uuid'], 'a')

    def test_parse_empty_list(self):
        """Test parsing an empty list."""
        json_str = json.dumps([])

        tasks = read_tasks_from_json(json_str)

        self.assertEqual(len(tasks), 0)


if __name__ == '__main__':
    unittest.main()
