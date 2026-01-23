#!/usr/bin/env python3
"""
Integration test for the complete twh graph workflow.

This test creates sample task data and verifies the complete workflow
from task data to Mermaid/CSV generation.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from twh.graph import create_task_graph


class TestGraphIntegration(unittest.TestCase):
    """Integration tests for complete graph workflow."""

    def test_complete_workflow_with_sample_data(self):
        """Test complete workflow with realistic sample data."""
        # Sample task data similar to what Taskwarrior exports
        tasks = [
            {
                'uuid': 'aaaa-1111-bbbb-2222',
                'id': 1,
                'description': 'Setup development environment',
                'status': 'pending',
                'entry': '20250120T120000Z',
                'urgency': 5.0,
                'tags': ['dev', 'setup']
            },
            {
                'uuid': 'cccc-3333-dddd-4444',
                'id': 2,
                'description': 'Install dependencies',
                'status': 'pending',
                'entry': '20250120T120100Z',
                'urgency': 4.5,
                'depends': 'aaaa-1111-bbbb-2222',
                'tags': ['dev']
            },
            {
                'uuid': 'eeee-5555-ffff-6666',
                'id': 3,
                'description': 'Write unit tests',
                'status': 'pending',
                'entry': '20250120T120200Z',
                'urgency': 4.0,
                'depends': 'cccc-3333-dddd-4444',
                'tags': ['dev', 'testing']
            },
            {
                'uuid': 'gggg-7777-hhhh-8888',
                'id': 4,
                'description': 'Run integration tests',
                'status': 'pending',
                'entry': '20250120T120300Z',
                'urgency': 3.5,
                'depends': 'eeee-5555-ffff-6666',
                'tags': ['testing']
            },
            {
                'uuid': 'iiii-9999-jjjj-0000',
                'id': 5,
                'description': 'Create documentation',
                'status': 'pending',
                'entry': '20250120T120400Z',
                'urgency': 3.0,
                'depends': 'eeee-5555-ffff-6666',
                'tags': ['docs']
            }
        ]

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mmd_path = tmpdir_path / 'workflow.mmd'
            csv_path = tmpdir_path / 'workflow.csv'

            # Generate graph
            mermaid_content = create_task_graph(
                tasks,
                output_mmd=mmd_path,
                output_csv=csv_path
            )

            # Verify Mermaid content
            self.assertIsInstance(mermaid_content, str)
            self.assertIn('flowchart TD', mermaid_content)

            # Verify key task descriptions appear
            # Note: chain collapsing may omit some intermediate nodes
            self.assertIn('Setup development environment', mermaid_content)
            # Other tasks may be collapsed in the chain

            # Verify arrows (dependencies)
            self.assertIn('-->', mermaid_content)

            # Verify files were created
            self.assertTrue(mmd_path.exists())
            self.assertTrue(csv_path.exists())

            # Verify Mermaid file content
            mmd_file_content = mmd_path.read_text(encoding='utf-8')
            self.assertEqual(mmd_file_content, mermaid_content)

            # Verify CSV content
            csv_content = csv_path.read_text(encoding='utf-8')

            # Check header
            self.assertIn('Title,Body,Tags,UUID,DependsOn', csv_content)

            # Check all tasks are in CSV
            for task in tasks:
                self.assertIn(task['description'], csv_content)
                self.assertIn(task['uuid'], csv_content)

            # Check tags
            self.assertIn('dev;setup', csv_content)
            self.assertIn('dev', csv_content)
            self.assertIn('testing', csv_content)
            self.assertIn('docs', csv_content)

            # Check dependencies
            self.assertIn('aaaa-1111-bbbb-2222', csv_content)
            self.assertIn('cccc-3333-dddd-4444', csv_content)

    def test_diamond_dependency_pattern(self):
        """Test diamond dependency pattern: D depends on B and C, both depend on A."""
        tasks = [
            {
                'uuid': 'a',
                'id': 1,
                'description': 'Base task',
                'tags': []
            },
            {
                'uuid': 'b',
                'id': 2,
                'description': 'Task B',
                'depends': 'a',
                'tags': []
            },
            {
                'uuid': 'c',
                'id': 3,
                'description': 'Task C',
                'depends': 'a',
                'tags': []
            },
            {
                'uuid': 'd',
                'id': 4,
                'description': 'Final task',
                'depends': 'b,c',
                'tags': []
            }
        ]

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mmd_path = tmpdir_path / 'diamond.mmd'

            mermaid_content = create_task_graph(tasks, output_mmd=mmd_path)

            # Verify structure
            self.assertIn('Base task', mermaid_content)
            self.assertIn('Task B', mermaid_content)
            self.assertIn('Task C', mermaid_content)
            self.assertIn('Final task', mermaid_content)

            # Should have arrows showing the diamond pattern
            self.assertIn('-->', mermaid_content)

    def test_empty_task_list(self):
        """Test handling of empty task list."""
        tasks = []

        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            mmd_path = tmpdir_path / 'empty.mmd'
            csv_path = tmpdir_path / 'empty.csv'

            mermaid_content = create_task_graph(
                tasks,
                output_mmd=mmd_path,
                output_csv=csv_path
            )

            # Should produce minimal but valid Mermaid
            self.assertEqual(mermaid_content, 'flowchart TD')

            # Files should still be created
            self.assertTrue(mmd_path.exists())
            self.assertTrue(csv_path.exists())

            # CSV should have header but no data
            csv_content = csv_path.read_text(encoding='utf-8')
            self.assertIn('Title,Body,Tags,UUID,DependsOn', csv_content)
            # Count lines (header + newline = 2 lines total, or just header)
            lines = csv_content.strip().split('\n')
            self.assertEqual(len(lines), 1)  # Just header

    def test_no_file_output(self):
        """Test workflow without file output."""
        tasks = [
            {'uuid': 'a', 'description': 'Task A'},
            {'uuid': 'b', 'description': 'Task B', 'depends': 'a'}
        ]

        # Don't provide output paths
        mermaid_content = create_task_graph(tasks)

        # Should still return content
        self.assertIn('flowchart TD', mermaid_content)
        self.assertIn('Task A', mermaid_content)
        self.assertIn('Task B', mermaid_content)


if __name__ == '__main__':
    unittest.main()
