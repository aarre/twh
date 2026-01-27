#!/usr/bin/env python3
"""
Integration tests for Graphviz graph generation.
"""

import pytest

import twh.graph as graph


@pytest.mark.integration
def test_generate_dot_with_sample_tasks():
    """
    Test DOT generation with realistic sample data.

    Returns
    -------
    None
        This test asserts on DOT output content.
    """
    tasks = [
        {
            "uuid": "aaaa-1111-bbbb-2222",
            "id": 1,
            "description": "Setup development environment",
            "status": "pending",
            "entry": "20250120T120000Z",
            "urgency": 5.0,
            "tags": ["dev", "setup"],
        },
        {
            "uuid": "cccc-3333-dddd-4444",
            "id": 2,
            "description": "Install dependencies",
            "status": "pending",
            "entry": "20250120T120100Z",
            "urgency": 4.5,
            "depends": "aaaa-1111-bbbb-2222",
            "tags": ["dev"],
        },
        {
            "uuid": "eeee-5555-ffff-6666",
            "id": 3,
            "description": "Write unit tests",
            "status": "pending",
            "entry": "20250120T120200Z",
            "urgency": 4.0,
            "depends": "cccc-3333-dddd-4444",
            "tags": ["dev", "testing"],
        },
        {
            "uuid": "gggg-7777-hhhh-8888",
            "id": 4,
            "description": "Run integration tests",
            "status": "pending",
            "entry": "20250120T120300Z",
            "urgency": 3.5,
            "depends": "eeee-5555-ffff-6666",
            "tags": ["testing"],
        },
    ]

    edges, by_uuid = graph.build_dependency_edges(tasks, reverse=False)
    dot_source = graph.generate_dot(by_uuid, edges, rankdir="LR")

    assert "digraph twh" in dot_source
    assert "Setup development environment" in dot_source
    assert "Install dependencies" in dot_source
    assert "Write unit tests" in dot_source
    assert "Run integration tests" in dot_source
    assert "Urg:" in dot_source
    assert "ID: 1" in dot_source
    assert "->" in dot_source
