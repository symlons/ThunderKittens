#!/usr/bin/env python3
"""Tests for dashboard generation — no GPU required."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate_dashboard

def test_generate_html():
    html = generate_dashboard.generate_html()
    assert isinstance(html, str)
    assert len(html) > 1000
    assert "<!DOCTYPE html>" in html
    assert "</html>" in html

def test_html_contains_gpu():
    html = generate_dashboard.generate_html()
    assert "gpu" in html.lower()

def test_html_contains_layers():
    html = generate_dashboard.generate_html()
    assert "kernel" in html.lower()

def test_html_contains_correctness():
    html = generate_dashboard.generate_html()
    assert "correctness" in html

def test_html_contains_table():
    html = generate_dashboard.generate_html()
    assert "<table>" in html
    assert "<tr>" in html

def test_graph_interaction_ui_present():
    html = generate_dashboard.generate_html()
    assert "initGraphDetails" in html
    assert "graph-detail" in html
    assert "detail-close" in html
    assert "related-in" in html
    assert "related-out" in html
    assert "innerHTML" not in html

def test_graph_uses_latex_and_impact():
    html = generate_dashboard.generate_html()
    assert "MathJax" in html
    assert "block impact" in html
    assert "Amdahl" in html
    assert "Little" in html

def test_correctness_ui_has_summary_and_margin():
    html = generate_dashboard.generate_html()
    assert "correctness-grid" in html
    assert "worst max diff" in html
    assert "tightest margin" in html
    assert "<th>margin</th>" in html


if __name__ == "__main__":
    test_funcs = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")
