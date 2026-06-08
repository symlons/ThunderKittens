import unittest

from dit_variants import (
    TraceEvent,
    packed_call_timeline_cells,
    time_marker_timeline_cells,
    trace_event_glyph,
)


class DiTTraceTimelineTests(unittest.TestCase):
    def test_trace_event_glyph_maps_known_subcomponents(self):
        cases = {
            "fused_adaln_modulation": "F",
            "unfused_attention": "A",
            "linear_gemm": "G",
            "layer_norm": "L",
            "adaln_modulation": "N",
            "gated_residual": "R",
            "mlp_gelu": "U",
            "patch_embed_conv": "P",
            "memset": "0",
            "other_cuda": ".",
        }
        for category, glyph in cases.items():
            with self.subTest(category=category):
                self.assertEqual(trace_event_glyph(category), glyph)

    def test_call_timeline_packs_visible_calls_without_inserted_spaces(self):
        events = (
            TraceEvent(0.0, 5.0, "linear_gemm", "gemm0"),
            TraceEvent(7.0, 2.0, "fused_adaln", "fused0"),
            TraceEvent(20.0, 1.0, "unfused_attention", "attn0"),
            TraceEvent(25.0, 1.0, "memset", "memset0"),
        )

        cells = packed_call_timeline_cells(events, lane_width=12, call_offset=0, call_window_size=12)

        self.assertEqual("".join(cells[:4]), "GFA0")
        self.assertEqual("".join(cells[4:]), " " * 8)

    def test_call_timeline_respects_offset_and_window(self):
        events = tuple(
            TraceEvent(float(idx), 1.0, category, category)
            for idx, category in enumerate(("linear_gemm", "fused_a", "unfused_attention", "mlp_gelu"))
        )

        cells = packed_call_timeline_cells(events, lane_width=8, call_offset=1, call_window_size=2)

        self.assertEqual("".join(cells[:2]), "FA")
        self.assertEqual("".join(cells[2:]), " " * 6)

    def test_time_timeline_marks_each_event_once(self):
        events = (
            TraceEvent(0.0, 80.0, "linear_gemm", "long_gemm"),
            TraceEvent(100.0, 10.0, "unfused_attention", "attn"),
        )

        cells = time_marker_timeline_cells(events, lane_width=20, offset_us=0.0, window_us=120.0)
        rendered = "".join(cells)

        self.assertEqual(rendered.count("G"), 1)
        self.assertEqual(rendered.count("A"), 1)
        self.assertGreater(rendered.count(" "), 10)


if __name__ == "__main__":
    unittest.main()
