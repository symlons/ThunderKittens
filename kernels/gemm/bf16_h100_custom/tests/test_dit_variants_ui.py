import unittest

from dit_variants import (
    TraceEvent,
    format_bytes,
    format_cuda_speedup,
    packed_call_timeline_cells,
    time_marker_timeline_cells,
    trace_event_footprint_lines,
    trace_event_glyph,
    trace_event_bytes,
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
            "memcpy": "M",
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

    def test_speedup_and_memory_formatters(self):
        self.assertEqual(format_cuda_speedup(63.0, 42.0), "speedup=+50.0%")
        self.assertEqual(format_cuda_speedup(0.0, 42.0), "speedup=n/a")
        self.assertEqual(format_bytes(4), "4B")
        self.assertEqual(format_bytes(2048), "2.00KiB")

    def test_trace_event_footprint_lines_include_profiler_args(self):
        event = TraceEvent(
            0.0,
            1.0,
            "linear_gemm",
            "gemm",
            {
                "registers per thread": 168,
                "shared memory": 164308,
                "grid": [4, 24, 1],
                "block": [384, 1, 1],
                "bytes": 4096,
                "memory bandwidth (GB/s)": 1.5,
                "device": 0,
                "stream": 7,
            },
        )

        lines = trace_event_footprint_lines(event)

        self.assertEqual(trace_event_bytes(event), 4096)
        self.assertTrue(any("registers per thread=168" in line for line in lines))
        self.assertTrue(any("shared memory=164308" in line for line in lines))
        self.assertTrue(any("bytes=4096" in line for line in lines))
        self.assertTrue(any("stream=7" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
