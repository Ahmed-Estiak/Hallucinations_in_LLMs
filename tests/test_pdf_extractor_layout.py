"""Tests for PDF layout-aware text block ordering."""

from __future__ import annotations

import unittest

from src.rag.pdf_extractor import (
    PdfTextBlock,
    classify_page_layout,
    detect_table_regions,
    format_page_with_table_regions,
    join_line_spans,
    order_page_blocks,
)


class PdfLayoutOrderingTests(unittest.TestCase):
    def test_join_line_spans_preserves_word_spaces_from_bbox_gaps(self) -> None:
        spans = [
            {"text": "in", "bbox": (0.0, 0.0, 8.0, 10.0)},
            {"text": "late", "bbox": (12.0, 0.0, 30.0, 10.0)},
            {"text": "2022", "bbox": (34.0, 0.0, 55.0, 10.0)},
        ]

        self.assertEqual(join_line_spans(spans), "in late 2022")

    def test_two_column_blocks_are_ordered_left_then_right(self) -> None:
        blocks = [
            PdfTextBlock(320, 100, 560, 120, "right 1"),
            PdfTextBlock(40, 100, 280, 120, "left 1"),
            PdfTextBlock(320, 140, 560, 160, "right 2"),
            PdfTextBlock(40, 140, 280, 160, "left 2"),
        ]

        ordered = order_page_blocks(blocks, page_width=600)

        self.assertEqual(
            [block.text for block in ordered],
            ["left 1", "left 2", "right 1", "right 2"],
        )

    def test_full_width_heading_stays_before_two_column_body(self) -> None:
        blocks = [
            PdfTextBlock(40, 40, 560, 70, "Heading"),
            PdfTextBlock(320, 100, 560, 120, "right 1"),
            PdfTextBlock(40, 100, 280, 120, "left 1"),
            PdfTextBlock(320, 140, 560, 160, "right 2"),
            PdfTextBlock(40, 140, 280, 160, "left 2"),
        ]

        ordered = order_page_blocks(blocks, page_width=600)

        self.assertEqual(
            [block.text for block in ordered],
            ["Heading", "left 1", "left 2", "right 1", "right 2"],
        )

    def test_three_column_blocks_are_ordered_left_middle_right(self) -> None:
        blocks = [
            PdfTextBlock(420, 100, 560, 120, "right 1"),
            PdfTextBlock(220, 100, 360, 120, "middle 1"),
            PdfTextBlock(40, 100, 180, 120, "left 1"),
            PdfTextBlock(420, 140, 560, 160, "right 2"),
            PdfTextBlock(220, 140, 360, 160, "middle 2"),
            PdfTextBlock(40, 140, 180, 160, "left 2"),
        ]

        ordered = order_page_blocks(blocks, page_width=600)

        self.assertEqual(
            [block.text for block in ordered],
            ["left 1", "left 2", "middle 1", "middle 2", "right 1", "right 2"],
        )

    def test_more_than_three_columns_falls_back_to_top_to_bottom_order(self) -> None:
        blocks = [
            PdfTextBlock(40, 100, 90, 120, "col1"),
            PdfTextBlock(170, 101, 220, 121, "col2"),
            PdfTextBlock(300, 102, 350, 122, "col3"),
            PdfTextBlock(430, 103, 480, 123, "col4"),
            PdfTextBlock(40, 140, 90, 160, "col1 row2"),
            PdfTextBlock(170, 141, 220, 161, "col2 row2"),
            PdfTextBlock(300, 142, 350, 162, "col3 row2"),
            PdfTextBlock(430, 143, 480, 163, "col4 row2"),
        ]

        ordered = order_page_blocks(blocks, page_width=600)

        self.assertEqual(
            [block.text for block in ordered],
            ["col1", "col2", "col3", "col4", "col1 row2", "col2 row2", "col3 row2", "col4 row2"],
        )

    def test_cover_like_short_blocks_use_plain_text_fallback(self) -> None:
        blocks = [
            PdfTextBlock(40, 100, 90, 120, "Sun"),
            PdfTextBlock(40, 140, 90, 160, "Earth"),
            PdfTextBlock(220, 100, 280, 120, "Mars"),
            PdfTextBlock(220, 140, 280, 160, "Jupiter"),
            PdfTextBlock(420, 100, 490, 120, "Saturn"),
            PdfTextBlock(420, 140, 490, 160, "Uranus"),
            PdfTextBlock(40, 200, 120, 220, "Neptune"),
            PdfTextBlock(220, 200, 300, 220, "Comets"),
            PdfTextBlock(420, 200, 500, 220, "Moons"),
        ]

        layout = classify_page_layout(blocks, page_width=600)

        self.assertEqual(layout["layout_class"], "poster_or_cover")
        self.assertEqual(layout["chosen_mode"], "plain_text")

    def test_table_like_blocks_use_row_order_lines(self) -> None:
        blocks = [
            PdfTextBlock(40, 100, 80, 120, "Saturn"),
            PdfTextBlock(160, 100, 180, 120, "82"),
            PdfTextBlock(260, 100, 300, 120, "2019"),
            PdfTextBlock(40, 140, 80, 160, "Jupiter"),
            PdfTextBlock(160, 140, 180, 160, "95"),
            PdfTextBlock(260, 140, 300, 160, "2023"),
            PdfTextBlock(40, 180, 80, 200, "Uranus"),
            PdfTextBlock(160, 180, 180, 200, "27"),
            PdfTextBlock(260, 180, 300, 200, "2021"),
        ]

        layout = classify_page_layout(blocks, page_width=600)

        self.assertEqual(layout["layout_class"], "table_like")
        self.assertEqual(layout["chosen_mode"], "row_order_lines")

    def test_table_region_inside_page_is_formatted_row_wise(self) -> None:
        blocks = [
            PdfTextBlock(40, 40, 560, 60, "Intro paragraph before the table."),
            PdfTextBlock(40, 100, 90, 120, "Saturn"),
            PdfTextBlock(160, 100, 180, 120, "82"),
            PdfTextBlock(260, 100, 300, 120, "2019"),
            PdfTextBlock(40, 140, 90, 160, "Jupiter"),
            PdfTextBlock(160, 140, 180, 160, "95"),
            PdfTextBlock(260, 140, 300, 160, "2023"),
            PdfTextBlock(40, 220, 560, 240, "Conclusion paragraph after the table."),
        ]
        lines = blocks

        bands = detect_table_regions(lines)
        text = format_page_with_table_regions(blocks, lines, 600, bands)

        self.assertEqual(len(bands), 1)
        self.assertIn("Intro paragraph before the table.", text)
        self.assertIn("Saturn | 82 | 2019", text)
        self.assertIn("Jupiter | 95 | 2023", text)
        self.assertIn("Conclusion paragraph after the table.", text)

    def test_single_column_blocks_keep_top_to_bottom_order(self) -> None:
        blocks = [
            PdfTextBlock(40, 140, 560, 160, "second"),
            PdfTextBlock(40, 100, 560, 120, "first"),
        ]

        ordered = order_page_blocks(blocks, page_width=600)

        self.assertEqual([block.text for block in ordered], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
