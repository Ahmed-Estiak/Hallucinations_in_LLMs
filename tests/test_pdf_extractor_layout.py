"""Tests for PDF layout-aware text block ordering."""

from __future__ import annotations

import unittest

from src.rag.pdf_extractor import PdfTextBlock, order_page_blocks


class PdfLayoutOrderingTests(unittest.TestCase):
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

    def test_single_column_blocks_keep_top_to_bottom_order(self) -> None:
        blocks = [
            PdfTextBlock(40, 140, 560, 160, "second"),
            PdfTextBlock(40, 100, 560, 120, "first"),
        ]

        ordered = order_page_blocks(blocks, page_width=600)

        self.assertEqual([block.text for block in ordered], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
