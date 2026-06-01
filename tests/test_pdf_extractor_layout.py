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

    def test_single_column_blocks_keep_top_to_bottom_order(self) -> None:
        blocks = [
            PdfTextBlock(40, 140, 560, 160, "second"),
            PdfTextBlock(40, 100, 560, 120, "first"),
        ]

        ordered = order_page_blocks(blocks, page_width=600)

        self.assertEqual([block.text for block in ordered], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
