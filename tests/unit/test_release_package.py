from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.check_release_package import (
    ReleasePackageError,
    check_release_package,
    markdown_headings,
    validate_links,
    validate_no_secrets,
)


class ReleasePackageTests(unittest.TestCase):
    def test_checked_release_package_is_complete(self) -> None:
        root = Path(__file__).resolve().parents[2]
        result = check_release_package(root)
        self.assertEqual(result.release_documents, 5)
        self.assertEqual(result.public_paths, 13)
        self.assertEqual(result.error_codes, 51)
        self.assertEqual(result.immutable_images, 3)
        self.assertEqual(result.provider_fingerprints, 2)
        self.assertEqual(result.prior_reports, 4)
        self.assertGreaterEqual(result.markdown_links, 10)

    def test_heading_parser_normalizes_markdown_levels(self) -> None:
        self.assertEqual(
            markdown_headings("# Title\n\n## Start and Stop\ntext\n"),
            {"Title", "Start and Stop"},
        )

    def test_link_validation_rejects_missing_relative_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.md").write_text("[missing](missing.md)\n", encoding="utf-8")
            with self.assertRaises(ReleasePackageError):
                validate_links(root, ("guide.md",))

    def test_secret_patterns_reject_bearer_and_non_placeholder_dsn(self) -> None:
        for value in (
            "Authorization: Bearer placeholder-token",
            "postgresql+asyncpg://user:placeholder-password@postgres/db",
            "not-a-real-api-key-placeholder",
        ):
            with self.subTest(value=value), self.assertRaises(ReleasePackageError):
                validate_no_secrets(value)

    def test_documented_placeholder_values_are_allowed(self) -> None:
        validate_no_secrets(
            "postgresql+asyncpg://user:replace-locally@postgres/db and choose-a-password"
        )


if __name__ == "__main__":
    unittest.main()
