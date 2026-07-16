from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from tools.check_frontend_lock import validate_frontend_lock


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FrontendLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = json.loads(
            (PROJECT_ROOT / "apps/web-test/package.json").read_text(encoding="utf-8")
        )
        cls.lock = json.loads(
            (PROJECT_ROOT / "apps/web-test/package-lock.json").read_text(
                encoding="utf-8"
            )
        )

    def test_current_frontend_lock_is_exact_and_reviewed(self) -> None:
        errors, direct_count, resolved_count = validate_frontend_lock(
            self.package,
            self.lock,
        )

        self.assertEqual(errors, [])
        self.assertEqual(direct_count, 14)
        self.assertEqual(resolved_count, 147)

    def test_floating_direct_version_is_rejected(self) -> None:
        package = copy.deepcopy(self.package)
        package["dependencies"]["react"] = "^19.2.7"

        errors, _, _ = validate_frontend_lock(package, self.lock)

        self.assertIn(
            "package.json dependency react is not exactly pinned",
            errors,
        )
        self.assertIn(
            "package-lock root dependencies differ from package.json",
            errors,
        )

    def test_missing_integrity_and_unreviewed_license_are_rejected(self) -> None:
        lock = copy.deepcopy(self.lock)
        path = next(path for path in lock["packages"] if path)
        lock["packages"][path]["integrity"] = "sha512-garbage"
        lock["packages"][path]["license"] = "GPL-3.0-only"

        errors, _, _ = validate_frontend_lock(self.package, lock)

        self.assertIn(f"lock entry {path} has no sha512 integrity value", errors)
        self.assertIn(
            f"lock entry {path} has an unreviewed license: GPL-3.0-only",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
