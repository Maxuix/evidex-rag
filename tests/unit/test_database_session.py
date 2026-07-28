from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from rag_kb.db import DatabaseProcess, create_database_resources


class DatabaseSessionTests(unittest.TestCase):
    @patch("rag_kb.db.session.async_sessionmaker")
    @patch("rag_kb.db.session.create_async_engine")
    def test_process_engines_receive_bounded_connection_server_settings(
        self,
        create_engine: MagicMock,
        sessionmaker: MagicMock,
    ) -> None:
        expected_statement_timeouts = {
            DatabaseProcess.API: "30000",
            DatabaseProcess.WORKER: "60000",
            DatabaseProcess.MAINTENANCE: "300000",
        }
        for process, expected_statement_timeout in (
            expected_statement_timeouts.items()
        ):
            with self.subTest(process=process):
                create_engine.reset_mock()
                sessionmaker.reset_mock()
                engine = MagicMock()
                sessions = MagicMock()
                create_engine.return_value = engine
                sessionmaker.return_value = sessions

                resources = create_database_resources(
                    "postgresql+asyncpg://runtime:secret@localhost/rag_kb",
                    pool_size=3,
                    max_overflow=2,
                    process=process,
                )

                self.assertIs(resources.engine, engine)
                self.assertIs(resources.sessions, sessions)
                arguments = create_engine.call_args.kwargs
                self.assertTrue(arguments["pool_pre_ping"])
                self.assertEqual(arguments["pool_size"], 3)
                self.assertEqual(arguments["max_overflow"], 2)
                self.assertEqual(
                    arguments["connect_args"]["server_settings"],
                    {
                        "application_name": f"rag-kb-{process.value}",
                        "statement_timeout": expected_statement_timeout,
                        "lock_timeout": "5000",
                        "idle_in_transaction_session_timeout": "30000",
                    },
                )

    @patch("rag_kb.db.session.async_sessionmaker")
    @patch("rag_kb.db.session.create_async_engine")
    def test_explicit_session_timeouts_are_stringified(
        self,
        create_engine: MagicMock,
        sessionmaker: MagicMock,
    ) -> None:
        create_engine.return_value = MagicMock()
        sessionmaker.return_value = MagicMock()

        create_database_resources(
            "postgresql+asyncpg://runtime:secret@localhost/rag_kb",
            pool_size=1,
            max_overflow=0,
            process=DatabaseProcess.API,
            statement_timeout_ms=101,
            lock_timeout_ms=102,
            idle_in_transaction_session_timeout_ms=103,
        )

        self.assertEqual(
            create_engine.call_args.kwargs["connect_args"]["server_settings"],
            {
                "application_name": "rag-kb-api",
                "statement_timeout": "101",
                "lock_timeout": "102",
                "idle_in_transaction_session_timeout": "103",
            },
        )

    def test_out_of_range_session_timeout_is_rejected_before_engine_creation(
        self,
    ) -> None:
        for field_name in (
            "statement_timeout_ms",
            "lock_timeout_ms",
            "idle_in_transaction_session_timeout_ms",
        ):
            arguments = {
                "statement_timeout_ms": 1,
                "lock_timeout_ms": 1,
                "idle_in_transaction_session_timeout_ms": 1,
            }
            for invalid_value in (0, 86_400_001):
                invalid_arguments = dict(arguments)
                invalid_arguments[field_name] = invalid_value
                with self.subTest(
                    field_name=field_name,
                    invalid_value=invalid_value,
                ), self.assertRaises(ValueError):
                    create_database_resources(
                        "postgresql+asyncpg://runtime:secret@localhost/rag_kb",
                        pool_size=1,
                        max_overflow=0,
                        process=DatabaseProcess.API,
                        **invalid_arguments,
                    )


if __name__ == "__main__":
    unittest.main()
