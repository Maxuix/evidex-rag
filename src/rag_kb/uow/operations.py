"""Helpers that make the end of a database-only transaction explicit."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from rag_kb.uow.mode import TransactionMode

if TYPE_CHECKING:
    from rag_kb.uow.sqlalchemy import SqlAlchemyUnitOfWork, SqlAlchemyUnitOfWorkFactory


ResultT = TypeVar("ResultT")


async def execute_in_transaction(
    factory: SqlAlchemyUnitOfWorkFactory,
    operation: Callable[[SqlAlchemyUnitOfWork], Awaitable[ResultT]],
    *,
    mode: TransactionMode = TransactionMode.READ_WRITE,
) -> ResultT:
    """Run database-only work, close its session, then return to the caller."""

    async with factory(mode=mode) as unit_of_work:
        result = await operation(unit_of_work)
        await unit_of_work.commit()
    return result
