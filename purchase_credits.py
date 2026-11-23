"""Persistent credit reservations for verified Gumroad purchases."""
from contextlib import closing, contextmanager
import sqlite3


class PurchaseCredits:
    def __init__(self, path):
        self.path = path
        with self.transaction() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS purchases (
                    sale_id TEXT PRIMARY KEY,
                    credits INTEGER NOT NULL CHECK (credits >= 0)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS reservations (
                    id TEXT PRIMARY KEY,
                    sale_id TEXT NOT NULL REFERENCES purchases(sale_id),
                    state TEXT NOT NULL CHECK (state IN ('pending', 'used', 'released'))
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS refinements (
                    card_id TEXT PRIMARY KEY REFERENCES reservations(id),
                    attempt_id TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL CHECK (state IN ('pending', 'used'))
                )
            """)

    @contextmanager
    def transaction(self):
        with closing(sqlite3.connect(self.path, timeout=10)) as connection, connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            yield connection

    def activate(self, sale_id):
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO purchases (sale_id, credits) VALUES (?, 7)", (sale_id,)
            )
            return connection.execute(
                "SELECT credits FROM purchases WHERE sale_id = ?", (sale_id,)
            ).fetchone()[0]

    def reserve(self, sale_id, card_id):
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT sale_id, state FROM reservations WHERE id = ?", (card_id,)
            ).fetchone()
            if existing:
                return False
            changed = connection.execute(
                "UPDATE purchases SET credits = credits - 1 WHERE sale_id = ? AND credits > 0",
                (sale_id,),
            ).rowcount
            if changed:
                connection.execute(
                    "INSERT INTO reservations VALUES (?, ?, 'pending')", (card_id, sale_id)
                )
            return bool(changed)

    def settle(self, sale_id, card_id, succeeded):
        with self.transaction() as connection:
            changed = connection.execute(
                "UPDATE reservations SET state = ? WHERE id = ? AND sale_id = ? AND state = 'pending'",
                ("used" if succeeded else "released", card_id, sale_id),
            ).rowcount
            if changed and not succeeded:
                connection.execute(
                    "UPDATE purchases SET credits = credits + 1 WHERE sale_id = ?", (sale_id,)
                )

    def reserve_refinement(self, sale_id, card_id, attempt_id):
        with self.transaction() as connection:
            return bool(connection.execute("""
                INSERT OR IGNORE INTO refinements
                SELECT id, ?, 'pending' FROM reservations
                WHERE id = ? AND sale_id = ? AND state = 'used'
            """, (attempt_id, card_id, sale_id)).rowcount)

    def settle_refinement(self, sale_id, card_id, attempt_id, succeeded):
        where = """WHERE card_id = ? AND attempt_id = ? AND state = 'pending'
                   AND EXISTS (SELECT 1 FROM reservations WHERE id = card_id AND sale_id = ?)"""
        operation = "UPDATE refinements SET state = 'used' " if succeeded else "DELETE FROM refinements "
        with self.transaction() as connection:
            connection.execute(operation + where, (card_id, attempt_id, sale_id))
