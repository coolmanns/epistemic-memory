"""Epistemic Synthesis — Claim Decay Processor

Downgrades claim confidence based on time since last reinforcement.
HIGH → MED (90 days) → LOW (60 days) → decayed (60 days)
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from . import config
from .schema import init_epistemic_db

logger = logging.getLogger("epistemic")


class ClaimDecay:
    """Processes claim confidence decay based on reinforcement timing."""

    def __init__(
        self,
        epistemic_db: Optional[Path] = None,
        high_days: Optional[int] = None,
        med_days: Optional[int] = None,
        low_days: Optional[int] = None,
        now: Optional[datetime] = None,
    ):
        self.epistemic_db = epistemic_db or config.EPISTEMIC_DB
        self.high_days = high_days if high_days is not None else config.CLAIM_DECAY_HIGH_DAYS
        self.med_days = med_days if med_days is not None else config.CLAIM_DECAY_MED_DAYS
        self.low_days = low_days if low_days is not None else config.CLAIM_DECAY_LOW_DAYS
        self._now = now  # allow injection for testing

    def _get_now(self) -> datetime:
        return self._now or datetime.utcnow()

    def run(self) -> dict:
        """Process decay for all claims. Returns stats."""
        conn = init_epistemic_db(self.epistemic_db)
        now = self._get_now()
        stats = {"high_to_med": 0, "med_to_low": 0, "low_to_decayed": 0, "total_processed": 0}

        # Process in REVERSE order to prevent same-run cascading:
        # LOW → decayed first, then MED → LOW, then HIGH → MED
        # This way a HIGH claim only drops to MED in one run, not all the way to decayed.

        # LOW → decayed
        cutoff = (now - timedelta(days=self.low_days)).strftime("%Y-%m-%dT%H:%M:%S")
        rows = conn.execute(
            """UPDATE claims SET status = 'decayed',
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
               WHERE confidence = 'LOW' AND status IN ('active', 'verified')
                 AND last_reinforced < ?""",
            (cutoff,)
        )
        stats["low_to_decayed"] = rows.rowcount

        # MED → LOW
        cutoff = (now - timedelta(days=self.med_days)).strftime("%Y-%m-%dT%H:%M:%S")
        rows = conn.execute(
            """UPDATE claims SET confidence = 'LOW',
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
               WHERE confidence = 'MED' AND status IN ('active', 'verified')
                 AND last_reinforced < ?""",
            (cutoff,)
        )
        stats["med_to_low"] = rows.rowcount

        # HIGH → MED
        cutoff = (now - timedelta(days=self.high_days)).strftime("%Y-%m-%dT%H:%M:%S")
        rows = conn.execute(
            """UPDATE claims SET confidence = 'MED',
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
               WHERE confidence = 'HIGH' AND status IN ('active', 'verified')
                 AND last_reinforced < ?""",
            (cutoff,)
        )
        stats["high_to_med"] = rows.rowcount

        conn.commit()
        stats["total_processed"] = stats["high_to_med"] + stats["med_to_low"] + stats["low_to_decayed"]

        conn.close()
        if stats["total_processed"] > 0:
            logger.info(f"Decay: {stats}")
        return stats

    @staticmethod
    def reinforce(claim_id: int, db_path: Optional[Path] = None):
        """Reinforce a claim — reset its last_reinforced timestamp."""
        path = db_path or config.EPISTEMIC_DB
        conn = sqlite3.connect(str(path))
        conn.execute(
            """UPDATE claims SET last_reinforced = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                   updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
               WHERE id = ?""",
            (claim_id,)
        )
        conn.commit()
        conn.close()
