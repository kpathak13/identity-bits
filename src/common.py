"""
Shared by every stage: paths, the field list, and the small helpers that more
than one stage needs. Stages are run as `python src/<stage>.py`, which puts
src/ on the import path, so they use `from common import ...`.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"

OFFLINE = ["given_name", "surname", "dob", "postal", "email", "email_hash", "phone"]
DIGITAL = ["ip", "device_id", "fpc", "tpc", "shared_id"]
FIELDS = OFFLINE + DIGITAL

MAX_PLAUSIBLE = 12          # more records than this for one person is suspect


class DSU:
    """Disjoint set union (union-find): merges records into components."""

    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def c2(n):
    """Number of unordered pairs among n items."""
    return n * (n - 1) // 2
