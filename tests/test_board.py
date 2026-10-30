import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from nlp_module import storage
from nlp_module.board import FUSION_VERSION, newest_decisions, snapshot_passages
from nlp_module.contract import build_snapshot
from nlp_module.pipeline import load_results

CUT = datetime(2026, 10, 29, 8, 0, tzinfo=timezone.utc)
M = "m1"


def fd(c, sym, base, h, ver=FUSION_VERSION, created="2026-10-29T08:00:00Z"):
    c.execute("INSERT INTO fusion_decision VALUES (?,?,?,?,NULL,NULL,NULL,0.4,'weak',1,0.1,'[]','r',?,?)",
              (sym, base, h, "NO_TRADE", ver, created))


def passage(c, i, sym, pub, proc, relation="direct", rep=1, text="ok", model=M):
    c.execute("INSERT OR IGNORE INTO document VALUES (?,?,?,?,?)", (f"d{i}", "s", f"u{i}", "news", pub))
    c.execute("INSERT INTO document_version VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
              (f"v{i}", f"d{i}", "h", f"t{i}", text, text, pub, None, pub, "k", None, rep))
    c.execute("INSERT INTO article_company_result (result_id, document_version_id, target_entity_id, symbol, relation,"
              " sentence_ids, passage_text, p_pos, p_neu, p_neg, nlp_model_version, processed_at)"
              " VALUES (?,?,?,?,?,'[]',?,0.6,0.3,0.1,?,?)", (f"r{i}", f"v{i}", "e", sym, relation, text, model, proc))


class BoardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t.db"
        self.c = storage.connect(self.path)

    def tearDown(self):
        self.c.close()
        self.tmp.cleanup()

    def test_newest_row_per_symbol_h_and_version_filter(self):
        fd(self.c, "AAA", "2026-10-28", 2)
        fd(self.c, "AAA", "2026-10-29", 2)
        fd(self.c, "AAA", "2026-10-29", 5)
        fd(self.c, "AAA", "2026-10-30", 5, ver="decision-v9")      # other version never leaks in
        self.c.commit()
        df = newest_decisions(self.c)
        self.assertEqual(list(zip(df["symbol"], df["h"], df["base_date"])),
                         [("AAA", 2, "2026-10-29"), ("AAA", 5, "2026-10-29")])

    def test_missing_table_is_empty(self):
        con = sqlite3.connect(":memory:")
        self.assertTrue(newest_decisions(con).empty)

    def test_passages_match_build_snapshot_lineage(self):
        passage(self.c, 1, "AAA", "2026-10-28T02:00:00Z", "2026-10-28T03:00:00Z", text="bị khởi tố vì thao túng")   # in
        passage(self.c, 2, "AAA", "2026-10-28T02:00:00Z", "2026-10-28T03:00:00Z", relation="market")          # market: out
        passage(self.c, 3, "AAA", "2026-10-28T02:00:00Z", "2026-10-29T09:00:00Z")                              # processed after cutoff: out
        passage(self.c, 4, "AAA", "2026-10-20T02:00:00Z", "2026-10-28T03:00:00Z")                              # late-collected, old publish: in
        passage(self.c, 5, "AAA", "2026-10-28T02:00:00Z", "2026-10-25T03:00:00Z")                              # processed > 72h before cutoff: out
        passage(self.c, 6, "AAA", "2026-10-28T02:00:00Z", "2026-10-28T03:00:00Z", rep=0)                       # repost: out
        passage(self.c, 7, "AAA", "2026-10-28T02:00:00Z", "2026-10-28T03:00:00Z", model="other")                # other model: out
        passage(self.c, 8, "BBB", "2026-10-28T02:00:00Z", "2026-10-28T03:00:00Z")                              # other symbol: out
        self.c.commit()
        df = snapshot_passages(self.c, "AAA", CUT, M)
        want = build_snapshot([r for r in load_results(self.c, M)], "AAA", CUT, "live", M).lineage
        self.assertEqual(sorted(df["result_id"]), sorted(want))
        self.assertEqual(sorted(df["result_id"]), ["r1", "r4"])
        self.assertIn("prosecution", df.set_index("result_id").loc["r1", "flags"])
