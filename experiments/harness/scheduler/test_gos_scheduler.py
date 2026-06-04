import unittest
from collections import deque
from collections import defaultdict
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spice_shallow_issuer_runtime as rt


class FakeIssuer:
    low_slots = [None] * 4

    def low_backlog_count(self):
        return 0

    def low_reserved_count(self):
        return 0


class FakeEvent:
    def __init__(self, complete):
        self.complete = complete

    def query(self):
        return self.complete


class GosSchedulerTest(unittest.TestCase):
    def make_shallow_issuer_without_cuda(self):
        issuer = rt.ShallowIssuer.__new__(rt.ShallowIssuer)
        issuer.intents = deque()
        issuer.active_low = []
        issuer.staged = {}
        issuer.pending = set()
        issuer.stats = defaultdict(float)
        issuer.released_slots = []
        issuer.release_low_slot = issuer.released_slots.append
        return issuer

    def test_dp_rebuilds_prefix_after_selected_duplicate_key(self):
        args = SimpleNamespace(
            gos_max_prefetch_per_target=2,
            gos_layer_slack_ms=10.0,
            gos_cpu_overlap_ms=0.0,
            gos_value_margin_ms=0.0,
        )
        # Make serving A for the earlier target much more valuable than choosing
        # A+B only for the later target. The later target must then rebuild its
        # prefix with B after A has already been selected.
        cost = {
            (1, 0): 100.0,
            (1, 1): 0.0,
            (2, 0): 101.0,
            (2, 1): 100.0,
            (2, 2): 99.0,
        }
        a_key = (1, 0)
        b_key = (1, 1)
        candidates = {
            (0, 1, 1): [a_key],
            (0, 2, 2): [a_key, b_key],
        }

        admitted = rt.choose_gos_admissions_dp(
            candidates,
            set(),
            FakeIssuer(),
            cost,
            t_fetch=1.0,
            t_gpu=0.0,
            args=args,
            stats=defaultdict(float),
        )

        self.assertEqual(admitted, [(a_key, 0, 1), (b_key, 0, 2)])

    def test_dp_zero_option_keeps_later_state_alive(self):
        args = SimpleNamespace(
            gos_max_prefetch_per_target=1,
            gos_layer_slack_ms=0.1,
            gos_cpu_overlap_ms=0.0,
            gos_value_margin_ms=0.0,
        )
        cost = {
            (1, 0): 100.0,
            (1, 1): 0.0,
        }
        a_key = (1, 0)
        b_key = (1, 1)
        candidates = {
            (0, 1, 1): [a_key],
            (0, 3, 20): [b_key],
        }

        admitted = rt.choose_gos_admissions_dp(
            candidates,
            set(),
            FakeIssuer(),
            cost,
            t_fetch=1.0,
            t_gpu=0.0,
            args=args,
            stats=defaultdict(float),
        )

        self.assertEqual(admitted, [(b_key, 0, 3)])

    def test_expired_active_completion_does_not_clear_newer_pending_copy(self):
        issuer = self.make_shallow_issuer_without_cuda()
        key = (3, 7)
        old_active = {
            "event": FakeEvent(True),
            "key": key,
            "slot": 0,
            "target_token": 0,
            "target_layer": 1,
            "expired": False,
            "pending_live": True,
        }
        issuer.active_low.append(old_active)
        issuer.pending.add(key)

        issuer.expire(ti=0, layer=2)
        self.assertTrue(old_active["expired"])
        self.assertFalse(old_active["pending_live"])
        self.assertNotIn(key, issuer.pending)

        issuer.add_intent(key, target_token=0, target_layer=5, resident_or_staged=set())
        self.assertIn(key, issuer.pending)
        self.assertEqual(len(issuer.intents), 1)

        issuer.poll()
        self.assertEqual(issuer.active_low, [])
        self.assertEqual(issuer.released_slots, [0])
        self.assertIn(key, issuer.pending)
        self.assertEqual(issuer.stats["prefetch_completed_expired"], 1)


if __name__ == "__main__":
    unittest.main()
