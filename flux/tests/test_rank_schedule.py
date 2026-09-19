import unittest

from model.rank_schedule import get_rank_by_timestep


class RankScheduleTest(unittest.TestCase):
    def test_default_decreasing_schedule_preserves_original_mapping(self):
        self.assertEqual(get_rank_by_timestep(0, 1000, 256, 64), 256)
        self.assertEqual(get_rank_by_timestep(500, 1000, 256, 64), 160)
        self.assertEqual(get_rank_by_timestep(1000, 1000, 256, 64), 64)

    def test_increasing_schedule_reverses_mapping(self):
        self.assertEqual(get_rank_by_timestep(0, 1000, 256, 64, "increasing"), 64)
        self.assertEqual(get_rank_by_timestep(500, 1000, 256, 64, "increasing"), 160)
        self.assertEqual(get_rank_by_timestep(1000, 1000, 256, 64, "increasing"), 256)

    def test_unknown_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported rank schedule"):
            get_rank_by_timestep(500, 1000, 256, 64, "unknown")


if __name__ == "__main__":
    unittest.main()
