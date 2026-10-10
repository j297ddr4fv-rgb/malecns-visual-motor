"""Prevent unsafe cross-specimen assignment and preserve conflict evidence."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_leg_motor_targets import crosswalk, integral_id


def published(body='23', group='10', name='MNml78', target='middle leg'):
    row = dict.fromkeys(('bodyid', 'group', 'type', 'systematic_type', 'subclass', 'exit_nerve', 'target',
                         'synonyms', 'match_certainty(1-5)', 'publication_match', 'line_match', 'match_notes'), '')
    row.update(bodyid=body, group=group, type=name, systematic_type=name, target=target)
    return row


class TargetCrosswalkTests(unittest.TestCase):
    def test_same_number_in_different_specimen_is_not_a_match(self):
        result = crosswalk({'bodyId': 23, 'type': None}, [published(target='Ti flexor')])
        self.assertEqual(result['published_manc_rows'], [])
        self.assertIsNone(result['specific_target_candidate'])

    def test_predicted_specific_target_is_not_curated(self):
        result = crosswalk({'mancBodyid': 23., 'mancGroup': None}, [published(target='Ti flexor')])
        self.assertEqual(result['specific_target_candidate'], 'Ti flexor')
        self.assertIsNone(result['curated_specific_target'])
        self.assertEqual(result['status'], 'predicted_candidate_requires_manual_review')

    def test_curated_coarse_target_kept_despite_conflicting_prediction(self):
        rows = [published(), published(body='24', group='11', name='Ti flexor MN', target='Ti flexor')]
        result = crosswalk({'mancBodyid': 24., 'mancGroup': 10., 'mancType': 'MNml78'}, rows)
        self.assertTrue(result['type_disagreement'])
        self.assertEqual(result['curated_group_targets'], ['middle leg'])
        self.assertIsNone(result['specific_target_candidate'])
        self.assertIsNone(result['curated_specific_target'])

    def test_invalid_float_ids_rejected(self):
        for value in (float('nan'), float('inf'), 1.25, -1):
            with self.assertRaises(ValueError):
                integral_id(value)


if __name__ == '__main__':
    unittest.main()
