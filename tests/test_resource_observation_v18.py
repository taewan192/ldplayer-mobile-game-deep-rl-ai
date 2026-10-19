import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

from live_lords.outcomes import OutcomePolicy,decision
from live_lords.vision import Vision


class ResourceObservationTests(unittest.TestCase):
    def test_two_gold_crops_confirm_the_leading_digit(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder);v.regions=Mock(side_effect=[{'vp':'117M','gold':'12.5G'},{'gold_verify':'12.5G'}])
            self.assertEqual(v.hud(Mock(),verify=True)['gold'],12.5e9)
            self.assertEqual(v.regions.call_count,2)

    def test_missing_leading_one_is_rejected_when_two_other_crops_agree(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder);v.regions=Mock(side_effect=[{'vp':'117M','gold':'2.5G'},
                {'gold_verify':'12.5G'},{'gold_confirm':'12.5G'}])
            self.assertEqual(v.hud(Mock(),verify=True)['gold'],12.5e9)

    def test_unresolved_gold_disagreement_is_unknown_instead_of_a_false_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder);v.regions=Mock(side_effect=[{'vp':'117M','gold':'2.5G'},
                {'gold_verify':'12.5G'},{'gold_confirm':'1.25G'}])
            self.assertIsNone(v.hud(Mock(),verify=True)['gold'])

    def test_evidence_correction_changes_credit_without_another_battle_or_sample(self):
        with sqlite3.connect(':memory:') as db:
            policy=OutcomePolicy(db);choice=decision('muster_source','shortfall:1','nearest_bulk')
            event=policy.begin('attack',[choice],{'vp':116e6,'gold':12.4e9},now=100)
            bad=policy.settle(event,True,{'vp':117e6,'gold':2.5e9,'troops_exposed':500e6,
                'troop_loss_lower_bound':244e6},'actual ownership',now=918,hold=True)
            fixed=policy.correct_resource_observation(event,'gold',12.5e9,'original-conquest-frame.png')
            self.assertAlmostEqual(fixed-bad,2)
            self.assertEqual(policy.summary()['evaluated'],1)
            self.assertEqual(policy.estimate(choice)['samples'],1)
            self.assertEqual(policy.get(event)['result'],1)
            self.assertEqual(policy.get(event)['updated'],918)
            self.assertEqual(policy.get(event)['after']['troop_loss_lower_bound'],244e6)
            self.assertEqual(policy.get(event)['after']['signals']['gold_delta'],100e6)
            self.assertEqual(policy.correct_resource_observation(event,'gold',12.5e9,'same-frame.png'),fixed)
            self.assertEqual(len(policy.get(event)['after']['observation_corrections']),1)


if __name__=='__main__':unittest.main()
