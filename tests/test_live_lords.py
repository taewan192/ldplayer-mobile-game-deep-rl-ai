import tempfile
import unittest
from pathlib import Path

from live_lords.driver import normalized_name
from live_lords.learning import Learner
from live_lords.planner import attack_power,required_attack,reinforcement_plan,Reinforcement
from live_lords.vision import amount


class LiveLordsTests(unittest.TestCase):
    def test_user_combat_example(self):
        self.assertEqual(attack_power(400e9,500),2.4e12)
        self.assertEqual(attack_power(1e6,277.25),3772500)

    def test_required_attack_and_reserve(self):
        needed=required_attack(2.70e9,277.25,1.5)
        self.assertGreaterEqual(attack_power(needed,277.25),2.70e9*1.5)
        self.assertGreater(needed,43.1e6*.35)

    def test_reinforcement_must_arrive_before_impact(self):
        sources=[Reinforcement('near',80,20),Reinforcement('late',1000,60),Reinforcement('second',50,30)]
        plan=reinforcement_plan(100,50,sources,12)
        self.assertEqual([(x.source,x.troops) for x in plan],[('near',80),('second',20)])
        self.assertEqual(reinforcement_plan(140,50,sources,12),[])

    def test_uncertain_ocr_fails_closed(self):
        self.assertEqual(amount('2.70G'),2.7e9)
        self.assertEqual(amount('400g'),400e9)
        self.assertIsNone(amount('2.706 troop'))
        self.assertIsNone(amount('4.2GG'))
        self.assertIsNone(amount('?'))
        self.assertEqual(normalized_name('유미 ( 전투용 ) '),normalized_name('유미(전투용)'))

    def test_learning_persists_actual_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'learn.sqlite3'
            learner=Learner(path)
            learner.record('upgrade_batch','one',1,{'level':83},{'level':84},'observed')
            learner.close()
            learner=Learner(path)
            self.assertEqual(learner.summary()['updates'],1)
            self.assertEqual(learner.choose('upgrade_batch',['one','two']),'two')
            learner.close()


if __name__=='__main__': unittest.main()
