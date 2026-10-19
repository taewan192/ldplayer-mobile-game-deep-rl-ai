import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from PIL import Image

from live_lords.defense import DefenseMixin
from live_lords.learning import Learner
from live_lords.planner import SupportRoute,defense_shortfall,support_plan
from live_lords.strategy import StrategyLedger
from live_lords.vision import Vision


class DefenseTests(unittest.TestCase):
    def test_unknown_threat_no_longer_stops_measuring_investment_candidates(self):
        obj=DefenseMixin(); obj.config={}; obj.strategy=Mock()
        obj.strategy.reference_attack.return_value=None
        obj.map=Mock(); obj.vision=Mock(); obj.log=Mock()
        obj.image=Mock(); obj.state={'owner':'Yumi'}; obj.vision.own_sources.return_value=[]
        obj.strategic_upgrade()
        obj.map.assert_called_once(); obj.vision.own_sources.assert_called_once()

    def defense_fixture(self):
        obj=DefenseMixin()
        obj.config={'max_send_fraction':.35}; obj.data=Path('.')
        obj.state={'owner':'Yumi'}; obj.image=Mock(); obj.learner=Mock()
        obj.learner.choose.return_value='fastest'
        obj.strategy=Mock(); obj.vision=Mock(); obj.threatened_names=set()
        obj.vision.own_sources.return_value=[((800,330),1000)]
        obj.vision.transform_point.side_effect=lambda matrix,pos:pos
        obj.capture=Mock(); obj.rebase_defense=Mock(); obj.map=Mock(); obj.back=Mock()
        obj.save_defenses=Mock(); obj.log=Mock(); obj.tap=Mock(); obj.counts={'supports':0}
        obj.defense_snapshot_valid=Mock(return_value=True)
        job={'status':'preparing','staging':(640,330),'staging_name':'Home','attack':100,'impact':200,
             'before':{'defense':0,'city_bonus':0,'guardian_bonus':0},'moves':[],'donors':[]}
        obj.defenses=[job]
        return obj,job

    def test_rechecks_real_deadline_after_slider_and_sends_nothing_if_route_now_late(self):
        obj,job=self.defense_fixture()
        preflight={'donor':{'name':'A','troops':1000},'duration':5}
        selected={'donor':{'name':'A','troops':1000},'duration':90,'observed':100,'troops':200,'button':(880,540)}
        obj.prepare_transfer=Mock(side_effect=[preflight,selected])
        with patch('live_lords.defense.time.time',return_value=100): obj.prepare_defense(job)
        obj.tap.assert_not_called()
        self.assertEqual(job['moves'],[])

    def test_unknown_send_outcome_already_has_intent_and_is_never_a_support_success(self):
        obj,job=self.defense_fixture()
        selected={'donor':{'name':'A','troops':1000},'duration':5,'observed':100,'troops':200,'button':(880,540)}
        obj.prepare_transfer=Mock(return_value=selected)
        def uncertain(*args,**kwargs):
            self.assertEqual(job['moves'][0]['status'],'intent')
            raise RuntimeError('connection interrupted after input')
        obj.tap.side_effect=uncertain
        with patch('live_lords.defense.time.time',return_value=100):
            with self.assertRaises(RuntimeError): obj.prepare_defense(job)
        self.assertEqual(obj.counts['supports'],0)
        self.assertIsNone(obj.active_defense)

    def test_defense_uses_city_and_guardian_additively_and_existing_wall_total(self):
        self.assertEqual(defense_shortfall(600,531,255,144,1.3),51)
        self.assertEqual(defense_shortfall(300,531,255,144,1.3),0)
        with self.assertRaises(ValueError): defense_shortfall(float('nan'),531,255,144)

    def test_deadline_includes_serial_setup_and_equality_is_late(self):
        route=SupportRoute('A',(500,300),100,30)
        self.assertIsNone(support_plan(10,162,100,[route],buffer=20,setup_seconds=12))
        self.assertEqual(support_plan(10,163,100,[route],buffer=20,setup_seconds=12)[0][1],10)
        routes=[route,SupportRoute('B',(700,300),100,30)]
        self.assertIsNone(support_plan(50,170,100,routes,buffer=20,setup_seconds=12))

    def test_rounding_reserves_duplicate_sources_and_insufficient_all_or_nothing(self):
        routes=[SupportRoute('A',(500,300),100,20),SupportRoute('B',(700,300),100,30)]
        plan=support_plan(45,1000,100,routes)
        self.assertEqual([take for _,take in plan],[30,20])
        self.assertTrue(all(take<=r.total*.3+.01 for r,take in plan))
        self.assertIsNone(support_plan(31,1000,100,[routes[0],routes[0]]))
        self.assertIsNone(support_plan(61,1000,100,routes))

    def test_no_learning_reward_for_send_or_unknown_ownership(self):
        obj=DefenseMixin(); now=time.time()
        obj.defenses=[{'id':'x','status':'waiting','impact':now-15,'view':'x.png','staging_name':'Home','moves':[]}]
        obj.map=Mock(); obj.defense_position=Mock(return_value=(640,330))
        obj.read_owned_details=Mock(return_value={'troops':100})
        obj.read_defense_result=Mock(return_value=None); obj.save_defenses=Mock()
        obj.log=Mock(); obj.learner=Mock(); obj.strategy=Mock()
        obj.check_defense_results()
        obj.learner.record.assert_not_called(); obj.strategy.outcome.assert_not_called()
        self.assertEqual(obj.defenses[0]['status'],'unverified')

    def test_restart_never_replays_uncertain_support_intent(self):
        with tempfile.TemporaryDirectory() as folder:
            obj=DefenseMixin(); obj.data=Path(folder); obj.learner=Mock()
            job={'status':'preparing','moves':[{'status':'intent','troops':10}]}
            (obj.data/'defense-journal.json').write_text(json.dumps([job]),'utf-8')
            obj.init_defense()
            self.assertEqual(obj.defenses[0]['status'],'waiting')
            self.assertEqual(obj.defenses[0]['moves'],job['moves'])

    def test_troop_number_never_becomes_unknown_enemy_attack_power(self):
        with tempfile.TemporaryDirectory() as folder:
            ocr=Mock(); v=Vision(ocr,Path(folder)); im=Image.new('RGB',(1280,720))
            ocr.read.return_value={'text':'4.36M 0m 22s','lines':[{'text':'4.36M 0m 22s','words':[]}]}
            self.assertIsNone(v.labeled_attack_power(im))
            ocr.read.return_value={'lines':[{'text':'총 공격력 2.4T','words':[]}]}
            self.assertEqual(v.labeled_attack_power(im),2.4e12)

    def test_inactive_defense_tab_never_reads_outgoing_as_incoming(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(Mock(),Path(folder)); im=Image.new('RGB',(1280,720),'#172b39')
            self.assertEqual(v.defense_rows(im),[])
            v.ocr.read.assert_not_called()

    def test_upgrade_expense_has_delayed_reward_and_later_loss_reverses_hold_reward(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); learner=Learner(p/'learn.db'); ledger=StrategyLedger(p/'strategy.json',learner)
            entry=ledger.invest('Home',10,'risk',{'gold':1000}); entry['status']='pending'; ledger.save()
            self.assertEqual(learner.summary()['updates'],0)
            ledger.outcome('Home',True,'held')
            ledger.outcome('Home',True,'held again')
            self.assertEqual(learner.summary()['updates'],1)
            ledger.outcome('Home',False,'actual new owner',event_id='battle1')
            ledger.outcome('Home',False,'duplicate battle',event_id='battle1')
            self.assertEqual(learner.summary()['updates'],2)
            self.assertLess(learner.summary()['reward'],0)
            self.assertEqual(entry['status'],'lost')
            row=learner.db.execute('SELECT after_json FROM experience ORDER BY id DESC LIMIT 1').fetchone()
            self.assertEqual(json.loads(row[0])['capital_lost'],10)
            learner.close()

    def test_no_uniform_upgrade_without_threat_baseline_and_loss_lock_is_hard(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=StrategyLedger(Path(folder)/'strategy.json',Mock())
            city={'name':'Home','defense':1000}
            self.assertFalse(ledger.investment_allowed(city,10,1000,100)[0])
            ledger.threat('Other',100,time.time()-60)
            self.assertTrue(ledger.investment_allowed(city,10,1000,100)[0])
            self.assertFalse(ledger.investment_allowed({'name':'Weak','defense':100},10,1000,100)[0])
            ledger.outcome('Home',False,'lost')
            self.assertFalse(ledger.investment_allowed(city,10,1000,100)[0])

    def test_ownership_learning_persists_across_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'strategy.json'; ledger=StrategyLedger(path,Mock())
            ledger.threat('Home',600,time.time()+90); ledger.outcome('Home',False,'lost')
            restored=StrategyLedger(path,Mock())
            self.assertEqual(restored.reference_attack(),600)
            self.assertFalse(restored.investment_allowed({'name':'Home','defense':9999},1,1000,10)[0])

    def test_unconfirmed_upgrade_intent_blocks_new_spending_after_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'strategy.json'; ledger=StrategyLedger(path,Mock())
            ledger.threat('Other',100,time.time()-60)
            ledger.invest('Home',10,'risk',{'gold':1000})
            restored=StrategyLedger(path,Mock())
            self.assertFalse(restored.investment_allowed({'name':'Home','defense':1000},10,1000,100)[0])


if __name__=='__main__': unittest.main()
