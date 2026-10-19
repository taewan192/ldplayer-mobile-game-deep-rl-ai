import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from PIL import Image
from live_lords.learning import Learner
from live_lords.outcomes import OutcomePolicy,decision,economic_reward,combat_result
from live_lords.runner import Bot
from live_lords.muster import MusterMixin
from live_lords.adaptive import AdaptiveMixin
from live_lords.strategy import StrategyLedger
import test_live_defense as defense_tests
import test_live_attack_path as attack_tests


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.db=sqlite3.connect(':memory:');self.policy=OutcomePolicy(self.db)
        self.before={'vp':100e6,'gold':1e9}

    def tearDown(self):self.db.close()

    def evaluate(self,choice,won=True,after=None,hold=False,now=100):
        event=self.policy.begin('attack',[choice],self.before,now=now)
        self.policy.settle(event,won,after or self.before,'verified test evidence',now=now+30,hold=hold)
        return event

    def test_verified_vp_and_gold_losses_change_target_order(self):
        near=decision('attack_target','army:3','near');far=decision('attack_target','army:3','far')
        self.assertEqual(self.policy.rank([('near',near,.1),('far',far,0)])[0][0],'near')
        self.evaluate(near,after={'vp':98e6,'gold':800e6})
        self.assertEqual(self.policy.rank([('near',near,.1),('far',far,0)])[0][0],'far')
        self.assertLess(self.policy.estimate(near)['mean'],0)

    def test_muster_strategy_changes_from_final_battle_feedback(self):
        near=decision('muster_source','shortfall:0','nearest_bulk')
        large=decision('muster_source','shortfall:0','largest_bulk')
        self.evaluate(near,False)
        self.evaluate(large,True,now=200)
        self.assertEqual(self.policy.choose('muster_source','shortfall:0',['nearest_bulk','largest_bulk'])['action'],'largest_bulk')

    def test_defense_choice_uses_results_in_its_pressure_and_deadline_context(self):
        context='pressure:2|deadline:1'
        self.evaluate(decision('defense_route',context,'fastest:base'),False)
        self.evaluate(decision('defense_route',context,'fewest:base'),True,now=200)
        self.assertEqual(self.policy.choose('defense_route',context,['fastest:base','fewest:base'])['action'],'fewest:base')

    def test_uncertain_result_does_not_update_any_model(self):
        c=decision('attack_target','context','target')
        event=self.policy.begin('attack',[c],self.before,now=10)
        self.assertIsNone(self.policy.settle(event,None,{},'unreadable'))
        self.assertEqual(self.policy.estimate(c)['samples'],0)
        self.assertEqual(self.policy.summary()['pending'],1)

    def test_restart_retains_pending_and_never_replays_a_send(self):
        with tempfile.TemporaryDirectory() as d:
            db=sqlite3.connect(Path(d)/'learning.db');p=OutcomePolicy(db)
            event=p.begin('attack',[decision('attack_target','context','target')],self.before,
                {'pending':{'target':[500,300],'arrival':9999999999,'before':self.before,'view':'saved.png'}},now=__import__('time').time())
            db.close();db=sqlite3.connect(Path(d)/'learning.db');p=OutcomePolicy(db)
            b=AdaptiveMixin();b.outcome_policy=p;b.pending=None;b.muster=None;b.tap=Mock()
            b.restore_outcome_attack()
            self.assertEqual(b.pending['episode_id'],event);b.tap.assert_not_called();db.close()

    def test_duplicate_result_cannot_inflate_sample_count(self):
        c=decision('attack_target','context','target');event=self.evaluate(c)
        self.assertIsNone(self.policy.settle(event,True,self.before,'duplicate'))
        self.assertEqual(self.policy.estimate(c)['samples'],1)

    def test_restart_between_muster_attack_intent_and_journal_save_cannot_resend(self):
        import time
        mission={'started':time.time()-100,'staging_name':'Anchor','status':'waiting'}
        pending={'muster':mission.copy(),'arrival':time.time()+30,'target':[800,300]}
        event=self.policy.begin('attack',[decision('muster_source','context','nearest_bulk')],self.before,
            {'pending':pending},now=time.time()-1)
        b=AdaptiveMixin();b.outcome_policy=self.policy;b.muster=mission;b.pending=None;b.tap=Mock()
        b.restore_outcome_attack()
        self.assertIsNone(b.muster);self.assertEqual(b.pending['episode_id'],event);b.tap.assert_not_called()

    def test_later_city_loss_replaces_the_win_instead_of_adding_a_sample(self):
        c=decision('attack_target','context','target');event=self.evaluate(c,hold=True)
        old=self.policy.estimate(c)['mean']
        new=self.policy.settle(event,False,{'vp':98e6,'gold':1e9,'capital_lost':100e6},'new owner confirmed',now=1000)
        self.assertLess(new,old);self.assertLess(self.policy.estimate(c)['mean'],0)
        self.assertEqual(self.policy.estimate(c)['samples'],1)
        self.assertEqual(self.policy.summary()['revisions'],1)
        self.assertIsNone(self.policy.settle(event,False,self.before,'duplicate loss'))

    def test_no_positive_reward_for_passive_gold_income(self):
        a,_=economic_reward(self.before,self.before,True)
        b,_=economic_reward(self.before,{**self.before,'gold':2e9},True)
        self.assertEqual(a,b)

    def test_lost_investment_is_penalized_even_if_income_masks_hud_drop(self):
        a,_=economic_reward(self.before,self.before,False)
        b,signals=economic_reward(self.before,{**self.before,'gold':1.1e9,'capital_lost':100e6},False)
        self.assertLess(b,a);self.assertLess(signals['components']['gold'],0)

    def test_overlapping_or_long_windows_do_not_credit_passive_positive_vp(self):
        after={**self.before,'vp':102e6}
        for elapsed,confounded in ((500,False),(30,True)):
            _,signals=economic_reward(self.before,after,True,elapsed,confounded)
            self.assertEqual(signals['components']['vp'],0)

    def test_unreadable_economy_and_army_are_unknown_not_fake_zero_loss(self):
        _,signals=economic_reward({}, {},False)
        self.assertIsNone(signals['vp_delta']);self.assertIsNone(signals['gold_delta'])
        self.assertIsNone(signals['troop_loss_lower_bound'])

    def test_combat_parser_requires_city_player_role_and_unambiguous_outcome(self):
        self.assertFalse(combat_result('Astev PRABOWO 공격 패배','Astev','PRABOWO','공격'))
        for text in ('Astev PRABOWO 정찰 승리','Other PRABOWO 공격 패배','Astev Other 공격 패배','Astev PRABOWO 공격 승리 패배'):
            self.assertIsNone(combat_result(text,'Astev','PRABOWO','공격'))

    def test_old_scout_rewards_are_not_imported_as_real_combat_training(self):
        with tempfile.TemporaryDirectory() as d:
            learner=Learner(Path(d)/'learning.db')
            learner.record('live:21','scout',1,{}, {},'old scout success')
            self.assertEqual(learner.summary()['updates'],1)
            self.assertEqual(learner.summary()['outcomes']['evaluated'],0)
            self.assertEqual(learner.outcomes.estimate(decision('attack_target','context','target'))['samples'],0)
            learner.close()

    def test_real_bulk_planner_changes_donor_using_learned_strategy(self):
        b=MusterMixin();b.outcome_policy=self.policy;b.config={'max_send_fraction':.35}
        b.muster={'required':200,'staging_before':100,'reserve':65,'moves':[],'staging':(500,300),
                  'policy':'single','single_transfer':True,'candidates':[{'position':(600,300),'troops':2000},{'position':(900,300),'troops':10000}]}
        context='shortfall:2'
        self.evaluate(decision('muster_source',context,'nearest_bulk'),False)
        self.evaluate(decision('muster_source',context,'largest_bulk'),True,now=200)
        plan=b.muster_plan()
        self.assertEqual(len(plan.contributions),1)
        self.assertEqual(plan.contributions[0].position,(900,300))

    def test_learned_defense_preference_cannot_override_capacity_or_deadline(self):
        b,job=defense_tests.DefenseTests().defense_fixture();b.outcome_policy=self.policy
        job['attack']=1000
        b.prepare_transfer=Mock(return_value={'donor':{'name':'A','troops':1000},'duration':5})
        with patch('live_lords.defense.time.time',return_value=100):b.prepare_defense(job)
        b.tap.assert_not_called();self.assertEqual(job['status'],'insufficient')

    def test_unreadable_walls_defer_investment_instead_of_stopping_autoplay(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=StrategyLedger(Path(d)/'strategy.json',Mock())
            self.assertFalse(ledger.investment_allowed({'name':'Home','defense':1000,'troops':100,'walls':None},10,1000,100)[0])

    def test_missing_decimal_retries_actual_numeric_bubble_without_guessing(self):
        b=attack_tests.AttackPathTests().slider_actor('289M')
        b.vision.troop_selection_amount.return_value=28.9e6
        self.assertEqual(b.select_troops(20.875e6,289e6,86.7e6),(28.9e6,22))
        b.vision.troop_selection_amount.assert_called_once()

    def test_numeric_retry_cannot_authorize_too_many_troops(self):
        b=attack_tests.AttackPathTests().slider_actor('289M')
        b.vision.troop_selection_amount.return_value=289e6
        self.assertIsNone(b.select_troops(20.875e6,289e6,86.7e6))

    def test_attack_send_intent_is_saved_before_click_and_uses_unique_reference(self):
        with tempfile.TemporaryDirectory() as d:
            b=Bot.__new__(Bot);b.data=Path(d);b.outcome_policy=self.policy
            b.config={'max_send_fraction':.35};b.state={'attack_bonus':500,'vp':100e6,'gold':1e9}
            b.image=Image.new('RGB',(1280,720));b.muster=None;b.pending=None
            b.map=Mock();b.log=Mock();b.wait=Mock();b.capture=Mock();b.record=Mock();b.incoming=Mock(return_value=False)
            b.select_owned=Mock(return_value={'name':'Anchor','troops':1000})
            b.last_scout_city='enemy';b.last_scout_owner='other'
            b.attack_choices=[decision('attack_target','context','target')]
            b.vision=Mock();b.vision.viewport_transform.return_value='matrix'
            b.vision.transform_point.side_effect=lambda matrix,point:point
            b.vision.match.side_effect=lambda image,name,*args,**kwargs:None if name=='loss_warning' else (880,540,1)
            b.vision.enemy_identity.return_value='other:enemy'
            b.skip_shielded=Mock(return_value=False);b.reject_restricted_attack=Mock(return_value=False)
            b.region=Mock(return_value={'prompt':'출발 도시 선택'});b.select_troops=Mock(return_value=(100,30))
            b.counts={'attacks':0}
            observed=[]
            def tap(*args,**kwargs):
                if b.pending and b.pending.get('episode_id'):
                    episode=self.policy.get(b.pending['episode_id']);observed.append(episode['status'])
                    self.assertTrue((b.data/episode['meta']['pending']['view']).exists())
            b.tap=Mock(side_effect=tap)
            self.assertTrue(b.dispatch_attack((800,330),(500,300),1000,100,300,self.before))
            self.assertEqual(observed,['intent'])
            self.assertEqual(self.policy.get(b.pending['episode_id'])['status'],'sent')
            self.assertTrue(b.pending['view'].startswith('episode-'))


if __name__=='__main__':unittest.main()
