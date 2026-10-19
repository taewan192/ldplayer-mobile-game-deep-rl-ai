"""Algorithm and live-adapter verification; all test data stays in temp folders."""
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from live_lords.deep_rl import DeepPolicy, QNetwork, OBS, ACTION, action_vector, pack, unpack
from live_lords.learning import Learner
from live_lords.outcomes import OutcomePolicy, decision
from live_lords.reinforcement import ReinforcementMixin
from live_lords.runner import Bot
from live_lords.strategy import StrategyLedger
from live_lords.focus_defense import FocusDefenseMixin


class DeepRLTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        self.db=sqlite3.connect(self.path/'learning.sqlite3')
        self.deep=DeepPolicy(self.db,self.path/'reinforcement')
        self.policy=OutcomePolicy(self.db,self.deep)
        self.image=Image.new('RGB',(160,90),(30,60,90))
        self.deep.observe(self.image,{'vp':1e8,'gold':1e9})

    def tearDown(self):self.db.close();self.temp.cleanup()

    def choice(self,domain='attack_target',actions=('near','far')):
        return self.deep.choose(domain,'test',list(actions))

    def battle(self,choices,won=True,hold=False):
        event=self.policy.begin('attack',choices,{'vp':1e8,'gold':1e9})
        self.policy.settle(event,won,{'vp':1e8 if won else 98e6,'gold':1e9 if won else 8e8},'verified test report',hold=hold)
        return event

    def replay(self):
        return self.db.execute('SELECT id,observation,action,reward,next_observation,next_actions,discount,terminal,priority FROM rl_replay ORDER BY ts,id').fetchall()

    def seed_verified(self,n=8):
        for _ in range(n):self.battle([self.choice()])

    def test_images_and_records_alone_never_update_neural_weights(self):
        before={k:v.copy() for k,v in self.deep.online.weights.items()}
        for _ in range(12):self.choice()
        self.assertFalse(self.deep.train_step());self.assertEqual(self.deep.rows(),0)
        for k,v in before.items():np.testing.assert_array_equal(v,self.deep.online.weights[k])

    def test_unreadable_outcome_is_not_a_training_example(self):
        event=self.policy.begin('attack',[self.choice()],{})
        self.policy.settle(event,None,{},'unreadable')
        self.assertEqual(self.deep.rows(),0)

    def test_verified_trace_preserves_delayed_credit_and_masked_successor(self):
        first=self.choice('strategy',('attack','explore'))
        second=self.choice('muster_source',('nearest_bulk','largest_bulk'))
        event=self.battle([first,second])
        rows=self.replay();self.assertEqual(len(rows),2)
        root=next(r for r in rows if r[0].endswith(first['_rl_id']))
        final=next(r for r in rows if r[0].endswith(second['_rl_id']))
        self.assertEqual(root[3],0);self.assertEqual(root[7],0)
        self.assertGreater(final[3],0);self.assertEqual(final[7],1)
        self.assertEqual(unpack(root[5]).size,2*ACTION)
        self.assertEqual(self.deep.summary()['combat_episodes'],1)
        self.assertIsNone(self.policy.settle(event,True,{},'duplicate'))
        self.assertEqual(len(self.replay()),2)

    def test_double_dqn_online_selects_and_target_evaluates_only_legal_actions(self):
        self.deep.online.forward=Mock(return_value=np.array([1.,8.]))
        self.deep.target.forward=Mock(return_value=np.array([50.,3.]))
        row=('id',None,None,2.,pack(np.zeros(OBS)),pack(np.zeros((2,ACTION))),.5,False,1.)
        self.assertAlmostEqual(self.deep.targets([row])[0],3.5)
        self.assertEqual(self.deep.online.forward.call_args.args[0].shape[0],2)

    def test_pixels_are_input_and_receive_actual_backpropagation(self):
        self.seed_verified()
        before=self.deep.online.weights['w0'][:576].copy()
        self.assertTrue(self.deep.train_step())
        self.assertFalse(np.array_equal(before,self.deep.online.weights['w0'][:576]))
        self.assertEqual(self.deep.online.step,1)
        self.assertTrue(np.isfinite(self.deep.last_loss))

    def test_real_economic_loss_changes_terminal_reward(self):
        self.battle([self.choice()],True)
        self.battle([self.choice()],False)
        rewards=sorted(r[3] for r in self.replay())
        self.assertLess(rewards[0],-2);self.assertGreater(rewards[1],0)

    def test_later_loss_relabels_and_rebuilds_instead_of_double_counting(self):
        event=self.battle([self.choice()],True,hold=True)
        self.seed_verified();self.deep.train_step();self.assertGreater(self.deep.online.step,0)
        before=len(self.replay())
        self.policy.settle(event,False,{'vp':9e7,'gold':8e8,'capital_lost':1e8},'owner loss verified')
        self.assertEqual(len(self.replay()),before)
        reward=self.db.execute('SELECT reward FROM rl_replay WHERE episode=?',(event,)).fetchone()[0]
        self.assertLess(reward,0);self.assertEqual(self.deep.online.step,0)
        self.assertEqual(self.deep.summary()['rebuilds'],1)

    def test_checkpoint_preserves_predictions_target_and_optimizer(self):
        self.seed_verified();self.deep.train_step();self.deep.save()
        x=np.concatenate((self.deep.observation,action_vector(decision('a','b','c'))))[None]
        expected=self.deep.online.forward(x)
        restored=DeepPolicy(self.db,self.path/'reinforcement')
        np.testing.assert_array_equal(expected,restored.online.forward(x))
        self.assertEqual(restored.online.step,1)
        for k,v in self.deep.target.weights.items():np.testing.assert_array_equal(v,restored.target.weights[k])
        for k,v in self.deep.online.m.items():np.testing.assert_array_equal(v,restored.online.m[k])

    def test_stale_checkpoint_cannot_restore_pre_correction_weights(self):
        event=self.battle([self.choice()],True,hold=True)
        self.seed_verified();self.deep.train_step();self.deep.save()
        self.policy.settle(event,False,{},'city lost')
        restored=DeepPolicy(self.db,self.path/'reinforcement')
        self.assertEqual(restored.online.step,0)
        self.assertIsNotNone(restored.checkpoint_error)

    def test_interrupted_checkpoint_is_quarantined_without_destroying_replay(self):
        self.seed_verified();self.deep.save()
        path=self.path/'reinforcement/visual-policy.npz'
        path.write_bytes(path.read_bytes()[:100])
        restored=DeepPolicy(self.db,self.path/'reinforcement')
        self.assertIsNotNone(restored.checkpoint_error)
        self.assertEqual(restored.rows(),8)
        self.assertEqual(restored.online.step,0)

    def test_evaluation_freezes_weights_and_disables_exploration_even_on_correction(self):
        event=self.battle([self.choice()],True,hold=True)
        self.seed_verified();self.deep.train_step();self.deep.mode='evaluate'
        before={k:v.copy() for k,v in self.deep.online.weights.items()}
        self.policy.settle(event,False,{},'city lost')
        self.assertFalse(self.deep.train_step());self.assertEqual(self.deep.epsilon,0)
        for k,v in before.items():np.testing.assert_array_equal(v,self.deep.online.weights[k])
        self.deep.mode='train';self.deep.train_step()
        self.assertEqual(self.deep.summary()['rebuilds'],1)

    def test_navigation_cost_bootstraps_on_next_actual_strategy_mask(self):
        c=self.choice('strategy',('explore',))
        self.deep.finish_navigation(c,30)
        self.assertEqual(self.deep.rows(),0)
        self.choice('attack_margin',('configured',))
        self.assertEqual(self.deep.rows(),0)
        self.choice('strategy',('attack',))
        self.assertEqual(self.deep.rows(),1)
        row=self.replay()[0]
        self.assertLess(row[3],0);self.assertEqual(unpack(row[5]).size,ACTION)
        self.assertEqual(self.deep.summary()['combat_episodes'],0)

    def test_missing_frame_cannot_authorize_a_neural_action(self):
        self.deep.observe(None,{})
        with self.assertRaises(RuntimeError):self.choice()

    def test_training_learns_different_actions_from_pixels_with_identical_stats(self):
        # Algorithm proof on isolated labelled terminal transitions, not live
        # performance evidence. No synthetic data is imported by the app.
        observations=[]
        for colour in ((230,20,20),(20,20,230)):
            self.deep.previous_pixels=None
            self.deep.observe(Image.new('RGB',(160,90),colour),{'vp':1e8,'gold':1e9})
            observations.append(self.deep.observation.copy())
        choices=[decision('attack_target','identical_context',a) for a in ('near','far')]
        with self.db:
            for i,obs in enumerate(observations):
                for j,c in enumerate(choices):
                    for n in range(8):self.deep.insert(f'{i}:{j}:{n}','unit-test','combat',pack(obs),pack(action_vector(c)),2 if i==j else -2,None,None,.97,True,True)
        for _ in range(350):self.deep.train_step(32)
        for i,obs in enumerate(observations):
            x=np.stack([np.concatenate((obs,action_vector(c))) for c in choices])
            q=self.deep.online.forward(x)
            self.assertEqual(int(np.argmax(q)),i)
            self.assertGreater(q[i]-q[1-i],1.)

    def test_old_bandit_records_are_preserved_but_not_invented_as_visual_experiences(self):
        with tempfile.TemporaryDirectory() as d:
            learner=Learner(Path(d)/'learning.db',neural=True)
            learner.record('old','scout',1,{}, {},'legacy record')
            self.assertEqual(learner.summary()['updates'],1)
            self.assertEqual(learner.summary()['reinforcement']['transitions'],0)
            learner.close()


class LiveAdapterTests(unittest.TestCase):
    def test_normal_play_also_uses_learned_troop_defense_with_protected_reserves(self):
        import test_focus_defense_v24 as focus_tests
        b,job=focus_tests.FocusDefenseTests().fixture()
        with tempfile.TemporaryDirectory() as d:
            db=sqlite3.connect(':memory:');deep=DeepPolicy(db,Path(d))
            deep.observe(Image.new('RGB',(160,90),'blue'),{})
            b.config={'focus_defense':False,'defense_enabled':True}
            b.reinforcement_policy=lambda:deep
            with patch('live_lords.focus_defense.time.time',return_value=100):b.prepare_focus_defense(job)
            self.assertEqual(b.counts['supports'],1)
            self.assertGreaterEqual(job['troop_goal'],150)
            self.assertTrue(job['focus_rl_choice']['_rl_id'])
            options=unpack(db.execute('SELECT candidates FROM rl_decisions LIMIT 1').fetchone()[0]).reshape(-1,ACTION)
            self.assertEqual(len(options),6)
            db.close()

    def test_pending_or_incoming_work_preempts_policy_and_training(self):
        b=Bot.__new__(Bot);b.pending={'arrival':time.time()+30};b.muster=None
        b.defense_busy=Mock(return_value=False);b.incoming=Mock(return_value=False)
        deep=Mock();b.reinforcement_policy=Mock(return_value=deep)
        b.reinforcement_progress();deep.order.assert_not_called()
        b.config={};b.publish_state=Mock();b.reinforcement_tick();deep.train_step.assert_not_called()

    def test_strategy_skill_is_executed_and_recorded_as_time_cost_without_fake_victory(self):
        with tempfile.TemporaryDirectory() as d:
            db=sqlite3.connect(':memory:');deep=DeepPolicy(db,Path(d))
            b=Bot.__new__(Bot);b.learner=Mock();b.learner.deep=deep;b.outcome_policy=OutcomePolicy(db,deep)
            b.config={'autonomous':True,'attack_enabled':True,'gold_reserve':1000}
            b.state={'owner':'Yumi','gold':0};b.image=Image.new('RGB',(160,90),'blue')
            b.pending=b.muster=None;b.defenses=[];b.active_defense=None
            deep.observer=b.reinforcement_observation
            b.incoming=Mock(return_value=False);b.defense_busy=Mock(return_value=False)
            b.map=Mock();b.decide=Mock();b.explore=Mock();b.targets=Mock();b.targets.blocked.return_value=False
            b.strategy=Mock();b.strategy.state={'investments':[]}
            b.vision=Mock();b.vision.own_sources.return_value=[];b.vision.enemy_candidates.return_value=[]
            b.reinforcement_progress();b.explore.assert_called_once()
            self.assertEqual(deep.summary()['sources'],{'navigation':1})
            self.assertEqual(deep.summary()['combat_episodes'],0)
            db.close()

    def test_investment_retention_is_not_a_combat_win_and_later_loss_is_corrected(self):
        with tempfile.TemporaryDirectory() as d:
            learner=Learner(Path(d)/'learning.db',neural=True)
            deep=learner.deep;deep.observe(Image.new('RGB',(160,90),'blue'),{})
            deep.root_choice=deep.choose('strategy','map',['invest'])
            ledger=StrategyLedger(Path(d)/'strategy.json',learner)
            entry=ledger.invest('Core',100,'upgrade_safety:known',{'gold':10000})
            entry['status']='pending'
            ledger.outcome('Core',True,'ownership verified')
            self.assertEqual(deep.summary()['combat_episodes'],0)
            first=learner.db.execute('SELECT reward FROM rl_replay').fetchone()[0]
            self.assertLess(first,0)
            ledger.outcome('Core',False,'new owner verified')
            second=learner.db.execute('SELECT reward FROM rl_replay').fetchone()[0]
            rebuilds=deep.summary()['rebuilds']
            learner.close()
            self.assertLess(second,first);self.assertEqual(rebuilds,1)


if __name__=='__main__':unittest.main()
