import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from live_lords.runner import Bot,Stopped
from live_lords.vision import Vision,percentage


class StatsV23Tests(unittest.TestCase):
    def test_terminal_percent_glyph_does_not_multiply_changed_attack_by_ten(self):
        for text,expected in [('+930/6',93),('+930/0',93),('+93 0/6',93),
                              ('+930%',930),('+9300/6',930),('+144.250/6',144.25),('+363%',363)]:
            self.assertEqual(percentage(text),expected,text)
        for text in ('+93/6','+930/8','930','+93% other','1G-10G'):
            self.assertIsNone(percentage(text),text)

    def vision(self,folder,readings):
        v=Vision(None,folder);v.match=Mock(side_effect=[(246,349,1),(246,237,1)])
        v.regions=Mock(side_effect=readings)
        return v

    def test_corrupt_percent_requires_independent_crop_agreement(self):
        with tempfile.TemporaryDirectory() as folder:
            v=self.vision(folder,[{'attack':'+930/6','defense':'+363%'},{'attack_verify':'+930/0'}])
            self.assertEqual(v.combat_bonuses(None),{'attack':93,'defense':363})
            self.assertNotEqual(v.regions.call_args_list[0].args[1]['attack'],v.regions.call_args_list[1].args[1]['attack_verify'])

    def test_conflicting_glyph_read_never_turns_a_possible_930_into_93(self):
        with tempfile.TemporaryDirectory() as folder:
            v=self.vision(folder,[{'attack':'+930/6','defense':'+363%'},{'attack_verify':'+930%'}])
            with self.assertRaisesRegex(RuntimeError,'불일치'):v.combat_bonuses(None)

    def bot(self):
        b=Bot.__new__(Bot);b.stop=threading.Event();b.config={};b.image=Mock();b.vision=Mock()
        b.state={'owner':'Yumi','attack_bonus':280,'defense_bonus':105};b.last_stats_update=10
        b.map=Mock();b.tap=Mock();b.capture=Mock();b.wait=Mock();b.log=Mock();b.publish_state=Mock()
        b.region=Mock(side_effect=lambda boxes: {'title':'플레이어 정보' if boxes['title'][0]==208 else '능력치 개요'}
            if 'title' in boxes else {'name':'Yumi'})
        return b

    def test_same_panel_retry_updates_both_stats_and_publishes_real_change(self):
        b=self.bot();b.vision.combat_bonuses.side_effect=[RuntimeError('transient'),{'attack':93,'defense':363}]
        with patch('live_lords.runner.time.time',return_value=100):b.sync_stats()
        self.assertEqual((b.state['attack_bonus'],b.state['defense_bonus']),(93,363))
        self.assertEqual([c.args for c in b.tap.call_args_list],[(70,55),(728,120),(424,545)])
        b.capture.assert_called_once();b.publish_state.assert_called_once()
        self.assertEqual(b.last_stats_update,100)
        self.assertTrue(any('스탯 변경 반영' in c.args[0] for c in b.log.call_args_list))

    def test_failed_reads_never_partially_replace_old_bonus_or_freshness(self):
        b=self.bot();b.vision.combat_bonuses.side_effect=RuntimeError('unreadable')
        with self.assertRaises(RuntimeError):b.sync_stats()
        self.assertEqual((b.state['attack_bonus'],b.state['defense_bonus']),(280,105))
        self.assertEqual(b.last_stats_update,10);b.publish_state.assert_not_called()
        self.assertEqual(b.vision.combat_bonuses.call_count,3)

    def test_initialized_loop_refreshes_old_stats_before_next_offensive_work(self):
        b=self.bot();b.stop=threading.Event();b.capture=Mock();b.hud=Mock();b.log=Mock();b.emit=Mock()
        b.config={'defense_enabled':False};b.learner=Mock();b.muster=None;b.pending=None
        b.check_defense_results=Mock();b.incoming=Mock(return_value=False)
        b.restore_outcome_attack=Mock();b.check_pending=Mock(return_value=False);b.check_muster=Mock(return_value=False)
        b.audit_investments=Mock();b.audit_learned_holdings=Mock();b.last_recruit_check=100
        b.progress_cycle=Mock();b.cycle_delay=Mock(return_value=1)
        def initial_then_due():
            if b.sync_stats.call_count==1:b.last_stats_update=0
            else:raise Stopped()
        b.sync_stats=Mock(side_effect=initial_then_due)
        with patch('live_lords.runner.time.time',return_value=100):b.run()
        self.assertEqual(b.sync_stats.call_count,2);b.progress_cycle.assert_not_called()


if __name__=='__main__':unittest.main()
