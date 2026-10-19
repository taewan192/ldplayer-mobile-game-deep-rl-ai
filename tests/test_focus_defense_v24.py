import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from live_lords.app import App
from live_lords.focus_defense import FocusDefenseMixin,incoming_troops,protected_level,troop_goal
from live_lords.runner import Bot,Stopped


class FocusDefenseTests(unittest.TestCase):
    def test_hidden_range_is_troops_not_fictitious_enemy_attack_power(self):
        row={'troops_min':1e9,'troops':1e10,'troops_exact':False}
        self.assertEqual(incoming_troops(row),9.99e9)
        self.assertEqual(troop_goal([row],110),14.985e9)
        self.assertEqual(troop_goal([row],109),12.987e9)
        self.assertEqual(incoming_troops({'troops':4e9,'troops_exact':True}),4e9)
        with self.assertRaises(ValueError):incoming_troops({'troops_min':10,'troops':1})

    def test_critical_castles_precede_other_destinations_even_if_others_arrive_first(self):
        b=FocusDefenseMixin();b.strategy=Mock();b.strategy.state={'cities':{}}
        normal=[{'city':'Small','level':109,'impact':110}]
        core=[{'city':'Core','level':115,'impact':200}]
        self.assertLess(b.defense_group_priority(core),b.defense_group_priority(normal))
        self.assertTrue(protected_level(110));self.assertTrue(protected_level(115));self.assertFalse(protected_level(116))

    def fixture(self,donor_level=90):
        b=FocusDefenseMixin();b.config={'focus_defense':True};b.state={'owner':'Yumi'};b.image=Mock()
        b.vision=Mock();b.vision.own_sources.return_value=[((800,330),1000)]
        b.map=Mock();b.back=Mock();b.rebase_defense=Mock();b.save_defenses=Mock()
        b.ensure_defense_episode=Mock();b.publish_state=Mock();b.log=Mock();b.emit=Mock();b.tap=Mock()
        b.read_owned_details=Mock(return_value={'name':'Donor','troops':1000,'level':donor_level})
        b.threatened_names={'target'};b.counts={'supports':0};b.defense_snapshot_valid=Mock(return_value=True)
        b.prepare_transfer=Mock(return_value={'donor':{'name':'Donor','troops':1000},'troops':100,
            'duration':5,'observed':100,'button':(880,540)})
        job={'id':'wave','staging_name':'Target','staging':(640,330),'impact':200,'before':{'troops':100,'level':110},
            'incoming_rows':[{'troops':100,'troops_exact':True}],'moves':[],'status':'preparing'}
        b.defenses=[job]
        return b,job

    def test_support_intent_precedes_click_and_never_awards_a_win(self):
        b,job=self.fixture()
        def click(*args,**kwargs):self.assertEqual(job['moves'][0]['status'],'intent')
        b.tap.side_effect=click
        with patch('live_lords.focus_defense.time.time',return_value=100):b.prepare_focus_defense(job)
        self.assertEqual(job['attack'],None);self.assertEqual(job['threat_basis'],'troops_only')
        self.assertEqual(job['moves'][0]['status'],'sent');self.assertEqual(b.counts['supports'],1)
        self.assertEqual(job['status'],'waiting');self.assertIsNone(b.active_defense)

    def test_protected_donor_keeps_ninety_percent(self):
        b,job=self.fixture(112)
        with patch('live_lords.focus_defense.time.time',return_value=100):b.prepare_focus_defense(job)
        self.assertEqual(job['support_fraction'],.1)
        self.assertEqual(job['moves'][0]['reserve_fraction'],.9)

    def test_protected_donor_over_cap_or_wrong_source_is_never_sent(self):
        for change in ({'troops':200},{'donor':{'name':'Other','troops':1000}}):
            b,job=self.fixture(112);b.prepare_transfer.return_value.update(change)
            with patch('live_lords.focus_defense.time.time',return_value=100):b.prepare_focus_defense(job)
            b.tap.assert_not_called();self.assertEqual(job['moves'],[])

    def test_late_arrival_and_changed_wave_never_send(self):
        for change in ('late','wave'):
            b,job=self.fixture()
            if change=='late':b.prepare_transfer.return_value['duration']=96
            else:b.defense_snapshot_valid.return_value=False
            with patch('live_lords.focus_defense.time.time',return_value=100):b.prepare_focus_defense(job)
            b.tap.assert_not_called();self.assertEqual(job['moves'],[])

    def test_already_threatened_donor_cannot_send_even_with_large_army(self):
        b,job=self.fixture();b.threatened_names.add('donor')
        with patch('live_lords.focus_defense.time.time',return_value=100):b.prepare_focus_defense(job)
        b.prepare_transfer.assert_not_called();b.tap.assert_not_called()

    def test_uncertain_existing_move_is_reserved_and_never_replayed(self):
        b,job=self.fixture();job['moves']=[{'source':'Donor','troops':100,'arrival':120,'status':'intent'}]
        with patch('live_lords.focus_defense.time.time',return_value=100):b.prepare_focus_defense(job)
        b.prepare_transfer.assert_not_called();b.tap.assert_not_called()

    def test_invalid_incoming_range_does_not_stop_monitor_or_leave_active_send(self):
        b,job=self.fixture();job['incoming_rows']=[{'troops':0}];b.active_defense=None
        b.prepare_focus_defense(job)
        self.assertEqual(job['status'],'unverified');self.assertIsNone(b.active_defense)
        b.tap.assert_not_called()

    def test_incomplete_capacity_sends_timely_partial_support_and_alerts_shortfall(self):
        b,job=self.fixture();job['incoming_rows']=[{'troops':1000,'troops_exact':True}]
        with patch('live_lords.focus_defense.time.time',return_value=100):b.prepare_focus_defense(job)
        self.assertEqual(b.counts['supports'],1);self.assertGreater(job['needed'],0)
        self.assertTrue(any('alert' in c.args[0] for c in b.emit.call_args_list))

    def test_focus_guards_block_attack_scout_and_offensive_transfer_entry_points(self):
        b=Bot.__new__(Bot);b.config={'focus_defense':True};b.tap=Mock();b.select_owned=Mock()
        self.assertFalse(b.dispatch_attack((400,300),(500,300),100,10,30,{}))
        self.assertIsNone(b.scout_target((400,300)));self.assertEqual(b.attack_screen_batch(),'defense')
        self.assertIsNone(b.prepare_transfer((500,300),100,{},Mock()))
        b.tap.assert_not_called();b.select_owned.assert_not_called()

    def test_focus_loop_never_enters_normal_exploration_or_offensive_work(self):
        b=Bot.__new__(Bot);b.config={'focus_defense':True};b.stop=threading.Event();b.capture=Mock()
        b.focus_defense_tick=Mock(side_effect=Stopped());b.progress_cycle=Mock();b.sync_stats=Mock()
        b.log=Mock();b.emit=Mock();b.learner=Mock();b.vision=Mock()
        b.run();b.focus_defense_tick.assert_called_once();b.progress_cycle.assert_not_called();b.sync_stats.assert_not_called()

    def test_focus_ui_overrides_autonomy_and_accepts_fractional_quarter_second(self):
        with patch.object(App,'refresh',lambda _:None):
            app=App();app.withdraw()
            try:
                app.rows={'test':['21','Yumi','0','0','1']};app.instance.set('test')
                app.focus_defense.set(True);app.autonomy_changed();app.interval.set(.25)
                config,_=app.config()
                self.assertTrue(config['focus_defense']);self.assertTrue(config['defense_enabled'])
                self.assertFalse(config['attack_enabled']);self.assertFalse(config['muster_enabled']);self.assertFalse(config['upgrade_enabled'])
                self.assertEqual(config['interval'],.25)
                app.interval.set(.2)
                with self.assertRaises(ValueError):app.config()
            finally:app.destroy()

    def test_hot_mode_and_quarter_second_speed_update_the_running_shared_config(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(App,'refresh',lambda _:None):
            app=App();app.withdraw()
            try:
                app.active_directory=Path(folder);app.active_config={'interval':1};shared=app.active_config
                app.focus_defense.set(True);app.focus_changed();app.interval.set(.25);app.apply_speed_settings()
                self.assertIs(shared,app.active_config);self.assertTrue(shared['focus_defense'])
                self.assertFalse(shared['attack_enabled']);self.assertEqual(shared['interval'],.25)
                saved=json.loads((Path(folder)/'config.json').read_text('utf-8'))
                self.assertTrue(saved['focus_defense']);self.assertEqual(saved['interval'],.25)
            finally:app.destroy()

    def test_manual_preferences_return_when_focus_mode_is_disabled(self):
        with patch.object(App,'refresh',lambda _:None):
            app=App();app.withdraw()
            try:
                app.autonomous.set(False);app.attack.set(True);app.muster_enabled.set(False)
                app.focus_defense.set(True);app.focus_changed();self.assertFalse(app.attack.get())
                app.focus_defense.set(False);app.focus_changed()
                self.assertTrue(app.attack.get());self.assertFalse(app.muster_enabled.get())
            finally:app.destroy()


if __name__=='__main__':unittest.main()
