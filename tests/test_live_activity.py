import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from live_lords.activity import ActivityMixin
from live_lords.strategy import StrategyLedger
from live_lords.vision import Vision,amount
from live_lords.runner import Bot


class ActivityTests(unittest.TestCase):
    def test_attack_rechecks_real_source_army_instead_of_trusting_map_ocr(self):
        obj=Bot.__new__(Bot); obj.config={'max_send_fraction':.35}; obj.muster=None
        obj.image=Mock(); obj.map=Mock(); obj.log=Mock(); obj.tap=Mock(); obj.select_troops=Mock()
        obj.select_owned=Mock(return_value={'troops':78.1e6})
        obj.vision=Mock(); obj.vision.transform_point.side_effect=lambda matrix,point:point
        obj.vision.is_shielded.return_value=False
        self.assertFalse(obj.dispatch_attack((900,330),(640,330),4e12,250e6,1.4e12,{}))
        obj.tap.assert_not_called(); obj.select_troops.assert_not_called()

    def test_trailing_troop_icon_cannot_create_a_fictitious_trillion_army(self):
        self.assertEqual(amount('78.1M4t'),78.1e6)
        self.assertIsNone(amount('unreadable M 4t'))
        self.assertIsNone(amount('78.1MUE'))

    def actor(self):
        obj=ActivityMixin(); obj.config={'autonomous':True,'attack_enabled':True,'wide_exploration':False}
        obj.state={}; obj.seen={}; obj.pending=None; obj.muster=None
        obj.emit=Mock(); obj.log=Mock(); obj.map=Mock(); obj.capture=Mock(); obj.wait=Mock()
        obj.driver=Mock(); obj.incoming=Mock(return_value=False); obj.defense_busy=Mock(return_value=False)
        obj.counts={'explorations':0}
        return obj

    def test_upgrade_deferred_does_not_stop_attack_check_or_next_exploration(self):
        obj=self.actor(); obj.attack_nearby=Mock(return_value='blocked'); obj.upgrade=Mock(return_value=False)
        obj.progress_cycle()
        obj.attack_nearby.assert_called_once(); obj.upgrade.assert_called_once()
        obj.driver.pan_map.assert_called_once()
        self.assertEqual(obj.state['decision']['action'],'주변 탐색')

    def test_active_attack_does_not_spend_gold_or_move_the_map(self):
        obj=self.actor(); obj.pending={'arrival':time.time()+50}
        obj.attack_nearby=Mock(return_value='dispatched'); obj.upgrade=Mock()
        obj.progress_cycle()
        obj.upgrade.assert_not_called(); obj.driver.pan_map.assert_not_called()

    def test_defense_preempts_investment_and_exploration(self):
        obj=self.actor(); obj.attack_nearby=Mock(return_value='defense'); obj.upgrade=Mock()
        obj.progress_cycle()
        obj.upgrade.assert_not_called(); obj.driver.pan_map.assert_not_called()

    def test_no_target_triggers_bounded_map_search_but_manual_mode_does_not(self):
        obj=self.actor(); obj.attack_nearby=Mock(return_value='no_targets'); obj.upgrade=Mock(return_value=False)
        obj.progress_cycle(); self.assertEqual(obj.counts['explorations'],1)
        dx,dy=obj.driver.pan_map.call_args.args
        self.assertLessEqual(abs(dx),260); self.assertLessEqual(abs(dy),180)
        obj.config['autonomous']=False; obj.driver.pan_map.reset_mock(); obj.progress_cycle()
        obj.driver.pan_map.assert_not_called()
        self.assertEqual(obj.state['decision']['action'],'감시 대기')

    def test_successful_investment_is_followed_by_result_tracking_not_exploration(self):
        obj=self.actor(); obj.attack_nearby=Mock(return_value='blocked'); obj.upgrade=Mock(return_value=True)
        obj.progress_cycle(); obj.driver.pan_map.assert_not_called()
        self.assertEqual(obj.state['decision']['action'],'투자 결과 추적')

    def test_gold_unit_g_misread_as_six_requires_matching_visual_glyph(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(Mock(),Path(folder)); image=Image.new('RGB',(1280,720))
            v.regions=Mock(return_value={'vp':'106M','gold':'3.956'}); v.match=Mock(return_value=(96,197,1))
            self.assertEqual(v.hud(image)['gold'],3.95e9)
            v.match.return_value=None
            self.assertIsNone(v.hud(image)['gold'])

    def test_unknown_threat_allows_only_one_small_core_trial_and_not_city_averaging(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=StrategyLedger(Path(folder)/'journal.json',Mock())
            city={'name':'Core','defense':478e6,'troops':43.8e6,'walls':261e6}
            self.assertTrue(ledger.investment_allowed(city,37.3e6,3.95e9,500e6)[0])
            self.assertFalse(ledger.investment_allowed(city,44.7e6,3.95e9,500e6)[0])
            entry=ledger.invest('Core',37.3e6,'upgrade_safety:unmeasured',{'gold':3.95e9})
            entry['status']='pending'
            self.assertFalse(ledger.investment_allowed(city,37.3e6,3.95e9,500e6)[0])
            ledger.outcome('Core',True,'held')
            other={**city,'name':'Other'}
            self.assertFalse(ledger.investment_allowed(other,1,3.95e9,500e6)[0])
            self.assertTrue(ledger.investment_allowed(city,37.3e6,3.95e9,500e6)[0])

    def test_lost_trial_city_cannot_be_retried_even_without_an_enemy_power_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=StrategyLedger(Path(folder)/'journal.json',Mock())
            city={'name':'Core','defense':478e6,'troops':43.8e6,'walls':261e6}
            entry=ledger.invest('Core',37.3e6,'upgrade_safety:unmeasured',{'gold':3.95e9}); entry['status']='pending'
            ledger.outcome('Core',False,'actual city loss')
            self.assertFalse(ledger.investment_allowed(city,1,3.95e9,500e6)[0])


if __name__=='__main__': unittest.main()
