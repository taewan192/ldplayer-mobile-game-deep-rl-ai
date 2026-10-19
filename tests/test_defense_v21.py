import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from PIL import Image,ImageDraw

from live_lords.defense import DefenseMixin
from live_lords.vision import Vision,march_countdown
from live_lords.driver import GameUnavailable,EmulatorBridgeUnavailable


FIXTURES=Path(__file__).parent/'fixtures'


class DefenseV21Tests(unittest.TestCase):
    def test_no_troop_column_click_even_when_ocr_returns_exact_or_missing_range_flag(self):
        for exact in (True,False,None):
            b=DefenseMixin();b.tap=Mock();b.vision=Mock();b.image=Mock()
            b.vision.labeled_attack_power.return_value=None
            row={'city':'Aunei','attacker':'Raja Humabon','row':148}
            if exact is not None:row['troops_exact']=exact
            self.assertIsNone(b.inspect_attack_power(row))
            b.tap.assert_not_called()
            self.assertEqual(b.vision.labeled_attack_power.call_count,int(exact is True))

    def test_mixed_shortcut_requires_incoming_red_badge(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(Mock(),folder);im=Image.open(FIXTURES/'incoming-mixed-map-v21.png').convert('RGB')
            self.assertIsNotNone(v.march_button(im));self.assertIsNotNone(v.incoming_button(im))
            ImageDraw.Draw(im).rectangle((1250,525,1278,553),fill='#293c50')
            self.assertIsNotNone(v.march_button(im));self.assertIsNone(v.incoming_button(im))

    def test_aunei_missing_leading_digit_is_reread_and_component_total_still_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(Mock(),folder);im=Image.open(FIXTURES/'own-aunei-v21.png')
            fields={'city_name':'Aunei','defense':'39.4G','troops':'.78G','guardian':'105%',
                    'city_bonus':'330%','walls':'29.9G','level':'110','city_vp':'2.84M','cost':'4.27G'}
            v.regions=Mock(side_effect=[fields,{'defense_troops':'1.78G'}])
            city=v.own_city_details(im,'Aunei')
            self.assertEqual(city['troops'],1.78e9)
            v.regions=Mock(side_effect=[dict(fields,troops='.78G'),{'defense_troops':'7.8G'}])
            self.assertIsNone(v.own_city_details(im,'Aunei'))

    def test_wrong_city_is_never_relabelled_as_attacked_city(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(Mock(),folder);im=Image.open(FIXTURES/'own-aunei-v21.png')
            v.regions=Mock(side_effect=[{'city_name':'Darayol'},
                {'short_name':'Darayol','long_name':'Darayol'}])
            self.assertIsNone(v.own_city_details(im,'Aunei'))

    def focus_fixture(self):
        b=DefenseMixin();b.image=Mock();b.vision=Mock();b.tap=Mock();b.wait=Mock();b.capture=Mock();b.log=Mock()
        b.vision.march_panel.side_effect=['defense',None,None,None]
        b.vision.defense_tab_selected.return_value=True
        b.vision.match.side_effect=[(650,148,1),(50,660,1),(50,660,1),(50,660,1)]
        return b,{'city':'Aunei','pin':(650,148)}

    def test_camera_must_arrive_and_settle_before_selecting_city(self):
        b,row=self.focus_fixture();b.vision.same_viewport.side_effect=[False,True]
        self.assertTrue(b.focus_defense_city(row))
        b.tap.assert_called_once_with(650,148,delay=1.2)
        self.assertEqual(b.capture.call_count,2)

    def test_missing_or_displaced_pin_never_falls_back_to_troop_column(self):
        for match in (None,(731,148,.99)):
            b,row=self.focus_fixture();b.vision.match.side_effect=None;b.vision.match.return_value=match
            self.assertFalse(b.focus_defense_city(row));b.tap.assert_not_called()

    def test_failed_city_focus_is_bounded_per_wave_but_new_wave_can_proceed(self):
        b=DefenseMixin();group=[{'city':'Aunei','attacker':'Raja','troops':1e10}]
        for now in (100,110,130):
            with patch('live_lords.defense.time.time',return_value=now):
                self.assertTrue(b.defense_focus_retry(group,500))
                b.defense_focus_retry(group,500,failed=True)
                self.assertFalse(b.defense_focus_retry(group,500))
        with patch('live_lords.defense.time.time',return_value=200):
            self.assertFalse(b.defense_focus_retry(group,500))
            self.assertTrue(b.defense_focus_retry(group,550))

    def test_live_clock_nine_confusion_only_in_seconds_slot(self):
        self.assertEqual(march_countdown('3m 1 gs'),199)
        for text in ('3g19s','3m6gs','1G-10G','3m g1s'):
            self.assertIsNone(march_countdown(text),text)

    def test_game_loss_inside_defense_read_reaches_outer_recovery(self):
        b=DefenseMixin();b.config={};b.last_defense_poll=0;b.map=Mock();b.image=Mock();b.defer_defense_read=Mock()
        b.inspect_defense_rows=Mock(side_effect=GameUnavailable('game closed'))
        with self.assertRaises(GameUnavailable):b.poll_defense(force=True)
        b.defer_defense_read.assert_not_called()

    def test_android_loss_inside_defense_read_reaches_outer_recovery(self):
        b=DefenseMixin();b.config={};b.last_defense_poll=0;b.map=Mock();b.image=Mock();b.defer_defense_read=Mock()
        b.inspect_defense_rows=Mock(side_effect=EmulatorBridgeUnavailable('bridge lost'))
        with self.assertRaises(EmulatorBridgeUnavailable):b.poll_defense(force=True)
        b.defer_defense_read.assert_not_called()


if __name__=='__main__':unittest.main()
