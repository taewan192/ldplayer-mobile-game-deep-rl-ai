import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from PIL import Image

from live_lords.defense import DefenseMixin
from live_lords.runner import Bot,Stopped
from live_lords.vision import Vision,march_countdown,troop_bounds


class DefenseRecoveryTests(unittest.TestCase):
    def test_reported_hidden_range_ocr_is_a_range_and_requires_whole_numeric_tokens(self):
        for raw in ('LOG-IOOG','10G–100G','10G—100G','lOG ~ lOOG'):
            self.assertEqual(troop_bounds(raw),(10e9,100e9),raw)
        for raw in ('OG-IOOG','10G-1G','LOG-IOOG Captain','gold-100G','10G-100G-1T',
                    '1G10G','10G-100G?','10G100G'):
            self.assertEqual(troop_bounds(raw),(None,None),raw)

    def test_reported_clock_confusions_are_constrained_to_complete_clocks(self):
        for raw,seconds in [('Irn18s',78),('0m 5Ss',55),('0m 438',43),('0m 278',27),('1m 30s',90)]:
            self.assertEqual(march_countdown(raw),seconds,raw)
        for raw in ('ltn 42-3','Itn18s','0m 99s','Captain 1m 30s','1m 3so'):
            self.assertIsNone(march_countdown(raw),raw)

    def test_failure_frame_retries_punctuation_city_and_preserves_hidden_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(Mock(),folder)
            im=Image.open(Path(__file__).parent/'fixtures/defense-captain-iqi.png')
            v.regions=Mock(side_effect=[{'march_city':'!','remaining':'0m 438',
                'troops':'LOG-IOOG','attacker':'Captain'},{'defense_city_tight':'Iqi'}])
            row,=v.defense_rows(im)
            self.assertEqual((row['city'],row['attacker'],row['remaining']),('Iqi','Captain',43))
            self.assertEqual((row['troops_min'],row['troops']),(10e9,100e9))
            self.assertFalse(row['troops_exact']);self.assertFalse(row['time_uncertain'])
            self.assertEqual(v.regions.call_args.args[1],{'defense_city_tight':(510,128,605,162)})

    def test_scaled_attachment_uses_pin_relative_crops_and_padded_short_name_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(Mock(),folder)
            im=Image.open(Path(__file__).parent/'fixtures/defense-captain-marked.png').resize((1280,720))
            v.regions=Mock(side_effect=[{'march_city':'','remaining':'0m 278',
                'troops':'IOG-IOOG','attacker':'Captain'},{'defense_city_tight':''},
                {'march_city_padded':'Iqi'}])
            row,=v.defense_rows(im)
            self.assertEqual((row['city'],row['remaining']),('Iqi',27))
            self.assertEqual(row['pin'],(646,146))
            self.assertEqual(v.regions.call_args.args[1],{'march_city_padded':(501,123,606,164)})

    def test_hidden_range_does_not_click_troop_column(self):
        b=DefenseMixin();b.tap=Mock();b.vision=Mock()
        self.assertIsNone(b.inspect_attack_power({'troops_exact':False,'row':148}))
        b.tap.assert_not_called();b.vision.labeled_attack_power.assert_not_called()

    def reader(self):
        b=DefenseMixin();b.image=Mock();b.vision=Mock();b.driver=Mock()
        b.tap=Mock();b.capture=Mock();b.wait=Mock();b.back=Mock();b.data=Path('.')
        b.vision.march_panel.return_value=True;b.vision.defense_tab_selected.return_value=True
        return b

    def test_retained_panel_is_read_without_reopening_tabbing_or_scrolling(self):
        b=self.reader();b.vision.defense_rows.return_value=[{'city':'Iqi','remaining':43}]
        with patch('live_lords.defense.time.time',return_value=100):rows=b.inspect_defense_rows(reuse=True)
        self.assertEqual(rows[0]['impact'],143)
        b.tap.assert_not_called();b.back.assert_not_called();b.driver.scroll_marches.assert_not_called()

    def test_retained_unreadable_panel_is_not_closed(self):
        b=self.reader();b.vision.defense_rows.return_value=[]
        with self.assertRaises(RuntimeError):b.inspect_defense_rows(reuse=True)
        b.tap.assert_not_called();b.back.assert_not_called()

    def test_later_page_retry_rescans_all_threatened_cities_from_top(self):
        b=self.reader();b.defense_read_page=1
        b.vision.defense_rows.return_value=[{'city':'Iqi','remaining':43}]
        b.inspect_defense_rows(reuse=True)
        self.assertEqual(b.driver.scroll_marches.call_count,3)
        b.tap.assert_not_called()

    def test_read_failures_back_off_without_map_return_or_log_spam(self):
        b=self.reader();b.config={'decision_speed':3};b.last_defense_poll=0
        b.defense_read_waiting=False;b.map=Mock();b.log=Mock();b.emit=Mock()
        b.inspect_defense_rows=Mock(side_effect=RuntimeError('unreadable incoming row'))
        with patch('live_lords.defense.time.time',return_value=100):self.assertTrue(b.poll_defense(force=True))
        self.assertEqual(b.defense_read_retry_at,102)
        with patch('live_lords.defense.time.time',return_value=101):self.assertTrue(b.poll_defense(force=True))
        self.assertEqual(b.inspect_defense_rows.call_count,1)
        with patch('live_lords.defense.time.time',return_value=104):self.assertTrue(b.poll_defense(force=True))
        self.assertEqual(b.defense_read_retry_at,108)
        b.map.assert_called_once();self.assertEqual(b.log.call_count,1)
        self.assertEqual(b.inspect_defense_rows.call_args.kwargs,{'reuse':True})

    def test_incoming_disappearing_clears_ocr_hold_without_turning_outgoing_into_a_threat(self):
        b=self.reader();b.config={};b.last_defense_poll=0;b.map=Mock()
        b.defenses=[];b.defense_read_waiting=True;b.defense_read_retry_at=102
        b.defense_unknown_until=112;b.inspect_defense_rows=Mock(return_value=[])
        b.vision.incoming_button.return_value=None
        with patch('live_lords.defense.time.time',return_value=104):self.assertFalse(b.poll_defense(force=True))
        self.assertFalse(b.defense_read_waiting);self.assertEqual(b.defense_unknown_until,0)

    def test_unreadable_clock_also_retains_the_panel_instead_of_returning_to_map(self):
        b=self.reader();b.config={};b.last_defense_poll=0;b.defense_unknown_until=0
        b.map=Mock();b.log=Mock();b.emit=Mock();b.muster=None;b.defenses=[]
        b.inspect_defense_rows=Mock(return_value=[{'city':'Iqi','time_uncertain':True}])
        with patch('live_lords.defense.time.time',return_value=100):self.assertTrue(b.poll_defense(force=True))
        b.map.assert_called_once();self.assertTrue(b.defense_read_waiting)
        self.assertEqual(b.defense_read_retry_at,102)

    def test_production_loop_preserves_panel_during_read_retry_and_remains_stoppable(self):
        b=Bot.__new__(Bot);b.stop=threading.Event();b.defense_read_waiting=True
        b.capture=Mock(side_effect=[None,Stopped()]);b.incoming=Mock(return_value=True)
        b.map=Mock();b.hud=Mock();b.decide=Mock();b.wait=Mock();b.log=Mock()
        b.learner=Mock();b.vision=Mock();b.emit=Mock()
        b.run()
        b.map.assert_not_called();b.hud.assert_not_called();b.wait.assert_called_once_with(2)
        b.log.assert_called_with('중지됨. 학습 기록 저장 완료')


if __name__=='__main__':unittest.main()
