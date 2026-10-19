import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
from PIL import Image

from live_lords.defense import DefenseMixin
from live_lords.driver import LDPlayer
from live_lords.vision import Vision


FIXTURES=Path(__file__).parent/'fixtures'


def frame(name):
    with Image.open(FIXTURES/name) as source:return source.convert('RGB')


class DefenseTouchTests(unittest.TestCase):
    def test_legacy_scroll_touch_matches_reported_red_circle(self):
        im=np.asarray(frame('mosdynin-reported-touch-v25.png'))
        with Image.open(Path('live_lords/assets/march_pin.png')) as source:template=np.asarray(source.convert('RGB'))
        template=cv2.resize(template,None,fx=.75,fy=.75)
        _,score,_,loc=cv2.minMaxLoc(cv2.matchTemplate(im,template,cv2.TM_CCOEFF_NORMED))
        self.assertGreater(score,.97)
        pin=(loc[0]+template.shape[1]/2,loc[1]+template.shape[0]/2)
        offset=(650*.75-pin[0],148*.75-pin[1])
        old_start=(592-offset[0],112-offset[1])
        self.assertTrue(400<old_start[0]<490 and 50<old_start[1]<115)

    def test_short_list_needs_no_scroll_but_full_list_still_does(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            self.assertFalse(v.march_list_scrollable(frame('defense-captain-iqi.png')))
            self.assertTrue(v.march_list_scrollable(frame('incoming-defense-list.png')))

    def test_production_short_list_opening_never_emits_reset_gestures(self):
        with tempfile.TemporaryDirectory() as folder:
            b=DefenseMixin();b.data=Path(folder);b.vision=Vision(None,folder)
            b.image=frame('incoming-shield-map.png');listing=frame('defense-captain-iqi.png')
            b.tap=Mock(side_effect=lambda *a,**k:setattr(b,'image',listing))
            b.driver=Mock();b.wait=Mock();b.capture=Mock();b.back=Mock()
            b.vision.defense_rows=Mock(return_value=[{'city':'Iqi','attacker':'Captain','troops':1e11,'remaining':43}])
            rows=b.inspect_defense_rows()
            self.assertEqual(rows[0]['city'],'Iqi')
            b.driver.scroll_marches.assert_not_called()
            self.assertEqual([c.args for c in b.tap.call_args_list],[(1228,571),(1036,57)])

    def test_old_page_marker_cannot_scroll_a_now_short_list(self):
        with tempfile.TemporaryDirectory() as folder:
            b=DefenseMixin();b.data=Path(folder);b.vision=Vision(None,folder)
            b.image=frame('defense-captain-iqi.png');b.defense_read_page=2
            b.tap=Mock();b.driver=Mock();b.wait=Mock();b.capture=Mock();b.back=Mock()
            b.vision.defense_rows=Mock(return_value=[{'city':'Iqi','attacker':'Captain','troops':1e11,'remaining':43}])
            b.inspect_defense_rows(reuse=True)
            b.driver.scroll_marches.assert_not_called();b.tap.assert_not_called()

    def driver(self,folder):
        d=LDPlayer.__new__(LDPlayer);d.data=Path(folder);d.index=21;d.resolution=(960,540)
        d.last_capture=time.monotonic();d.last_image=frame('defense-captain-iqi.png');d.capture_id=1
        d.run=Mock(return_value=b'');d.foreground=Mock(return_value=True)
        return d

    def test_android_dispatch_blocks_legacy_swipe_before_sending_it(self):
        with tempfile.TemporaryDirectory() as folder:
            d=self.driver(folder)
            with self.assertRaisesRegex(RuntimeError,'스와이프 차단'):
                d.shell('input','swipe',592,112,592,472,220)
            d.run.assert_not_called()
            entry=json.loads((d.data/'input-audit.jsonl').read_text('utf-8'))
            self.assertTrue(entry['blocked']);self.assertEqual(entry['kind'],'swipe')

    def test_actual_scroll_gesture_stays_in_blank_column(self):
        with tempfile.TemporaryDirectory() as folder:
            d=self.driver(folder);d.scroll_marches(to_top=True)
            self.assertEqual(d.run.call_args.args[3:],('input','swipe',315,146,315,469,300))
            entry=json.loads((d.data/'input-audit.jsonl').read_text('utf-8'))
            self.assertFalse(entry['blocked'])
            self.assertTrue(all(abs(point[0]-420)<1 for point in entry['normalized']))

    def test_guard_also_catches_a_swipe_crossing_troop_column(self):
        with tempfile.TemporaryDirectory() as folder:
            d=self.driver(folder)
            with self.assertRaises(RuntimeError):d.shell('input','swipe',480,260,675,260,450)
            d.run.assert_not_called()

    def test_location_pin_and_unrelated_profile_click_remain_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            d=self.driver(folder);d.tap(650,148)
            self.assertEqual(d.run.call_count,1)
            d.last_image=frame('incoming-mixed-map-v21.png');d.tap(728,120)
            self.assertEqual(d.run.call_count,2)


if __name__=='__main__':unittest.main()
