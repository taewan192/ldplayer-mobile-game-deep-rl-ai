import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
from PIL import Image

from live_lords.exploration import ExplorationMemory
from live_lords.runner import Bot
from live_lords.vision import Vision
import test_live_activity


class ExplorationTests(unittest.TestCase):
    def test_region_change_does_not_interrupt_existing_attack_muster_or_defense(self):
        for job in ('pending','muster','defense'):
            obj=test_live_activity.ActivityTests().actor();obj.tap=Mock()
            if job=='defense':obj.defense_busy.return_value=True
            else:setattr(obj,job,{'status':'waiting'})
            self.assertFalse(obj.next_region());obj.tap.assert_not_called();obj.driver.pan_map.assert_not_called()

    def test_perspective_sized_own_flags_preserve_large_army_and_donors(self):
        with tempfile.TemporaryDirectory() as directory:
            image=Image.new('RGB',(1280,720))
            with Image.open(Path(__file__).with_name('fixtures')/'own-flags-depth.png') as flags:
                image.paste(flags,(455,195))
            v=Vision(None,Path(directory));v.regions=Mock(side_effect=[
                {'troops':'2.93G'},{'troops':'2.83G'},{'troops':'3.94G'}])
            sources=v.own_sources(image,'Yumi')
            self.assertEqual(len(sources),3)
            self.assertEqual(max(n for _,n in sources),3.94e9)
            self.assertEqual(sum(n for _,n in sources),9.7e9)

    def fixture(self):
        with Image.open(Path(__file__).with_name('fixtures')/'city-interaction-map.png') as image:
            return image.convert('RGB')

    def test_actual_round_bubbles_are_filtered_without_hiding_unshielded_castles(self):
        with tempfile.TemporaryDirectory() as directory:
            v=Vision(None,Path(directory));image=self.fixture()
            self.assertTrue(v.is_shielded(image,(974,270)))
            self.assertTrue(v.is_shielded(image,(1036,383)))
            self.assertFalse(v.is_shielded(image,(373,269)))
            self.assertFalse(v.is_shielded(image,(915,536)))
            raw=v.enemy_candidates(image,True);safe=v.enemy_candidates(image)
            self.assertIn((974,270),raw);self.assertNotIn((974,270),safe)
            self.assertIn((373,269),safe);self.assertEqual(v.last_shield_exclusions,2)

    def test_shield_detection_tracks_real_bubble_after_pan_and_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            v=Vision(None,Path(directory));image=np.array(self.fixture())
            for scale,dx,dy in [(1,40,30),(.9,30,20),(1.1,-180,-30)]:
                matrix=np.float32([[scale,0,dx],[0,scale,dy]])
                moved=Image.fromarray(cv2.warpAffine(image,matrix,(1280,720)))
                self.assertTrue(v.is_shielded(moved,(974*scale+dx,270*scale+dy)))

    def shield_actor(self,directory):
        b=Bot.__new__(Bot);b.vision=Vision(None,Path(directory));b.image=self.fixture()
        b.map=Mock();b.tap=Mock();b.decide=Mock();b.counts={};b.select_owned=Mock()
        return b

    def test_shielded_target_is_never_clicked_even_for_forced_rescout(self):
        with tempfile.TemporaryDirectory() as directory:
            for force in (False,True):
                b=self.shield_actor(directory)
                self.assertIsNone(b.scout_target((974,270),force=force))
                b.tap.assert_not_called();self.assertEqual(b.last_scout_failure,'shielded')

    def test_attack_rechecks_bubble_before_selecting_source_or_target(self):
        with tempfile.TemporaryDirectory() as directory:
            b=self.shield_actor(directory)
            self.assertFalse(b.dispatch_attack((974,270),(640,319),100,10,35,{}))
            b.tap.assert_not_called();b.select_owned.assert_not_called()

    def test_page_first_route_covers_all_slots_and_persists_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            memory=ExplorationMemory(Path(directory))
            first=[memory.reserve_anchor(9) for _ in range(3)]
            self.assertEqual(first,[(2,0),(3,0),(4,0)])
            memory=ExplorationMemory(Path(directory))
            self.assertEqual(memory.reserve_anchor(9),(5,0))
            remaining=[memory.reserve_anchor(9) for _ in range(41)]
            self.assertEqual(len(set(first+[(5,0)]+remaining)),45)

    def test_return_to_city_does_not_restart_the_same_pan_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            obj=test_live_activity.ActivityTests().actor();obj.data=Path(directory)
            obj.explore('scan');first=obj.driver.pan_map.call_args.args
            obj.exploration_step=0;obj.explore('after return');second=obj.driver.pan_map.call_args.args
            self.assertNotEqual(first,second)
            restored=ExplorationMemory(Path(directory))
            self.assertEqual(restored.state['direction'],2)

    def test_recent_identical_region_is_skipped_but_translated_region_is_new(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);memory=ExplorationMemory(path);image=self.fixture();v=Vision(None,path)
            memory.remember('City',image);restored=ExplorationMemory(path)
            self.assertTrue(restored.recently_seen(image,v))
            moved=Image.new('RGB',image.size);moved.paste(image,(220,-80))
            self.assertFalse(restored.recently_seen(moved,v))

    def test_page_parser_uses_exact_alternate_when_slash_is_misread_as_one(self):
        obj=test_live_activity.ActivityTests().actor();obj.region=Mock(return_value={'page':'119','page_wide':'1/9'})
        self.assertEqual(obj.city_list_page(),(1,9))
        obj.region.return_value={'page':'119','page_wide':'unknown'}
        self.assertIsNone(obj.city_list_page())
