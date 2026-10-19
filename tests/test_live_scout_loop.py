import random
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image,ImageDraw

from live_lords.runner import Bot
from live_lords.targets import TargetMemory
from live_lords.vision import Vision,numeric_amount
import test_live_activity


class ScoutLoopTests(unittest.TestCase):
    def test_real_camera_pan_with_six_percent_scale_change_projects_the_same_city(self):
        with tempfile.TemporaryDirectory() as directory:
            v=Vision(None,Path(directory)); fixtures=Path(__file__).with_name('fixtures')
            before=Image.open(fixtures/'camera-before.png'); after=Image.open(fixtures/'camera-after.png')
            transform=v.viewport_transform(before,after)
            self.assertIsNotNone(transform)
            x,y=v.transform_point(transform,(925,260))
            self.assertLess(abs(x-780),8); self.assertLess(abs(y-180),8)
            self.assertFalse(v.same_viewport(after,before))

    def test_completed_exploration_cycle_advances_region_instead_of_repeating_home(self):
        obj=test_live_activity.ActivityTests().actor(); obj.exploration_step=8
        obj.next_region=Mock(return_value=True)
        self.assertTrue(obj.explore('no target'))
        obj.driver.pan_map.assert_not_called()
        obj.next_region.assert_called_once()

    def test_explicit_unit_and_numeric_context_resolve_capital_i_without_unit_guessing(self):
        self.assertEqual(numeric_amount('I.09G'),1.09e9)
        self.assertEqual(numeric_amount('l.09G'),1.09e9)
        self.assertIsNone(numeric_amount('I.09'))
        self.assertIsNone(numeric_amount('I09G'))

    def test_actual_report_layout_returns_defense_and_city_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            v=Vision(None,Path(directory))
            v.regions=Mock(return_value={'title':'세부 보고서','defense':'I.09G','city_name':'Toroi','attacker':'BAGUHAN'})
            report=v.scout_report(Image.new('RGB',(1280,720)))
            self.assertEqual(report['defense'],1.09e9)
            self.assertEqual(report['city'],'Toroi')

    def terrain(self):
        im=Image.new('RGB',(1280,720),'#ba8235'); draw=ImageDraw.Draw(im); rng=random.Random(91)
        for _ in range(180):
            x,y=rng.randrange(220,1050),rng.randrange(170,550)
            draw.rectangle((x,y,x+12,y+12),fill=tuple(rng.randrange(256) for _ in range(3)))
        return im

    def test_same_city_cooldown_survives_actual_map_translation_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); vision=Vision(None,path); reference=self.terrain()
            memory=TargetMemory(path); memory.add(reference,(640,330),'baguhan:toroi')
            moved=Image.new('RGB',reference.size); moved.paste(reference,(80,-40))
            restored=TargetMemory(path)
            self.assertTrue(restored.blocked(moved,(720,290),vision))
            self.assertFalse(restored.blocked(moved,(900,400),vision))
            self.assertTrue(restored.identity_blocked('baguhan:toroi'))

    def test_cooldown_expires_instead_of_permanently_excluding_a_city(self):
        with tempfile.TemporaryDirectory() as directory:
            memory=TargetMemory(Path(directory)); vision=Mock()
            memory.add(Image.new('RGB',(1280,720)),(640,330),'x:y',seconds=-1)
            self.assertFalse(memory.identity_blocked('x:y'))
            self.assertFalse(memory.blocked(None,(640,330),vision))
            vision.viewport_transform.assert_not_called()

    def test_report_ocr_failure_is_not_misreported_as_map_movement(self):
        obj=Bot.__new__(Bot); obj.config={'attack_enabled':True}; obj.last_stats_update=time.time()
        obj.map=Mock(); obj.incoming=Mock(return_value=False); obj.image=Mock(); obj.state={'owner':'Yumi'}
        obj.vision=Mock(); obj.vision.own_sources.return_value=[((640,330),100)]
        obj.vision.transform_point.side_effect=lambda m,p:p
        obj.vision.enemy_candidates.return_value=[(900,330)]
        obj.targets=Mock(); obj.targets.blocked.return_value=False
        obj.capture=Mock(); obj.align_view=Mock(return_value=True); obj.decide=Mock()
        def unreadable(point): obj.last_scout_failure='report_unreadable'; return None
        obj.scout_target=Mock(side_effect=unreadable)
        self.assertEqual(obj.attack_nearby(),'report_unreadable')

    def test_repeated_alignment_failure_cannot_generate_continuous_map_panning(self):
        obj=test_live_activity.ActivityTests().actor(); obj.attack_nearby=Mock(return_value='unknown_view'); obj.upgrade=Mock()
        for _ in range(4): obj.progress_cycle()
        self.assertEqual(obj.attack_nearby.call_count,3)
        self.assertGreater(obj.alignment_retry_at,time.time())
        obj.driver.pan_map.assert_not_called(); obj.upgrade.assert_not_called()

    def test_alignment_retry_waits_for_camera_settling_without_input(self):
        obj=Bot.__new__(Bot); obj.vision=Mock(); obj.image=Mock(); obj.capture=Mock(); obj.wait=Mock()
        matrix=object(); obj.vision.muster_transform.side_effect=[None,None,matrix]
        reference=Mock()
        self.assertIs(obj.align_view(reference),matrix)
        self.assertEqual(obj.capture.call_count,2)
        reference.save.assert_not_called()


if __name__=='__main__': unittest.main()
