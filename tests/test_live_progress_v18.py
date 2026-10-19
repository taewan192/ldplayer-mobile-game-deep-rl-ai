import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from live_lords.vision import Vision,numeric_amount,amount
from live_lords.activity import ActivityMixin
from live_lords.batch import BatchMixin
import test_live_activity
import test_wide_exploration


class LiveProgressV18Tests(unittest.TestCase):
    def test_numeric_decimal_confusions_require_an_explicit_unit_and_do_not_change_generic_amounts(self):
        self.assertEqual(numeric_amount("2.OOG'W'"),2e9)
        self.assertEqual(numeric_amount("2.ooG'"),2e9)
        self.assertEqual(numeric_amount('I.09G'),1.09e9)
        for raw in ('I09G','I.09','2.OOGarbage','Captain 2.00G'):
            self.assertIsNone(numeric_amount(raw),raw)
        self.assertEqual(numeric_amount('2.006'),2.006)  # Never guessed as 2.00G.
        self.assertIsNone(amount('2.OOG'))

    def test_real_mil_201g_banner_rereads_the_value_without_the_attached_troop_icon(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            image=Image.open(Path(__file__).parent/'fixtures/owned-main-201g.png').convert('RGB')
            v.regions=Mock(side_effect=[{'city_name':'Mil','troops':"2.01GJU'"},{'troops':'2.01G'}])
            own=v.selected_own_city(image,'Yumi')
            self.assertEqual(own['name'],'Mil');self.assertAlmostEqual(own['troops'],2.01e9,delta=1)
            self.assertEqual(v.regions.call_args.args[1],{'troops':(662,209,730,238)})

    def test_transient_wrong_homography_does_not_hide_a_later_verified_large_map_pan(self):
        with tempfile.TemporaryDirectory() as folder:
            b=test_live_activity.ActivityTests().actor();b.data=Path(folder)
            fixtures=Path(__file__).parent/'fixtures'
            before=Image.open(fixtures/'shore-before-v18.png').convert('RGB')
            after=Image.open(fixtures/'shore-after-v18.png').convert('RGB')
            b.image=before;b.vision=Vision(None,folder)
            def capture():b.image=after
            b.capture=Mock(side_effect=capture)
            b.vision.viewport_transform=Mock(side_effect=[np.float32([[1,0,-80],[0,1,80]]),None])
            self.assertTrue(b.sweep_pan((260,-180)))
            b.driver.pan_map.assert_called_once()

    def test_uncertain_camera_does_not_skip_the_unfinished_direction(self):
        with tempfile.TemporaryDirectory() as folder:
            b=test_live_activity.ActivityTests().actor();b.data=Path(folder)
            b.sweep_stationary=False
            for _ in range(3):b.sweep_pan_failed('inconclusive')
            route=b.exploration_memory().state['sweep']
            self.assertEqual(route['cursor'],0);self.assertNotIn('skipped',route)

    def test_optional_upgrade_screen_failure_does_not_restart_scouting_the_same_region(self):
        with tempfile.TemporaryDirectory() as folder:
            b=test_wide_exploration.WideExplorationTests().actor(folder)
            b.attack_nearby=Mock(return_value='no_targets')
            b.upgrade=Mock(side_effect=RuntimeError('unrecognized city details'))
            b.recover_navigation=Mock(return_value=True)
            b.progress_cycle();b.progress_cycle()
            self.assertEqual(b.exploration_memory().state['sweep']['cursor'],2)
            b.upgrade.assert_called_once();b.recover_navigation.assert_called_once()

    def test_unreadable_defense_report_still_records_the_real_excluded_player_and_location(self):
        with tempfile.TemporaryDirectory() as folder:
            b=BatchMixin();b.data=Path(folder);b.image=Image.new('RGB',(1280,720))
            b.decide=Mock();b.settled_map=Mock();b.fixed_point=Mock(return_value=(650,330))
            b.world_position=Mock(return_value=True);b.skip_shielded=Mock(return_value=False)
            b.enemy_interaction=lambda point:point;b.tap=Mock();b.vision=Mock();b.targets=Mock();b.log=Mock()
            b.vision.match.return_value=(800,400,1)
            b.vision.scout_report.return_value={'city':'Onet','owner':'VikingBdg','defense':None}
            b.offensive_excluded=Mock(return_value=True)
            reports=b.read_batch_reports(b.image,[{'point':(650,330),'name':'onel','owner':'badocr'}])
            self.assertEqual(reports,[])
            b.offensive_excluded.assert_called_once_with('vikingbdg:onet')
            self.assertEqual(b.targets.add.call_args.args[2],'vikingbdg:onet')

    def test_sea_dominated_real_camera_move_is_verified_from_islands_without_counting_wave_animation(self):
        with tempfile.TemporaryDirectory() as folder:
            fixtures=Path(__file__).parent/'fixtures'
            before=Image.open(fixtures/'sea-before-v18.png').convert('RGB')
            after=Image.open(fixtures/'sea-after-v18.png').convert('RGB')
            self.assertLess(Vision.sweep_frame_delta(before,after)[0],.45)
            self.assertGreater(Vision.sweep_motion_delta(before,after)[0],.45)
            self.assertEqual(Vision.sweep_motion_delta(after,after),(0,0))
            b=test_live_activity.ActivityTests().actor();b.data=Path(folder);b.image=before;b.vision=Vision(None,folder)
            def capture():b.image=after
            b.capture=Mock(side_effect=capture)
            self.assertTrue(b.sweep_pan((260,0)))
            b.driver.pan_map.assert_called_once_with(260,0)

    def test_fixed_map_edge_pointer_cannot_claim_a_large_real_pan_was_stationary(self):
        with tempfile.TemporaryDirectory() as folder:
            fixtures=Path(__file__).parent/'fixtures'
            before=Image.open(fixtures/'static-icon-pan-before-v18.png').convert('RGB')
            after=Image.open(fixtures/'static-icon-pan-after-v18.png').convert('RGB')
            v=Vision(None,folder);matrix=v.viewport_transform(before,after)
            if matrix is not None:
                self.assertGreater(np.linalg.norm(np.array(v.transform_point(matrix,(640,350)))-[640,350]),40)
            b=test_live_activity.ActivityTests().actor();b.data=Path(folder);b.image=before;b.vision=v
            def capture():b.image=after
            b.capture=Mock(side_effect=capture)
            self.assertTrue(b.sweep_pan((-260,0)))
            b.driver.pan_map.assert_called_once_with(-260,0)

    def test_visible_enemy_badge_selects_the_real_offcenter_castle_without_clicking_empty_ground(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            image=Image.open(Path(__file__).parent/'fixtures/offcenter-enemy-flag-v18.png').convert('RGB')
            self.assertIn((223,491),v.enemy_candidates(image))
            self.assertEqual(v.enemy_click_point(image,(223,491)),(223,436))
            v.protection_circles=Mock(return_value=[(285,491,60)])
            self.assertTrue(v.is_shielded(image,(223,491)))

    def test_sparse_shore_move_is_not_lost_in_the_gray_water_background(self):
        with tempfile.TemporaryDirectory() as folder:
            fixtures=Path(__file__).parent/'fixtures'
            before=Image.open(fixtures/'water-shader-before-v18.png').convert('RGB')
            after=Image.open(fixtures/'water-shader-after-v18.png').convert('RGB')
            self.assertLess(Vision.sweep_frame_delta(before,after)[0],.45)
            self.assertGreater(Vision.sweep_motion_delta(before,after)[0],.45)
            b=test_live_activity.ActivityTests().actor();b.data=Path(folder);b.image=before;b.vision=Vision(None,folder)
            def capture():b.image=after
            b.capture=Mock(side_effect=capture)
            self.assertTrue(b.sweep_pan((260,180)))

    def test_issued_camera_gesture_over_featureless_sea_advances_once_without_claiming_a_new_view_or_learning(self):
        with tempfile.TemporaryDirectory() as folder:
            b=test_wide_exploration.WideExplorationTests().actor(folder)
            b.sweep_pan.return_value=False;b.sweep_pan_issued=True;b.sweep_interrupted=False
            b.vision.match.side_effect=lambda image,name,*args,**kwargs:(49,669,1) if name=='map_home' else None
            self.assertTrue(b.explore('sea'))
            self.assertEqual(b.exploration_memory().state['sweep']['cursor'],1)
            self.assertEqual(b.counts['explorations'],0)
            self.assertEqual(b.counts['exploration_commands'],1)
            self.assertFalse(b.state['wide_sweep']['movement_verified'])
            b.attack_nearby=Mock();b.progress_cycle()
            b.attack_nearby.assert_not_called()
            self.assertEqual(b.exploration_memory().state['sweep']['cursor'],2)

    def test_short_native_scout_city_name_is_read_with_padding_before_using_the_defense(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            image=Image.open(Path(__file__).parent/'fixtures/scout-nin-native-v18.png').convert('RGB')
            v.regions=Mock(side_effect=[{'title':'세부 보고서','city_name':'','attacker':'House Of K','defense':'1.21G'},
                                       {'scout_city_padded':'Nin'}])
            report=v.scout_report(image)
            self.assertEqual(report['city'],'Nin');self.assertEqual(report['defense'],1.21e9)

    def test_startup_current_view_can_dispatch_before_camera_returns_to_sweep_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            b=test_wide_exploration.WideExplorationTests().actor(folder)
            b.startup_combat_check=True;b.sweep_view_matches.return_value=False
            b.vision.own_flags.return_value=[(600,300,1,1)]
            def send():b.pending={'arrival':999};return 'dispatched'
            b.attack_nearby=Mock(side_effect=send)
            b.progress_cycle()
            b.attack_nearby.assert_called_once();b.center_sweep_main.assert_not_called()
            self.assertFalse(b.startup_combat_check)

    def test_report_total_animation_is_reread_without_sending_another_scout(self):
        b=BatchMixin();b.image=Mock();b.wait=Mock();b.capture=Mock();b.vision=Mock()
        blank={'defense':None,'city':'Nin','owner':'House Of K'}
        good={**blank,'defense':1.21e9}
        b.vision.scout_report.return_value=good
        self.assertEqual(b.ready_scout_report(blank),good)
        b.wait.assert_called_once_with(.5);b.capture.assert_called_once()


if __name__=='__main__':unittest.main()
