import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from live_lords.exploration import ExplorationMemory,SWEEP_RAYS
from live_lords.vision import Vision
from live_lords.batch import BatchMixin
import test_live_activity


class WideExplorationTests(unittest.TestCase):
    def actor(self,folder):
        b=test_live_activity.ActivityTests().actor();b.config['wide_exploration']=True
        b.data=Path(folder);b.image=Image.new('RGB',(1280,720));b.vision=Mock()
        b.sweep_view_matches=Mock(return_value=True)
        b.center_sweep_main=Mock(side_effect=lambda:b.exploration_memory().sweep_bind_main('Main') or True)
        b.sweep_pan=Mock(return_value=True)
        return b

    def test_full_route_is_72_unique_positions_in_eight_rays_with_nine_consecutive_moves(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.actor(folder)
            for _ in range(72):self.assertTrue(b.explore('scan'))
            self.assertEqual(b.center_sweep_main.call_count,8)
            expected=[(dx,dy) for _,dx,dy in SWEEP_RAYS for _ in range(9)]
            self.assertEqual([c.args[0] for c in b.sweep_pan.call_args_list],expected)
            route=ExplorationMemory(folder).state['sweep']
            self.assertEqual(len({(v['cycle'],v['ray'],v['step']) for v in route['visits']}),72)
            self.assertEqual((route['cursor'],route['cycle']),(0,2))
            self.assertEqual(b.counts['explorations'],72)

    def test_midray_camera_reset_replays_from_main_without_counting_replay_as_new_exploration(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.actor(folder)
            for _ in range(5):b.explore('scan')
            b.sweep_pan.reset_mock();b.sweep_view_matches.return_value=False
            b.explore('resume')
            self.assertEqual(b.sweep_pan.call_count,6)
            self.assertEqual(b.counts['explorations'],6)
            self.assertEqual(b.exploration_memory().state['sweep']['cursor'],6)
            self.assertEqual(b.state['wide_sweep']['step'],6)

    def test_restart_preserves_direction_and_distance_instead_of_restarting_below_main(self):
        with tempfile.TemporaryDirectory() as folder:
            memory=ExplorationMemory(folder)
            for _ in range(31):memory.sweep_commit()
            restored=ExplorationMemory(folder);point=restored.sweep_next()
            self.assertEqual((point['direction'],point['step']),('오른쪽',5))

    def test_failed_pan_never_advances_or_counts_a_visit_and_repeated_failure_skips_only_that_ray(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.actor(folder);b.sweep_pan.return_value=False;b.sweep_stationary=True
            for _ in range(3):self.assertFalse(b.explore('scan'))
            route=b.exploration_memory().state['sweep']
            self.assertEqual(b.counts['explorations'],0);self.assertEqual(route['visits'],[])
            self.assertEqual(route['cursor'],9);self.assertEqual(len(route['skipped']),1)

    def test_defense_interruption_does_not_consume_exploration_or_mark_the_ray_failed(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.actor(folder);b.sweep_pan.return_value=False;b.sweep_interrupted=True
            self.assertFalse(b.explore('scan'))
            route=b.exploration_memory().state['sweep']
            self.assertEqual(route['cursor'],0);self.assertNotIn('failures',route)

    def test_no_owned_castles_does_not_cut_off_the_ray_after_two_moves(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.actor(folder);b.attack_nearby=Mock(return_value='no_sources');b.upgrade=Mock()
            b.next_region=Mock();b.goto_combat_bank=Mock()
            for _ in range(7):b.progress_cycle()
            self.assertEqual(b.exploration_memory().state['sweep']['cursor'],7)
            b.next_region.assert_not_called();b.goto_combat_bank.assert_not_called();b.upgrade.assert_not_called()

    def test_no_target_does_not_redirect_to_old_nearby_troop_banks(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.actor(folder);b.attack_nearby=Mock(return_value='no_targets')
            b.outcome_policy=Mock();b.upgrade=Mock(return_value=False);b.goto_combat_bank=Mock()
            b.progress_cycle()
            b.goto_combat_bank.assert_not_called();self.assertEqual(b.state['wide_sweep']['step'],1)

    def test_remote_enemy_region_is_not_scouted_without_an_owned_departure_castle(self):
        b=BatchMixin();b.wide_exploration_enabled=lambda:True;b.settled_map=Mock()
        b.image=Mock();b.vision=Mock();b.vision.own_flags.return_value=[]
        self.assertEqual(b.attack_screen_batch(),'no_sources')
        b.vision.enemy_candidates.assert_not_called()

    def test_actual_pan_waits_for_camera_to_settle_without_repeating_the_drag(self):
        with tempfile.TemporaryDirectory() as folder:
            b=test_live_activity.ActivityTests().actor();b.data=Path(folder)
            b.image=Image.new('RGB',(1280,720));b.vision=Mock()
            b.vision.sweep_frame_delta.return_value=(0,0)
            b.vision.sweep_motion_delta.return_value=(0,0)
            b.vision.viewport_transform.side_effect=[None,np.float32([[1,0,0],[0,1,-180]])]
            b.vision.transform_point.side_effect=Vision.transform_point
            self.assertTrue(b.sweep_pan((0,-180)))
            b.driver.pan_map.assert_called_once_with(0,-180)
            self.assertEqual(b.capture.call_count,3)

    def test_actual_pan_rejects_unchanged_map_and_reverse_direction(self):
        for dy in (0,180):
            with tempfile.TemporaryDirectory() as folder:
                b=test_live_activity.ActivityTests().actor();b.data=Path(folder)
                b.image=Image.new('RGB',(1280,720));b.vision=Mock()
                b.vision.sweep_frame_delta.return_value=(0,0)
                b.vision.sweep_motion_delta.return_value=(0,0)
                b.vision.viewport_transform.return_value=np.float32([[1,0,0],[0,1,dy]])
                b.vision.transform_point.side_effect=Vision.transform_point
                self.assertFalse(b.sweep_pan((0,-180)))
                b.driver.pan_map.assert_called_once()

    def test_boundary_ninth_position_is_checked_before_returning_to_main(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.actor(folder)
            for _ in range(9):b.explore('scan')
            b.attack_nearby=Mock(return_value='no_sources')
            b.progress_cycle()
            b.attack_nearby.assert_called_once()
            self.assertEqual(b.state['wide_sweep']['direction'],'위')
            self.assertEqual(b.state['wide_sweep']['step'],1)

    def test_actual_3d_camera_pan_is_accepted_without_using_its_pixels_as_combat_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            b=test_live_activity.ActivityTests().actor();b.data=Path(folder)
            fixtures=Path(__file__).parent/'fixtures'
            before=Image.open(fixtures/'sweep-before-wide-pan.png').convert('RGB')
            after=Image.open(fixtures/'sweep-after-wide-pan.png').convert('RGB')
            b.image=before;b.vision=Vision(None,folder)
            def capture():b.image=after
            b.capture=Mock(side_effect=capture)
            # The travel verifier may now also obtain valid terrain geometry;
            # it must accept this real move even when the strict fit fails.
            self.assertTrue(b.sweep_pan((0,-180)))
            b.driver.pan_map.assert_called_once()
            self.assertEqual(b.vision.sweep_frame_delta(after,after),(0,0))

    def test_main_castle_is_found_on_first_page_via_house_icon_and_verified_name(self):
        with tempfile.TemporaryDirectory() as folder:
            b=test_live_activity.ActivityTests().actor();b.data=Path(folder);b.tap=Mock()
            b.image=Image.new('RGB',(1280,720));b.vision=Mock()
            b.vision.match.side_effect=[(280,130,1),(265,264,1)]
            b.city_list_page=Mock(side_effect=[(4,8),(1,8)])
            b.region=Mock(return_value={'city_name':'Mil'})
            b.select_owned=Mock(return_value={'name':'Neighbour'})
            b.select_owned_named=Mock(return_value={'name':'Mil'})
            self.assertTrue(b.center_sweep_main())
            self.assertEqual([c.args for c in b.tap.call_args_list],
                             [(1047,665),(959,122),(574,578),(574,578),(574,578),(265,264)])
            self.assertEqual(b.exploration_memory().state['sweep']['main'],'Mil')
            b.select_owned_named.assert_called_once_with((640,330),'Mil')

    def test_unverified_main_castle_does_not_send_the_route_from_a_guessed_city(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.actor(folder);b.center_sweep_main.return_value=False;b.center_sweep_main.side_effect=None
            self.assertFalse(b.explore('scan'))
            b.sweep_pan.assert_not_called();self.assertEqual(b.counts['explorations'],0)

    def test_main_navigation_accepts_confirmed_identity_without_turning_unreadable_army_into_dispatch_capacity(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            image=Image.open(Path(__file__).parent/'fixtures/owned-main-unreadable-army.png').convert('RGB')
            v.regions=Mock(return_value={'city_name':'Mil','troops':'unreadable army'})
            self.assertIsNone(v.selected_own_city(image,'Yumi'))
            own=v.selected_own_city(image,'Yumi',require_troops=False)
            self.assertEqual(own['name'],'Mil');self.assertIsNone(own['troops'])


if __name__=='__main__':unittest.main()
