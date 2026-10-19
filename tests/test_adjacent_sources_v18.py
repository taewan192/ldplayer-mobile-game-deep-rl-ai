import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from live_lords.batch import BatchMixin
from live_lords.muster import MusterMixin
from live_lords.runner import Bot
from live_lords.vision import Vision
import test_live_attack_path as attack_path_tests
import test_muster_workflow as muster_tests


class AdjacentSourcesTests(unittest.TestCase):
    def actor(self):
        b=BatchMixin()
        b.config={'max_send_fraction':.35,'combat_margin':1.5}
        b.state={'attack_bonus':300}
        b.world_position=MusterMixin.world_position
        b.image=Mock();b.driver=Mock();b.wait=Mock();b.capture=Mock()
        b.incoming=Mock(return_value=False);b.settled_map=Mock();b.decide=Mock()
        b.vision=Mock();b.vision.transform_point.side_effect=Vision.transform_point
        b.align_view=Mock(return_value=np.eye(3))
        b.verify_sources=Mock(return_value=('source',[]))
        return b

    def test_enough_actual_capacity_does_not_move_or_remeasure(self):
        b=self.actor()
        reports=[{'point':(600,350),'defense':1e9}]
        sources=[{'name':'Bank','position':(400,350),'troops':2e9}]
        self.assertEqual(b.extend_batch_sources(reports,sources),(reports,sources))
        b.driver.pan_map.assert_not_called();b.verify_sources.assert_not_called()

    def test_adjacent_actual_source_is_projected_and_stops_search(self):
        b=self.actor()
        reports=[{'point':(400,300),'defense':1e9}]
        sources=[{'name':'Front','position':(600,300),'troops':100e6}]
        b.align_view.side_effect=[np.array([[1,0,80],[0,1,0]])]*3+[np.eye(3),np.eye(3)]
        b.verify_sources.return_value=('source',[{'name':'Bank','position':(250,350),'troops':2e9}])
        result,measured=b.extend_batch_sources(reports,sources)
        self.assertEqual(result[0]['point'],(640,300))
        self.assertEqual({s['name']:s['position'] for s in measured},{'Front':(840,300),'Bank':(250,350)})
        self.assertEqual(b.driver.pan_map.call_count,3)
        self.assertTrue(all(call.args==(80,0) for call in b.driver.pan_map.call_args_list))

    def test_failed_registration_cannot_provide_guessed_donors(self):
        b=self.actor();b.align_view.return_value=None
        self.assertIsNone(b.extend_batch_sources([{'point':(400,300),'defense':1e9}],
            [{'name':'Front','position':(600,300),'troops':100e6}]))
        b.verify_sources.assert_not_called()
        b.driver.pan_map.assert_called_once_with(80,0)

    def test_no_donors_are_bounded_to_two_views_without_rescout(self):
        b=self.actor()
        reports=[{'point':(400,300),'defense':1e9}]
        sources=[{'name':'Front','position':(600,300),'troops':100e6}]
        self.assertEqual(b.extend_batch_sources(reports,sources),(reports,sources))
        self.assertEqual(b.driver.pan_map.call_count,5)
        self.assertEqual(b.verify_sources.call_count,2)
        for call in b.driver.pan_map.call_args_list:
            self.assertLessEqual(abs(call.args[0]),260)
            self.assertLessEqual(abs(call.args[1]),180)

    def test_one_city_with_different_ocr_name_cannot_contribute_twice(self):
        b=self.actor()
        reports=[{'point':(400,300),'defense':1e9}]
        sources=[{'name':'Ualeyr','position':(600,300),'troops':100e6,'qa_frontline':True}]
        b.verify_sources.return_value=('source',[{'name':'Uateyr','position':(604,302),'troops':100e6}])
        _,measured=b.extend_batch_sources(reports,sources)
        self.assertEqual(len(measured),1)
        self.assertEqual(measured[0]['name'],'Uateyr')
        self.assertTrue(measured[0]['qa_frontline'])

    def test_perspective_projections_resolving_to_one_blue_flag_are_not_two_armies(self):
        b=self.actor()
        b.vision.own_flag_for_castle.return_value=(850,200)
        b.verify_sources.return_value=('source',[{'name':'Uateyr','position':(655,300),'troops':100e6}])
        _,measured=b.extend_batch_sources([{'point':(400,300),'defense':1e9}],
            [{'name':'Ualeyr','position':(600,300),'troops':100e6}])
        self.assertEqual(len(measured),1)
        self.assertEqual(measured[0]['troops'],100e6)

    def test_stage_centering_uses_small_registered_steps(self):
        b=self.actor()
        b.align_view.return_value=np.array([[1,0,0],[0,1,70]])
        self.assertEqual(b.centre_combat_pair((500,190),(700,180)),((500,330),(700,320)))
        self.assertEqual(b.driver.pan_map.call_count,2)
        self.assertTrue(all(call.args==(40,70) for call in b.driver.pan_map.call_args_list))
        self.assertEqual(len(b.combat_pan_matrices),2)

    def test_all_donors_follow_each_verified_centering_step_even_when_offscreen(self):
        b=self.actor();b.combat_pan_matrices=[np.array([[1,0,0],[0,1,70]]),np.array([[1.1,0,0],[0,1.1,70]])]
        b.centre_combat_pair=Mock(return_value=((500,300),(700,350)))
        b.begin_muster=Mock(return_value=True)
        report={'name':'Target','owner':'Rival','point':(500,100)}
        origin={'name':'Front','position':(700,150),'troops':100e6}
        donors=[origin,{'name':'Bank','position':(300,500),'troops':2e9}]
        self.assertTrue(b.start_batch_muster(report,origin,donors,400e6,{}))
        self.assertEqual(b.begin_muster.call_args.args[3],[((770,312),100e6),((330,697),2e9)])
        self.assertEqual(b.verified_source_names[(330,697)],'Bank')
        b.align_view.assert_not_called()

    def test_hud_occluded_city_suffix_cannot_become_a_staging_name(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            with Image.open(Path(__file__).parent/'fixtures/own-staging-after-arrival.png') as image:im=image.convert('RGB')
            x,y,_=v.match(im,'move_troops',.84) or v.match(im,'move_troops_city88',.84)
            shifted=Image.new('RGB',(1280,720));shifted.paste(im,(1000-x,245-y))
            v.regions=Mock(return_value={'city_name':'ley','troops':'655M'})
            self.assertIsNone(v.selected_own_city(shifted,'Yumi'))
            v.regions.assert_not_called()

    def test_transient_donor_focus_failure_retries_three_times_without_sending(self):
        b=MusterMixin();donor={'position':(500,500),'troops':345e6}
        b.muster={'status':'preparing','staging_name':'Front','moves':[],'donors':[donor],'candidates':[]}
        b.muster_plan=Mock(return_value=Mock(contributions=(1,)))
        b.focus_muster=Mock(return_value=False);b.save_muster=Mock();b.log=Mock();b.transfer_donor=Mock()
        for _ in range(4):b.advance_muster()
        self.assertEqual(b.focus_muster.call_count,3)
        self.assertTrue(donor['done']);b.transfer_donor.assert_not_called()

    def test_visible_owned_destination_is_used_before_panning_to_an_offscreen_projection(self):
        b=attack_path_tests.AttackPathTests().transfer_actor({'city_tight':'laaril','city_wide':'laaril'})
        b.state={'owner':'Yumi'};b.select_owned.return_value['position']=(640,330)
        b.vision.own_sources.return_value=[((640,330),100),((940,190),655e6)]
        navigate=Mock(return_value=False)
        result=b.prepare_transfer((640,330),10,{'staging':(1200,50),'staging_name':'laaril'},Mock(),navigate)
        self.assertEqual(result['troops'],10)
        navigate.assert_not_called()
        self.assertEqual(b.tap.call_args_list[1].args,(940,190))

    def test_visible_blue_castle_still_requires_exact_destination_name(self):
        b=attack_path_tests.AttackPathTests().transfer_actor({'city_tight':'Other','city_wide':'Wrong'})
        b.state={'owner':'Yumi'};b.select_owned.return_value['position']=(640,330)
        b.vision.own_sources.return_value=[((940,190),655e6)]
        self.assertIsNone(b.prepare_transfer((640,330),10,{'staging':(1200,50),'staging_name':'laaril'},Mock(),Mock()))
        b.select_troops.assert_not_called()

    def rescout_actor(self):
        b=Bot.__new__(Bot);b.image=Mock();b.map=Mock();b.tap=Mock();b.log=Mock();b.save_muster=Mock()
        b.skip_shielded=Mock(return_value=False);b.targets=Mock();b.targets.excluded.return_value=False;b.targets.add.return_value={}
        b.vision=Mock();b.vision.enemy_candidates.return_value=[(526,395),(600,235)]
        b.vision.transform_point.side_effect=Vision.transform_point;b.vision.match.return_value=None
        b.enemy_interaction=Mock(side_effect=lambda p:p);b.rebase_muster=Mock()
        b.wait=Mock();b.capture=Mock()
        b.muster={'target':(554,395),'fixed_reference':'anchor.png','anchor_matrix':np.eye(3).tolist(),
                  'anchors':{'target':(554,395)},'target_name':'nin','target_owner':'houseofk'}
        return b

    def test_rescout_reanchors_to_observed_enemy_flag_instead_of_drifting_terrain_point(self):
        b=self.rescout_actor()
        self.assertIsNone(b.scout_target((554,395),force=True))
        self.assertEqual(b.muster['anchors']['target'],(526,395))
        b.tap.assert_called_once_with(526,395,delay=.8)

    def test_rescout_cannot_send_to_another_city_even_when_its_flag_is_nearest(self):
        b=self.rescout_actor();b.vision.match.return_value=(500,400,1)
        b.vision.enemy_identity.return_value='houseofk:Other'
        self.assertIsNone(b.scout_target((554,395),force=True))
        self.assertEqual(b.last_scout_failure,'target_identity_mismatch')
        b.tap.assert_called_once_with(526,395,delay=.8)

    def test_arrived_army_gets_bounded_rescout_retries_without_more_transfers(self):
        b=muster_tests.MusterWorkflowTests().fixture()
        b.muster={'version':2,'status':'waiting','arrival':0,'staging':(640,330),'staging_name':'Anchor',
                  'staging_before':100,'target':(800,300),'reserve':65,'moves':[{'status':'sent','troops':35}]}
        b.rebase_muster=Mock();b.focus_muster=Mock(return_value=True);b.sync_stats=Mock()
        b.scout_target=Mock(return_value=None);b.cancel_muster=Mock();b.transfer_donor=Mock();b.record=Mock()
        for _ in range(3):b.check_muster()
        self.assertEqual(b.scout_target.call_count,3);b.cancel_muster.assert_called_once()
        b.transfer_donor.assert_not_called();b.record.assert_not_called()

    def test_owner_garbage_is_retried_in_tight_crop_and_requires_exact_observed_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            v.regions=Mock(side_effect=[{'city_name':'Nin','attacker':'idhlpuseigfkj'},
                                       {'owner_middle':'House Of K','owner_tight':'House Of K'}])
            self.assertEqual(v.enemy_identity(Mock(),(500,400),'nin','houseofk'),'houseofk:nin')

    def test_identity_banner_can_finish_rendering_before_a_scout_is_rejected(self):
        b=self.actor();b.vision.match.return_value=(500,400,1)
        b.vision.enemy_identity.return_value='houseofk:nin'
        self.assertEqual(b.ready_enemy_identity('houseoftk:nin','nin','houseofk'),'houseofk:nin')
        b.capture.assert_called_once();b.wait.assert_called_once_with(.4)

    def test_owned_name_retry_accepts_only_an_exact_name_observed_in_an_alternate_crop(self):
        b=MusterMixin();b.map=Mock();b.capture=Mock();b.image=Mock()
        b.select_owned=Mock(return_value={'name':'Uateyr','troops':1e9,'move_button':(800,400)})
        b.region=Mock(return_value={'scout_city_padded':'Ualeyr','owned_name_tight':'Uateyr'})
        self.assertEqual(b.select_owned_named((640,330),'Ualeyr')['name'],'Ualeyr')

    def test_owned_name_retry_does_not_substitute_the_requested_name_without_observing_it(self):
        b=MusterMixin();b.map=Mock();b.capture=Mock();b.image=Mock()
        b.select_owned=Mock(return_value={'name':'Neighbour','troops':1e9,'move_button':(800,400)})
        b.region=Mock(return_value={'scout_city_padded':'Neighbour','owned_name_tight':'Other'})
        b.vision=Mock();b.vision.own_flags.return_value=[]
        self.assertIsNone(b.select_owned_named((640,330),'Ualeyr'))


if __name__=='__main__':unittest.main()
