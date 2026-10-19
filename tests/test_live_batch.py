import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from live_lords.batch import BatchMixin
from live_lords.muster import MusterMixin
from live_lords.vision import Vision
from live_lords.runner import Bot
from live_lords.defense import DefenseMixin
from live_lords.learning import Learner
from live_lords.outcomes import decision
import test_muster_workflow as workflow
import test_live_attack_path as attack_tests


class BatchTests(unittest.TestCase):
    def test_owned_controls_near_top_are_not_rejected_before_reading_name(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            with Image.open(Path(__file__).parent/'fixtures'/'own-staging-after-arrival.png') as source:im=source.convert('RGB')
            button=v.match(im,'move_troops',.84) or v.match(im,'move_troops_city88',.84)
            shifted=Image.new('RGB',(1280,720));shifted.paste(im,(0,245-button[1]))
            v.regions=Mock(return_value={'city_name':'Eyrra','troops':'426M'})
            self.assertEqual(v.selected_own_city(shifted,'Yumi')['name'],'Eyrra')

    def test_owned_city_lookup_requires_own_tab_and_exact_name_after_pin(self):
        b=Bot.__new__(Bot);b.map=Mock();b.tap=Mock();b.capture=Mock();b.image=Mock();b.log=Mock()
        b.vision=Mock();b.vision.match.return_value=(265,328,1)
        b.city_list_page=Mock(return_value=(1,2))
        b.region=Mock(return_value={'0':'Mil','1':'Eyrra','2':'Other','3':'Other','4':'Other'})
        b.select_owned=Mock(return_value={'name':'Eyrra','troops':426e6});b.select_owned_named=Mock(return_value=None)
        self.assertEqual(b.locate_owned_city('eyrra')['name'],'Eyrra')
        self.assertEqual(b.tap.call_args_list[1].args,(959,122))
        b.select_owned.return_value={'name':'Neighbour'}
        self.assertIsNone(b.locate_owned_city('eyrra'))

    def test_missing_camera_registration_recovers_real_result_through_owned_list(self):
        with tempfile.TemporaryDirectory() as folder:
            b=attack_tests.AttackPathTests().result_actor(None);b.data=Path(folder)
            b.image=Image.new('RGB',(1280,720));b.image.save(b.data/'episode.png')
            b.pending.update(view='episode.png',reference_target=[400,300],target_name='eyrra')
            b.vision.viewport_transform.return_value=None
            b.locate_owned_city=Mock(return_value={'name':'Eyrra','troops':426e6})
            b.check_pending()
            b.locate_owned_city.assert_called_once_with('eyrra')
            self.assertEqual(b.counts['conquests'],1)
            self.assertTrue(b.record.call_args.args[3])

    def test_visible_flag_with_missing_map_army_is_still_measured(self):
        with tempfile.TemporaryDirectory() as folder:
            b=Bot.__new__(Bot);b.data=Path(folder);b.image=Image.new('RGB',(1280,720));b.state={'owner':'Yumi'}
            b.settled_map=Mock();b.fixed_point=Mock(side_effect=lambda ref,p:p);b.log=Mock()
            b.vision=Mock();b.vision.own_sources.return_value=[((700,300),100),((800,300),90)]
            b.vision.own_flags.return_value=[(600,200,1,1)]
            b.vision.muster_transform.return_value=np.eye(3);b.align_view=Mock(return_value=np.eye(3))
            b.vision.transform_point.side_effect=Vision.transform_point
            b.select_owned=Mock(side_effect=lambda p:{'name':'Big' if p[0]==603 else str(p),
                'troops':1e9 if p[0]==603 else 100,'position':p})
            _,sources=b.verify_sources(limit=3)
            self.assertTrue(any(s['name']=='Big' and s['troops']==1e9 for s in sources))

    def test_empty_owner_crop_retries_tighter_line_before_skipping_scout(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder);v.regions=Mock(side_effect=[{'city_name':'Eyrra','attacker':''},
                {'owner_middle':'House of K','owner_tight':'{House Of K'}])
            v.enemy_city_ink=Mock(return_value='eyrra')
            self.assertEqual(v.enemy_identity(Image.new('RGB',(1280,720)),(405,255)), 'houseofk:eyrra')

    def test_bulk_nearest_policy_uses_one_capable_city_before_combining_small_ones(self):
        b=MusterMixin();b.config={'max_send_fraction':.35};b.outcome_policy=Mock()
        b.outcome_policy.choose.return_value={'action':'nearest_bulk'}
        b.muster={'bulk_once':True,'required':1000,'staging_before':100,'reserve':65,'moves':[],
            'staging':(500,300),'policy':'fewest','candidates':[
                *[{'position':(550+i*10,300),'troops':1000} for i in range(4)],
                {'position':(900,300),'troops':10000}]}
        plan=b.muster_plan();self.assertEqual(len(plan.contributions),1)
        self.assertEqual(plan.contributions[0].position,(900,300))

    def test_unreadable_eta_is_retried_from_numeric_bubble_before_cancelling(self):
        b=attack_tests.AttackPathTests().slider_actor(10)
        b.region.return_value={'send':'10','time':'am 179'}
        b.vision.troop_selection_timer.return_value=17
        self.assertEqual(b.select_troops(10,100,35),(10,17))

    def test_verified_result_updates_ui_and_saved_counter_before_next_scout_cycle(self):
        with tempfile.TemporaryDirectory() as folder:
            b=Bot.__new__(Bot);b.data=Path(folder);b.learner=Learner(b.data/'learning.db')
            try:
                b.outcome_policy=b.learner.outcomes;b.state={'vp':100,'gold':100};b.counts={'conquests':1}
                b.emit=Mock();b.log=Mock()
                episode=b.outcome_policy.begin('attack',[decision('attack_target','army','target')],b.state,
                    {'troops_sent':100})
                b.learn_attack_result({'episode_id':episode},True,{'troops':90})
                saved=json.loads((b.data/'status.json').read_text('utf-8'))
                self.assertEqual(saved['learning']['outcomes']['evaluated'],1)
                self.assertEqual(b.emit.call_args.args[0]['learning']['outcomes']['evaluated'],1)
            finally:b.learner.close()

    def test_owned_banner_covering_enemy_uses_visible_enemy_level_badge(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            with Image.open(Path(__file__).parent/'fixtures'/'banner-overlap.png') as source:im=source.convert('RGB')
            self.assertEqual(v.enemy_click_point(im,(327,275)),(327,220))
            self.assertEqual(v.enemy_click_point(im,(663,236)),(663,181))

    def test_named_owned_selection_rejects_neighbour_banner(self):
        b=MusterMixin();b.map=Mock();b.capture=Mock();b.tap=Mock();b.image=Mock();b.state={'owner':'Yumi'}
        b.vision=Mock();b.vision.own_flags.return_value=[(500,200,1,1),(510,300,1,1)]
        b.vision.muster_transform.return_value=np.eye(3);b.vision.transform_point.side_effect=Vision.transform_point
        b.vision.selected_own_city.side_effect=[{'name':'Neighbour'},{'name':'Expected','troops':100}]
        self.assertEqual(b.select_owned_named((500,300),'Expected')['name'],'Expected')
        self.assertEqual([call.args for call in b.tap.call_args_list],[(500,300),(500,200)])

    def test_last_outgoing_march_disappearing_is_not_a_defense_block(self):
        b=DefenseMixin();b.vision=Mock();b.image=Mock();b.tap=Mock();b.wait=Mock();b.capture=Mock()
        b.vision.march_button.side_effect=[(1224,570,1),None]
        b.vision.march_panel.return_value=None;b.vision.match.return_value=(50,670,1)
        b.vision.incoming_button.return_value=None
        self.assertEqual(b.inspect_defense_rows(),[])
        b.capture.assert_called_once()

    def test_unknown_march_panel_with_real_incoming_alert_still_blocks(self):
        b=DefenseMixin();b.vision=Mock();b.image=Mock();b.tap=Mock();b.wait=Mock();b.capture=Mock()
        b.vision.march_button.return_value=(1224,570,1)
        b.vision.march_panel.return_value=None
        with self.assertRaises(RuntimeError):b.inspect_defense_rows()

    def test_edge_candidate_cannot_pan_away_from_the_batch_reference(self):
        b=self.batch_actor();b.targets=Mock();b.targets.excluded.return_value=False
        b.vision.match.return_value=None;b.driver=Mock()
        b.inspect_batch_targets('reference',[(1050,550),(600,300)])
        self.assertEqual([call.args for call in b.tap.call_args_list],[(1050,550),(600,300)])
        b.driver.pan_map.assert_not_called()

    def test_merged_g_unit_and_troop_icon_is_reread_without_icon(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            im=Image.new('RGB',(1280,720))
            with Image.open(Path(__file__).parent/'fixtures'/'own-staging-after-arrival.png') as source:im=source.convert('RGB')
            v.regions=Mock(side_effect=[{'city_name':'Amin','troops':'2.956M'},{'troops':'2.95G'}])
            self.assertEqual(v.selected_own_city(im,'Yumi')['troops'],2.95e9)

    def test_fixed_reference_does_not_accumulate_camera_offsets(self):
        with tempfile.TemporaryDirectory() as folder:
            b=MusterMixin();b.data=Path(folder);b.image=Image.new('RGB',(1280,720))
            b.image.save(b.data/'anchor.png');b.save_muster=Mock()
            b.muster={'fixed_reference':'anchor.png','anchors':{'staging':[500,300]},
                'donors':[{'position':[700,300],'anchor_position':[700,300]}]}
            b.vision=Mock();b.vision.muster_transform.return_value=np.array([[1,0,5],[0,1,0],[0,0,1]])
            b.vision.transform_point.side_effect=Vision.transform_point
            for _ in range(20):b.rebase_muster()
            self.assertEqual(b.muster['staging'],(505,300))
            self.assertEqual(b.muster['donors'][0]['position'],(705,300))

    def test_banner_occlusion_uses_last_verified_frame_and_canonical_points(self):
        with tempfile.TemporaryDirectory() as folder:
            b=MusterMixin();b.data=Path(folder);b.image=Image.new('RGB',(1280,720));b.save_muster=Mock()
            b.image.save(b.data/'anchor.png');b.image.save(b.data/'muster-view.png')
            previous=np.array([[1,0,10],[0,1,0],[0,0,1]])
            b.muster={'fixed_reference':'anchor.png','anchors':{'staging':[500,300]},'anchor_matrix':previous.tolist()}
            b.vision=Mock();b.vision.muster_transform.return_value=None
            b.vision.viewport_transform.return_value=np.array([[1,0,5],[0,1,0]])
            b.vision.transform_point.side_effect=Vision.transform_point
            b.rebase_muster();self.assertEqual(b.muster['staging'],(515,300))

    def test_failed_transfer_cannot_repeat_forever_through_navigation_recovery(self):
        b=workflow.MusterWorkflowTests().fixture()
        b.muster={'status':'preparing','staging_name':'Anchor','staging':[500,300],
            'moves':[],'donors':[{'position':(700,300),'troops':30}],'candidates':[]}
        b.muster_plan=Mock(return_value=Mock(contributions=(1,)));b.focus_muster=Mock(return_value=True)
        b.transfer_donor=Mock(side_effect=RuntimeError('occluded terrain'))
        for _ in range(5):b.advance_muster()
        self.assertEqual(b.transfer_donor.call_count,3)
        self.assertTrue(b.muster['donors'][0]['done'])

    def test_incomplete_bulk_plan_never_starts_speculative_gather(self):
        b=workflow.MusterWorkflowTests().fixture();b.prevalidated_bulk=True
        self.assertFalse(b.begin_muster((800,300),(640,330),1000,
            [((640,330),100),((900,400),100)],{}))
        self.assertIsNone(b.muster);b.driver.pan_map.assert_not_called()

    def test_bulk_sends_each_planned_donor_once_before_waiting(self):
        b=workflow.MusterWorkflowTests().fixture()
        b.muster={'bulk_once':True,'status':'preparing','staging_name':'Anchor','staging':[500,300],
            'moves':[],'donors':[{'position':(700,300),'troops':30},{'position':(850,300),'troops':40}],
            'candidates':[]}
        b.muster_plan=Mock(return_value=Mock(contributions=(1,)))
        b.focus_muster=Mock(return_value=True)
        def send(position,troops):
            b.muster['moves'].append({'position':position,'troops':troops,'status':'sent'})
            return True
        b.transfer_donor=Mock(side_effect=send)
        b.save_muster.side_effect=lambda status=None:b.muster.update(status=status) if status else None
        b.advance_muster();b.advance_muster();b.advance_muster()
        self.assertEqual(b.transfer_donor.call_count,2)
        self.assertEqual(b.muster['status'],'waiting')
        self.assertTrue(all(d['done'] for d in b.muster['donors']))

    def test_restart_preserves_remaining_bulk_donor_but_never_replays_intent(self):
        with tempfile.TemporaryDirectory() as folder:
            b=MusterMixin();b.data=Path(folder)
            saved={'version':4,'bulk_once':True,'status':'preparing','donors':[
                {'troops':30,'done':True},{'troops':40}],
                'moves':[{'status':'sent','troops':30}]}
            path=b.data/'muster-journal.json';path.write_text(json.dumps(saved),'utf-8')
            b.init_muster();self.assertEqual(b.muster['status'],'preparing')
            self.assertTrue(b.muster['donors'][0]['done'])
            saved['moves'][0]['status']='intent';path.write_text(json.dumps(saved),'utf-8')
            b.init_muster();self.assertEqual(b.muster['status'],'waiting')

    def test_wrong_measured_donor_name_blocks_send(self):
        b=workflow.MusterWorkflowTests().fixture();b.back=Mock();b.tap=Mock()
        b.muster={'staging_name':'Anchor','arrival':0,'moves':[],
            'donors':[{'position':(700,300),'expected_name':'Verified'}]}
        b.prepare_transfer=Mock(return_value={'donor':{'name':'Different','troops':100},
            'troops':30,'duration':20,'button':(880,540)})
        self.assertFalse(b.transfer_donor((700,300),30));b.tap.assert_not_called()
        self.assertEqual(b.muster['moves'],[])

    def batch_actor(self):
        b=BatchMixin();b.settled_map=Mock();b.fixed_point=Mock(side_effect=lambda ref,p:p)
        b.skip_shielded=Mock(return_value=False);b.tap=Mock();b.wait=Mock();b.log=Mock()
        b.save_batch_intent=Mock();b.offensive_excluded=Mock(return_value=False);b.decide=Mock()
        b.world_position=MusterMixin.world_position
        b.vision=Mock();b.vision.match.return_value=(300,300,1)
        b.vision.enemy_identity.return_value='allowed:verified';b.image=Mock();b.counts={'scouts':0}
        return b

    def test_parallel_scout_dispatches_all_before_one_wait(self):
        b=self.batch_actor();timeline=[]
        b.tap.side_effect=lambda *args,**kw:timeline.append('tap')
        b.wait.side_effect=lambda seconds:timeline.append(('wait',seconds))
        targets=[{'point':(500,300),'name':'a'},{'point':(800,300),'name':'b'}]
        self.assertEqual(b.targeted_parallel_scout('ref',targets),targets)
        self.assertEqual(timeline,['tap','tap','tap','tap',('wait',12)])
        self.assertEqual(b.counts['scouts'],2)

    def test_known_excluded_player_blocks_native_area_command(self):
        b=self.batch_actor();b.targets=Mock();b.world_position=Mock(return_value=True)
        b.targets.projected.return_value=[({'identities':['prabowo:astev']},(600,300))]
        b.targets._excluded_entry.return_value=True
        self.assertFalse(b.native_multi_scout('ref',[{},{}]))
        b.tap.assert_not_called()

    def native_actor(self,title):
        b=self.batch_actor();b.targets=Mock();b.targets.projected.return_value=[]
        b.vision.enemy_candidates.return_value=[(600,300),(700,300)]
        b.vision.own_sources.return_value=[((500,300),1000)];b.state={'owner':'Yumi'}
        b.align_view=Mock(return_value=np.eye(3));b.vision.transform_point.side_effect=Vision.transform_point
        b.select_owned=Mock(return_value={'name':'Own'})
        b.vision.match.side_effect=[(450,350,1),(600,430,1)]
        b.region=Mock(return_value={'prompt':title});b.back=Mock()
        return b

    def test_native_multi_scout_confirms_verified_mode_then_waits_once(self):
        b=self.native_actor('다중 정찰 확인')
        targets=[{'point':(600,300)},{'point':(700,300)}]
        self.assertTrue(b.native_multi_scout('ref',targets))
        self.assertEqual([call.args for call in b.tap.call_args_list],[(450,350),(600,430)])
        b.wait.assert_called_once_with(12)

    def test_same_green_confirmation_never_confirms_regrouping(self):
        b=self.native_actor('재편성 확인')
        self.assertFalse(b.native_multi_scout('ref',[{'point':(600,300)},{'point':(700,300)}]))
        b.tap.assert_called_once_with(450,350,delay=.7);b.wait.assert_not_called()

    def attack_actor(self,folder,reports):
        b=Bot.__new__(Bot);b.data=Path(folder);b.image=Image.new('RGB',(1280,720))
        b.config={'max_send_fraction':.35,'combat_margin':1.5};b.state={'attack_bonus':100}
        b.settled_map=Mock();b.vision=Mock();b.vision.enemy_candidates.return_value=[r['point'] for r in reports]
        b.vision.transform_point.side_effect=lambda matrix,p:tuple(p)
        b.targets=Mock();b.targets.blocked.return_value=False
        b.inspect_batch_targets=Mock(return_value=reports);b.native_multi_scout=Mock(return_value=False)
        b.targeted_parallel_scout=Mock(return_value=reports);b.read_batch_reports=Mock(return_value=reports)
        b.verify_sources=Mock(return_value=(b.image,[{'name':'Origin','troops':1000,'position':(500,300)}]))
        b.align_view=Mock(return_value=np.eye(3));b.centre_combat_pair=Mock(side_effect=lambda t,o:(t,o))
        b.choose_attack_requirement=Mock(return_value=(75,1.5));b.hud=Mock();b.decide=Mock()
        b.dispatch_attack=Mock(return_value=True)
        def rank(points,*args):
            b.attack_rank_choices={tuple(p):({'domain':'attack_target','action':str(p),'context':'army'},
                {'score':r.get('score',0)}) for p,r in zip(points,reports)}
            return points
        b.rank_attack_targets=Mock(side_effect=rank)
        return b

    def test_strong_first_report_does_not_block_attack_on_feasible_second_city(self):
        with tempfile.TemporaryDirectory() as folder:
            reports=[{'point':(700,300),'owner':'a','name':'strong','defense':1000},
                     {'point':(800,300),'owner':'b','name':'weak','defense':100}]
            b=self.attack_actor(folder,reports)
            self.assertEqual(b.attack_screen_batch(),'dispatched')
            self.assertEqual(b.dispatch_attack.call_args.args[0],(800,300))
            self.assertEqual(b.last_scout_city,'weak')

    def test_learned_target_score_changes_choice_among_feasible_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            reports=[{'point':(700,300),'owner':'a','name':'safer','defense':100,'score':1.5},
                     {'point':(800,300),'owner':'b','name':'weaker','defense':50,'score':0}]
            b=self.attack_actor(folder,reports)
            self.assertEqual(b.attack_screen_batch(),'dispatched')
            self.assertEqual(b.last_scout_city,'safer')
            self.assertEqual(b.attack_choices[0]['action'],str((700,300)))


if __name__=='__main__':unittest.main()
