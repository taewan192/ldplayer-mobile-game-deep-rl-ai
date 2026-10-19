import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from PIL import Image

from live_lords.muster import MusterMixin
from live_lords.planner import plan_muster,MusterDonor
from live_lords.vision import Vision


class MusterWorkflowTests(unittest.TestCase):
    def fixture(self):
        b=MusterMixin();b.config={'max_send_fraction':.35};b.data=Path('.')
        b.image=Mock();b.map=Mock();b.capture=Mock();b.wait=Mock();b.emit=Mock();b.log=Mock();b.decide=Mock()
        b.state={'owner':'Yumi'};b.learner=Mock();b.learner.choose.return_value='nearest'
        b.vision=Mock();b.vision.muster_transform.return_value='verified'
        b.vision.transform_point.side_effect=lambda matrix,pos:pos
        b.select_owned=Mock(return_value={'name':'Anchor','troops':100})
        b.save_muster=Mock();b.driver=Mock();b.last_scout_city='enemy';b.muster=None
        return b

    def begin(self,b):
        return b.begin_muster((800,300),(640,330),1000,[((640,330),100),((900,400),100)],{'vp':10})

    def test_insufficient_visible_army_creates_persistent_fixed_anchor_search(self):
        b=self.fixture();self.assertTrue(self.begin(b))
        self.assertEqual(b.muster['status'],'discovering')
        self.assertEqual(b.muster['staging_name'],'Anchor')
        self.assertEqual(b.muster['target_name'],'enemy')
        self.assertEqual(b.muster['required'],1000)
        self.assertEqual(b.muster['moves'],[])

    def test_search_keeps_goal_and_reaches_transfer_phase_when_new_donor_is_found(self):
        b=self.fixture();self.begin(b)
        b.vision.own_sources.return_value=[((850,350),10000)]
        b.discover_muster()
        self.assertEqual(b.muster['status'],'discovering')  # mocked saver does not mutate status
        b.save_muster.assert_called_with('preparing')
        self.assertEqual(b.muster['staging_name'],'Anchor')
        self.assertTrue(b.muster['donors'])
        self.assertEqual(b.muster['donors'][0]['position'],(850,350))

    def test_search_pans_for_additional_donors_without_switching_anchor(self):
        b=self.fixture();self.begin(b);b.rebase_muster=Mock()
        b.vision.own_sources.return_value=[]
        self.assertTrue(b.discover_muster())
        b.driver.pan_map.assert_called_once_with(120,0)
        self.assertEqual(b.muster['scan_step'],1)
        self.assertEqual(b.muster['staging_name'],'Anchor')
        self.assertEqual(b.muster['required'],1000)

    def test_failed_donor_does_not_cancel_goal_or_repeat_that_source(self):
        b=self.fixture();self.begin(b)
        b.muster['status']='preparing';b.muster['donors']=[{'position':(900,400),'troops':30}]
        b.focus_muster=Mock(return_value=True);b.transfer_donor=Mock(return_value=False)
        self.assertTrue(b.advance_muster())
        self.assertIsNotNone(b.muster)
        self.assertTrue(b.muster['donors'][0]['done'])
        self.assertTrue(b.muster['candidates'][0]['unavailable'])

    def test_restart_resumes_search_with_the_same_target(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.fixture();self.begin(b);saved=b.muster;b.data=Path(folder)
            (b.data/'muster-journal.json').write_text(json.dumps(saved),'utf-8')
            b.init_muster()
            self.assertEqual(b.muster['status'],'discovering')
            self.assertEqual(b.muster['target'],[800,300])
            self.assertEqual(b.muster['staging_name'],'Anchor')

    def test_restart_never_replays_a_donor_that_already_has_send_intent(self):
        b=self.fixture();self.begin(b);b.muster['status']='preparing'
        b.muster['donors']=[{'position':(900,400),'troops':30}]
        b.muster['moves']=[{'position':(900,400),'troops':30,'status':'intent'}]
        b.muster_plan=Mock(return_value=None);b.focus_muster=Mock();b.transfer_donor=Mock()
        b.advance_muster()
        b.focus_muster.assert_not_called();b.transfer_donor.assert_not_called()

    def test_reserve_does_not_grow_when_previous_transfers_are_counted(self):
        plan=plan_muster(100,200,[],.35,quantum=.1,staging_reserve=65)
        self.assertEqual(plan.staging_reserve,65)
        self.assertEqual(plan.contributions,())

    def test_camera_translation_does_not_cancel_in_progress_search(self):
        b=self.fixture();self.begin(b)
        b.rebase_muster=Mock();b.advance_muster=Mock(return_value=True)
        b.verify_muster_view=Mock(return_value=False);b.cancel_muster=Mock()
        self.assertTrue(b.check_muster())
        b.rebase_muster.assert_called_once();b.advance_muster.assert_called_once()
        b.cancel_muster.assert_not_called()

    def arrived(self):
        b=self.fixture();self.begin(b)
        b.muster['status']='waiting';b.muster['arrival']=0
        b.rebase_muster=Mock();b.focus_muster=Mock(return_value=True);b.sync_stats=Mock()
        b.select_owned=Mock(return_value={'name':'Anchor','troops':1500})
        b.state['attack_bonus']=500;b.config['combat_margin']=1.5
        b.scout_target=Mock(return_value=5000);b.record=Mock()
        b.dispatch_attack=Mock(return_value=True);b.pending={'arrival':9999999999}
        return b

    def test_arrival_rechecks_staging_then_rescouts_and_dispatches_from_fixed_anchor(self):
        b=self.arrived()
        self.assertTrue(b.check_muster())
        b.scout_target.assert_called_once_with((800,300),force=True)
        b.dispatch_attack.assert_called_once()
        self.assertEqual(b.dispatch_attack.call_args.args[:5],((800,300),(640,330),1500,1250,1435))
        b.save_muster.assert_called_with('dispatched')
        self.assertIsNone(b.muster)

    def test_stronger_rescout_keeps_anchor_for_another_gather_round(self):
        b=self.arrived();b.scout_target.return_value=8000
        b.muster['single_transfer']=False  # legacy journal behavior only
        self.assertTrue(b.check_muster())
        b.dispatch_attack.assert_not_called()
        self.assertEqual(b.muster['staging_name'],'Anchor')
        self.assertEqual(b.muster['planning_base'],1500)
        self.assertEqual(b.muster['required'],2000)
        b.save_muster.assert_called_with('discovering')

    def test_single_bulk_never_drips_more_troops_after_stronger_rescout(self):
        b=self.arrived();b.scout_target.return_value=8000
        self.assertFalse(b.check_muster())
        b.dispatch_attack.assert_not_called()
        self.assertIsNone(b.muster)
        b.save_muster.assert_called_with('cancelled')

    def test_unrelated_camera_jump_is_retried_but_never_an_infinite_loop(self):
        b=self.fixture();self.begin(b);b.rebase_muster=Mock(side_effect=RuntimeError('unrelated map'))
        b.cancel_muster=Mock();b.advance_muster=Mock()
        self.assertTrue(b.check_muster());self.assertTrue(b.check_muster());self.assertFalse(b.check_muster())
        b.cancel_muster.assert_called_once();b.advance_muster.assert_not_called()

    def test_real_pan_that_used_to_abort_can_be_registered(self):
        p=Path(__file__).parent/'fixtures'
        v=Vision(None,p)
        h=v.muster_transform(Image.open(p/'muster-tracking-reference.png'),Image.open(p/'muster-tracking-after.png'))
        self.assertIsNotNone(h)
        x,y=v.transform_point(h,(719,325))
        self.assertLess(abs(x-552),12);self.assertLess(abs(y-325),12)

    def test_failed_pan_reverses_camera_and_requires_verified_registration(self):
        b=self.fixture();b.rebase_muster=Mock(side_effect=[RuntimeError('weak features'),None])
        self.assertFalse(b.pan_muster(120,0))
        self.assertEqual([c.args for c in b.driver.pan_map.call_args_list],[(120,0),(-120,0)])
        self.assertEqual(b.rebase_muster.call_count,2)

    def test_slider_rounding_never_creates_a_negative_contribution(self):
        plan=plan_muster(100,100,[MusterDonor((1,1),1000,1),MusterDonor((2,2),100,2)],.35,quantum=.1)
        self.assertEqual(len(plan.contributions),1)
        self.assertEqual(plan.contributions[0].troops,100)


if __name__=='__main__':unittest.main()
