import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
from PIL import Image

from live_lords.exclusions import PlayerExclusions
from live_lords.targets import TargetMemory
from live_lords.planner import MusterDonor,plan_muster
from live_lords.runner import Bot
from live_lords.vision import Vision
from live_lords.app import App
import test_live_attack_path as attack_tests


class BulkExclusionTests(unittest.TestCase):
    def test_bulk_selects_one_large_source_and_ignores_nearby_small_sources(self):
        donors=[MusterDonor((100,100),100,1),MusterDonor((500,300),10000,500)]
        plan=plan_muster(1000,100,donors,.35,'single',quantum=.1)
        self.assertEqual(len(plan.contributions),1)
        self.assertEqual(plan.contributions[0].position,(500,300))
        self.assertGreaterEqual(plan.contributions[0].troops+100-plan.staging_reserve,1000)

    def test_total_capacity_of_small_sources_does_not_authorize_split_sends(self):
        self.assertIsNone(plan_muster(1000,100,[MusterDonor((i,1),1000,i) for i in range(5)],.35,'single',quantum=.1))

    def test_actual_short_donor_never_clamps_requested_bulk_to_small_send(self):
        b=attack_tests.AttackPathTests().transfer_actor({})
        self.assertIsNone(b.prepare_transfer((640,330),50,{'single_transfer':True,'staging_name':'Anchor'},Mock()))
        b.tap.assert_not_called();b.select_troops.assert_not_called()

    def test_seed_player_and_new_restricted_owner_survive_restart(self):
        with tempfile.TemporaryDirectory() as d:
            e=PlayerExclusions(d)
            self.assertTrue(e.blocked('-PRABOW0-'))
            self.assertTrue(e.blocked('grab6wo'))
            self.assertTrue(e.blocked('prabolfo'))
            self.assertTrue(e.blocked_city('astv'))
            e.exclude('Other Player','병력 배치 제한',city='OtherCity')
            restored=PlayerExclusions(d)
            self.assertTrue(restored.blocked_identity('otherplayer:newcity'))
            self.assertFalse(restored.blocked('Unrelated'))

    def test_excluded_known_map_location_cannot_be_clicked_even_for_force_rescout(self):
        with tempfile.TemporaryDirectory() as d:
            b=Bot.__new__(Bot);b.data=Path(d);b.image=Image.new('RGB',(1280,720))
            b.map=Mock();b.tap=Mock();b.skip_shielded=Mock(return_value=False)
            b.targets=TargetMemory(d);b.targets.exclusions=PlayerExclusions(d)
            b.targets.add(b.image,(400,300),'prabowo:astev',seconds=-1)
            b.vision=Mock();b.vision.viewport_transform.return_value='verified'
            b.vision.transform_point.return_value=(400,300)
            self.assertIsNone(b.scout_target((400,300),force=True))
            b.tap.assert_not_called()

    def test_penalty_tooltip_persists_player_and_cancels_before_send(self):
        with tempfile.TemporaryDirectory() as d:
            b=Bot.__new__(Bot);b.data=Path(d);b.exclusions=PlayerExclusions(d)
            b.vision=Mock();b.vision.deployment_restriction.return_value='병력 배치는 제한됩니다'
            b.image=Mock();b.back=Mock();b.map=Mock();b.log=Mock();b.tap=Mock()
            self.assertTrue(b.reject_restricted_attack('NewPlayer','NewCity'))
            self.assertTrue(b.exclusions.blocked('NewPlayer'));b.tap.assert_not_called()
            b.back.assert_called_once()

    def test_selected_excluded_owner_never_receives_a_scout_even_when_forced(self):
        with tempfile.TemporaryDirectory() as d:
            b=Bot.__new__(Bot);b.data=Path(d);b.image=Image.new('RGB',(1280,720));b.muster=None
            b.map=Mock();b.tap=Mock();b.log=Mock();b.skip_shielded=Mock(return_value=False)
            b.exclusions=PlayerExclusions(d);b.targets=TargetMemory(d);b.targets.exclusions=b.exclusions
            b.align_view=Mock(return_value='matrix');b.vision=Mock()
            b.vision.transform_point.return_value=(400,300);b.vision.match.return_value=(500,300,1)
            b.vision.enemy_identity.return_value='grab6wo:astv'
            self.assertIsNone(b.scout_target((400,300),force=True))
            b.tap.assert_called_once_with(400,300,delay=.8)  # identity selection only; no scout dispatch
            self.assertEqual(b.last_scout_failure,'excluded_player')

    def test_excluded_owner_cannot_begin_attack_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            b=Bot.__new__(Bot);b.exclusions=PlayerExclusions(d);b.last_scout_owner='prabowo'
            b.select_owned=Mock();b.log=Mock()
            self.assertFalse(b.dispatch_attack((400,300),(500,300),100,10,30,{}))
            b.select_owned.assert_not_called()

    def test_captured_penalty_tooltip_uses_korean_ocr_and_rejects_normal_text(self):
        with tempfile.TemporaryDirectory() as d:
            v=Vision(None,d);v.regions=Mock(return_value={'restriction0':'상대의 영웅 점수가 적을 경우 병력 배치는 제한됩니다'})
            image=Image.open(Path(__file__).parent/'fixtures'/'deployment-restriction.png')
            self.assertIsNotNone(v.deployment_restriction(image))
            v.regions.return_value={'restriction0':'일반 안내'}
            self.assertIsNone(v.deployment_restriction(image))

    def test_live_interval_one_applies_without_restarting_worker(self):
        with tempfile.TemporaryDirectory() as d,patch.object(App,'refresh',lambda _:None):
            app=App();app.withdraw()
            try:
                app.active_config={'interval':6,'exploration_speed':4,'decision_speed':3}
                app.active_directory=Path(d);shared=app.active_config
                app.interval.set(1);app.apply_speed_settings()
                self.assertIs(shared,app.active_config);self.assertEqual(shared['interval'],1)
                self.assertTrue((Path(d)/'config.json').exists())
            finally:app.destroy()

    def test_feature_and_projection_results_are_reused(self):
        with tempfile.TemporaryDirectory() as d:
            image=Image.new('RGB',(1280,720));v=Vision(None,d)
            self.assertIs(v.terrain_features(image,'800'),v.terrain_features(image.copy(),'800'))
            memory=TargetMemory(d);memory.add(image,(400,300),'normal:city')
            vision=Mock();vision.viewport_transform.return_value='verified';vision.transform_point.return_value=(400,300)
            self.assertTrue(memory.blocked(image,(400,300),vision))
            self.assertFalse(memory.blocked(image,(900,300),vision))
            vision.viewport_transform.assert_called_once()

    def test_known_excluded_locations_survive_regular_history_limit(self):
        with tempfile.TemporaryDirectory() as d:
            memory=TargetMemory(d);memory.exclusions=PlayerExclusions(d)
            memory.entries=[{'view':'permanent.png','point':[400,300],'until':0,'identities':['prabowo:astev']}]
            memory.entries.extend({'view':str(i)+'.png','point':[500,300],'until':0,'identities':['normal:city'+str(i)]} for i in range(60))
            memory.save()
            self.assertEqual(len(memory.entries),49)
            self.assertEqual(memory.entries[0]['identities'],['prabowo:astev'])


if __name__=='__main__':unittest.main()
