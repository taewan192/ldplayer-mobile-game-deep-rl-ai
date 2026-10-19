import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from live_lords.vision import Vision
from live_lords.runner import Bot
from live_lords.planner import plan_muster,MusterDonor
from live_lords.muster import MusterMixin


class AttackPathTests(unittest.TestCase):
    def test_captured_level88_city_uses_its_actual_move_button_variant(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,Path(folder)); v.regions=Mock(return_value={'city_name':'Aughestou','troops':'268M'})
            image=Image.new('RGB',(1280,720))
            with Image.open(Path(__file__).with_name('fixtures')/'city88-controls.png') as controls:
                image.paste(controls,(220,270))
            self.assertIsNone(v.match(image,'move_troops',.84))
            city=v.selected_own_city(image,'Yumi')
            self.assertEqual(city['name'],'Aughestou'); self.assertEqual(city['move_button'],(486,513))

    def test_owned_flag_fallback_identifies_the_same_castle_banner(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,Path(folder))
            with Image.open(Path(__file__).with_name('fixtures')/'city-interaction-map.png') as im:
                self.assertEqual(v.own_flag_for_castle(im,(640,319)),(640,224))
                self.assertIsNone(v.own_flag_for_castle(im,(373,269)))

    def result_actor(self,city):
        b=Bot.__new__(Bot); b.pending={'target':(400,300),'arrival':0,'before':{}}
        b.map=Mock(); b.select_owned=Mock(return_value=city); b.capture=Mock(); b.wait=Mock(); b.hud=Mock()
        b.vision=Mock(); b.vision.selected_own_city.return_value=city
        b.vision.own_sources.return_value=[]
        b.image=Mock(); b.state={'owner':'Yumi'}; b.counts={'conquests':0}; b.log=Mock(); b.record=Mock()
        return b

    def test_unreadable_arrival_does_not_train_a_false_defeat(self):
        b=self.result_actor(None); self.assertFalse(b.check_pending())
        b.record.assert_not_called(); self.assertEqual(b.counts['conquests'],0); self.assertIsNone(b.pending)

    def test_arrival_waits_for_actual_owned_city_controls_before_learning_win(self):
        b=self.result_actor({'name':'Aughestou','troops':330e6}); b.check_pending()
        b.select_owned.assert_called_once_with((400,300))
        self.assertEqual(b.counts['conquests'],1); self.assertTrue(b.record.call_args.args[3])

    def test_actual_outlined_city_name_isolated_read_still_requires_exact_match(self):
        with tempfile.TemporaryDirectory() as folder:
            image=Image.new('RGB',(1280,720))
            with Image.open(Path(__file__).with_name('fixtures')/'enemy-name-ink.png') as ink:
                image.paste(ink,(390,305))
            ocr=Mock(); ocr.read.return_value={'text':'Aughestou'}
            v=Vision(ocr,Path(folder)); v.regions=Mock(side_effect=[
                {'city_name':'Abughstou','attacker':'Raja Humabon'},
                {'city_wide':'Abughstou','city_tight':'ghestou','city_baseline':'Abughstou'}])
            self.assertEqual(v.enemy_identity(image,(315,512),'aughestou'),'rajahumabon:aughestou')
            ocr.read.assert_called_once()

    def slider_actor(self,value):
        b=MusterMixin(); b.capture=Mock(); b.wait=Mock(); b.driver=Mock(); b.log=Mock(); b.image=Mock()
        b.vision=Mock(); b.vision.troop_slider_handle.return_value=457
        b.region=Mock(return_value={'send':str(value),'time':'0m 22s'})
        return b

    def test_slider_cannot_silently_send_twice_the_requested_game_unit(self):
        b=self.slider_actor(20)
        self.assertIsNone(b.select_troops(10,100,35))
        b=self.slider_actor(10)
        self.assertEqual(b.select_troops(10,100,35),(10,22))
        b=self.slider_actor(20)
        self.assertEqual(b.select_troops(19,100,35),(20,22))

    def test_slider_partial_gesture_is_corrected_from_observed_quantity(self):
        b=self.slider_actor(50)
        b.region.side_effect=[{'send':str(value),'time':'0m 22s'} for value in (50,20,10)]
        self.assertEqual(b.select_troops(10,100,35),(10,22))
        self.assertEqual(b.driver.drag_troop_slider.call_count,3)

    def test_arrived_staging_army_with_ocr_icon_suffix_is_not_lost(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,Path(folder)); v.regions=Mock(side_effect=[
                {'city_name':'laaril','troops':"1.11GØ'"},{'troops':'1.11G}U'}])
            with Image.open(Path(__file__).with_name('fixtures')/'own-staging-after-arrival.png') as im:
                city=v.selected_own_city(im,'Yumi')
            self.assertEqual(city['name'],'laaril'); self.assertEqual(city['troops'],1.11e9)

    def transfer_actor(self,alternatives):
        b=MusterMixin(); b.config={'max_send_fraction':.35}; b.map=Mock(); b.tap=Mock(); b.back=Mock()
        b.select_owned=Mock(return_value={'name':'Donor','troops':100,'move_button':(700,390)})
        b.region=Mock(side_effect=[{'prompt':'목표 선택'},{'city_name':'laarit'},alternatives])
        b.vision=Mock(); b.vision.match.return_value=(880,545,1)
        b.image=Mock(); b.data=Path('.'); b.log=Mock(); b.select_troops=Mock(return_value=(10,20))
        return b

    def test_transfer_crop_retry_requires_exact_destination_before_send_preparation(self):
        b=self.transfer_actor({'city_tight':'laaril','city_wide':'laaril'})
        result=b.prepare_transfer((640,330),10,{'staging':(400,300),'staging_name':'laaril'},Mock())
        self.assertEqual(result['troops'],10); b.back.assert_not_called()
        b=self.transfer_actor({'city_tight':'Other','city_wide':'laarit'})
        self.assertIsNone(b.prepare_transfer((640,330),10,{'staging':(400,300),'staging_name':'laaril'},Mock()))
        b.back.assert_called_once(); b.select_troops.assert_not_called()

    def test_player_city_transition_waits_for_transfer_form_without_cancelling_route(self):
        b=self.transfer_actor({'city_tight':'laaril','city_wide':'laaril'})
        b.wait=Mock();b.capture=Mock()
        frames=iter([None,None,(880,545,1)])
        b.vision.match.side_effect=lambda *args,**kwargs:next(frames,(880,545,1))
        result=b.prepare_transfer((640,330),10,{'staging':(400,300),'staging_name':'laaril'},Mock())
        self.assertEqual(result['troops'],10)
        self.assertEqual(b.wait.call_count,2)
        b.back.assert_not_called()

    def test_icon_merged_value_retries_narrow_crop_instead_of_dropping_donor(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,Path(folder)); v.regions=Mock(side_effect=[{'troops':''},{'troops':'2.85G'}])
            with Image.open(Path(__file__).with_name('fixtures')/'city-interaction-map.png') as im:
                sources=v.own_sources(im,'Yumi')
            self.assertEqual(sources,[((640,319),2.85e9)])
            self.assertEqual(v.regions.call_args.args[1]['troops'],(666,208,749,239))

    def test_observed_muster_shortfall_is_one_donor_unit_with_staging_reserve(self):
        plan=plan_muster(378926442,541e6,[MusterDonor((640,319),2850e6,270)],.35,quantum=.1)
        self.assertEqual(len(plan.contributions),1)
        self.assertEqual(plan.contributions[0].troops,285e6)
        total=541e6+285e6
        self.assertGreaterEqual(total*.5,plan.required)
        self.assertGreaterEqual(total*.5,plan.staging_reserve)

    def test_real_map_own_city_coordinate_hits_castle_instead_of_right_hand_ground(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,Path(folder)); v.regions=Mock(return_value={'troops':'3.41G'})
            with Image.open(Path(__file__).with_name('fixtures')/'city-interaction-map.png') as im:
                sources=v.own_sources(im,'Yumi')
            self.assertEqual(sources,[((640,319),3.41e9)])

    def test_revealed_enemy_troop_banner_does_not_shift_castle_hit_point(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,Path(folder))
            with Image.open(Path(__file__).with_name('fixtures')/'city-interaction-map.png') as im:
                targets=v.enemy_candidates(im)
            self.assertIn((373,269),targets)
            self.assertNotIn((413,269),targets)

    def test_name_retry_requires_exact_report_name_in_observed_crop(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,Path(folder)); v.regions=Mock(side_effect=[
                {'city_name':'laotit','attacker':'Raja Humabon'},
                {'city_wide':'laarit','city_tight':'laarit'}])
            self.assertEqual(v.enemy_identity(None,(311,333),'laarit'),'rajahumabon:laarit')
            v.regions=Mock(side_effect=[{'city_name':'Other','attacker':'Raja Humabon'},
                                      {'city_wide':'Other','city_tight':'Another'}])
            self.assertNotEqual(v.enemy_identity(None,(311,333),'laarit'),'rajahumabon:laarit')

    def test_candidate_without_scout_does_not_starve_the_next_enemy(self):
        b=Bot.__new__(Bot); b.config={'attack_enabled':True,'combat_margin':1.5,'max_send_fraction':.35}
        b.state={'owner':'Yumi','attack_bonus':100}; b.image=Mock(); b.map=Mock(); b.capture=Mock()
        b.incoming=Mock(return_value=False); b.align_view=Mock(return_value=True)
        b.vision=Mock(); b.vision.own_sources.return_value=[((640,319),1000)]
        b.vision.transform_point.side_effect=lambda m,p:p
        b.vision.enemy_candidates.return_value=[(373,269),(378,455)]
        b.targets=Mock(); b.targets.blocked.return_value=False; b.decide=Mock()
        b.last_stats_update=float('inf'); b.last_scout_transform=True; b.last_scout_failure='no_scout_button'
        b.scout_target=Mock(side_effect=[None,100]); b.dispatch_attack=Mock(return_value=True)
        self.assertEqual(b.attack_nearby(),'dispatched')
        self.assertEqual(b.scout_target.call_count,2)
        self.assertEqual(b.dispatch_attack.call_args.args[0],(378,455))
