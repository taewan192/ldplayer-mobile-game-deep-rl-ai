import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from PIL import Image,ImageDraw

from live_lords.muster import MusterMixin
from live_lords.planner import MusterDonor,plan_muster,march_seconds
from live_lords.vision import Vision,percentage


class MusterTests(unittest.TestCase):
    def test_combat_percentage_ocr_variant_does_not_turn_269_into_2690(self):
        self.assertEqual(percentage('+2690/0'),269)
        self.assertEqual(percentage('+144.25%'),144.25)
        self.assertIsNone(percentage('245'))

    def test_bonus_rows_are_located_by_icon_not_sort_order(self):
        with tempfile.TemporaryDirectory() as directory:
            v=Vision(None,Path(directory))
            v.match=Mock(side_effect=[(247,274,1),(247,350,1)])
            v.regions=Mock(return_value={'attack':'+245%','defense':'+144.25%'})
            self.assertEqual(v.combat_bonuses(None),{'attack':245,'defense':144.25})
            self.assertEqual(v.regions.call_args.args[1]['attack'][1],257)

    def test_moves_only_shortfall_and_retains_original_staging_reserve(self):
        plan=plan_muster(100,200,[MusterDonor((1,1),200,1)],.35)
        self.assertEqual(plan.staging_reserve,130)
        self.assertEqual(sum(c.troops for c in plan.contributions),30)
        self.assertEqual(200-plan.staging_reserve+30,100)

    def test_each_donor_keeps_reserve_and_no_partial_mission_when_total_insufficient(self):
        donors=[MusterDonor((1,1),100,1),MusterDonor((2,2),100,2)]
        plan=plan_muster(100,100,donors,.35)
        self.assertEqual([c.troops for c in plan.contributions],[35,30])
        self.assertTrue(all(c.troops<=c.available for c in plan.contributions))
        self.assertIsNone(plan_muster(106,100,donors,.35))

    def test_no_transfer_when_staging_alone_is_sufficient(self):
        self.assertEqual(plan_muster(70,200,[],.35).contributions,())

    def test_duplicate_source_not_counted_twice(self):
        donor=MusterDonor((1,1),100,1)
        self.assertIsNone(plan_muster(80,100,[donor,donor],.35))

    def test_learning_choices_only_change_order_not_reserve_or_required_amount(self):
        donors=[MusterDonor((1,1),100,1),MusterDonor((2,2),300,10)]
        nearest=plan_muster(100,100,donors,.35,'nearest')
        fewest=plan_muster(100,100,donors,.35,'fewest')
        self.assertEqual(len(nearest.contributions),2)
        self.assertEqual(len(fewest.contributions),1)
        self.assertEqual(sum(x.troops for x in nearest.contributions),sum(x.troops for x in fewest.contributions))
        self.assertEqual(nearest.staging_reserve,fewest.staging_reserve)

    def test_unknown_march_time_is_not_zero(self):
        self.assertEqual(march_seconds('1h 2m 3s'),3723)
        self.assertEqual(march_seconds('0m 22s'),22)
        self.assertIsNone(march_seconds('???'))

    def test_game_slider_units_preserve_reserves_and_make_attack_selectable(self):
        donors=[MusterDonor((1,1),200,1),MusterDonor((2,2),100,2)]
        plan=plan_muster(100,100,donors,.35,quantum=.1)
        self.assertIsNotNone(plan)
        total=100+sum(c.troops for c in plan.contributions)
        self.assertTrue(any(total*i/10>=100 and total*(1-i/10)>=plan.staging_reserve
                            for i in range(1,10)))
        self.assertTrue(all(c.troops<=c.available for c in plan.contributions))

    def test_restart_waits_for_submitted_intent_without_sending_again(self):
        with tempfile.TemporaryDirectory() as directory:
            obj=MusterMixin(); obj.data=Path(directory)
            saved={'status':'preparing','arrival':1000,'moves':[{'status':'intent','troops':10}]}
            (obj.data/'muster-journal.json').write_text(json.dumps(saved),'utf-8')
            obj.init_muster()
            self.assertEqual(obj.muster['status'],'waiting')
            self.assertEqual(obj.muster['moves'],saved['moves'])

    def test_restart_preserves_attack_wait_instead_of_gathering_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            obj=MusterMixin(); obj.data=Path(directory); obj.pending=None
            saved={'status':'dispatched','attack_arrival':1000,'target':[500,400],'before':{'vp':10}}
            (obj.data/'muster-journal.json').write_text(json.dumps(saved),'utf-8')
            obj.init_muster()
            self.assertIsNone(obj.muster)
            self.assertEqual(obj.pending['arrival'],1000)

    def test_camera_translation_projects_city_coordinates_and_detects_manual_pan(self):
        with tempfile.TemporaryDirectory() as directory:
            v=Vision(None,Path(directory)); image=Image.new('RGB',(1280,720),'#ba8235')
            draw=ImageDraw.Draw(image); rng=random.Random(91)
            for _ in range(180):
                x,y=rng.randrange(220,1050),rng.randrange(170,550)
                draw.rectangle((x,y,x+12,y+12),fill=tuple(rng.randrange(256) for _ in range(3)))
            moved=Image.new('RGB',image.size); moved.paste(image,(80,-40))
            matrix=v.viewport_transform(image,moved)
            self.assertIsNotNone(matrix)
            x,y=v.transform_point(matrix,(640,330))
            self.assertLessEqual(abs(x-720),2); self.assertLessEqual(abs(y-290),2)
            self.assertFalse(v.same_viewport(moved,image))
            self.assertTrue(v.same_viewport(image,image))

    def test_increased_enemy_defense_prevents_attack_after_gathering(self):
        obj=MusterMixin()
        obj.muster={'arrival':0,'staging':(100,100),'target':(200,200),'staging_name':'Home',
                    'staging_before':100,'reserve':65}
        obj.map=Mock(); obj.verify_muster_view=Mock(return_value=True)
        obj.rebase_muster=Mock()
        obj.sync_stats=Mock()
        obj.select_owned=Mock(return_value={'name':'Home','troops':200})
        obj.scout_target=Mock(return_value=1e12)
        obj.state={'attack_bonus':500}; obj.config={'combat_margin':1.5}
        obj.log=Mock(); obj.cancel_muster=Mock(); obj.dispatch_attack=Mock()
        self.assertFalse(obj.check_muster())
        obj.cancel_muster.assert_called_once()
        obj.dispatch_attack.assert_not_called()


if __name__=='__main__': unittest.main()
