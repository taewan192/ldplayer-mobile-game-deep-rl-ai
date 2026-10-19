import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np

from PIL import Image

from live_lords.defense_logistics import DefenseLogisticsMixin,explicit_army,defense_waves
from live_lords.focus_defense import troop_goal
from live_lords.runner import Bot,Stopped,DefensePreempted
from live_lords.vision import Vision
from live_lords.defense import DefenseMixin


class DefenseLogisticsTests(unittest.TestCase):
    def test_unitless_ocr_g_as_six_is_not_an_army(self):
        self.assertIsNone(explicit_army('81.16'))
        self.assertEqual(explicit_army('81.1G'),81.1e9)
        self.assertIsNone(explicit_army('9.72Gjl'))
        self.assertEqual(explicit_army('9.72G'),9.72e9)

    def test_future_wave_does_not_inflate_first_deadline_goal(self):
        rows=[dict(impact=100,troops=10e9,troops_min=1e9,troops_exact=False),
              dict(impact=1000,troops=100e9,troops_min=10e9,troops_exact=False)]
        waves=defense_waves(rows)
        self.assertEqual(len(waves),2)
        self.assertEqual(troop_goal(waves[0],84),12_987_000_000)
        self.assertEqual(troop_goal(waves[1],84),129_870_000_000)

    def test_wave_window_is_anchored_not_transitively_unbounded(self):
        waves=defense_waves([dict(impact=t) for t in (160,100,125)])
        self.assertEqual([[r['impact'] for r in w] for w in waves],[[100,125],[160]])

    def test_uncertain_clock_has_no_invented_wave_deadline(self):
        self.assertEqual(defense_waves([{'time_uncertain':True}]),[])

    def test_city_i_l_alias_requires_known_unique_owned_name(self):
        b=DefenseLogisticsMixin();b.strategy=Mock();b.strategy.state={'cities':{'a':{'name':'Tiaoughar'}}}
        rows=b.canonical_defense_rows([{'city':'Tlaoughar'}])
        self.assertEqual(rows[0]['city'],'Tiaoughar')
        b.strategy.state['cities']['b']={'name':'Tlaoughar'}
        self.assertEqual(b.canonical_defense_rows([{'city':'Tlaoughar'}])[0]['city'],'Tlaoughar')

    def test_blue_support_bubble_is_detected_and_border_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            with Image.open(Path(__file__).parent/'fixtures/support-blue-selection-v27.png') as im:
                self.assertEqual(v.troop_bubble_box(im),(596,474,112,66))
                def fields(image,boxes):
                    self.assertGreaterEqual(boxes['selection_number'][0],605)
                    return {'selection_number':'56.4M'}
                v.regions=Mock(side_effect=fields)
                self.assertEqual(v.troop_selection_amount(im),56.4e6)

    def test_dark_blue_timer_uses_dark_glyph_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder)
            v.regions=Mock(side_effect=[{'troop_eta':''},{'troop_eta':''},{'troop_eta_dark':'0m 27s'}])
            with Image.open(Path(__file__).parent/'fixtures/support-blue-selection-v27.png') as im:
                self.assertEqual(v.troop_selection_timer(im),27)

    def test_visible_garrisons_use_one_ocr_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(None,folder);v.own_flags=Mock(return_value=[(400,250,1,1),(700,250,1,1)])
            v.regions=Mock(return_value={'source0':'81.16','source0_wide':'81.1G','source1':'9.72G','source1_wide':''})
            result=v.own_sources_batch(Image.new('RGB',(1280,720)),'Yumi')
            self.assertEqual([n for _,n in result],[81.1e9,9.72e9]);self.assertEqual(v.regions.call_count,1)

    def controller(self):
        b=DefenseLogisticsMixin();b.config={};b.map=Mock();b.tap=Mock();b.wait=Mock();b.capture=Mock()
        b.log=Mock();b.save_defenses=Mock();b.back=Mock();b.image=Image.new('RGB',(1280,720))
        b.vision=Mock();b.vision.match.return_value=None
        b.vision.march_button.return_value=(1225,570,1);b.vision.march_panel.return_value='행군'
        return b

    def test_map_return_without_matching_march_is_not_counted_as_sent(self):
        b=self.controller();b.vision.outgoing_support_rows.return_value=[]
        move={'status':'intent','troops':100,'source':'Donor'}
        self.assertFalse(b.confirm_defense_dispatch({'staging_name':'Target'},move,{'duration':50}))
        self.assertEqual(move['status'],'intent')

    def test_march_requires_target_amount_and_eta_to_acknowledge(self):
        b=self.controller();move={'status':'intent','troops':100,'source':'Donor'}
        b.vision.outgoing_support_rows.return_value=[dict(city='Target',troops=100,remaining=48,pin=(650,148),row=148)]
        with patch('live_lords.defense_logistics.time.time',return_value=100):
            self.assertTrue(b.confirm_defense_dispatch({'staging_name':'Target','impact':200},move,{'duration':50}))
        self.assertEqual(move['status'],'sent');self.assertEqual(move['arrival'],148)
        self.assertEqual(move['confirmed_by'],'outgoing_city_troops_eta')

    def test_different_target_or_quantity_never_acknowledges(self):
        for change in ({'city':'Wrong'},{'troops':1000},{'remaining':200}):
            b=self.controller();row=dict(city='Target',troops=100,remaining=48,pin=(650,148),row=148);row.update(change)
            b.vision.outgoing_support_rows.return_value=[row]
            move={'status':'intent','troops':100,'source':'Donor'}
            self.assertFalse(b.confirm_defense_dispatch({'staging_name':'Target','impact':200},move,{'duration':50}))
            self.assertEqual(move['status'],'intent')

    def test_unchanged_form_is_a_failed_send_not_an_intent_reservation(self):
        b=self.controller();b.vision.match.return_value=(880,540,1)
        move={'status':'intent','troops':100,'source':'Donor'}
        self.assertFalse(b.confirm_defense_dispatch({'staging_name':'Target'},move,{'duration':50}))
        self.assertEqual(move['status'],'not_sent');b.vision.outgoing_support_rows.assert_not_called()

    def test_no_speedup_control_never_clicks_troop_label(self):
        b=self.controller();b.vision.march_acceleration_button.return_value=None
        self.assertFalse(b.accelerate_defense_march({}, {}, {'row':148}))
        b.tap.assert_not_called()

    def test_speedup_does_not_spend_item_when_still_late(self):
        b=self.controller();b.vision.march_acceleration_button.return_value=(1050,148,1)
        b.vision.acceleration_option.return_value={'remaining_after':60,'button':(500,500)}
        with patch('live_lords.defense_logistics.time.time',return_value=100):
            self.assertFalse(b.accelerate_defense_march({'impact':150},{},{'row':148}))
        self.assertEqual(b.tap.call_count,1)

    def test_consumed_speedup_requires_observed_countdown_change(self):
        b=self.controller();b.vision.march_acceleration_button.return_value=(1050,148,1)
        b.vision.acceleration_option.return_value={'remaining_after':20,'button':(500,500)}
        b.vision.outgoing_support_rows.return_value=[];move={}
        with patch('live_lords.defense_logistics.time.time',return_value=100):
            self.assertFalse(b.accelerate_defense_march({'staging_name':'Target','impact':150},move,{'row':148,'remaining':60}))
        self.assertIn('acceleration_intent',move);self.assertNotIn('accelerated',move)

    def test_standby_does_not_compete_with_active_incoming(self):
        b=self.controller();b.strategy=Mock();b.strategy.state={};b.defense_busy=Mock(return_value=True)
        b.defense_city_inventory=Mock();b.prepare_defense_standby();b.defense_city_inventory.assert_not_called()

    def test_pinned_main_city_is_not_mistaken_for_largest_garrison(self):
        b=self.controller();b.city_list_page=Mock(return_value=(1,1));b.vision.match.return_value=(250,125,1)
        b.strategy=Mock();b.strategy.state={'cities':{}}
        fields={f'{key}{i}':value for i,(name,army) in enumerate((('Main','76.1M'),('Bank','81.1G')))
                for key,value in (('name',name),('army',army),('level','110'))}
        b.region=Mock(return_value=fields);b.vision.city_list_army.return_value=None
        inventory=b.defense_city_inventory()
        self.assertEqual([c['name'] for c in inventory],['Bank','Main'])

    def test_recovery_handler_does_not_escape_on_defense_preemption(self):
        b=Bot.__new__(Bot);b.config={'focus_defense':True};b.stop=threading.Event()
        b.capture=Mock();b.focus_defense_tick=Mock(side_effect=[RuntimeError('OCR'),Stopped()])
        b.recover_navigation=Mock(return_value=True);b.recovery_pause=Mock();b.wait=Mock(side_effect=DefensePreempted())
        b.decide=Mock();b.log=Mock();b.emit=Mock();b.learner=Mock();b.vision=Mock()
        b.run();self.assertEqual(b.focus_defense_tick.call_count,2)
        b.recovery_pause.assert_called_once_with(1);b.wait.assert_not_called()
        self.assertFalse(any('error' in c.args[0] for c in b.emit.call_args_list))

    def test_missing_alert_requires_repeated_map_observation_before_releasing(self):
        b=self.controller();b.vision.march_button.return_value=None;b.vision.match.return_value=(50,670,1)
        b.defenses=[dict(impact=500,status='unverified',moves=[])];b.threatened_names={'target'}
        b.reconcile_defense_absence(100);self.assertEqual(b.defenses[0]['status'],'unverified')
        b.reconcile_defense_absence(102);self.assertEqual(b.defenses[0]['status'],'unverified')
        b.reconcile_defense_absence(104);self.assertEqual(b.defenses[0]['status'],'cancelled')

    def test_disappearance_never_cancels_a_submitted_support(self):
        b=self.controller();b.vision.march_button.return_value=None;b.vision.match.return_value=(50,670,1)
        b.defenses=[dict(impact=500,status='waiting',moves=[{'status':'sent'}])];b.threatened_names={'target'}
        b.reconcile_defense_absence(100);b.reconcile_defense_absence(104)
        self.assertEqual(b.defenses[0]['status'],'waiting')

    def test_expired_panel_retry_does_not_hold_an_empty_map_forever(self):
        b=self.controller();b.vision.march_button.return_value=None;b.vision.match.return_value=(50,670,1)
        b.defenses=[];b.threatened_names=set();b.defense_read_waiting=True
        b.reconcile_defense_absence(100);self.assertTrue(b.defense_read_waiting)
        b.reconcile_defense_absence(104);self.assertFalse(b.defense_read_waiting)

    def test_camera_bridge_composes_only_observed_transforms(self):
        with tempfile.TemporaryDirectory() as folder:
            b=self.controller();b.data=Path(folder)
            red=Image.new('RGB',(30,30),'red');green=Image.new('RGB',(30,30),'green');blue=Image.new('RGB',(30,30),'blue')
            b.remember_defense_view('Middle',green)
            def transform(a,z):
                pair=(a.getpixel((0,0)),z.getpixel((0,0)))
                if pair==(red.getpixel((0,0)),green.getpixel((0,0))):return np.array([[1.,0,100],[0,1,0]])
                if pair==(green.getpixel((0,0)),blue.getpixel((0,0))):return np.array([[1.,0,50],[0,1,0]])
                return None
            b.vision.muster_transform.side_effect=transform
            result=b.linked_defense_transform(red,blue)
            np.testing.assert_array_equal(result,np.array([[1.,0,150],[0,1,0],[0,0,1]]))

    def test_late_route_requires_measured_free_speedup_capability(self):
        b=self.controller();b.strategy=Mock();b.strategy.state={}
        with patch('live_lords.defense_logistics.time.time',return_value=100):
            self.assertFalse(b.support_arrival_feasible(100,170))
            b.strategy.state['free_support_speedup_ratio']=.5
            self.assertTrue(b.support_arrival_feasible(100,170))
            b.config['defense_acceleration_enabled']=False
            self.assertFalse(b.support_arrival_feasible(100,170))

    def test_free_boost_requires_matching_amount_and_observed_faster_timer(self):
        b=self.controller();b.vision.match.return_value=(319,239,1);b.strategy=Mock();b.strategy.state={}
        b.vision.outgoing_support_rows.return_value=[{'city':'Target','troops':100,'remaining':40}]
        move={'troops':100}
        with patch('live_lords.defense_logistics.time.time',return_value=100):
            self.assertTrue(b.accelerate_defense_march({'staging_name':'Target'},move,{'row':239,'remaining':100}))
        self.assertEqual(move['arrival'],140);self.assertTrue(move['accelerated'])
        self.assertLess(b.strategy.state['free_support_speedup_ratio'],.5)

    def test_rebase_accepts_named_list_donors_before_they_have_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            b=DefenseMixin();b.data=Path(folder);b.image=Image.new('RGB',(1280,720));b.image.save(b.data/'view.png')
            b.active_defense={'view':'view.png','staging':(600,300),'target':(600,300),
                'donors':[{'name':'Bank','from_city_list':True},{'position':None},{'position':(500,300)}]}
            b.vision=Vision(None,b.data);b.vision.viewport_transform=Mock(return_value=np.array([[1.,0,10],[0,1,0]]))
            b.save_defenses=Mock();b.rebase_defense()
            self.assertNotIn('position',b.active_defense['donors'][0])
            self.assertEqual(b.active_defense['donors'][2]['position'],(510,300))

    def test_threatened_large_garrison_can_help_only_after_its_own_reserve(self):
        b=self.controller();b.threatened_names={'bank'};b.latest_defense_rows_at=100
        b.latest_defense_rows=[{'city':'Bank','troops':10e9,'troops_min':1e9,'troops_exact':False}]
        with patch('live_lords.defense_logistics.time.time',return_value=100):
            self.assertEqual(b.defense_donor_capacity({'name':'Bank','troops':81e9,'level':110}),8.1e9)
            self.assertEqual(b.defense_donor_capacity({'name':'Bank','troops':10e9,'level':110}),0)

    def test_threatened_donor_with_stale_or_unknown_attack_stays_excluded(self):
        b=self.controller();b.threatened_names={'bank'};b.latest_defense_rows_at=50;b.latest_defense_rows=[]
        with patch('live_lords.defense_logistics.time.time',return_value=100):
            self.assertEqual(b.defense_donor_capacity({'name':'Bank','troops':81e9,'level':110}),0)

    def test_troop_defense_does_not_require_unrelated_wall_ocr(self):
        b=self.controller();b.select_owned_named=Mock(return_value={'name':'Elta','troops':1e9,'position':(640,330)})
        b.read_owned_details=Mock(side_effect=AssertionError('Wall OCR is irrelevant'))
        b.strategy=Mock();b.strategy.state={'cities':{}}
        city=b.read_focus_defense_target({'city':'Elta','level':84})
        self.assertEqual(city['troops'],1e9);self.assertEqual(city['level'],84)
        b.read_owned_details.assert_not_called()

    def test_missing_target_banner_cannot_be_replaced_by_incoming_label(self):
        b=self.controller();b.select_owned_named=Mock(return_value=None);b.strategy=Mock()
        self.assertIsNone(b.read_focus_defense_target({'city':'Elta','level':84}))
        b.strategy.observe_city.assert_not_called()


if __name__=='__main__':unittest.main()
