import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from PIL import Image

from live_lords.defense import DefenseMixin
from live_lords.runner import Bot,DefensePreempted
from live_lords.vision import Vision,troop_bounds,march_countdown


class IncomingDefenseTests(unittest.TestCase):
    def test_live_shield_variant_detected_and_outgoing_button_still_works(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(Mock(),Path(folder))
            im=Image.open(Path(__file__).parent/'fixtures/incoming-shield-map.png')
            self.assertIsNone(v.match(im,'march_button',.84,area=(1180,510,1280,615)))
            self.assertEqual(v.incoming_button(im)[:2],(1228,571))
            self.assertEqual(v.march_button(im)[:2],(1228,571))
            blank=Image.new('RGB',(1280,720))
            self.assertIsNone(v.incoming_button(blank))
            blank.paste(Image.open(v.assets/'march_button.png'),(1205,550))
            self.assertIsNotNone(v.march_button(blank))
            self.assertIsNone(v.incoming_button(blank))

    def test_hidden_ranges_never_become_an_exact_troop_number(self):
        self.assertEqual(troop_bounds('1G-10G'),(1e9,1e10))
        self.assertEqual(troop_bounds('IOOM-IG'),(1e8,1e9))
        self.assertEqual(troop_bounds('IG-IOG'),(1e9,1e10))
        self.assertEqual(troop_bounds('10G-1G'),(None,None))
        self.assertEqual(troop_bounds('4.36M'),(4.36e6,4.36e6))

    def test_real_clock_ocr_requires_a_whole_clock(self):
        for raw,seconds in [('3m 42s',222),('3m 428',222),('5m 598',359),
                            ('6rn 6s',366),('6m 273',387),('6m 498',409)]:
            self.assertEqual(march_countdown(raw),seconds,raw)
        for raw in ('3so','310','563','3m {2s','1m 99s'):
            self.assertIsNone(march_countdown(raw),raw)

    def test_live_list_fields_and_hidden_troop_ranges(self):
        with tempfile.TemporaryDirectory() as folder:
            v=Vision(Mock(),Path(folder))
            im=Image.open(Path(__file__).parent/'fixtures/incoming-defense-list.png')
            names={148:'laaril',239:'Uisse',329:'lehatuj',420:'Yo',511:'Oiachu',601:'Droo'}
            times={148:'2m 20s',239:'5m 598',329:'6m 6s',420:'6m 273',511:'6m 498',601:'6m 538'}
            def fields(image,boxes):
                y=boxes['march_city'][1]+21
                return {'march_city':names[y],'remaining':times[y],
                        'troops':'IG-IOG' if y==148 else 'IOOM-IG','attacker':'Raja Humabon'}
            v.regions=fields
            rows=v.defense_rows(im)
            self.assertEqual([r['city'] for r in rows],list(names.values()))
            self.assertEqual([r['remaining'] for r in rows],[140,359,366,387,409,413])
            self.assertTrue(all(not r['troops_exact'] for r in rows))

    def test_scout_wait_is_preempted_after_first_two_second_check(self):
        b=Bot.__new__(Bot);b.config={};b.stop=threading.Event();b.image=Mock()
        b.stop=Mock();b.stop.wait.return_value=False;b.capture=Mock();b.vision=Mock();b.log=Mock()
        b.vision.incoming_button.return_value=(1228,571,1)
        with self.assertRaises(DefensePreempted): b.wait(12)
        b.stop.wait.assert_called_once();b.capture.assert_called_once()
        self.assertLessEqual(b.stop.wait.call_args.args[0],2)

    def precaution_fixture(self):
        b=DefenseMixin();b.config={'max_send_fraction':.35,'defense_buffer':20};b.image=Mock()
        b.state={'owner':'Yumi'};b.vision=Mock();b.threatened_names=set();b.counts={'supports':0}
        b.vision.own_sources.return_value=[((800,330),1000)]
        b.save_defenses=Mock();b.rebase_defense=Mock();b.map=Mock();b.back=Mock();b.log=Mock();b.tap=Mock()
        b.defense_snapshot_valid=Mock(return_value=True)
        job={'status':'preparing','attack':None,'staging':(640,330),'staging_name':'Home',
             'impact':300,'before':{'troops':200},'moves':[]}
        prepared={'donor':{'name':'Donor','troops':1000},'troops':200,'duration':10,
                  'observed':100,'button':(880,540)}
        b.prepare_transfer=Mock(return_value=prepared)
        return b,job

    def test_hidden_power_still_sends_one_bounded_timely_support_with_intent_first(self):
        b,job=self.precaution_fixture()
        def sent(*args,**kwargs): self.assertEqual(job['moves'][0]['status'],'intent')
        b.tap.side_effect=sent
        with patch('live_lords.defense.time.time',return_value=100): b.prepare_defense(job)
        b.tap.assert_called_once_with(880,540,delay=.5)
        self.assertEqual(b.prepare_transfer.call_args.args[1],200)
        self.assertEqual(job['moves'][0]['status'],'sent')
        self.assertIsNone(job['attack']);self.assertEqual(b.counts['supports'],1)
        self.assertIsNone(b.active_defense)

    def test_hidden_power_excludes_threatened_donor_and_late_arrival(self):
        for reason in ('threatened','late'):
            b,job=self.precaution_fixture()
            if reason=='threatened': b.threatened_names={'donor'}
            else: job['impact']=125
            with patch('live_lords.defense.time.time',return_value=100): b.prepare_defense(job)
            b.tap.assert_not_called();self.assertEqual(job['moves'],[])
            self.assertEqual(job['status'],'insufficient')

    def test_changed_wave_cancels_precautionary_send(self):
        b,job=self.precaution_fixture();b.defense_snapshot_valid.return_value=False
        with patch('live_lords.defense.time.time',return_value=100): b.prepare_defense(job)
        b.tap.assert_not_called();self.assertEqual(job['moves'],[])

    def test_defense_menu_collects_more_than_six_attacks_and_timestamps_each_page(self):
        b=DefenseMixin();b.image=Mock();b.vision=Mock();b.driver=Mock();b.wait=Mock()
        b.capture=Mock();b.tap=Mock();b.back=Mock();b.log=Mock();b.vision.march_button.return_value=(1228,571,1)
        first=[{'city':f'C{i}','attacker':'Enemy','troops':100,'remaining':200} for i in range(6)]
        last=[{'city':'C6','attacker':'Enemy','troops':100,'remaining':300}]
        b.vision.defense_rows.side_effect=[first,last]
        with patch('live_lords.defense.time.time',side_effect=[100,110]): rows=b.inspect_defense_rows()
        self.assertEqual(len(rows),7)
        self.assertEqual(rows[0]['impact'],300);self.assertEqual(rows[-1]['impact'],410)
        self.assertTrue(b.defense_list_complete)
        self.assertEqual(b.driver.scroll_marches.call_count,4)

    def test_focus_reopens_the_page_with_the_requested_city(self):
        b=DefenseMixin();b.image=Mock();b.vision=Mock();b.driver=Mock();b.wait=Mock()
        b.capture=Mock();b.tap=Mock();b.back=Mock();b.log=Mock();b.vision.march_button.return_value=(1228,571,1)
        first=[{'city':f'C{i}','attacker':'Enemy','troops':100,'remaining':200} for i in range(6)]
        last=[{'city':'Home','attacker':'Enemy','troops':100,'remaining':300,'pin':(650,148)}]
        b.vision.defense_rows.side_effect=[first,last]
        rows=b.inspect_defense_rows(focus='home')
        self.assertEqual(rows,last)
        self.assertEqual(rows[0]['pin'],(650,148))


if __name__=='__main__': unittest.main()
