import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from live_lords.vision import Vision
import test_live_activity


class BatchOCR:
    def __init__(self): self.calls=[]
    def read_many(self,requests):
        self.calls.append(list(requests))
        return [{'text':'도시 목록' if lang=='ko' else '12M'} for _,lang in requests]


class SpeedTests(unittest.TestCase):
    def test_delay_ratios_and_explicit_ui_factors(self):
        obj=test_live_activity.ActivityTests().actor(); obj.config['interval']=12
        obj.decide('주변 탐색','scan'); self.assertEqual(obj.cycle_delay(),3)
        obj.decide('감시 대기','wait'); self.assertEqual(obj.cycle_delay(),4.8)
        obj.config['decision_speed']=3; self.assertEqual(obj.cycle_delay(),4)

    def test_missing_own_cities_returns_instead_of_panning_farther_or_scanning_upgrades(self):
        obj=test_live_activity.ActivityTests().actor()
        obj.attack_nearby=Mock(return_value='no_sources'); obj.upgrade=Mock()
        obj.next_region=Mock(return_value=True)
        obj.progress_cycle(); obj.progress_cycle()
        self.assertEqual(obj.driver.pan_map.call_count,1)
        obj.next_region.assert_called_once(); obj.upgrade.assert_not_called()

    def test_pending_action_prevents_home_navigation(self):
        obj=test_live_activity.ActivityTests().actor(); obj.pending={'arrival':100}
        self.assertFalse(obj.return_to_territory()); obj.driver.pan_map.assert_not_called()

    def test_home_button_requires_verified_city_list_and_own_city_after_return(self):
        obj=test_live_activity.ActivityTests().actor(); obj.tap=Mock(); obj.region=Mock()
        obj.region.return_value={'title':'도시 목록','prompt':'주요 도시'}
        obj.image=Image.new('RGB',(1280,720)); obj.vision=Mock()
        obj.vision.own_sources.return_value=[((640,330),100)]
        self.assertTrue(obj.return_to_territory())
        self.assertEqual([call.args for call in obj.tap.call_args_list],[(1047,665),(265,264)])
        obj.tap.reset_mock(); obj.vision.match.return_value=None
        self.assertFalse(obj.return_to_territory()); self.assertEqual(obj.tap.call_count,1)

    def test_batch_crops_keep_languages_and_cache_only_identical_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            ocr=BatchOCR(); vision=Vision(ocr,Path(directory)); image=Image.new('RGB',(1280,720))
            boxes={'title':(0,0,40,20),'troops':(40,0,80,20)}
            expected={'title':'도시 목록','troops':'12M'}
            self.assertEqual(vision.regions(image,boxes),expected)
            self.assertEqual([lang for _,lang in ocr.calls[0]],['ko','en-US'])
            self.assertEqual(len(set(path for path,_ in ocr.calls[0])),2)
            self.assertEqual(vision.regions(image,boxes),expected); self.assertEqual(len(ocr.calls),1)
            image.putpixel((45,5),(255,255,255)); vision.regions(image,boxes)
            self.assertEqual(len(ocr.calls),2); self.assertEqual(len(ocr.calls[1]),1)

    def test_blank_batch_result_is_not_cached_and_uses_numeric_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            ocr=BatchOCR(); ocr.read_many=Mock(return_value=[{'text':''}])
            vision=Vision(ocr,Path(directory)); image=Image.new('RGB',(1280,720))
            vision.regions(image,{'troops':(40,0,80,20)})
            self.assertEqual(ocr.read_many.call_count,2)
            self.assertEqual(len(vision.ocr_cache),0)
