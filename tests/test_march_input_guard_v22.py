import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from live_lords.runner import Bot,NavigationBlocked
from live_lords.vision import Vision


class MarchInputGuardTests(unittest.TestCase):
    def bot(self):
        b=Bot.__new__(Bot);b.stop=threading.Event();b.config={};b.driver=Mock()
        b.wait=Mock();b.capture=Mock();b.state={'owner':'Yumi'};b.map=Mock()
        with Image.open(Path(__file__).parent/'fixtures/incoming-defense-list.png') as source:b.image=source.convert('RGB')
        b.vision=Vision(None,Path('.'))
        return b

    def test_march_troop_column_blocks_stale_profile_and_defense_inputs(self):
        for point in ((728,120),(731,148),(731,601)):
            b=self.bot()
            with self.assertRaises(NavigationBlocked):b.tap(*point)
            b.driver.tap.assert_not_called()

    def test_location_pin_remains_clickable(self):
        b=self.bot();b.tap(650,148,delay=1.2)
        b.driver.tap.assert_called_once_with(650,148)

    def test_unexpected_march_cancels_stats_sequence_and_preserves_owner(self):
        b=self.bot();b.tap=Mock();b.region=Mock(return_value={'title':'이동'})
        with self.assertRaises(NavigationBlocked):b.sync_stats()
        b.tap.assert_called_once_with(70,55)
        self.assertEqual(b.state['owner'],'Yumi')

    def test_unverified_profile_never_continues_fixed_coordinate_sequence(self):
        b=self.bot()
        with Image.open(Path(__file__).parent/'fixtures/incoming-mixed-map-v21.png') as source:b.image=source.convert('RGB')
        b.tap=Mock();b.region=Mock(return_value={'title':'길드'})
        with self.assertRaises(NavigationBlocked):b.sync_stats()
        b.tap.assert_called_once_with(70,55)
        self.assertEqual(b.state['owner'],'Yumi')

    def test_info_only_city_selection_is_cleared_before_returning_map(self):
        b=self.bot()
        with Image.open(Path(__file__).parent/'fixtures/selected-info-only-v22.png') as source:b.image=source.convert('RGB')
        with Image.open(Path(__file__).parent/'fixtures/incoming-mixed-map-v21.png') as source:plain=source.convert('RGB')
        b.back=Mock(side_effect=lambda:setattr(b,'image',plain))
        b.map=Bot.map.__get__(b);b.log=Mock()
        b.map();b.back.assert_called_once()


if __name__=='__main__':unittest.main()
