import tempfile
import unittest
import threading
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from live_lords.runner import Bot,NavigationBlocked,Stopped
from live_lords.vision import Vision


class NavigationTests(unittest.TestCase):
    def bot_at(self,screens):
        bot=Bot.__new__(Bot)
        bot.image=screens[0]
        bot.driver=Mock(); bot.driver.foreground.return_value=True
        bot.log=Mock()
        bot.back=Mock(side_effect=lambda:setattr(bot,'image',screens.pop(1) if len(screens)>1 else bot.image))
        bot.vision=Mock()
        bot.vision.navigation_panel.side_effect=lambda image: {'profile':'상대 프로필','details':'도시 상세정보'}.get(image)
        bot.vision.navigation_panel_text.return_value=None
        bot.vision.match.side_effect=lambda image,name,*a,**kw: (50,670,1) if image=='map' and name=='map_home' else None
        return bot

    def test_nested_enemy_profile_and_city_details_return_to_map(self):
        bot=self.bot_at(['profile','details','map'])
        bot.map()
        self.assertEqual(bot.image,'map')
        self.assertEqual(bot.back.call_count,2)

    def test_unknown_screen_does_not_press_back(self):
        bot=self.bot_at(['unknown'])
        bot.region=lambda boxes:{'popup':''}
        with self.assertRaisesRegex(RuntimeError,'화면을 확인'):
            bot.map()
        bot.back.assert_not_called()

    def test_stuck_known_panel_has_bounded_retries(self):
        bot=self.bot_at(['profile'])
        with self.assertRaisesRegex(RuntimeError,'패널 닫기 실패'):
            bot.map()
        self.assertEqual(bot.back.call_count,8)

    def test_mid_action_panel_recovery_observes_again_without_repeating_tap(self):
        bot=self.bot_at(['profile','map'])
        bot.capture=Mock()
        bot.tap=Mock()
        bot.pending={'arrival':10}
        bot.seen={'old':100}
        self.assertTrue(bot.recover_navigation(RuntimeError('도시 레벨 OCR 실패')))
        bot.capture.assert_called_once()
        bot.tap.assert_not_called()
        self.assertEqual(bot.pending,{'arrival':10})
        self.assertEqual(bot.seen,{})

    def test_unregistered_game_window_uses_back_then_recovers_without_taps(self):
        bot=self.bot_at(['unknown','map']); bot.capture=Mock(); bot.tap=Mock()
        bot.region=lambda boxes:{'popup':''}; bot.seen={}; bot.pending={'arrival':10}
        self.assertTrue(bot.recover_navigation(NavigationBlocked('unknown panel')))
        bot.back.assert_called_once(); bot.tap.assert_not_called()
        self.assertEqual(bot.pending,{'arrival':10})

    def test_external_app_gets_no_back_or_input(self):
        bot=self.bot_at(['unknown']); bot.capture=Mock(); bot.tap=Mock()
        bot.driver.foreground.return_value=False
        self.assertFalse(bot.recover_navigation(RuntimeError('foreground changed')))
        bot.back.assert_not_called(); bot.tap.assert_not_called(); bot.capture.assert_not_called()

    def test_unknown_panel_retries_are_bounded_and_dont_accept_confirmation(self):
        bot=self.bot_at(['unknown']); bot.capture=Mock(); bot.tap=Mock(); bot.map=Mock(side_effect=NavigationBlocked('unknown'))
        bot.image=Mock(); bot.data=Path('.'); bot.seen={}
        self.assertFalse(bot.recover_navigation(RuntimeError('panel')))
        self.assertEqual(bot.back.call_count,2); self.assertEqual(bot.map.call_count,3)
        bot.tap.assert_not_called()

    def test_readonly_recovery_preserves_muster_and_defense_jobs(self):
        bot=self.bot_at(['profile','map']); bot.capture=Mock(); bot.seen={}; bot.pending=None
        bot.muster={'status':'waiting','moves':[{'status':'sent','troops':100}]}
        bot.defenses=[{'status':'waiting'}]
        self.assertTrue(bot.recover_navigation(RuntimeError('panel')))
        self.assertEqual(bot.muster['moves'][0]['status'],'sent')
        self.assertEqual(bot.defenses,[{'status':'waiting'}])

    def test_worker_waits_and_retries_even_if_initial_window_cannot_close(self):
        bot=Bot.__new__(Bot); bot.stop=threading.Event(); bot.learner=Mock(); bot.emit=Mock()
        bot.capture=Mock(side_effect=RuntimeError('unknown startup menu'))
        bot.recover_navigation=Mock(return_value=False); bot.decide=Mock(); bot.log=Mock()
        delays=[]
        def wait(seconds):
            delays.append(seconds)
            if len(delays)==7: raise Stopped()
        bot.recovery_pause=wait
        bot.run()
        self.assertEqual(delays,[2,4,8,16,30,30,30])
        self.assertEqual(bot.recover_navigation.call_count,7)
        self.assertFalse(any('error' in call.args[0] for call in bot.emit.call_args_list))
        bot.learner.close.assert_called_once()

    def test_startup_recovers_then_initializes_normally(self):
        bot=Bot.__new__(Bot); bot.stop=threading.Event(); bot.learner=Mock(); bot.emit=Mock()
        bot.capture=Mock(side_effect=[RuntimeError('harvester'),None])
        bot.recover_navigation=Mock(return_value=True); bot.decide=Mock(); bot.log=Mock()
        bot.wait=Mock(); bot.map=Mock(); bot.hud=Mock(); bot.state={'vp':100}
        bot.config={}; bot.vision=Mock(); bot.vision.incoming_button.return_value=None; bot.image=Mock()
        bot.sync_stats=Mock(side_effect=Stopped())
        bot.run()
        bot.recover_navigation.assert_called_once(); bot.sync_stats.assert_called_once()
        self.assertFalse(any('error' in call.args[0] for call in bot.emit.call_args_list))

    def test_live_cycle_panel_failure_waits_without_clearing_submitted_attack(self):
        bot=Bot.__new__(Bot); bot.stop=threading.Event(); bot.learner=Mock(); bot.emit=Mock()
        bot.capture=Mock(); bot.map=Mock(); bot.hud=Mock(); bot.sync_stats=Mock(); bot.state={'vp':100}
        bot.config={}; bot.vision=Mock(); bot.vision.incoming_button.return_value=None; bot.image=Mock()
        bot.check_defense_results=Mock(side_effect=RuntimeError('shop interrupted cycle'))
        bot.recover_navigation=Mock(return_value=False); bot.decide=Mock(); bot.log=Mock()
        bot.muster=None;bot.pending={'arrival':100,'target':(500,400)}; bot.recovery_pause=Mock(side_effect=Stopped())
        bot.run()
        self.assertEqual(bot.pending,{'arrival':100,'target':(500,400)})
        bot.recovery_pause.assert_called_once_with(2)
        self.assertFalse(any('error' in call.args[0] for call in bot.emit.call_args_list))

    def test_menu_titles_cover_requested_popups(self):
        with tempfile.TemporaryDirectory() as directory:
            vision=Vision(None,Path(directory))
            for title in ('수확자','상점','설정','공지','게임 소식','상자','길드','보관함','이벤트'):
                with self.subTest(title=title):
                    vision.regions=Mock(return_value={'title':title})
                    self.assertEqual(vision.navigation_panel_text(None),title)

    def test_english_profile_header_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            vision=Vision(None,Path(directory))
            vision.regions=Mock(side_effect=[{'title':'VikingBdg'}, {'profile':"VikingBdg's profile"}])
            self.assertEqual(vision.navigation_panel_text(Image.new('RGB',(1280,720))),'상대 프로필')


if __name__=='__main__': unittest.main()
