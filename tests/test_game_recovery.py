import threading
import subprocess
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

from live_lords.driver import LDPlayer,GameUnavailable,EmulatorBridgeUnavailable,PACKAGE
from live_lords.runner import Bot,Stopped


class GameRecoveryTests(unittest.TestCase):
    def bot(self):
        b=Bot.__new__(Bot);b.stop=threading.Event();b.driver=Mock();b.driver.index=21
        b.log=Mock();b.decide=Mock();b.emit=Mock();b.capture=Mock();b.map=Mock()
        b.recovery_pause=Mock();b.seen={'old':1};b.counts={};b.vision=Mock()
        b.vision.match.return_value=None;b.muster=None;b.pending=None
        b.outcome_policy=None;b.save_muster=Mock();b.restore_outcome_attack=Mock()
        b.begin_game_recovery()
        return b

    def test_closed_game_is_launched_only_on_selected_emulator_with_backoff(self):
        b=self.bot();b.driver.game_state.return_value='closed'
        with patch('live_lords.recovery.time.monotonic',side_effect=[100,110,131]):
            for _ in range(3):self.assertFalse(b.recover_game())
        self.assertEqual(b.driver.launch.call_count,2)
        self.assertEqual(b.game_recovery_next_launch,191)
        self.assertEqual(b.counts['restarts'],2)
        b.capture.assert_not_called();b.map.assert_not_called()

    def test_launch_error_cannot_spam_launches_or_stop_recovery(self):
        b=self.bot();b.driver.game_state.return_value='closed'
        b.driver.launch.side_effect=RuntimeError('launcher busy')
        with patch('live_lords.recovery.time.monotonic',side_effect=[100,101]):
            self.assertFalse(b.recover_game());self.assertFalse(b.recover_game())
        b.driver.launch.assert_called_once();self.assertTrue(b.game_recovering)

    def test_loading_screen_only_waits_and_never_runs_map_navigation(self):
        b=self.bot();b.driver.game_state.return_value='foreground'
        self.assertFalse(b.recover_game())
        b.capture.assert_called_once();b.map.assert_not_called();b.driver.launch.assert_not_called()
        b.recovery_pause.assert_called_once_with(2)

    def test_another_active_app_does_not_get_clicked_or_replaced(self):
        b=self.bot();b.driver.game_state.return_value='background'
        self.assertFalse(b.recover_game())
        b.driver.launch.assert_not_called();b.capture.assert_not_called();b.map.assert_not_called()

    def test_map_recovery_preserves_pending_attack_and_uncertain_transfer_without_replay(self):
        b=self.bot();b.driver.game_state.return_value='foreground';b.vision.match.return_value=(50,670,1)
        pending={'episode_id':'actual-intent','arrival':123};b.pending=pending
        b.muster={'status':'preparing','moves':[{'status':'intent','troops':345000000}]}
        b.outcome_policy=Mock()
        def save(status):b.muster['status']=status
        b.save_muster.side_effect=save
        self.assertTrue(b.recover_game())
        self.assertIs(b.pending,pending);self.assertEqual(b.muster['status'],'waiting')
        b.restore_outcome_attack.assert_called_once();self.assertEqual(b.seen,{})
        self.assertFalse(b.game_recovering);b.driver.launch.assert_not_called()

    def test_crash_during_recovery_capture_is_retried_without_a_click(self):
        b=self.bot();b.driver.game_state.side_effect=['foreground','closed']
        b.capture.side_effect=GameUnavailable('closed again')
        self.assertFalse(b.recover_game());self.assertFalse(b.recover_game())
        b.driver.launch.assert_called_once();b.map.assert_not_called()

    def test_uncertain_defense_support_waits_for_result_instead_of_repeating_send(self):
        b=self.bot();b.driver.game_state.return_value='foreground';b.vision.match.return_value=(50,670,1)
        b.defenses=[{'status':'preparing','moves':[{'status':'intent','troops':1000000}]}]
        b.active_defense=b.defenses[0];b.save_defenses=Mock()
        self.assertTrue(b.recover_game())
        self.assertEqual(b.defenses[0]['status'],'waiting')
        self.assertEqual(b.defenses[0]['moves'][0]['status'],'intent')
        b.save_defenses.assert_called_once();self.assertIsNone(b.active_defense)

    def test_stop_prevents_any_relaunch(self):
        b=self.bot();b.stop.set();b.recovery_pause=Bot.recovery_pause.__get__(b)
        with self.assertRaises(Stopped):b.recover_game()
        b.driver.game_state.assert_not_called();b.driver.launch.assert_not_called()

    def test_bridge_timeout_still_requests_game_launch_without_a_working_android_probe(self):
        b=self.bot();b.driver.game_state.side_effect=EmulatorBridgeUnavailable('dumpsys timeout')
        self.assertFalse(b.recover_game())
        b.driver.launch.assert_called_once();b.driver.reboot_selected.assert_not_called()
        b.capture.assert_not_called();b.map.assert_not_called()

    def test_repeated_bridge_failures_reboot_only_selected_vm_with_cooldown(self):
        b=self.bot();b.driver.game_state.side_effect=EmulatorBridgeUnavailable('bridge timeout')
        clock=[0]
        with patch('live_lords.recovery.time.monotonic',side_effect=lambda:clock[0]), \
                patch('live_lords.recovery.time.time',side_effect=lambda:1000+clock[0]):
            for t in (0,8,16,25,33,41,58):
                clock[0]=t;self.assertFalse(b.recover_game())
        b.driver.reboot_selected.assert_called_once()
        self.assertEqual(b.counts['emulator_restarts'],1)
        self.assertEqual(b.game_recovery_next_reboot,1316)

    def test_capture_timeouts_are_not_erased_by_a_working_foreground_probe(self):
        b=self.bot();b.driver.game_state.return_value='foreground'
        b.capture.side_effect=EmulatorBridgeUnavailable('screencap timeout')
        clock=[0]
        with patch('live_lords.recovery.time.monotonic',side_effect=lambda:clock[0]), \
                patch('live_lords.recovery.time.time',return_value=1000):
            for t in (0,8,16):clock[0]=t;self.assertFalse(b.recover_game())
        b.driver.reboot_selected.assert_called_once();b.map.assert_not_called()

    def test_worker_unwinds_a_crashed_action_and_initializes_from_a_new_observation(self):
        b=Bot.__new__(Bot);b.stop=threading.Event();b.driver=Mock();b.learner=Mock();b.emit=Mock();b.log=Mock()
        b.capture=Mock(side_effect=[GameUnavailable('game died'),None]);b.map=Mock();b.hud=Mock()
        b.config={};b.vision=Mock();b.vision.incoming_button.return_value=None;b.image=Mock()
        b.state={'vp':100};b.pending={'episode_id':'sent'};b.muster=None
        b.sync_stats=Mock(side_effect=Stopped())
        def recovered():b.game_recovering=False;return True
        b.recover_game=Mock(side_effect=recovered)
        b.run()
        self.assertEqual(b.capture.call_count,2);b.sync_stats.assert_called_once()
        self.assertEqual(b.pending,{'episode_id':'sent'})
        self.assertFalse(any('error' in c.args[0] for c in b.emit.call_args_list))


class GameDriverRecoveryTests(unittest.TestCase):
    def driver(self):
        d=LDPlayer.__new__(LDPlayer);d.bind=Mock();d.shell=Mock();d.run=Mock()
        d.foreground=Mock(return_value=False);d.index=21
        return d

    def test_missing_game_process_is_closed(self):
        d=self.driver();d.shell.return_value=''
        self.assertEqual(d.game_state(),'closed');d.bind.assert_called_once()
        d.shell.assert_called_once_with('pidof',PACKAGE,timeout=4)

    def test_launcher_with_live_process_also_restores_game(self):
        d=self.driver();d.shell.side_effect=['12345','mCurrentFocus=Window{ com.ldmnq.launcher3/com.android.launcher3.Launcher }']
        self.assertEqual(d.game_state(),'launcher')

    def test_other_foreground_app_with_game_running_is_background(self):
        d=self.driver();d.shell.side_effect=['12345','mCurrentFocus=Window{ com.example.app/.Main }']
        self.assertEqual(d.game_state(),'background')

    def test_unknown_process_output_is_not_permission_to_relaunch(self):
        d=self.driver();d.shell.return_value='pidof: not found'
        with self.assertRaises(RuntimeError):d.game_state()
        d.run.assert_not_called()

    def test_launch_rebinds_instance_and_always_uses_million_lords_package(self):
        d=self.driver();d.launch()
        d.bind.assert_called_once()
        d.run.assert_called_once_with('ldconsole.exe','runapp','--index',21,'--packagename',PACKAGE,timeout=10)

    def test_closed_game_prevents_capture_and_back_input(self):
        d=self.driver()
        with self.assertRaises(GameUnavailable):d.capture()
        with self.assertRaises(GameUnavailable):d.back()
        d.shell.assert_not_called()

    def test_vm_reboot_rebinds_and_targets_only_the_selected_index(self):
        d=self.driver();d.reboot_selected()
        d.bind.assert_called_once()
        d.run.assert_called_once_with('ldconsole.exe','reboot','--index',21,timeout=10)

    def test_android_command_timeout_never_waits_on_inherited_stdout_pipes(self):
        d=LDPlayer.__new__(LDPlayer);d.install=Path('.')
        proc=Mock();proc.wait.side_effect=[subprocess.TimeoutExpired('ld.exe',.1),0]
        with patch('live_lords.driver.subprocess.Popen',return_value=proc) as launch:
            with self.assertRaises(EmulatorBridgeUnavailable):d.run('ld.exe','-s',21,'pidof',PACKAGE,timeout=.1)
        proc.kill.assert_called_once();proc.communicate.assert_not_called()
        self.assertNotEqual(launch.call_args.kwargs['stdout'],subprocess.PIPE)
        self.assertNotEqual(launch.call_args.kwargs['stderr'],subprocess.PIPE)

    def test_host_timeout_is_bounded_without_being_mistaken_for_android_state(self):
        d=LDPlayer.__new__(LDPlayer);d.install=Path('.')
        proc=Mock();proc.wait.side_effect=[subprocess.TimeoutExpired('ldconsole.exe',.1),0]
        with patch('live_lords.driver.subprocess.Popen',return_value=proc):
            with self.assertRaises(subprocess.TimeoutExpired):d.run('ldconsole.exe','list2',timeout=.1)
        proc.communicate.assert_not_called()


if __name__=='__main__':unittest.main()
