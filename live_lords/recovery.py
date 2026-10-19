"""Relaunch a lost game; never resume the interrupted click sequence."""
import time
import json
from subprocess import TimeoutExpired
from .driver import EmulatorBridgeUnavailable


class GameRecoveryMixin:
    def begin_game_recovery(self):
        if getattr(self,'game_recovering',False):return
        self.game_recovering=True
        self.game_recovery_attempts=0
        self.game_recovery_next_launch=0
        self.game_recovery_notice=None
        self.game_bridge_failures=0
        self.game_bridge_since=None
        self.game_recovery_next_reboot=getattr(self,'game_recovery_next_reboot',0)
        if getattr(self,'data',None):
            try:
                saved=json.loads((self.data/'recovery-state.json').read_text('utf-8'))
                self.game_recovery_next_reboot=max(self.game_recovery_next_reboot,saved.get('last_emulator_reboot',0)+300)
            except (OSError,ValueError,TypeError):pass
        self.image=None
        self.driver.last_capture=0
        self.log('게임 종료/전면 이탈 감지 · 현재 조작 중단 · 행군/집결 기록 유지')

    def recovery_pause(self,seconds):
        # Bot.wait polls screenshots for defense. It must not run while the
        # game is absent or from an exception handler: another capture failure
        # there would escape the retry loop and stop the worker.
        if self.stop.wait(seconds):
            from .runner import Stopped
            raise Stopped()

    def game_recovery_notice_changed(self,state,text):
        self.decide('게임 복구',text)
        if getattr(self,'game_recovery_notice',None)!=state:
            self.game_recovery_notice=state
            self.log(text)

    def request_game_launch(self):
        now=time.monotonic()
        if now<getattr(self,'game_recovery_next_launch',0):return
        attempt=getattr(self,'game_recovery_attempts',0)+1
        self.game_recovery_attempts=attempt
        self.game_recovery_next_launch=now+min(120,30*2**min(attempt-1,2))
        if self.stop.is_set():self.recovery_pause(0)
        # This path uses ldconsole (host control), not the unresponsive
        # Android dumpsys/pidof/screenshot channel.
        self.driver.launch()
        self.counts['restarts']=self.counts.get('restarts',0)+1
        self.game_recovery_notice_changed('launch:'+str(attempt),
            f'Million Lords 자동 재실행 {attempt}회 · 선택한 LD플레이어 {self.driver.index} · 로딩 대기')

    def recover_android_bridge(self,error):
        self.game_bridge_failures=getattr(self,'game_bridge_failures',0)+1
        if getattr(self,'game_bridge_since',None) is None:self.game_bridge_since=time.monotonic()
        self.game_recovery_notice_changed('bridge','LD Android 응답 지연 · 화면 조회와 별도로 게임 재실행 시도')
        try:self.request_game_launch()
        except (RuntimeError,OSError,TimeoutExpired) as exc:
            self.game_recovery_notice_changed('launch-error','게임 실행 요청 응답 대기: '+str(exc))
        if (self.game_bridge_failures<3 or time.monotonic()-self.game_bridge_since<15 or
                time.time()<getattr(self,'game_recovery_next_reboot',0)):
            return
        if self.stop.is_set():self.recovery_pause(0)
        # Record before the request: an unknown command result must not cause
        # repeated VM restarts. No other emulator, shared ADB server or data is
        # touched. The selected name is rebound again by reboot_selected.
        now=time.time();self.game_recovery_next_reboot=now+300
        if getattr(self,'data',None):
            path=self.data/'recovery-state.json';temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'last_emulator_reboot':now},ensure_ascii=False),'utf-8')
            temporary.replace(path)
        self.log(f'LD Android 응답 없음 {self.game_bridge_failures}회 · 선택한 LD플레이어 {self.driver.index}만 재시작 요청 · 행군/학습 유지')
        self.driver.reboot_selected()
        self.counts['emulator_restarts']=self.counts.get('emulator_restarts',0)+1
        self.game_recovery_next_launch=0
        self.game_bridge_failures=0;self.game_bridge_since=None

    def recover_game(self):
        if self.stop.is_set():
            self.recovery_pause(0)
        try:
            state=self.driver.game_state()
            if state=='foreground':
                self.capture()
                self.game_bridge_failures=0;self.game_bridge_since=None
                if not self.vision.match(self.image,'map_home',.72,area=(0,615,100,720)):
                    self.game_recovery_notice_changed('loading','Million Lords 실행 중 · 지도 로딩 완료 대기')
                    self.recovery_pause(2);return False
                self.map()
                # A submitted/uncertain intent cannot be sent again after the
                # relaunch, even when the crash happened before its UI result.
                if getattr(self,'muster',None) and any(m['status']=='intent' for m in self.muster.get('moves',[])):
                    self.save_muster('waiting')
                defense_changed=False
                for job in getattr(self,'defenses',[]):
                    if job.get('status') in ('preparing','insufficient') and any(
                            move['status']=='intent' for move in job.get('moves',[])):
                        job['status']='waiting';defense_changed=True
                if defense_changed:self.save_defenses()
                self.active_defense=None
                if getattr(self,'outcome_policy',None):self.restore_outcome_attack()
                self.seen.clear()
                self.origin_troops=None;self.expected_origin_name=None;self.attack_choices=[]
                self.last_stats_update=0;self.last_defense_poll=0
                self.alignment_retry_at=0;self.exploration_pause_until=0
                self.defense_read_waiting=False
                self.startup_combat_check=False
                self.game_recovering=False
                self.log('Million Lords 지도 복귀 확인 · 기존 행군/집결 결과 확인부터 자동 플레이 재개')
                return True
            if state in ('closed','launcher'):
                self.game_bridge_failures=0;self.game_bridge_since=None
                self.request_game_launch()
            else:
                self.game_bridge_failures=0;self.game_bridge_since=None
                self.game_recovery_notice_changed('background','다른 앱이 전면에 있습니다 · 게임 복귀/종료 상태 확인 중')
        except EmulatorBridgeUnavailable as exc:
            try:self.recover_android_bridge(exc)
            except (RuntimeError,OSError,TimeoutExpired) as recovery_error:
                self.game_recovery_notice_changed('bridge-error','LD 연결 복구 요청 재확인 중: '+str(recovery_error))
        except (RuntimeError,OSError,TimeoutExpired) as exc:
            self.game_recovery_notice_changed('error:'+str(exc),'게임 연결/로딩 재확인 중: '+str(exc))
        self.recovery_pause(2)
        return False
