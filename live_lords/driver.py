from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
import tempfile
import queue
import threading
from pathlib import Path

from PIL import Image

PACKAGE = 'com.millionvictories.games.millionlords'
DEFAULT_NAME = '유미 ( 전투용 )'


class GameUnavailable(RuntimeError):
    """The current action must unwind before the game can be relaunched."""


class EmulatorBridgeUnavailable(RuntimeError):
    """Android's command bridge did not respond; host recovery is required."""


def normalized_name(name):
    return ''.join(name.split())


class LDPlayer:
    def __init__(self, install: Path, name: str, data: Path):
        self.install, self.name, self.data = Path(install), name.strip(), Path(data)
        self.data.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.bind()

    def run(self, exe, *args, timeout=20):
        command=[str(self.install / exe), *map(str, args)]
        # Do not use PIPE+communicate: a hung descendant may inherit the pipe
        # and keep communicate blocked even after its parent is terminated.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            proc=subprocess.Popen(command,stdout=output,stderr=errors,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            try:proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:pass
                if exe=='ld.exe':raise EmulatorBridgeUnavailable(f'LD Android 명령 응답 없음: {args[:3]}') from exc
                raise
            if proc.returncode:
                errors.seek(0)
                raise RuntimeError(errors.read(4096).decode('utf-8','replace')[:500])
            output.seek(0)
            return output.read(2*1024*1024)

    def bind(self):
        raw = self.run('ldconsole.exe', 'list2',timeout=5)
        for _ in range(2):
            if raw.strip(): break
            time.sleep(.15)
            raw=self.run('ldconsole.exe','list2',timeout=5)
        # Current LDPlayer emits UTF-8; some older versions use the system ACP.
        text = raw.decode('utf-8', 'replace')
        if '\ufffd' in text:
            text = raw.decode('mbcs', 'replace')
        matches = [line.split(',') for line in text.splitlines()
                   if len(line.split(',')) >= 10 and normalized_name(line.split(',')[1]) == normalized_name(self.name)]
        if len(matches) != 1:
            raise RuntimeError(f'대상 인스턴스를 한 개 찾을 수 없습니다: {self.name}')
        row = matches[0]
        if row[4] != '1':
            raise RuntimeError('대상 LD플레이어를 먼저 실행해 주세요.')
        self.index = int(row[0])
        self.resolution = (int(row[7]), int(row[8]))
        if self.resolution[0] * 9 != self.resolution[1] * 16:
            raise RuntimeError('16:9 가로 화면을 사용해 주세요.')
        config = json.loads((self.install / 'vms' / 'config' / f'leidian{self.index}.config').read_text('utf-8-sig'))
        self.shared = Path(config['statusSettings.sharedPictures'])

    def shell(self, *args,timeout=8):
        if args and args[0]=='input':self.guard_and_audit_input(args)
        raw = self.run('ld.exe', '-s', self.index, *args,timeout=timeout)
        text = raw.decode('utf-8', 'replace')
        if 'not found emulator' in text:raise EmulatorBridgeUnavailable(text)
        if 'Permission denied' in text:
            raise RuntimeError(text)
        return text

    def guard_and_audit_input(self,args):
        """Cover every Android touch, including swipes that bypass Bot.tap."""
        points=[];kind=args[1] if len(args)>1 else 'unknown'
        if kind in ('tap','swipe'):
            count=2 if kind=='tap' else 4
            values=[float(v) for v in args[2:2+count]]
            points=[(values[i]*1280/self.resolution[0],values[i+1]*720/self.resolution[1])
                    for i in range(0,len(values),2)]
        blocked=False
        im=getattr(self,'last_image',None)
        if points and im is not None:
            from .vision import Vision
            if getattr(self,'input_vision',None) is None:self.input_vision=Vision(None,self.data)
            if self.input_vision.match(im,'march_sort',.86,area=(760,35,850,95)):
                xs,ys=zip(*points)
                blocked=min(xs)<=820 and max(xs)>=680 and min(ys)<=660 and max(ys)>=100
        if getattr(self,'data',None):
            try:
                path=self.data/'input-audit.jsonl'
                if path.exists() and path.stat().st_size>1024*1024:path.replace(path.with_name('input-audit.previous.jsonl'))
                entry={'ts':time.time(),'instance':self.index,'kind':kind,'native':list(args[2:]),
                       'normalized':points,'blocked':blocked,'capture_id':getattr(self,'capture_id',None)}
                if (self.data/'trace-inputs.enabled').exists() and (self.data/'latest.png').exists():
                    folder=self.data/'input-frames';folder.mkdir(exist_ok=True)
                    frame=folder/f"{getattr(self,'capture_id',0)%32:02}.png"
                    shutil.copyfile(self.data/'latest.png',frame);entry['frame']=str(frame)
                with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
            except OSError:pass
        if blocked:raise RuntimeError('행군 병력 숫자 열의 클릭/스와이프 차단 · 새 화면에서 다시 판단')

    def foreground(self):
        for attempt in range(3):
            text = self.shell('dumpsys', 'window', 'windows',timeout=4)
            focus = next((line for line in text.splitlines() if 'mCurrentFocus=' in line), '')
            if PACKAGE in focus: return True
            # Android momentarily reports null during in-game transitions. Never
            # treat a different focused app as the game or send input to it.
            if focus and 'mCurrentFocus=null' not in focus: return False
            if attempt<2: time.sleep(.15)
        return False

    def launch(self):
        self.bind()
        self.run('ldconsole.exe', 'runapp', '--index', self.index, '--packagename', PACKAGE,timeout=10)

    def reboot_selected(self):
        self.bind()
        self.run('ldconsole.exe','reboot','--index',self.index,timeout=10)

    def game_state(self):
        """Inspect the selected emulator without sending any Android input."""
        self.bind()
        if self.foreground():return 'foreground'
        pids=self.shell('pidof',PACKAGE,timeout=4).strip()
        if not pids:return 'closed'
        if not all(part.isdigit() for part in pids.split()):
            raise RuntimeError('Million Lords 프로세스 상태를 확인하지 못했습니다.')
        windows=self.shell('dumpsys','window','windows')
        focus=next((line for line in windows.splitlines() if 'mCurrentFocus=' in line),'')
        if any(package in focus for package in ('com.ldmnq.launcher3/', 'com.android.launcher3/',
                                                'com.android.launcher/', 'com.ldplayer.launcher/')):
            return 'launcher'
        return 'background'

    def capture(self):
        self.bind()
        if not self.foreground():
            raise GameUnavailable('Million Lords가 전면 앱에서 사라졌습니다.')
        filename = 'mlai-' + uuid.uuid4().hex + '.png'
        path = self.shared / filename
        self.shell('screencap', '-p', '/sdcard/Pictures/' + filename)
        for _ in range(30):
            if path.exists() and path.stat().st_size > 100:
                break
            time.sleep(.1)
        with Image.open(path) as source:
            image = source.convert('RGB')
            image.load()
        if image.size != self.resolution:
            raise RuntimeError(f'게임 화면 크기 변경: {image.size}')
        image = image.resize((1280, 720), Image.Resampling.LANCZOS)
        image.save(self.data / 'latest.png')
        path.unlink(missing_ok=True)
        self.last_capture = time.monotonic()
        self.last_image=image
        self.capture_id=getattr(self,'capture_id',0)+1
        return image

    def tap(self, x, y):
        if time.monotonic() - getattr(self, 'last_capture', 0) > 15:
            raise RuntimeError('화면 관측이 오래되어 조작을 중단했습니다.')
        if not self.foreground():
            raise GameUnavailable('전면 앱이 바뀌어 현재 조작을 중단했습니다.')
        if not (0 <= x < 1280 and 0 <= y < 720):
            raise ValueError('좌표 범위 초과')
        self.shell('input', 'tap', round(x * self.resolution[0] / 1280), round(y * self.resolution[1] / 720))

    def back(self):
        if not self.foreground():
            raise GameUnavailable('전면 앱 변경')
        self.shell('input', 'keyevent', 4)

    def drag_troop_slider(self,start_x,end_x):
        if not self.foreground():raise GameUnavailable('병력 슬라이더 조작 중 게임이 사라졌습니다.')
        if time.monotonic()-getattr(self,'last_capture',0)>15:
            raise RuntimeError('병력 슬라이더 관측/전면 앱 확인 실패')
        if not (360<=start_x<=795 and 360<=end_x<=795):
            raise ValueError('병력 슬라이더 좌표 범위 초과')
        sx=self.resolution[0]/1280; sy=self.resolution[1]/720
        self.shell('input','swipe',round(start_x*sx),round(560*sy),round(end_x*sx),round(560*sy),220)

    def pan_map(self,dx,dy):
        if not self.foreground():raise GameUnavailable('지도 이동 중 게임이 사라졌습니다.')
        if time.monotonic()-getattr(self,'last_capture',0)>15:
            raise RuntimeError('지도 탐색 관측/전면 앱 확인 실패')
        if abs(dx)>260 or abs(dy)>180: raise ValueError('지도 탐색 거리 초과')
        sx=self.resolution[0]/1280; sy=self.resolution[1]/720
        self.shell('input','swipe',round(640*sx),round(355*sy),round((640+dx)*sx),round((355+dy)*sy),450)

    def scroll_marches(self,to_top=False):
        if not self.foreground():raise GameUnavailable('행군 확인 중 게임이 사라졌습니다.')
        if time.monotonic()-getattr(self,'last_capture',0)>15:
            raise RuntimeError('행군 목록 관측/전면 앱 확인 실패')
        sx=self.resolution[0]/1280; sy=self.resolution[1]/720
        # Use the blank gap between the shield and city level columns. The old
        # x=790/y=150 gesture started in the first row's troop range column.
        start,end=(195,625) if to_top else (625,195)
        self.shell('input','swipe',round(420*sx),round(start*sy),round(420*sx),round(end*sy),300)


class OCR:
    def __init__(self):
        self.script = Path(__file__).with_name('ocr.ps1')
        self.process=None
        self.lock=threading.Lock()

    def close(self):
        process=self.process;self.process=None
        if process is None:return
        if process.stdin:
            try:process.stdin.close()
            except OSError:pass
        try:process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=2)
        for stream in (process.stdout,process.stderr):
            if stream:stream.close()

    def start(self):
        if self.process is not None and self.process.poll() is None:return
        self.close();self.replies=queue.Queue()
        self.process=subprocess.Popen(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass',
            '-File',str(self.script),'-Server'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW)
        stream=self.process.stdout;replies=self.replies
        def read_output():
            try:
                for line in stream:replies.put(line)
            except (OSError,ValueError):pass
            finally:replies.put(None)
        threading.Thread(target=read_output,daemon=True).start()

    def read(self, path, language='ko'):
        return self.read_many([(path,language)])[0]

    def read_many(self, requests):
        if not requests: return []
        payload=[{'path':str(Path(path).resolve()),'language':language} for path,language in requests]
        with self.lock:
            try:
                self.start()
                self.process.stdin.write(json.dumps(payload,ensure_ascii=False)+'\n');self.process.stdin.flush()
                line=self.replies.get(timeout=max(25,len(payload)*5))
                if line is None:raise RuntimeError('OCR 작업 프로세스가 종료되었습니다.')
                results=json.loads(line)
                if isinstance(results,dict) and 'error' in results:raise RuntimeError('OCR 오류: '+results['error'])
            except (OSError,ValueError,queue.Empty,RuntimeError) as exc:
                self.close()
                raise RuntimeError(f'OCR 재연결 필요: {exc}') from exc
        if not isinstance(results,list) or len(results)!=len(payload):
            raise RuntimeError('OCR 일괄 처리 결과 개수 불일치')
        return results
