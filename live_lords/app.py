from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import tkinter as tk
from tkinter import ttk,messagebox,filedialog
from pathlib import Path

from PIL import Image,ImageTk

from .driver import DEFAULT_NAME,normalized_name
from .runner import Bot
from .vision import pretty
from .outcomes import choice_label

DATA=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'MillionLordsAI'


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Million Lords AI v0.26 · 화면 기반 강화학습')
        self.geometry('1100x810'); self.minsize(930,720)
        self.configure(bg='#101b2b')
        self.columnconfigure(0,weight=1)
        self.rowconfigure(7,weight=1,minsize=70)
        DATA.mkdir(parents=True,exist_ok=True)
        self.messages=queue.Queue(); self.stop_event=threading.Event(); self.worker=None
        self.active_config=None;self.active_directory=None
        self.install=tk.StringVar(value=r'C:\LDPlayer\LDPlayer9')
        self.instance=tk.StringVar()
        self.target=tk.IntVar(value=100)
        self.reserve=tk.DoubleVar(value=500)
        self.margin=tk.DoubleVar(value=1.5)
        self.send=tk.IntVar(value=35)
        self.interval=tk.DoubleVar(value=.25)
        self.focus_defense=tk.BooleanVar(value=False)
        self.learning_mode=tk.StringVar(value='train')
        self.exploration_speed=tk.DoubleVar(value=4)
        self.decision_speed=tk.DoubleVar(value=3)
        self.attack=tk.BooleanVar(value=True)
        self.autonomous=tk.BooleanVar(value=True)
        self.muster_enabled=tk.BooleanVar(value=True)
        self.defense_enabled=tk.BooleanVar(value=True)
        self.upgrade_enabled=tk.BooleanVar(value=True)
        self.defense_margin=tk.DoubleVar(value=1.3)
        self.defense_buffer=tk.IntVar(value=20)
        self.status=tk.StringVar(value='인스턴스를 선택한 뒤 시작하세요.')
        self.decision=tk.StringVar(value='현재 판단 · 연결 대기')
        self.metrics=tk.StringVar(value='VP —   |   골드 —   |   결과 학습 0 · 기록 0')
        style=ttk.Style(self); style.theme_use('clam')
        style.configure('TFrame',background='#101b2b'); style.configure('TLabel',background='#101b2b',foreground='#e8eef7')
        style.configure('Title.TLabel',font=('맑은 고딕',18,'bold'))
        style.configure('TButton',padding=7)
        top=ttk.Frame(self,padding=(15,10)); top.grid(row=0,column=0,sticky='ew')
        ttk.Label(top,text='Million Lords AI',style='Title.TLabel').pack(side='left')
        ttk.Label(top,text='방어 우선 · 도착시간 검증 · 성 보유/투자 손실 학습').pack(side='right')
        setup=ttk.Frame(self,padding=(15,0,15,8)); setup.grid(row=1,column=0,sticky='ew')
        ttk.Label(setup,text='LD플레이어 폴더').grid(row=0,column=0,sticky='w')
        ttk.Entry(setup,textvariable=self.install,width=36).grid(row=0,column=1,padx=6)
        ttk.Button(setup,text='찾기',command=self.browse).grid(row=0,column=2)
        self.instances=ttk.Combobox(setup,textvariable=self.instance,width=32,state='readonly')
        self.instances.grid(row=0,column=3,padx=8)
        ttk.Button(setup,text='목록 새로고침',command=self.refresh).grid(row=0,column=4)
        options=ttk.Frame(self,padding=(15,0,15,8)); options.grid(row=2,column=0,sticky='ew')
        fields=[('투자 도시 Lv. 상한',self.target,1,200),('남길 골드(M)',self.reserve,0,100000),('공격 우세 배율',self.margin,1.1,10),('성별 동원(%)',self.send,10,70),('기준 간격(초)',self.interval,.25,120)]
        for col,(name,var,lo,hi) in enumerate(fields):
            ttk.Label(options,text=name).grid(row=0,column=col,padx=8,sticky='w')
            ttk.Spinbox(options,textvariable=var,from_=lo,to=hi,width=11,increment=.25 if var is self.interval else .1 if var is self.margin else 1).grid(row=1,column=col,padx=8)
        self.manual_controls=[]
        for label,variable,placement in (
            ('정찰 후 조건 충족 시 공격',self.attack,dict(row=1,column=5,padx=8)),
            ('부족분 자동 집결',self.muster_enabled,dict(row=2,column=5,padx=8,sticky='w')),
            ('도착 전 지원 방어',self.defense_enabled,dict(row=2,column=0,columnspan=2,sticky='w',padx=8)),
            ('방어 조건 확인 후 성 투자',self.upgrade_enabled,dict(row=2,column=2,columnspan=2,sticky='w',padx=8))):
            control=ttk.Checkbutton(options,text=label,variable=variable); control.grid(**placement)
            self.manual_controls.append(control)
        ttk.Label(options,text='방어 우세 / 도착 여유(초)').grid(row=3,column=0,columnspan=2,sticky='w',padx=8)
        ttk.Spinbox(options,textvariable=self.defense_margin,from_=1.1,to=3,increment=.1,width=9).grid(row=3,column=2)
        ttk.Spinbox(options,textvariable=self.defense_buffer,from_=15,to=120,width=9).grid(row=3,column=3)
        ttk.Checkbutton(options,text='AI 자율 판단: 방어·정찰·공격·집결·핵심 성 투자 선택',variable=self.autonomous,command=self.autonomy_changed).grid(row=4,column=0,columnspan=6,sticky='w',padx=8,pady=4)
        ttk.Label(options,text='탐색 속도(배) / 판단 속도(배)').grid(row=5,column=0,columnspan=2,sticky='w',padx=8)
        ttk.Spinbox(options,textvariable=self.exploration_speed,from_=1,to=4,increment=.5,width=9).grid(row=5,column=2)
        ttk.Spinbox(options,textvariable=self.decision_speed,from_=1,to=3,increment=.5,width=9).grid(row=5,column=3)
        ttk.Checkbutton(options,text='집중 방어: 새 공격·공격 집결 중단 / Lv.110~115 우선 보호',variable=self.focus_defense,command=self.focus_changed).grid(row=6,column=0,columnspan=6,sticky='w',padx=8,pady=3)
        ttk.Radiobutton(options,text='강화학습하며 플레이',variable=self.learning_mode,value='train',command=self.learning_mode_changed).grid(row=7,column=0,columnspan=2,sticky='w',padx=8)
        ttk.Radiobutton(options,text='저장된 정책으로 플레이 (신경망 고정)',variable=self.learning_mode,value='evaluate',command=self.learning_mode_changed).grid(row=7,column=2,columnspan=4,sticky='w',padx=8)
        self.autonomy_changed()
        buttons=ttk.Frame(self,padding=(15,0,15,8)); buttons.grid(row=3,column=0,sticky='ew')
        self.start_button=ttk.Button(buttons,text='자동 플레이 · 학습 시작',command=self.start)
        self.start_button.pack(side='left')
        ttk.Button(buttons,text='중지',command=self.stop).pack(side='left',padx=8)
        ttk.Button(buttons,text='게임 열기',command=self.open_game).pack(side='left')
        ttk.Button(buttons,text='학습 현황',command=self.show_learning).pack(side='left',padx=8)
        ttk.Button(buttons,text='학습 폴더',command=lambda:os.startfile(DATA)).pack(side='right')
        ttk.Label(self,textvariable=self.metrics,font=('맑은 고딕',12,'bold'),padding=(15,5)).grid(row=4,column=0,sticky='ew')
        self.status_label=tk.Label(self,textvariable=self.status,height=2,anchor='w',justify='left',bg='#101b2b',fg='#e8eef7',wraplength=1040)
        self.status_label.grid(row=5,column=0,sticky='ew',padx=15)
        self.decision_label=tk.Label(self,textvariable=self.decision,height=2,anchor='w',justify='left',bg='#101b2b',fg='#e8eef7',wraplength=1040)
        self.decision_label.grid(row=6,column=0,sticky='ew',padx=15)
        self.preview_frame=ttk.Frame(self,height=220,width=640)
        self.preview_frame.grid(row=7,column=0,sticky='nsew',padx=15,pady=5)
        self.preview_frame.grid_propagate(False)
        self.preview=tk.Label(self.preview_frame,bg='#07111f',text='게임 화면 미리보기',fg='#a6b5ca')
        self.preview.place(relwidth=1,relheight=1)
        logs=ttk.Frame(self,height=145);logs.grid(row=8,column=0,sticky='ew',padx=15,pady=5)
        logs.grid_propagate(False);logs.columnconfigure(0,weight=1);logs.rowconfigure(0,weight=1)
        self.logbox=tk.Text(logs,height=8,wrap='word',bg='#081320',fg='#d4e3f5',font=('맑은 고딕',9),state='disabled')
        self.logbox.grid(row=0,column=0,sticky='nsew')
        scroll=ttk.Scrollbar(logs,orient='vertical',command=self.logbox.yview);scroll.grid(row=0,column=1,sticky='ns')
        self.logbox.configure(yscrollcommand=scroll.set)
        ttk.Label(self,text='신경망 갱신 = 실제 가중치 학습 횟수 / 결과 = 확인된 전투 평가 수\n화면·수치 → 행동 선택 → 결과·손실 → 경험 재학습 / 학습 완료 여부는 실전 성적으로 확인해야 합니다.',padding=(15,5)).grid(row=9,column=0,sticky='ew')
        self.current_state={}; self.current_learning={}; self.current_counts={}
        self.protocol('WM_DELETE_WINDOW',self.close)
        self.after(200,self.poll)
        self.refresh()
        self.instances.bind('<<ComboboxSelected>>',lambda _:self.load_instance_mode())
        self.load_instance_mode()

    def log(self,text):
        self.logbox.configure(state='normal'); self.logbox.insert('end',text+'\n')
        if int(self.logbox.index('end-1c').split('.')[0])>600:self.logbox.delete('1.0','101.0')
        self.logbox.see('end'); self.logbox.configure(state='disabled')

    def render_preview(self,path):
        try:
            with Image.open(path) as im:thumb=im.copy()
            width=max(1,min(960,self.preview_frame.winfo_width()-4))
            height=max(1,min(280,self.preview_frame.winfo_height()-4))
            thumb.thumbnail((width,height))
            self.photo=ImageTk.PhotoImage(thumb);self.preview.configure(image=self.photo,text='')
        except OSError:pass

    def autonomy_changed(self):
        enabled=self.autonomous.get()
        focused=self.focus_defense.get()
        if focused:
            if not getattr(self,'_focus_controls_applied',False):
                self._focus_previous_controls=[v.get() for v in (self.attack,self.muster_enabled,self.defense_enabled,self.upgrade_enabled)]
            self._focus_controls_applied=True
            self.attack.set(False);self.muster_enabled.set(False);self.upgrade_enabled.set(False);self.defense_enabled.set(True)
        else:
            variables=(self.attack,self.muster_enabled,self.defense_enabled,self.upgrade_enabled)
            if enabled:
                for variable in variables:variable.set(True)
            elif getattr(self,'_focus_controls_applied',False):
                for variable,value in zip(variables,self._focus_previous_controls):variable.set(value)
            self._focus_controls_applied=False
        for control in self.manual_controls: control.configure(state='disabled' if enabled or focused else 'normal')

    def load_instance_mode(self):
        if self.worker and self.worker.is_alive():return
        row=getattr(self,'rows',{}).get(self.instance.get())
        if not row:return
        try:saved=json.loads((DATA/f'instance-{row[0]}'/'config.json').read_text('utf-8'))
        except (OSError,ValueError):return
        self.focus_defense.set(bool(saved.get('focus_defense',False)))
        if saved.get('learning_mode') in ('train','evaluate'):self.learning_mode.set(saved['learning_mode'])
        interval=saved.get('interval',.25)
        if isinstance(interval,(int,float)) and .25<=interval<=120:self.interval.set(interval)
        self.autonomy_changed()

    def focus_changed(self):
        self.autonomy_changed()
        if not self.active_config:return
        focused=self.focus_defense.get()
        self.active_config.update(focus_defense=focused,attack_enabled=self.attack.get(),
            muster_enabled=self.muster_enabled.get(),upgrade_enabled=self.upgrade_enabled.get(),defense_enabled=self.defense_enabled.get())
        temporary=self.active_directory/'config.tmp'
        temporary.write_text(json.dumps(self.active_config,ensure_ascii=False,indent=2),'utf-8')
        temporary.replace(self.active_directory/'config.json')
        self.log('집중 방어 '+('켜짐 · 새 공격/공격용 집결 중단' if focused else '꺼짐 · 일반 판단 재개'))

    def learning_mode_changed(self):
        if not self.active_config:return
        self.active_config['learning_mode']=self.learning_mode.get()
        temporary=self.active_directory/'config.tmp'
        temporary.write_text(json.dumps(self.active_config,ensure_ascii=False,indent=2),'utf-8')
        temporary.replace(self.active_directory/'config.json')
        self.log('강화학습 모드: '+('실제 결과로 신경망 갱신' if self.learning_mode.get()=='train' else '저장된 신경망 고정 · 탐험/가중치 갱신 중단'))

    def browse(self):
        selected=filedialog.askdirectory(initialdir=self.install.get())
        if selected: self.install.set(selected); self.refresh()

    def refresh(self):
        if self.worker and self.worker.is_alive(): return
        try:
            result=subprocess.run([str(Path(self.install.get())/'ldconsole.exe'),'list2'],capture_output=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
            text=result.stdout.decode('utf-8','replace')
            if '\ufffd' in text: text=result.stdout.decode('mbcs','replace')
            self.rows={}
            for line in text.splitlines():
                row=line.split(',')
                if len(row)<10: continue
                label=f"{row[0]} · {row[1].strip()} {'[실행 중]' if row[4]=='1' else '[종료]'}"
                self.rows[label]=row
            self.instances['values']=list(self.rows)
            if self.instance.get() not in self.rows:
                selected=next((label for label,row in self.rows.items() if normalized_name(row[1])==normalized_name(DEFAULT_NAME)),next(iter(self.rows),''))
                self.instance.set(selected)
        except Exception as exc: self.log('목록 읽기 실패: '+str(exc))

    def config(self):
        row=self.rows[self.instance.get()]
        if row[4]!='1': raise ValueError('선택한 LD플레이어를 먼저 실행해 주세요.')
        config={'name':row[1].strip(),'install':self.install.get(),'target_level':self.target.get(),
                'gold_reserve':self.reserve.get()*1e6,'combat_margin':self.margin.get(),
                'max_send_fraction':self.send.get()/100,'interval':self.interval.get(),'attack_enabled':self.attack.get(),
                'muster_enabled':self.muster_enabled.get(),'defense_enabled':self.defense_enabled.get(),
                'upgrade_enabled':self.upgrade_enabled.get(),'defense_margin':self.defense_margin.get(),
                'defense_buffer':self.defense_buffer.get(),'autonomous':self.autonomous.get(),
                'exploration_speed':self.exploration_speed.get(),'decision_speed':self.decision_speed.get(),
                'focus_defense':self.focus_defense.get(),'learning_mode':self.learning_mode.get()}
        if config['autonomous']:
            config.update(attack_enabled=True,muster_enabled=True,defense_enabled=True,upgrade_enabled=True)
        if config['focus_defense']:
            config.update(attack_enabled=False,muster_enabled=False,defense_enabled=True,upgrade_enabled=False)
        if not 1<=config['target_level']<=200 or config['gold_reserve']<0 or not 1.1<=config['combat_margin']<=10 or not .1<=config['max_send_fraction']<=.70 or not .25<=config['interval']<=120: raise ValueError('설정값 범위를 확인해 주세요.')
        if not 1.1<=config['defense_margin']<=3 or not 15<=config['defense_buffer']<=120: raise ValueError('방어 우세/도착 여유 설정값 범위를 확인해 주세요.')
        if not 1<=config['exploration_speed']<=4 or not 1<=config['decision_speed']<=3: raise ValueError('탐색/판단 속도 범위를 확인해 주세요.')
        return config,int(row[0])

    def open_game(self):
        try:
            config,index=self.config()
            subprocess.Popen([str(Path(config['install'])/'ldconsole.exe'),'runapp','--index',str(index),'--packagename','com.millionvictories.games.millionlords'],creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as exc: messagebox.showerror('실행 오류',str(exc))

    def start(self):
        if self.worker and self.worker.is_alive(): return
        try: config,index=self.config()
        except Exception as exc: messagebox.showerror('설정 오류',str(exc)); return
        directory=DATA/f'instance-{index}'
        directory.mkdir(exist_ok=True)
        (directory/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),'utf-8')
        self.active_config=config;self.active_directory=directory
        self.stop_event.clear(); self.start_button.configure(state='disabled'); self.instances.configure(state='disabled')
        self.status.set('연결 및 능력치 확인 중…')
        def work():
            try: Bot(config,directory,self.stop_event,self.messages.put).run()
            except Exception as exc: self.messages.put({'error':str(exc),'stopped':True})
        self.worker=threading.Thread(target=work,daemon=True); self.worker.start()

    def stop(self): self.stop_event.set(); self.status.set('중지 요청됨 · 현재 관측 후 종료합니다.')

    def apply_speed_settings(self):
        if not self.active_config:return
        try:
            values={'interval':self.interval.get(),'exploration_speed':self.exploration_speed.get(),'decision_speed':self.decision_speed.get()}
        except (tk.TclError,ValueError):return
        if not (.25<=values['interval']<=120 and 1<=values['exploration_speed']<=4 and 1<=values['decision_speed']<=3):return
        if all(self.active_config.get(k)==v for k,v in values.items()):return
        self.active_config.update(values)
        temporary=self.active_directory/'config.tmp'
        temporary.write_text(json.dumps(self.active_config,ensure_ascii=False,indent=2),'utf-8')
        temporary.replace(self.active_directory/'config.json')
        self.log(f"속도 즉시 적용 · 기준 {values['interval']}초 / 탐색 {values['exploration_speed']}배 / 판단 {values['decision_speed']}배")

    def show_learning(self):
        panel=tk.Toplevel(self);panel.title('실전 결과 학습 현황');panel.geometry('850x540')
        box=tk.Text(panel,wrap='word',font=('맑은 고딕',10));box.pack(fill='both',expand=True)
        domains={'attack_target':'공격 대상 유형','attack_margin':'공격 여유 병력','muster_source':'한 번 집결 지원 성',
                 'defense_route':'방어 경로·여유 병력','defense_blind':'공격력 비공개 방어 경로'}
        def refresh_learning():
            if not panel.winfo_exists():return
            data=self.current_learning;out=data.get('outcomes',{});rl=data.get('reinforcement') or {}
            lines=[f"누적 기록 {data.get('updates',0)}건 / 실제 결과 평가 {out.get('evaluated',0)}건 / 확인 대기 {out.get('pending',0)}건",
                   f"누적 결과 평가 {out.get('net_reward',0):+.2f} / 점령 후 손실로 수정 {out.get('revisions',0)}건",
                   '정찰·전송만으로는 아래 선택 정책에 보상을 주지 않습니다.',
                   'VP·골드 증감은 전투 전후 HUD 관측값입니다. 자연 수급·동시 전투가 섞일 수 있습니다.',
                   '골드 자연 수급에는 보상을 주지 않고, 손실·확인된 투자 자본 손실을 반영합니다.','', '선택별 실제 성적']
            lines[:0]=[f"강화학습: {rl.get('algorithm','연결 대기')}",
                f"상태 {rl.get('phase','연결 대기')} / 모드 {rl.get('mode','—')} / 신경망 갱신 {rl.get('gradient_updates',0)}회",
                f"재학습 가능 전이 {rl.get('transitions',0)} / 연결된 실제 전투 {rl.get('combat_episodes',0)} / 출처 {rl.get('sources',{})}",
                f"Loss {rl.get('loss')} / TD 오차 {rl.get('td_error')} / 탐험율 {rl.get('epsilon',0):.1%} / 손실 교정 재학습 {rl.get('rebuilds',0)}회",
                f"모델 파일: {rl.get('checkpoint','—')}",
                '초기에는 검증된 행동의 기본 순서를 사용합니다. 신경망 갱신 횟수는 승률이나 학습 완료율이 아닙니다.',
                '전투 보고서를 읽지 못한 경우 결과를 만들어 학습하지 않습니다. 탐색 전이는 시간 비용만 반영합니다.',
                '아래 선택별 평균 성적은 기존 정책의 진단 통계이며 신경망 가중치와 별개입니다.','']
            perf=rl.get('performance',{})
            for label,key in (('최근 20전','recent'),('그 이전 20전','previous')):
                p=perf.get(key,{})
                if p.get('battles'):
                    lines.insert(6,f"{label}: 확인 {p['battles']}전 / 승률 {p['win_rate']:.1%} / 평균 평가 {p['mean_reward']:+.2f} (서로 다른 상황의 관측치)")
            for model in sorted(out.get('models',[]),key=lambda m:-m['samples']):
                lines.append(f"{domains.get(model['domain'],model['domain'])} | {choice_label(model['domain'],model['action'])} : {model['samples']}건, 평균 {model['mean']:+.2f}")
            if not out.get('models'):lines.append('아직 새 학습 구조에서 결과가 확인된 전투가 없습니다.')
            lines.extend(['','최근 평가 결과'])
            for event in out.get('recent',[]):
                signals=event.get('signals',{})
                lines.append(f"{event['kind']} {'성공' if event['result'] else '실패/성 상실'} | 평가 {event['reward']:+.2f} | VP 변화 {pretty(signals.get('vp_delta'))} / 골드 변화 {pretty(signals.get('gold_delta'))}")
                lines.append(str(signals.get('components',{})))
            box.configure(state='normal');box.delete('1.0','end');box.insert('end','\n'.join(lines));box.configure(state='disabled')
            panel.after(2000,refresh_learning)
        refresh_learning()

    def poll(self):
        self.apply_speed_settings()
        latest_image=None
        try:
            for _ in range(200):
                event=self.messages.get_nowait()
                if 'log' in event: self.log(event['log']); self.status.set(event['log'])
                if 'decision' in event: self.decision.set('현재 판단 · '+event['decision'])
                if 'error' in event: self.status.set('중단: '+event['error'])
                if 'alert' in event: self.bell(); self.status.set(event['alert'])
                if 'notice' in event: self.status.set(event['notice'])
                if 'muster' in event: self.status.set(event['muster'])
                if 'state' in event: self.current_state.update(event['state'])
                if 'learning' in event: self.current_learning=event['learning']
                if 'counts' in event: self.current_counts=event['counts']
                if 'image' in event:
                    latest_image=event['image']
                if event.get('stopped'):
                    self.active_config=None
                    self.start_button.configure(state='normal'); self.instances.configure(state='readonly')
                s=self.current_state; c=self.current_counts
                out=self.current_learning.get('outcomes',{})
                rl=self.current_learning.get('reinforcement') or {}
                self.metrics.set(f"VP {pretty(s.get('vp'))} | 골드 {pretty(s.get('gold'))} | 신경망 갱신 {rl.get('gradient_updates',0)} · 결과 {out.get('evaluated',0)} · 대기 {out.get('pending',0)} | 공격 {c.get('attacks',0)} · 방어 {c.get('defenses',0)}")
        except queue.Empty: pass
        if latest_image:self.render_preview(latest_image)
        width=max(200,self.winfo_width()-30)
        self.status_label.configure(wraplength=width);self.decision_label.configure(wraplength=width)
        self.after(100,self.poll)

    def close(self):
        self.stop_event.set()
        if self.worker and self.worker.is_alive(): self.status.set('저장 후 종료 중…'); self.after(400,self.close)
        else: self.destroy()


def main():
    import sys
    app=App()
    if '--focus-defense' in sys.argv:app.focus_defense.set(True);app.autonomy_changed()
    if '--fast' in sys.argv:app.interval.set(.25)
    if '--evaluate' in sys.argv:app.learning_mode.set('evaluate')
    if '--train' in sys.argv:app.learning_mode.set('train')
    if '--start' in sys.argv: app.after(1000,app.start)
    app.mainloop()

if __name__=='__main__': main()
