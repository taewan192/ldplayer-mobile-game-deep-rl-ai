# LD 플레이어 모바일 게임전용 딥 강화학습 AI

**LDPlayer Mobile Game Deep Reinforcement Learning AI · v0.26**

[English](#english) · [한국어](#한국어) · [Algorithm analysis / 알고리즘 분석](docs/ALGORITHMS.md) · [Results / 검증 결과](docs/RESULTS.md)

The full English and Korean descriptions are included below on this page.
이 페이지에서 영문과 한글 설명을 모두 읽을 수 있습니다.

## English

A local Windows deep reinforcement learning project that observes an LDPlayer game through screenshots and OCR, executes verified game skills, and updates its policy from observed outcomes. The current game adapter targets **Million Lords**, including attack selection, troop assembly, defensive reinforcement, and investment review. Training and inference run on the CPU without an external AI API key or GPU.

**Status:** an experimental learning system. A neural policy learns to choose among hand-written, validated skills. It does not discover every game mechanic from scratch or work with arbitrary mobile games without a new adapter. Working training code is not evidence of strong live-game performance.

### Implemented capabilities

| Capability | Implementation and scope |
| --- | --- |
| Emulator selection | Select an installation and running instance in the UI; keep separate data per instance. |
| Observation | Combine screenshots, OpenCV templates and map registration, and Windows OCR. |
| Attack review | Filter shields and excluded players; generate feasible options using observed scouting information. |
| Troop assembly | Calculate a staging city's shortage, verify donor capacity and reserves, and execute transfers. Candidate generation favors a donor capable of supplying the entire shortage when available. |
| Defensive reinforcement | Prioritize incoming alerts and plan support using arrival deadlines and available troops. |
| Focus defense | Block new offensive attacks and assembly; prioritize level 110–115 cities. Successful protection is not guaranteed. |
| Exploration | Persist an eight-direction, nine-step-per-direction route anchored to the main city, with observed movement checks. |
| Investment review | Check reserves and defensive conditions; track verified spending and subsequent retention or loss. |
| Outcome learning | Associate verified results with decisions and account for observable VP, gold, troop losses, and time. |
| Recovery | Attempt known-panel recovery and selected-instance game/bridge recovery. |
| Monitoring | Display gradient updates, ready transitions, actual combat episodes, loss, recent results, and logs. |

These are implemented code paths, not claims that every capability has succeeded in unrestricted live play. See the separate [validation report](docs/RESULTS.md).

### Why this is deep reinforcement learning

Machine learning is the broad family of methods that learn from data. Deep learning uses multilayer neural networks. Reinforcement learning improves decisions using rewards obtained after actions. This project combines them: a neural network approximates action values and is trained from transitions and outcomes.

The primary policy uses **Double DQN with Prioritized Experience Replay**, implemented directly in NumPy. REA was used to investigate game artifacts and behavior; it is neither the learning algorithm nor a source of account-specific training outcomes.

### Learning loop

1. Capture the game screen and collect readable state values.
2. Generate feasible actions under shield, exclusion, reserve, timing, and identity checks.
3. Evaluate the state and each candidate with the Q-network and select an action.
4. Execute the existing scouting, transfer, assembly, attack, or support skill; verify critical details before dispatch.
5. Link verified battle or ownership observations to the decisions that were actually made. Do not invent a win or loss when the result is unreadable.
6. Store transitions and train small minibatches once at least eight transitions are ready.
7. Save model and optimizer state so learning can resume locally.

Scouting counts, exploration counts, and image counts are not victory rewards. Navigation-like transitions receive a cost for real elapsed time and bootstrap from a later observed decision. Navigation experience can update weights without a battle, so **gradient updates and combat-learning episodes are different metrics**.

### Algorithms and rationale

| Component | Why it was chosen | What it enables |
| --- | --- | --- |
| State-action MLP | Lightweight CPU execution and a shared scorer for variable-sized candidate sets. | Approximate Q-values from coarse pixels, frame differences, OCR state, and action features. |
| Double DQN | Separate action selection from target evaluation to mitigate maximization bias. | Update estimated future rewards of attack, assembly, and defense choices. |
| Target network | Reduce the movement of bootstrapped learning targets. | Copy online parameters every 100 gradient updates. |
| Prioritized replay | Live interaction is slow; reuse transitions with larger TD errors more often. | Revisit surprising outcomes rather than treating all stored transitions equally. |
| Importance weights | Partially correct the sampling bias introduced by prioritized replay. | Adjust the influence of frequently selected samples. |
| Hierarchical skills | Reduce the action space and connect strategic choices to delayed outcomes. | Learn a sequence such as strategy, staging city, target, troop margin, and route. |
| Bounded epsilon-greedy exploration | Try alternative feasible options after training begins. | Decay exploration from approximately 8% toward a 1% floor. |
| Huber loss, Adam, gradient clipping | Manage large TD errors and optimize a small CPU network. | Perform actual backpropagation and parameter updates. |
| Economic reward shaping | Avoid judging an action solely by a short-lived conquest or VP increase. | Account for observable resource losses and elapsed time. |
| Feasibility filters | Keep forbidden or impossible actions outside the policy's choice set. | Enforce shields, reserves, identity checks, and arrival deadlines. These checks are coded constraints, not learned rules. |

See [the detailed analysis](docs/ALGORITHMS.md) for equations, dimensions, hyperparameters, and reward limitations.

### Installation

Requirements:

- Windows, LDPlayer 9, and a running Million Lords instance.
- Landscape 16:9 resolution; coordinates are normalized to a 1280×720 reference.
- Windows OCR language support for Korean and English. The current adapter is tuned for the Korean game UI and bundled templates.
- Development validation used Python 3.14, NumPy 2.5.4, OpenCV 5.0.0, and Pillow 12.3.0. Other runtime combinations have not all been validated.

In PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-live.txt
.\.venv\Scripts\python.exe launch_live.py
```

Select the LDPlayer installation directory, refresh the instance list, and choose your instance. Open the game map, select training or frozen-policy mode, then start. The instance-name constant in the source is a historical development default; select your own instance in the UI.

The minimum base interval is 0.25 seconds. It does not guarantee four complete decisions per second: OCR, UI transitions, map registration, LD commands, and travel waits add latency.

#### Training versus frozen-policy play

- **Train while playing:** collect experience and update weights from ready transitions.
- **Play with saved policy:** freeze network updates and disable random exploration. Observation and execution records may still be written.
- With no trained checkpoint, the controller uses explicit initial priorities. Zero gradient updates do not indicate a trained agent.
- Focus defense is a separate mode that blocks new offensive work regardless of learning mode.

#### Local persistence

Data lives under `%LOCALAPPDATA%\MillionLordsAI\instance-<index>`.

| File | Purpose |
| --- | --- |
| `learning.sqlite3` | Observations, choices, outcomes, replay transitions, and statistics |
| `reinforcement/visual-policy.npz` | Online/target networks, Adam state, and correction generation |
| `config.json` | Per-instance settings |
| Screen, march, and event records | OCR, outcome verification, recovery, and duplicate-dispatch prevention |

A folder of images alone does not train this policy. Observations must be connected to actions, rewards, and successors. Private runtime databases and trained weights are not included in this repository.

### Tests and observed results

```powershell
python -m unittest discover -s tests -v
```

Tests use temporary data and screen fixtures. They do not launch a live emulator or send troops. Passing tests verify implementation behavior, not live win rate or VP gains.

An archived v0.26 installation audit observed real weight changes and checkpoint persistence. That snapshot reported two gradient updates, nine ready transitions, and zero combat episodes linked to the new neural policy. This validates learning and persistence plumbing; it does not demonstrate a strong combat policy. These are historical observations, not the current running agent's latest metrics. See [results and interpretation](docs/RESULTS.md).

### Applying the framework to another game

The policy network, replay system, and checkpoint format can be reused. Observation extraction, skill executors, feasible-action generation, menu recovery, result verification, and reward design are game-specific. Another game needs a new adapter; changing the Android package name is insufficient.

### Source map

| Module | Responsibility |
| --- | --- |
| `app.py` | Windows UI and instance configuration |
| `driver.py`, `vision.py` | Emulator bridge, screenshots, OCR, number parsing, templates, and registration |
| `deep_rl.py` | MLP, Double DQN, prioritized replay, and checkpoints |
| `reinforcement.py` | Live observations and hierarchical policy scheduling |
| `outcomes.py`, `learning.py` | Verified rewards, SQLite history, and retained legacy bandit components |
| `muster.py`, `batch.py` | Troop assembly and multi-scout execution paths |
| `defense.py`, `focus_defense.py` | Support, investment conditions, and defensive priorities |
| `exploration.py`, `recovery.py` | Persisted exploration and game/bridge recovery |
| `tests/` | Algorithm, workflow, and screen-regression tests |

### Build a Windows executable

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\build-live.ps1 -Python .\.venv\Scripts\python.exe
```

Output: `dist\MillionLordsAI\MillionLordsAI.exe`. The build bundles the OCR script and image templates. This is a source distribution, not a release of proven combat weights.

### Distribution scope

The repository excludes APKs, original native game binaries, extracted metadata, payment details, private runtime databases/checkpoints, and raw live-session logs. Game UI templates and test screen fixtures are included for recognition and regression testing; their underlying rights remain with their respective owners. See [NOTICE](NOTICE.md).

---

## 한국어

Windows PC에서 LD플레이어의 게임 화면을 캡처하고, 화면·OCR 상태와 실제 행동 결과를 이용해 정책을 갱신하는 로컬 딥 강화학습 프로젝트입니다. 현재 구현된 게임 어댑터는 **Million Lords**이며, 공격 대상 선정, 병력 집결, 방어 지원, 투자 검토를 대상으로 합니다. CPU에서 실행하며 외부 AI API 키나 GPU가 필요하지 않습니다.

**현재 상태:** 학습 가능한 실험적 시스템입니다. 사람이 작성한 게임 조작 기술과 검증 규칙 안에서 신경망이 선택을 학습합니다. 게임 전체를 처음부터 이해하거나, 임의의 모바일 게임에 그대로 적용되는 범용 AI는 아닙니다. 학습 코드가 작동한다는 검증과 높은 실전 승률을 얻었다는 검증은 구분합니다.

### 무엇을 할 수 있나요?

| 기능 | 구현 방식과 범위 |
| --- | --- |
| LD플레이어 선택 | 설치 폴더와 실행 중인 인스턴스를 UI에서 선택하고 인스턴스별 데이터를 분리합니다. |
| 화면 관측 | 스크린샷, OpenCV 템플릿·지도 정합, Windows OCR로 화면과 숫자를 읽습니다. |
| 공격 검토 | 보호막·제외 대상 등을 필터링하고, 정찰로 확인한 정보를 바탕으로 실행 가능한 후보를 만듭니다. |
| 병력 집결 | 기점 성의 부족분을 계산하고, 지원 성의 실제 병력·잔류량·경로를 확인한 후 집결을 수행합니다. 충분한 단일 지원 성에서 부족분 전체를 옮기는 후보를 우선 구성합니다. |
| 방어 지원 | 침입 알림을 우선 처리하고 도착 시간과 동원 가능한 병력을 확인해 지원을 계획합니다. |
| 집중 방어 | 새 공격과 공격 집결을 금지하고 방어를 우선하며, Lv.110~115 성을 우선 보호 대상으로 다룹니다. 보호 성공을 보장하는 기능은 아닙니다. |
| 탐색 | 메인 성을 기준으로 8방향 × 각 9단계 순번을 저장합니다. 실제 지도 이동 확인과 재정렬을 함께 사용합니다. |
| 투자 검토 | 골드 예비량과 방어 조건을 검토하고 지출 및 이후 보유·상실을 평가합니다. |
| 결과 학습 | 확인된 전투 결과와 관측 가능한 VP·골드·병력 손실, 소요 시간을 보상에 반영합니다. |
| 복구 | 알려진 창을 닫고, 게임 종료·Android 연결 문제를 감지하면 선택한 인스턴스에 한해 복구를 시도합니다. |
| 학습 확인 | 신경망 갱신, 준비된 경험, 실제 전투 수, 손실 함수, 최근 성적과 실행 로그를 표시합니다. |

이 표는 코드에 구현된 경로를 설명합니다. 모든 상황에서의 실전 성공을 의미하지 않으며, 검증 수준은 [결과 문서](docs/RESULTS.md)에 따로 기록합니다.

### 머신러닝, 딥러닝, 강화학습의 관계

머신러닝은 데이터에서 판단 규칙을 학습하는 넓은 범주입니다. 딥러닝은 여러 층의 신경망으로 표현과 예측을 학습하는 방법입니다. 강화학습은 행동 이후 받은 보상으로 다음 행동 선택을 개선하는 학습 문제입니다.

이 프로젝트는 **여러 층의 신경망으로 행동 가치를 근사하는 강화학습**, 즉 딥 강화학습을 사용합니다. 주요 정책은 **Double DQN + Prioritized Experience Replay**이며 NumPy로 직접 구현했습니다. REA는 게임 파일과 동작을 이해하는 분석 도구로 사용되었으며, 학습 알고리즘이나 보상 데이터베이스가 아닙니다.

### 학습이 이루어지는 과정

1. 현재 게임 화면을 캡처하고 읽을 수 있는 상태를 수집합니다.
2. 예비 병력, 실드, 제외 유저, 이동 시간, 목적지 확인 등의 조건으로 가능한 행동만 남깁니다.
3. 신경망이 상태와 각 후보 행동의 장기 가치를 평가하고 한 행동을 선택합니다.
4. 기존 실행기가 정찰, 이동, 집결, 공격 또는 지원을 수행하고 출병 직전 정보를 다시 확인합니다.
5. 관측된 전투·성 소유권 결과를 실제 선택 기록과 연결합니다. 결과가 불명확하면 승패를 만들어 넣지 않습니다.
6. 결과 보상과 다음 상태를 경험 버퍼에 저장하고, 준비된 경험이 8개 이상이면 작은 미니배치로 역전파합니다.
7. 모델과 옵티마이저 상태를 저장하고 다음 실행에서 이어갑니다.

탐색·정찰 횟수나 이미지 개수 자체를 승리 보상으로 사용하지 않습니다. 탐색 계열 경험에는 실제 경과 시간 비용을 주고 이후 관측의 가치와 연결합니다. 전투가 없는 상태에서도 탐색 경험으로 가중치가 갱신될 수 있으므로 **신경망 갱신 수가 전투 학습 수와 같지는 않습니다.**

### 사용한 알고리즘과 선택 이유

| 구성 | 왜 사용했나요? | 무엇을 학습하거나 처리하나요? |
| --- | --- | --- |
| 상태·행동 입력 MLP | CPU에서 가볍게 실행하고 개수가 바뀌는 후보를 같은 신경망으로 평가하기 위해 사용합니다. | 작은 RGB 화면, 화면 변화, OCR 숫자와 행동 특징의 조합에서 Q값을 근사합니다. |
| Double DQN | 최댓값 선택에서 발생할 수 있는 가치 과대평가를 줄이도록 행동 선택과 평가를 분리합니다. | 공격·집결·방어 후보의 미래 보상 예측을 갱신합니다. |
| 타깃 신경망 | 학습할 때 정답 역할을 하는 값이 매번 함께 움직이는 문제를 완화합니다. | 100회 갱신마다 온라인 신경망의 가중치를 복사합니다. |
| 우선순위 경험 재학습 | 실제 게임 경험이 느리고 귀하므로 TD 오차가 큰 경험을 더 자주 재사용합니다. | 예상과 달랐던 결과를 자주 다시 학습합니다. |
| 중요도 가중치 | 우선순위 표본 추출로 생기는 학습 편향을 보정하는 데 사용합니다. | 자주 뽑힌 경험의 영향력을 조절합니다. |
| 계층형 행동 기술 | 픽셀 좌표를 무작위로 누르는 행동 공간을 줄이고 결과를 전략 선택과 연결합니다. | 전략 → 기점·목표 → 병력량·경로처럼 여러 선택을 전투 결과에 연결합니다. |
| 제한된 ε-greedy | 현재 가치가 높은 행동 외의 실행 가능한 후보도 일부 시도합니다. | 학습 후 탐험 확률은 약 8%에서 최소 1%로 감소합니다. |
| Huber 손실·Adam·기울기 제한 | 큰 예측 오차와 불안정한 경사에 대응하며 작은 CPU 신경망을 최적화합니다. | 가중치를 실제로 변경합니다. |
| 경제 손실을 포함한 보상 | 점령 직후 VP만 높고 이후 성과 골드를 잃는 선택을 무조건 좋게 평가하지 않도록 합니다. | 전투 결과와 관측 가능한 손실·시간의 절충을 학습합니다. |
| 합법 행동 필터와 재확인 | 불가능하거나 금지된 행동은 학습 정책의 선택 대상에서 제외해야 하기 때문입니다. | 실드, 예비 병력, 목적지, 도착 마감 등의 제약을 적용합니다. 이 부분은 학습된 규칙이 아닙니다. |

수식, 정확한 입력 차원, 하이퍼파라미터, 보상 한계는 [알고리즘 상세](docs/ALGORITHMS.md)를 참고하세요.

### 실행 환경과 설치

- Windows PC, LDPlayer 9, 실행 중인 Million Lords 인스턴스.
- 가로 16:9 해상도. 내부 화면 좌표는 1280×720 기준으로 정규화합니다.
- Windows OCR 한국어·영어 인식 언어 지원. 현재 게임 어댑터는 한국어 UI와 해당 템플릿에 맞춰져 있습니다.
- 개발 검증 환경: Python 3.14, NumPy 2.5.4, OpenCV 5.0.0, Pillow 12.3.0. 모든 Python/LDPlayer 버전을 지원한다고 검증하지 않았습니다.

PowerShell에서 다음을 실행합니다.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-live.txt
.\.venv\Scripts\python.exe launch_live.py
```

UI에서 LD플레이어 설치 경로를 지정하고 목록을 새로고침한 뒤 대상 인스턴스를 선택합니다. 게임 지도를 연 상태에서 **강화학습하며 플레이** 또는 **저장된 정책으로 플레이**를 선택하고 시작합니다. 코드에 들어 있는 인스턴스 이름 기본값은 로컬 개발 당시의 예시이며, 다른 사용자는 자신의 인스턴스를 선택해야 합니다.

기준 간격의 최솟값은 0.25초입니다. 이 값은 전체 행동 처리 시간이나 초당 4회 판단을 보장하지 않습니다. OCR, 화면 전환, 지도 정합, LD 명령과 이동 대기는 추가 시간이 필요합니다.

#### 학습 모드와 실행 모드

- **강화학습하며 플레이:** 경험을 수집하고 준비된 경험으로 신경망을 갱신합니다.
- **저장된 정책으로 플레이:** 신경망 가중치를 고정하고 무작위 탐험을 끕니다. 관측과 실행 기록은 계속 남을 수 있습니다.
- 학습한 모델이 없으면 초기 우선순위를 사용합니다. UI에 갱신 0이라고 보이는 상태를 이미 학습된 AI라고 해석하면 안 됩니다.
- 집중 방어는 별도 선택이며, 학습 여부와 무관하게 새 공격을 차단합니다.

#### 로컬 데이터

기본 저장 위치는 `%LOCALAPPDATA%\MillionLordsAI\instance-<index>`입니다.

| 파일 | 역할 |
| --- | --- |
| `learning.sqlite3` | 기록, 선택 근거, 결과, 재학습 경험과 통계 |
| `reinforcement/visual-policy.npz` | 온라인·타깃 신경망, Adam 상태, 모델 세대 |
| `config.json` | 인스턴스별 설정 |
| 화면·행군·이벤트 기록 | OCR, 결과 대조, 복구와 중복 출병 방지 |

이미지 폴더를 모아 두기만 해서는 신경망이 학습하지 않습니다. 실제 관측·선택·보상·다음 상태가 연결되어야 합니다. 각 인스턴스의 원본 데이터와 가중치는 이 저장소에 포함하지 않았습니다.

### 검증과 현재 결과

```powershell
python -m unittest discover -s tests -v
```

테스트는 임시 데이터와 테스트 화면을 사용합니다. 이 명령은 실제 LD플레이어를 켜거나 군대를 보내는 실전 플레이 검증이 아닙니다. 테스트가 통과해도 실전 승률·VP 상승이 입증되는 것은 아닙니다.

보관된 v0.26 설치 검증에서는 실제 가중치 변경과 저장을 확인했습니다. 당시 스냅샷은 신경망 갱신 2회, 준비된 경험 9개, 새 신경망에 연결된 전투 0건이었습니다. 이는 **학습 배선과 저장이 작동한다는 증거**이며, 강한 전투 정책을 학습했다는 증거는 아닙니다. 이 수치는 현재 실행 중인 AI의 최신 상태가 아닙니다. 자세한 해석과 재현 결과는 [결과 문서](docs/RESULTS.md)에 있습니다.

### 다른 모바일 게임에도 사용할 수 있나요?

신경망·재학습·저장 구조는 재사용할 수 있습니다. 하지만 현재 관측기, 조작기, 전투 보상, 실행 가능한 행동 생성기는 Million Lords용입니다. 다른 게임에는 화면 인식, 상태 정의, 메뉴 복구, 행동 기술, 성공·실패 판정과 보상을 새로 구현해야 합니다. 패키지 이름만 바꾸면 동작하는 제품은 아닙니다.

### 소스 구조

```text
live_lords/
  app.py                 # Windows UI와 인스턴스 설정
  driver.py              # LD 명령, 캡처, OCR 연결과 입력 검사
  vision.py              # 숫자 해석, 템플릿·지도 정합
  deep_rl.py             # MLP, Double DQN, 우선순위 경험 재학습
  reinforcement.py       # 게임 상태와 학습 정책의 연결
  outcomes.py            # 확인된 결과와 경제 보상
  learning.py            # SQLite 기록과 이전 밴딧 학습 보존
  muster.py / batch.py    # 집결·다중 정찰 실행 경로
  defense.py             # 지원 및 투자 조건
  focus_defense.py       # 집중 방어
  exploration.py         # 8방향 탐색 기억
  recovery.py            # 게임·연결 복구
tests/                   # 알고리즘·실행 경로·화면 회귀 테스트
docs/                    # 알고리즘 분석과 검증 결과
validation/              # 비식별 과거 검증 요약과 이번 테스트 결과
```

#### 실행 파일 빌드

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\build-live.ps1 -Python .\.venv\Scripts\python.exe
```

결과는 `dist\MillionLordsAI\MillionLordsAI.exe`입니다. OCR 스크립트와 이미지 템플릿도 함께 묶습니다. 이 저장소는 소스 배포이며 검증된 전투 가중치를 제공하지 않습니다.

### 배포 범위

APK, 원본 게임 네이티브 바이너리, 추출 메타데이터, 결제 정보, 개인 실행 DB·모델과 원시 실행 로그는 제외했습니다. `live_lords/assets`와 `tests/fixtures`의 게임 UI 이미지 조각·테스트 화면은 인식 및 회귀 검증에 필요한 자료이며, 관련 권리는 해당 권리자에게 있습니다. 자세한 내용은 [NOTICE](NOTICE.md)를 참고하세요.
