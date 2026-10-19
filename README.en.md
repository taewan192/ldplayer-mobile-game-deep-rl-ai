# LDPlayer Mobile Game Deep Reinforcement Learning AI

**v0.26 · LD 플레이어 모바일 게임전용 딥 강화학습 AI**

[한국어](README.md#한국어) · [Algorithm analysis](docs/ALGORITHMS.md) · [Validation and results](docs/RESULTS.md)

A local Windows deep reinforcement learning project that observes an LDPlayer game through screenshots and OCR, executes verified game skills, and updates its policy from observed outcomes. The current game adapter targets **Million Lords**, including attack selection, troop assembly, defensive reinforcement, and investment review. Training and inference run on the CPU without an external AI API key or GPU.

**Status:** an experimental learning system. A neural policy learns to choose among hand-written, validated skills. It does not discover every game mechanic from scratch or work with arbitrary mobile games without a new adapter. Working training code is not evidence of strong live-game performance.

## Implemented capabilities

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

## Why this is deep reinforcement learning

Machine learning is the broad family of methods that learn from data. Deep learning uses multilayer neural networks. Reinforcement learning improves decisions using rewards obtained after actions. This project combines them: a neural network approximates action values and is trained from transitions and outcomes.

The primary policy uses **Double DQN with Prioritized Experience Replay**, implemented directly in NumPy. REA was used to investigate game artifacts and behavior; it is neither the learning algorithm nor a source of account-specific training outcomes.

## Learning loop

1. Capture the game screen and collect readable state values.
2. Generate feasible actions under shield, exclusion, reserve, timing, and identity checks.
3. Evaluate the state and each candidate with the Q-network and select an action.
4. Execute the existing scouting, transfer, assembly, attack, or support skill; verify critical details before dispatch.
5. Link verified battle or ownership observations to the decisions that were actually made. Do not invent a win or loss when the result is unreadable.
6. Store transitions and train small minibatches once at least eight transitions are ready.
7. Save model and optimizer state so learning can resume locally.

Scouting counts, exploration counts, and image counts are not victory rewards. Navigation-like transitions receive a cost for real elapsed time and bootstrap from a later observed decision. Navigation experience can update weights without a battle, so **gradient updates and combat-learning episodes are different metrics**.

## Algorithms and rationale

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

## Installation

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

### Training versus frozen-policy play

- **Train while playing:** collect experience and update weights from ready transitions.
- **Play with saved policy:** freeze network updates and disable random exploration. Observation and execution records may still be written.
- With no trained checkpoint, the controller uses explicit initial priorities. Zero gradient updates do not indicate a trained agent.
- Focus defense is a separate mode that blocks new offensive work regardless of learning mode.

### Local persistence

Data lives under `%LOCALAPPDATA%\MillionLordsAI\instance-<index>`.

| File | Purpose |
| --- | --- |
| `learning.sqlite3` | Observations, choices, outcomes, replay transitions, and statistics |
| `reinforcement/visual-policy.npz` | Online/target networks, Adam state, and correction generation |
| `config.json` | Per-instance settings |
| Screen, march, and event records | OCR, outcome verification, recovery, and duplicate-dispatch prevention |

A folder of images alone does not train this policy. Observations must be connected to actions, rewards, and successors. Private runtime databases and trained weights are not included in this repository.

## Tests and observed results

```powershell
python -m unittest discover -s tests -v
```

Tests use temporary data and screen fixtures. They do not launch a live emulator or send troops. Passing tests verify implementation behavior, not live win rate or VP gains.

An archived v0.26 installation audit observed real weight changes and checkpoint persistence. That snapshot reported two gradient updates, nine ready transitions, and zero combat episodes linked to the new neural policy. This validates learning and persistence plumbing; it does not demonstrate a strong combat policy. These are historical observations, not the current running agent's latest metrics. See [results and interpretation](docs/RESULTS.md).

## Applying the framework to another game

The policy network, replay system, and checkpoint format can be reused. Observation extraction, skill executors, feasible-action generation, menu recovery, result verification, and reward design are game-specific. Another game needs a new adapter; changing the Android package name is insufficient.

## Source map

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

## Build a Windows executable

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\build-live.ps1 -Python .\.venv\Scripts\python.exe
```

Output: `dist\MillionLordsAI\MillionLordsAI.exe`. The build bundles the OCR script and image templates. This is a source distribution, not a release of proven combat weights.

## Distribution scope

The repository excludes APKs, original native game binaries, extracted metadata, payment details, private runtime databases/checkpoints, and raw live-session logs. Game UI templates and test screen fixtures are included for recognition and regression testing; their underlying rights remain with their respective owners. See [NOTICE](NOTICE.md).
