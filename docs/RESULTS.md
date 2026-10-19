# 검증 결과와 해석 / Validation results and interpretation

## 검증 범위 / Scope

소스 배포 준비일은 2026-10-19(KST)입니다. 아래 과거 실전 요약과 이번 소스 테스트는 다른 종류의 검증입니다. 업로드 준비 중에는 실행 중인 게임을 조작하거나 새 전투·학습 결과를 만들지 않았습니다.

The source export was prepared on 2026-10-19 (KST). Historical live observations and current source tests are different kinds of evidence. No live game actions or new combat outcomes were generated while preparing this repository.

## A. 자동 테스트 / Automated tests

재현 명령 / Reproduction command:

```powershell
python -m unittest discover -s tests -v
```

이번 내보내기에는 현재 실전 컨트롤러의 테스트를 포함합니다. 별도 초기 Arena 시뮬레이터와 그 전용 테스트 3개 파일은 포함하지 않았습니다. 따라서 원래 전체 작업 폴더의 과거 테스트 총수와 이 저장소의 테스트 총수는 다를 수 있습니다.

This export contains the live-controller test suite. The separate early Arena simulator and its three dedicated test files are excluded. Consequently the repository's test count can differ from an older full-workspace count. Machine-readable execution results are recorded in `validation/source-tests.json` after running the suite from this exported directory.

**2026-10-19 내보낸 소스 검증: 328개 테스트 통과, 실패 0, 오류 0 (23.061초).**

**Export validation on 2026-10-19: 328 tests passed, zero failures, zero errors (23.061 seconds).**

| 검증 / Check | 무엇을 증명하나요? / What it establishes |
| --- | --- |
| Actual pixel-weight backpropagation | 화면 입력에 연결된 가중치가 실제로 바뀜 / Pixel-connected weights actually update. |
| Double DQN target calculation | 온라인 선택과 타깃 평가가 분리됨 / Online selection and target evaluation are separate. |
| Delayed credit and successor masks | 다단계 선택이 결과와 합법적 다음 후보에 연결됨 / Traces preserve delayed credit and feasible successors. |
| Economic loss reward | 확인된 손실이 평가값에 반영됨 / Verified losses affect reward. |
| Unknown and duplicate outcomes | 결과 불명·중복이 가짜 새 전투가 되지 않음 / Unknown or duplicate observations do not become fabricated battles. |
| Later-loss correction | 기존 성공 라벨을 교정하고 재학습 요청 / Later loss can relabel an episode and trigger rebuilding. |
| Frozen evaluation and checkpoint | 실행 모드의 가중치 고정과 저장 복원 / Frozen weights and checkpoint persistence. |
| OCR and interaction regression | 알려진 숫자·지도·병력 열 클릭 오류에 대한 회귀 검사 / Regression checks for known parsing and interaction failures. |
| Defensive priority and reserves | 방어 우선 처리, 지원 기한과 잔류량 조건 / Defensive priority, deadlines, and reserves. |

테스트의 모의 전투는 알고리즘 검사용이며 실전 승률에 포함하지 않습니다. 테스트 통과는 무한 루프가 영원히 없거나 모든 상대 공격을 막는다는 증명이 아닙니다.

Synthetic test episodes validate code and are never counted as live performance. Passing tests does not establish the absence of all future loops or guarantee defense against every attack.

## B. 보관된 v0.26 설치 검증 / Archived installation audit

원본 설치 감사에서 개인정보 경로·인스턴스 식별자를 제거한 요약을 `validation/archived-v0.26-audit.json`에 보관합니다. 아래는 당시 값이며 현재 상태를 조회한 결과가 아닙니다.

| 항목 / Metric | 과거 값 / Archived value |
| --- | ---: |
| In-memory gradient updates | 2 |
| Ready replay transitions | 9 |
| Navigation transitions, including not-yet-ready | 10 |
| Combat episodes linked to new neural policy | **0** |
| Loss at snapshot | 0.15837283432483673 |
| Mean absolute TD error at snapshot | 0.6224150061607361 |
| Saved checkpoint gradient updates | 1 |
| Saved weights changed | true |

준비된 경험 9개와 navigation 행 10개는 모순이 아닙니다. 다음 전략 관측에 아직 연결되지 않은 행은 준비된 학습 경험 수에 포함되지 않습니다. 메모리 갱신 2회와 저장 갱신 1회의 차이는 그 스냅샷에서 체크포인트가 뒤처져 있었다는 뜻이며, 종료 저장까지 최신이었음을 입증하는 수치는 아닙니다.

The audit demonstrates that observed live navigation fed actual neural updates and changed persisted weights. It does not demonstrate learned combat. A lagging checkpoint in that snapshot is reported explicitly rather than described as fully synchronized.

## C. 보관된 약 3분 실행 관측 / Archived approximately three-minute run

| 항목 / Metric | 값 / Value |
| --- | ---: |
| Duration | Approximately 182.3 seconds |
| Captures | 55 |
| Exploration commands | 3 |
| Attacks, conquests, supports, defenses | 0 in that window |

55회 캡처가 55회 독립적인 전략 판단이나 학습 갱신을 뜻하지 않습니다. 전투가 없었던 짧은 실행으로 공격·집결·방어의 완주나 성능 향상을 주장할 수 없습니다.

This short run is an observation of controller activity, not a throughput benchmark or a successful attack/assembly/defense demonstration. Earlier legacy outcome counters are not imported as verified visual-policy combat episodes without observation/action provenance.

## D. 아직 입증되지 않은 것 / Not established

- 기본 규칙 정책보다 높은 실전 승률 또는 VP/시간 / Better live win rate or VP per hour than the baseline.
- 골드·병력 손실의 지속적 감소 / Sustained reductions in gold or troop losses.
- 학습 완료, 최적 정책, 사람 수준 판단 / Convergence, optimality, or human-level judgment.
- Lv.110~115 성의 무조건적 유지 / Guaranteed retention of priority cities.
- 임의 모바일 게임으로의 자동 전이 / Automatic transfer to arbitrary mobile games.

## E. 실전 평가 시 확인할 지표 / Recommended live evaluation metrics

공격은 정찰 대비 출병 비율, 집결 시작 대비 완료 비율, 실제 점령·유지 비율을 확인합니다. 방어는 위협 감지 지연, 기한 내 지원 도착, 성 유지 여부를 봅니다. 경제 성과는 VP/시간, 골드·병력 손실, 투자 후 성 유지로 나누어 봅니다. 이것들을 신경망 갱신 수·Loss와 별도로 기록해야 합니다.

For a future evaluation, compare a frozen learned policy against the baseline over documented conditions. Track scouting-to-attack conversion, assembly completion, conquest retention, timely support, defense outcomes, VP per hour, and resource losses. Report sample counts and unknown outcomes. These comparisons are proposed evaluation work, not results already obtained.
