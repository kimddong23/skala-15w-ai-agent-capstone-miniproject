# 데이터

FitBit과 Stanford 데이터의 출처, 전처리, 컬럼 사전과 라이선스를 모았다. 제3자 자료 전체는 [NOTICE](../NOTICE.md)에 있다. 개요는 [README](../README.md)에 있다.

## 데이터

데이터셋은 둘이다. FitBit은 서비스와 개발 평가에 쓰는 기본 데이터이고 Stanford는 동결한 탐지기를 한 번 적용해 보는 외부 검증용이라 탐지기 튜닝에는 쓰지 않는다. 앱의 같은 `VitalsStore`가 두 데이터셋을 모두 읽는다.

| | FitBit (개발·서비스) | Stanford COVID-19 웨어러블 (외부 검증) |
|---|---|---|
| 역할 | 에이전트·화면의 기본 데이터. 평가용 이상을 합성 주입 | 실제 질병 사건(증상·진단·회복일)에 탐지기를 튜닝 없이 적용 |
| 사용자 ID | `U01`\~`U15` (가명) | `S001`\~ (가명) |
| 라이선스 표기 | 원본 Zenodo CC BY 4.0 (Kaggle 재배포본에는 다른 표기가 있으나 원본의 CC BY 4.0을 따라 저작자 표시) | **표기 없음** |
| 저장소 포함 | 가공본 포함 (`data/processed/`, 약 8MB) | 포함하지 않음 (`data/external/`, git 제외). 다운로드·변환 스크립트와 집계 지표만 공개 |

### FitBit (개발·서비스)

**출처**
- 이름: Fitbit Fitness Tracker Data.
- **원본 출처**: Furberg, R., Brinton, J., Keating, M., Ortiz, A. (2016). *Crowd-sourced Fitbit datasets 03.12.2016-05.12.2016*. Zenodo. [doi:10.5281/zenodo.53894](https://doi.org/10.5281/zenodo.53894). 라이선스 CC BY 4.0 (2026-10-20 Zenodo API로 확인).
- **재배포처**: Kaggle [`arashnic/fitbit`](https://www.kaggle.com/datasets/arashnic/fitbit)에서 내려받았다. Kaggle의 표기는 원본과 다르므로 원본의 CC BY 4.0을 따른다.
- **조건 준수**: Kaggle의 표기가 원본과 다르므로 원본의 CC BY 4.0을 따른다. 원저자를 이 문서와 `data/README.md`에 밝히고 변경 사항(초 단위 심박의 분 단위 집계, 걸음·수면 결합, 사용자 가명화, 평가용 합성 이상과 정답 라벨 주입)을 명시한다. 가공본에는 주입된 합성 이상이 섞여 있으므로 원자료로 오해하지 않도록 `vitals_minute_clean.parquet`(주입 없음)을 따로 둔다.

**기간·참가자·기기·해상도**

| 항목 | 값 |
|---|---|
| 수집 방식 | Amazon Mechanical Turk 설문 응답자 중 30명이 추적기 자료를 제출 (Zenodo 설명) |
| 내보내기 기간 | 2016-03-12 \~ 2016-05-12 (두 폴더: 3.12\~4.11, 4.12\~5.12) |
| 기기 | 여러 종류의 Fitbit 추적기 (모델은 자료에 표기되지 않음) |
| 우리가 쓴 참가자 | 심박 기록이 있는 15명 (원본 심박 파일의 `Id` 15개) |
| 우리 가공본의 심박 기간 | 2016-03-29 \~ 2016-05-12, 사용자별 일수·커버리지 편차 큼 (한계 절 참고) |
| 원본 해상도 | 심박은 초 단위(표본은 약 5초 간격), 걸음·수면은 1분 단위 |
| 원본 크기 | 심박 3,638,339행 (중복 제거 뒤 3,614,915행), 걸음 2,770,620행, 수면 387,080행 (두 기간 합) |
| 가공본 크기 | 482,507행(원본 그대로) / 482,078행(주입 포함), 심박이 있는 분 471,810개 |

**사용한 원본 파일** (`data/raw/*/*/`, 두 기간 폴더의 같은 이름 파일을 이어 붙인다)

| 파일 | 컬럼 | 타입·의미 | 쓰임 |
|---|---|---|---|
| `heartrate_seconds_merged.csv` | `Id` | 정수, 원본 참가자 ID | 사용자 키. 가명화 대상 |
| | `Time` | 문자열 `m/d/Y h:mm:ss AM/PM`, 기기 로컬 시각 | 분 단위로 내림해 집계 |
| | `Value` | 정수 bpm | 분별 평균·최소·최대·표본 수 |
| `minuteStepsNarrow_merged.csv` | `Id`, `ActivityMinute`, `Steps` | ID, 같은 형식의 분 시각, 분당 걸음 | 걸음이 0보다 큰 분만 사용 (심박이 있는 사용자만) |
| `minuteSleep_merged.csv` | `Id`, `date`, `value`, `logId` | ID, 분 시각(초가 `:30`인 행이 있어 분으로 내림), 수면 상태(1 수면, 2 뒤척임, 3 깨어 있음), 기록 번호 | `value == 1` 인 분만 `asleep=True`. `logId`는 쓰지 않음 |

**전처리 단계** (`app/data/prepare.py`, 시드 고정이라 같은 결과가 나온다)
1. 두 기간 폴더의 같은 이름 CSV를 이어 붙인다.
2. 시각 문자열을 시간대 변환 없이 naive 로컬 시각으로 읽고 분 단위로 내린다.
3. 심박은 `(Id, Time)` 중복을 지우고 분마다 평균·최소·최대·표본 수로 집계한다 (초 단위에서 분 단위로).
4. 걸음은 `(Id, 분)` 중복을 지우고 걸음이 0보다 큰 분만 남긴다. 심박 기록이 없는 사용자의 걸음은 버린다.
5. 수면은 `(Id, 분)` 중복을 지우고 `value == 1` 인 분만 수면으로 표시한다.
6. 심박·걸음을 합치고 수면을 붙인다. 사용자의 심박 기록 구간 밖의 행은 버린다.
7. **희소 저장**: 심박이 있는 분과 걸음이 있는 분만 저장한다. 걸음만 있는 분은 `hr_mean`=NaN, `hr_n`=0이다. 이 분이 센서 결측과 미착용을 구분하는 근거다. 걸음 없는 분의 걸음은 0으로 복원할 수 있다.
8. **주입 라벨**: 평가용 합성 이상 14건을 실제 데이터 위에 덮어쓴다 (추세 2, 안정 시 빈맥 3, 서맥 3, 개인 기준 이탈 3, 센서 결측 3). 사용자는 시드 고정 라운드 로빈으로 정하고 이벤트 사이를 90분 이상 띄우며 한 사용자에 최대 3건이다. 길이·크기는 `prepare.SPECS`에 있다.
9. **가명화**: 정렬한 원본 `Id`를 `U01`\~`U15`로 바꾼다 (`id_map.json`).
10. 산출물 4개를 쓴다: `vitals_minute_clean.parquet`(주입 없음), `vitals_minute.parquet`(주입 포함), `labels.json`, `id_map.json`.

서머타임은 가공 단계에서 고치지 않는다. 탐지기(`app/detection/baseline.py`)가 미국 봄 전환일(3월 둘째 일요일) 02:00\~02:59를 존재하지 않는 시간으로 보고 격자에서 뺀다. 이 가공본의 심박 기록은 3월 29일 이후라 해당 날짜(2016-03-13)가 들어 있지 않다.

**가공 컬럼 사전** (`vitals_minute.parquet`. 분 단위 희소 형식, 행 순서는 `user_id`, `ts`)

| 컬럼 | 타입 | 단위·의미 | 결측 처리 |
|---|---|---|---|
| `user_id` | 문자열 | 가명 사용자 `U01`\~`U15` | 없음 |
| `ts` | datetime64[ns] | 기기 로컬 시각(시간대 없음), 분 단위. 조회 구간은 (시작, 끝] | 없음 |
| `hr_mean` | float64 | 그 분의 심박 평균 (bpm). 원본 범위는 36.8\~202.2 | 심박이 없는 분은 NaN |
| `hr_min` | float64 | 그 분의 심박 최소 (bpm) | 같음 |
| `hr_max` | float64 | 그 분의 심박 최대 (bpm) | 같음 |
| `hr_n` | int32 | 그 분에 들어 있던 초 단위 심박 표본 수 (최대 54) | 심박이 없으면 0 |
| `steps` | int32 | 분당 걸음 수 | 행이 없는 분은 0으로 해석 |
| `asleep` | bool | 수면 기록 상태(`minuteSleep.value == 1`) | 수면 기록이 없으면 False |
| `injected` | 문자열 또는 None | 주입 이벤트 ID(`EVT-01`...). 평가 전용 | 주입이 아니면 None |

`injected`는 정답 라벨이라 앱(`VitalsStore`)이 읽을 때 제거하고 탐지기에도 전달하지 않는다 (라벨 누수 방지). 앱이 읽는 컬럼은 `ts`, `hr_mean`, `hr_min`, `hr_max`, `hr_n`, `steps`, `asleep`이다.

**`labels.json`** (주입 이벤트 14건의 정답): 배열이고 항목은 아래 필드를 가진다.

| 필드 | 의미 |
|---|---|
| `event_id` | `EVT-01`\~. `injected` 컬럼과 같은 값 |
| `user_id` | 가명 사용자 |
| `type` | `resting_tachycardia`, `bradycardia`, `personal_deviation`, `elevated_resting_trend`, `sensor_gap` |
| `start`, `end` | ISO 시각(naive). `end`는 마지막 분을 포함 |
| `severity` | `warning`, `critical`, `info` (기대 심각도) |
| `params` | 주입 상세. `kind`(`awake`/`sleep`/`gap`/`trend`)와 길이(`duration_min`, 추세는 `days`), 목표 심박(`level`) 또는 상승폭(`lift_bpm`), `ramp_min`, 추세의 `n_minutes`·`mean_before`, 결측의 `gap_minutes` |

**`id_map.json`**: 가명 `U01`\~`U15` 대 원본 `Id`. 원본 `Id`는 이미 공개 데이터의 익명 번호다.

### Stanford COVID-19 웨어러블 (외부 검증)

**출처**
- 이름: COVID-19 Wearables. 논문: Mishra, T., Wang, M., Metwally, A.A. et al. Pre-symptomatic detection of COVID-19 from smartwatch data. *Nature Biomedical Engineering* 4, 1208-1220 (2020). [doi:10.1038/s41551-020-00640-6](https://doi.org/10.1038/s41551-020-00640-6).
- 원시 자료: `https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip` (약 378MB). 라벨: 논문 Supplementary Data 1 (`SuppTable3_Fig2a_COVID-19` 시트).
- **라이선스: 표기 없음.** 원본 zip과 논문 Data availability 문구 어디에도 데이터 라이선스가 없다. "공개 다운로드 가능"이지 "재배포 허가"가 아니므로 원시 파일과 분 단위 변환본은 커밋하지 않고(`data/external/`은 git 제외) 이 저장소에는 스크립트와 집계 지표만 둔다.

**기간·참가자·기기·해상도**

| 항목 | 값 |
|---|---|
| 참가자 | 전체 118명 중 우리가 받은 75명: 라벨 참가자 45명 + 건강 대조군 표본 30명 |
| 사건 | 라벨 시트 47건 (COVID-19 32건, 기타 질병 15건). 증상 시작·진단·회복일 |
| 기기·해상도 | 웨어러블(Fitbit 형식) 심박은 불규칙한 초 단위 간격, 걸음은 1분, 수면은 구간(`stage`) 단위 |
| 수면 파일 | 있는 참가자 32명. 없는 참가자는 수면을 알 수 없다 |
| 기간 | 참가자당 심박이 있는 날 30\~200일. **날짜는 개인정보 보호로 이동된 값**이라 실제 연도가 아니다 |
| 변환본 크기 | 분 단위 7,292,870행 |

**사용한 원본 파일** (`data/external/stanford/raw/`, 참가자 ID 7자)

| 파일 | 컬럼 | 쓰임 |
|---|---|---|
| `{ID}_hr.csv` (`{ID}_1_hr.csv`처럼 조각난 경우 이어 붙임) | `user`, `datetime`, `heartrate` | 0 bpm은 측정 없음으로 버리고 분 단위로 집계 |
| `{ID}_steps.csv` | `user`, `datetime`, `steps` | 분 단위. 같은 분의 중복은 합산, 0보다 큰 분만 사용 |
| `{ID}_sleep.csv` (32명만) | `user`, `datetime`, `stage_duration`(초), `stage`(`wake`/`light`/`deep`/`rem`) | `wake`가 아닌 구간을 분 단위로 펼쳐 수면으로 표시 |
| `SuppData1.xlsx` 의 `SuppTable3_Fig2a_COVID-19` | `ParticipantID`, `Category`, `Symptom_dates`, `covid_diagnosis_dates`, `recovery_dates` | 사건 라벨. 날짜 셀이 `[Timestamp('...')]` 문자열 목록이라 파싱 |

**전처리 단계** (`eval/external/stanford_download.py`, `stanford_adapter.py`)
1. 다운로드: HTTP Range로 zip의 중앙 디렉터리와 필요한 멤버만 받는다 (전체 378MB를 받지 않는다). 이미 받은 파일은 건너뛴다.
2. 심박은 분으로 내려 평균·최소·최대·표본 수를 구한다. 조각난 파일은 이어 붙이고 같은 분은 한 번만 쓴다.
3. 걸음은 심박 기록 구간 안의 분만 남긴다. 수면은 `wake`가 아닌 구간을 `ceil(stage_duration / 60)`분으로 펼친다.
4. 같은 희소 규칙으로 저장한다 (심박이 있는 분과 걸음이 있는 분만).
5. 참가자 ID를 `S001`\~ 로 가명화하고 `id_map.json`에 원본 ID를 둔다.
6. 라벨 시트는 같은 참가자의 사건 여러 개를 쌓고 라벨이 없는 참가자는 `control`(건강 대조군)로 분류한다.
7. 시각은 이동된 값 그대로 쓰고 시간대 변환은 하지 않는다. 라벨도 같은 이동을 쓰므로 반드시 이 zip과 짝지어 쓴다.

**변환 컬럼 사전** (`data/external/stanford/vitals_minute.parquet`): FitBit 가공본과 같은 컬럼에서 `injected`가 빠지고 `source`가 붙는다.

| 컬럼 | 타입 | 단위·의미 | 결측 처리 |
|---|---|---|---|
| `user_id` | 문자열 | 가명 참가자 `S001`\~ | 없음 |
| `ts`, `hr_mean`, `hr_min`, `hr_max`, `hr_n`, `steps` | FitBit 가공본과 같음 | 같음 | 같음 |
| `asleep` | nullable boolean | 수면 구간이면 True, 수면 파일이 있는데 구간 밖이면 False | 수면 파일이 없는 참가자는 NA (행의 약 53%). 탐지기는 NA를 깨어 있음으로 처리해 수면 맥락 규칙이 사실상 꺼진다 |
| `source` | 문자열 | 항상 `stanford` | 없음 |

**사건 라벨** (앱 표시용 `events.json`, 평가용 `participants.json`)

| 파일 | 필드 | 의미 |
|---|---|---|
| `events.json` | `user_id`, `type`(`symptom_onset`/`diagnosis`/`recovery`), `date`(ISO), `illness`(`covid`/`other`) | 기록 구간 안의 사건만 한 줄씩 펼친 것. 연구 라벨이며 진단이 아니다 |
| `participants.json` | `group`(`labeled`/`control`), `start`, `end`, `days`, `has_sleep`, `events`(`kind`, `symptom`, `diagnosis`, `recovery` 날짜 목록) | 평가 하니스가 읽는 참가자 요약 |

**앱과의 관계**: `VitalsStore`는 `data/external/stanford/vitals_minute.parquet`가 있을 때만 이 데이터를 불러와 별도 데이터셋(`S001`\~)으로 둔다. 없으면 조용히 건너뛰므로 CI와 데모에는 나타나지 않는다. 이 데이터로는 탐지기를 튜닝하지 않는다 (`eval/external/stanford_eval.py`).

```bash
.venv/bin/python -m eval.external.stanford_download [--controls 30]   # 필요한 파일만 받기 (라이선스 표기 없음, 커밋 금지)
.venv/bin/python -m eval.external.stanford_adapter                    # 분 단위 변환 + 사건 라벨
.venv/bin/python -m eval.external.stanford_eval                       # 동결한 탐지기를 하루씩 재생해 평가
```

## 데이터와 라이선스

- **출처와 라이선스 전체 목록은 [NOTICE.md](../NOTICE.md)에 있다** (FitBit, Stanford, 가이드라인 문서, 모델, 연구 방법, 로고).
- **데이터**: 출처, 라이선스, 컬럼 사전, 전처리는 위 [데이터](#데이터) 절에 있다. FitBit은 원본 Zenodo(Furberg 외, CC BY 4.0)를 Kaggle [`arashnic/fitbit`](https://www.kaggle.com/datasets/arashnic/fitbit)에서 받은 것으로, 원본의 CC BY 4.0에 맞춰 원저자와 변경 사항을 밝히고 가공본을 저장소에 포함한다. Stanford COVID-19 웨어러블은 라이선스 표기가 없어 저장소에 넣지 않는다.
  평가용 이상은 FitBit 실데이터 위에 합성 주입한 것이다. 원본 사용자 ID는 `U01`..`U15`로 가명화해 노출한다 (매핑은 `data/processed/id_map.json`).
- **코드 라이선스**: [MIT](../LICENSE) (Copyright (c) 2026 kimddong23). 소스 코드와 문서에 적용된다. 데이터, 모델 같은 제3자 자료는 각자의 라이선스를 그대로 따른다.
- **가공 데이터(`data/processed/`)**: CC BY 4.0 데이터셋의 2차 저작물이라 MIT가 아닌 CC BY 4.0의 조건(원저자 표시, 변경 사항 표시)을 유지한다. 표시는 [NOTICE.md](../NOTICE.md)와 `data/README.md`에 있다.

## 감사의 말

- 연구 참고: Mishra et al., 2020 (*Nat. Biomed. Eng.*, 웨어러블 안정시 심박 기반 조기 이상 탐지). 공개 코드 `gireeshkbogu/AnomalyDetect`(MIT)의 RHR-AD·HROS-AD 계열(시간 단위 EllipticEnvelope)을 `app/detection/research.py`에 선택적으로 다시 구현했다. 논문의 RHR-Diff는 구현하지 않았다.
  Alavi et al., 2022 (*Nat. Med.*, 웨어러블 실시간 알림; CuSum 계열이라고 알려져 있으나 공개 소스를 찾지 못해 표준 단측 CuSum을 직접 구현했고 논문 파라미터를 재현한 것이 아니다).
- 데이터: FitBit 데이터 제공자(Furberg 외), Kaggle 재배포자(`arashnic`), Stanford COVID-19 웨어러블 연구진(Mishra 외).

## 면책

Vital Sentinel은 **의료기기가 아니며** 질병을 진단·치료·예방하지 않는다. 출력은 웨어러블 심박 데이터에 대한 참고용 분석이다.
탐지 임계값은 일반 가이드라인 수준의 보수적 기준이고 임상 검증을 거치지 않았으며 평가 라벨은 합성이다. 증상이 있거나 걱정되면 의료진과 상담하고 응급 증상이 있으면 즉시 119에 연락한다.
