# NOTICE: 제3자 자료의 출처와 라이선스

Vital Sentinel은 아래 자료를 쓰거나 참고했다. 이 문서는 어떤 자료가 저장소에 들어 있고 어떤 자료는 들어 있지 않은지, 각 자료를 어떤 조건으로 쓰는지를 밝힌다. 마지막 확인일은 2026-10-20이다.

**이 저장소의 소스 코드와 문서는 MIT 라이선스([`LICENSE`](LICENSE), Copyright (c) 2026 kimddong23)를 따른다. 아래 제3자 자료는 MIT의 적용을 받지 않고 각자의 라이선스를 유지한다.** 특히 `data/processed/`는 CC BY 4.0 데이터셋의 2차 저작물이므로 CC BY 4.0의 저작자 표시와 변경 표시 조건이 계속 적용된다.

| 자료 | 저장소에 포함 | 라이선스 | 절 |
|---|---|---|---|
| FitBit 심박·걸음·수면 (Furberg 외, 2016) | 가공본 포함 | CC BY 4.0 | [1](#1-fitbit-크라우드소싱-데이터) |
| Stanford COVID-19 웨어러블 (Mishra 외, 2020) | **포함하지 않음** | 표기 없음 | [2](#2-stanford-covid-19-웨어러블-데이터) |
| 건강 가이드라인 문서 19개 | 포함 (직접 쓴 일반 상식 요약, 검증·의료 검토 없음) | 이 저장소의 자체 저작물 | [3](#3-건강-가이드라인-문서) |
| Qwen3, bge-m3, Ollama | 포함하지 않음 (실행 때 내려받음) | Apache-2.0, MIT, MIT | [4](#4-모델과-실행-도구) |
| 연구 방법 (Mishra 2020, Alavi 2022) | 직접 다시 구현한 코드 | 논문·공개 코드 | [5](#5-연구-방법) |
| 기술 로고 | 구성도 그림 안 | 각 소유자의 상표 | [6](#6-로고와-상표) |

## 1. FitBit 크라우드소싱 데이터

**원본 인용**

> Furberg, R., Brinton, J., Keating, M., & Ortiz, A. (2016). *Crowd-sourced Fitbit datasets 03.12.2016-05.12.2016* [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.53894

- 라이선스: Creative Commons Attribution 4.0 International (CC BY 4.0). Zenodo 기록 페이지에서 확인했다.
- 내용: Amazon Mechanical Turk 설문으로 모은 Fitbit 사용자 30명의 분 단위 활동·심박·수면 기록이다. 이 프로젝트는 심박이 있는 15명을 쓴다 (기록 기간 2016-03-29 \~ 2016-05-12).
- 받은 곳: Kaggle 재배포본 [`arashnic/fitbit`](https://www.kaggle.com/datasets/arashnic/fitbit). Kaggle의 표기는 원본과 다르므로 원본의 CC BY 4.0을 따른다. 원저자를 밝히고 변경 사항을 적는다.

**변경 사항 (CC BY 4.0 Section 3(a)(1)에 따라 밝힌다)**

1. 초 단위 심박을 분 단위로 집계했다 (평균, 최소, 최대, 표본 수).
2. 걸음·수면 기록을 결합하고 중복 행을 지웠다.
3. 사용자 ID를 `U01`\~`U15`로 가명화했다. 매핑은 `data/processed/id_map.json`에 있다.
4. 평가용 **합성 이상**을 실제 데이터 위에 주입하고 정답 라벨(`labels.json`)을 만들었다.

`data/processed/`는 이 변경을 거친 **2차 저작물**이다. `vitals_minute.parquet`에는 우리가 주입한 합성 이상이 섞여 있으므로 원자료나 임상 기록으로 쓰면 안 된다. 주입이 없는 `vitals_minute_clean.parquet`을 함께 둔다. 저장소 루트의 MIT 라이선스는 이 데이터에 적용되지 않는다. 이 데이터를 다시 쓰거나 재배포하는 사람도 위 원저자 표시와 변경 사항 표시를 유지해야 한다. 원저자와 Zenodo 기록은 이 변경을 승인하거나 보증하지 않는다. 원본 CSV(`data/raw/`)는 저장소에 넣지 않는다.

## 2. Stanford COVID-19 웨어러블 데이터

**인용**

> Mishra, T., Wang, M., Metwally, A.A., Bogu, G.K., Brooks, A.W., Bahmani, A., Alavi, A., et al. (2020). Pre-symptomatic detection of COVID-19 from smartwatch data. *Nature Biomedical Engineering* 4, 1208–1220. https://doi.org/10.1038/s41551-020-00640-6

- 공개 다운로드: <https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip> (논문의 Data availability 문구에 적힌 주소다). 사건 라벨은 논문 Supplementary Data 1(Springer 보충 자료)이다.
- 원본 zip과 논문의 Data availability 문구에는 **데이터 라이선스 표기가 없다.** 내려받을 수 있다는 사실은 재배포를 허락한다는 뜻이 아니므로 **이 저장소는 원시 파일도 변환본도 재배포하지 않는다** (`data/external/`은 git에서 제외).
- 직접 받는 방법: `.venv/bin/python -m eval.external.stanford_download` (필요한 참여자 파일만 내려받는다), 이어서 `.venv/bin/python -m eval.external.stanford_adapter`. 이용 조건은 내려받는 사람이 원출처에서 확인한다.
- 이 저장소가 공개하는 것은 스크립트(`eval/external/`)와 **집계 지표**(`eval/results/external_stanford.md`)뿐이다.
- 원자료의 날짜는 개인정보 보호를 위해 출처가 이동(shift)시킨 값이다. 실제 달력 날짜가 아니고 사건 라벨도 같은 이동을 쓴다. 이 데이터는 탐지기 튜닝에 쓰지 않았고 동결한 탐지기를 한 번 적용하는 외부 검증에만 썼다.

## 3. 건강 가이드라인 문서

`app/mcp_server/guidelines/*.md` 19개 문서(약 4,500단어)는 이 프로젝트에서 **직접 쓴 한국어 요약문**이다. 일반 상식 수준의 내용을 기억에 의존해 정리했고 특정 기관의 문서를 보고 쓰거나 번역하거나 옮겨 적지 않았다. 어떤 기관의 자료와도 대조해 검증하지 않았다.

- 일부 수치는 대략적인 값이고 검증하지 않았다.
- 이 문서들은 **의학적 조언이 아니며** 의료진이나 임상 전문가의 검토를 받지 않았다.
- 이 서비스의 임계값(예: 안정 시 빈맥 100 bpm 초과, critical 130 bpm 이상)을 설명하는 문서(`service-detection-rules`, `heart-rate-ranges`, `red-flags`, `disclaimer`)는 이 서비스의 코드와 안내 문구를 풀어 쓴 것이다.
- 어떤 기관과도 제휴하거나 승인받지 않았다. 의학적 근거가 필요하면 해당 분야의 공식 자료와 의료진을 직접 확인해야 한다.

## 4. 모델과 실행 도구

모델과 도구는 저장소에 들어 있지 않고 실행할 때 사용자가 내려받는다 (`bash sh/setup_ollama.sh`).

| 이름 | 용도 | 제공 | 라이선스 | 확인한 곳 |
|---|---|---|---|---|
| Qwen3 `qwen3:0.6b`, `qwen3:8b` | fast, quality 언어 모델 | Qwen (Alibaba Cloud) | Apache-2.0 | Hugging Face 모델 카드, Ollama 라이브러리 |
| BGE-M3 `bge-m3` | 가이드라인 의미 검색 임베딩 | BAAI | MIT | Hugging Face 모델 카드 |
| Ollama | 로컬 LLM 실행 서버 | Ollama | MIT | GitHub `ollama/ollama` LICENSE |

선택 사항인 OpenAI 백엔드(`LLM_BACKEND=openai`)와 LangSmith 추적은 각 서비스의 이용약관을 따른다. 기본값은 둘 다 꺼짐이다.

이 밖의 소프트웨어 의존성은 `requirements.txt`와 `frontend/package.json`에 있고 확인한 주요 패키지는 MIT, BSD-3-Clause, Apache-2.0 계열이다. 화면 글꼴 IBM Plex Mono와 IBM Plex Sans KR(`@fontsource`, SIL OFL 1.1), 아이콘 `react-bootstrap-icons`(MIT)를 쓴다. 구성도 HTML 파일(`docs/architecture/`)은 Archify(MIT)로 만들었고 글꼴 JetBrains Mono(SIL OFL 1.1)를 포함하며 그 라이선스 전문이 파일 안에 들어 있다.

## 5. 연구 방법

`app/detection/research.py`는 아래 연구의 방법을 **이 프로젝트에서 다시 구현한 코드**다. 원본 코드를 복사하지 않았으므로 우리 코드는 이 저장소의 MIT 라이선스를 따른다. 원저자와 이 프로젝트는 아무 관계가 없고 원저자의 승인을 받지 않았다.

- **Mishra 외 (2020)** (위 2절의 인용). 공개 코드 [`gireeshkbogu/AnomalyDetect`](https://github.com/gireeshkbogu/AnomalyDetect)(MIT 라이선스)의 RHR-AD, HROS-AD 접근을 읽고 알고리즘과 파라미터(안정시 심박 정의, 1시간 평균, 표준화 뒤 EllipticEnvelope)를 가져와 새로 작성했다. 의도적으로 바꾼 점은 코드 머리말에 적었다. 논문의 RHR-Diff 방법은 구현하지 않았다.
- **Alavi 외 (2022)**. Alavi, A., Bogu, G.K., Wang, M. et al. Real-time alerting system for COVID-19 and other stress events using wearable data. *Nature Medicine* (2022, 온라인 공개 2021-11-29). https://doi.org/10.1038/s41591-021-01593-2. 이 논문의 실시간 알림이 CuSum 계열을 쓴다는 점만 참고했다. CuSum은 Page(1954)의 표준 단측 정의로 직접 구현했고 논문의 파라미터를 재현한 것이 아니다.

탐지 결과는 이 연구들의 성능을 재현한 것이 아니며 임상 성능을 뜻하지 않는다.

## 6. 로고와 상표

`docs/architecture/`의 구성도에 들어 있는 기술 로고(React, FastAPI, Ollama, Docker, Python, SQLite, OpenAI, nginx, Kaggle, Playwright, pytest, pandas, Zod, LangChain 계열 등)와 README의 배지는 각 소유자의 상표나 저작물이다. 어떤 기술을 쓰는지 알리는 용도로만 썼고 소유자와의 제휴나 보증을 뜻하지 않는다. 일부 로고 도형은 Simple Icons(CC0 1.0)의 SVG로 바꿔 넣었다.

## 7. 의료기기가 아니다

Vital Sentinel은 **의료기기가 아니며** 질병을 진단, 치료, 예방하지 않는다. 출력은 웨어러블 심박 데이터에 대한 참고용 분석이고 임상 검증을 거치지 않았다. 증상이 있거나 걱정되면 의료진과 상담하고 응급 증상이 있으면 즉시 119에 연락한다.
