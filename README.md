# Vital Sentinel

**웨어러블 심박 데이터에서 이상 징후를 찾아 근거와 함께 설명하는 에이전트.** 탐지는 결정론적 코드가 하고 해석은 로컬 LLM이 하며 위험 안내는 코드가 강제한다.

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/code-MIT-green)
![Data](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)
![Tests](https://img.shields.io/badge/tests-pytest%201408%20%2B%20vitest%2060%20%2B%20e2e%20160-blue)

![Vital Sentinel 콘솔](frontend/docs/screenshots/desktop-light.png)

## 무엇인가

손목 밴드가 기록한 분 단위 심박·걸음·수면 데이터를 규칙으로 점검하고 LangGraph 에이전트가 결과를 한국어로 설명하는 서비스다. 채팅창에 "U05 최근 30일 심박 이상 점검해줘"처럼 물으면 에이전트가 데이터를 직접 조회해 이상 구간을 찾고 가이드라인 근거와 함께 설명한다. 데이터는 가명 처리한 공개 FitBit 데이터이고 LLM은 기본적으로 내 컴퓨터(Ollama)에서 돈다.

## 무엇을 해 주나

- **이상 탐지**: 안정 시 빈맥, 서맥, 개인 기준 이탈, 안정시 심박 상승 추세, 센서 결측 5가지를 LLM 없이 규칙과 통계로 찾는다. 운동 중 상승은 이상으로 치지 않는다.
- **근거 있는 설명**: 인용한 수치는 도구 결과에서만 가져오고 검증기가 대조한다. 어긋나면 다시 쓰거나 템플릿 리포트로 대체한다.
- **위험 안내는 코드가 강제**: critical 이상이면 LLM이 무엇을 쓰든 119 안내문이 리포트 맨 앞에 붙는다. 보호자 알림은 사람이 승인해야 발송된다(시뮬레이션).
- **대화와 비교**: 후속 질문, 이름·나이 기억, 최대 3명 비교를 지원한다.
- **실시간 화면**: 라우팅, 계획, 조회, 검증 과정을 단계별로 보여 주는 웹 콘솔.
- **LLM 교체와 장애 내성**: `fake`(LLM 없음), `ollama`(로컬), `openai`(과금)를 바꿔 꽂고 LLM이 죽어도 끝까지 동작한다.

## 누구를 위한가

- **사용자**: 스마트워치로 심박을 기록하면서 스스로 알아채지 못한 이상 신호를 미리 알고 싶은 사람과, 위험 수치가 나왔을 때 알림을 받는 보호자.
- **개발자·학습자**: 건강 데이터처럼 틀리면 위험한 영역에서 LLM 에이전트(LangGraph, MCP, RAG, SSE)를 안전하게 만드는 방법을 참고하려는 사람.

> **의료기기가 아니다.** 질병을 진단·치료·예방하지 않고 임계값은 임상 검증을 거치지 않았다. 증상이 있거나 걱정되면 의료진과 상담하고 응급 증상이 있으면 즉시 119에 연락한다.

## 빠른 시작

필요한 것: Git, Python 3.11, Node.js 22 이상. 로컬 LLM을 쓰려면 Ollama, Docker로 띄우려면 Docker를 더한다. Ollama 없이도 `fake` 모드로 바로 돌아간다.

```bash
git clone https://github.com/kimddong23/skala-15w-ai-agent-capstone-miniproject.git
cd skala-15w-ai-agent-capstone-miniproject
bash sh/setup_venv.sh          # 가상환경과 의존성 (최초 1회)
bash sh/run_agent_fake.sh      # 백엔드 (LLM 없이 규칙 기반 리포트)
bash sh/run_frontend.sh        # 다른 터미널에서: http://localhost:8600
```

Docker로 한 번에: `docker compose up -d --build` 후 `http://localhost:8600`.
Windows, 로컬 Ollama, 문제 해결은 [시작하기](docs/getting-started.md)에 있다.

## 동작 방식

![시스템 구성도](docs/architecture/system-architecture.png)

- 화면은 SSE로 에이전트와 통신하고 에이전트는 MCP 서버 2개(`vitals`, `guidelines`)로 데이터와 가이드라인을 조회한다.
- 이상 탐지는 코드가, 해석과 서술은 LLM이 맡고 검증기와 안전 게이트가 그 사이를 지킨다.
- 구조 상세, 노드 그래프, 유스케이스별 흐름은 [아키텍처](docs/architecture.md)에 있다.

## 핵심 결과

모든 수치는 단일 실행이고 `eval/results/`의 파일에서 옮겼다. 조건과 한계는 [평가와 한계](docs/evaluation.md)에 있다.

- **안전**: critical 질의에서 119 안내문이 붙은 비율은 개발 4/4, 홀드아웃 3/3이고 비-critical 오탐은 0건이다. 안내문은 코드가 넣는다.
- **일반화**: 코드를 보지 않고 만든 블라인드 홀드아웃 16건의 첫 실행은 14/16(88%, 95% 구간 64\~96%, 측정 당시 fast `qwen2.5:3b`)이다. 수정 뒤 재실행은 블라인드가 아니다.
- **v1 대 v2**(40건): 진단·복약 단정 2건에서 0건, 질문 분류 정확도 0.526에서 1.0. 대가로 지연 p50이 8.04초에서 12.56초, LLM 호출이 1.0회에서 3.08회로 늘었다.
- **탐지의 약점**: 개인 기준 이탈 재현율은 0.77(13/17)로 목표 0.90에 못 미치고 표본이 작아 유의하다고 말할 수 없다. 정상 데이터의 알림은 사용자·일당 0.0938이다.
- **외부 데이터**: Stanford COVID-19 웨어러블에서 사건 27/32(84%)를 잡았지만 우연 기대가 65%라 약한 신호다. 질병 조기 탐지용이 아니다.
- **테스트**: pytest 1408, vitest 60, Playwright 160 통과 (2026-10-20).

## 문서

- [시작하기와 설정](docs/getting-started.md): 사전 준비, 운영체제별 실행, 스크립트, 문제 해결, Docker, 환경변수
- [사용자 가이드](docs/manual_ollma.md): 켜 둔 뒤 무엇을 물어볼 수 있나 (화면 캡처 포함)
- [회귀 테스트 쿼리](docs/regression_test_queries.md): 손으로 쳐 보는 질문 목록
- [아키텍처](docs/architecture.md): 구성도, 에이전트 그래프, 유스케이스별 흐름, 디렉터리 구조 ([인터랙티브 다이어그램](docs/architecture/system-architecture.html))
- [데이터](docs/data.md): FitBit과 Stanford 데이터, 전처리, 컬럼 사전
- [평가와 한계](docs/evaluation.md): KPI, Before/After, 모델 벤치마크, 한계와 로드맵
- [API](docs/api.md): HTTP 엔드포인트와 SSE 이벤트
- [개발과 품질](docs/development.md): 테스트, 보안, 품질 보증, 설계 결정
- [NOTICE](NOTICE.md): 제3자 자료의 출처와 라이선스
- [data/README](data/README.md), [frontend/README](frontend/README.md)

## 데이터와 라이선스

코드는 [MIT](LICENSE)다. 가공 데이터(`data/processed/`)는 CC BY 4.0인 FitBit 크라우드소싱 데이터(Furberg 외, 2016)의 2차 저작물이라 원저자 표시와 변경 표시를 유지한다. Stanford COVID-19 웨어러블은 라이선스 표기가 없어 저장소에 넣지 않고 직접 내려받는 스크립트만 둔다. 자세한 출처와 조건은 [NOTICE](NOTICE.md)와 [데이터](docs/data.md)에 있다.
