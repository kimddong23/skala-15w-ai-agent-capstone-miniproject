# 가이드라인 검색 Before/After: 키워드 섹션 매칭 vs RAG

- 질문 45개 (구어체, 오타, 영어 혼용 포함), 코퍼스 19개 문서 / 52개 청크, 임베딩 bge-m3:latest (dim 1024)
- 적중 = 상위 k 개 문서 안에 정답 문서가 하나라도 있음. MRR = 첫 정답 순위의 역수(10위 밖 0)
- **측정**: 2026-10-20 22:29, 현재 코드·코퍼스. 직전 실행(22:20)은 RAG 질의 p50(20.67 ms)이 그 안에 들어 있는 질의 임베딩 단독 p50(85.97 ms)보다 작게 나와 앞뒤가 맞지 않아 버리고 다시 쟀다(정확도는 두 번 같다). 로컬 Ollama를 다른 모델과 같이 쓰는 상태라 지연은 실행마다 흔들린다.
- **한계**: 질문과 코퍼스를 같은 작성자가 만들었다(정답을 아는 상태에서 질문을 씀) - 절대 수치는 낙관적이며, 방식 간 상대 비교를 보는 용도다.

## 1. 정확도 (전체 질문)

| 방식 | 후보 문서 수 | hit@1 | hit@3 | MRR | 지연 p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|
| A 기존 키워드(2문서) | 2 | 0.111 | 0.156 | 0.133 | 0.14 / 0.42 |
| B 키워드(확장 코퍼스) | 19 | 0.467 | 0.756 | 0.63 | 2.67 / 4.29 |
| C RAG(bge-m3) | 19 | 0.867 | 0.956 | 0.908 | 53.75 / 69.51 |
| D RAG 폴백(키워드) | 19 | 0.444 | 0.756 | 0.62 | 0.15 / 0.24 |

- hit@k = 상위 k 개 문서 안에 정답 문서가 *하나라도* 있는 질문의 비율 (정답이 여러 문서일 수 있어 recall@k 가 아님).
- **A 의 hit@3 은 비교 지표가 아니다**: 후보 문서가 2개뿐이라 정답이 그 안에 있는 질문은 순위와 무관하게 상위 3개에 들어간다. A 는 hit@1 과 MRR 로 보라.
- 키워드 방식은 질문과 겹치는 단서가 전혀 없으면(이번 질문 중 1개) 모든 점수가 0 이고 순위는 문서 id 순이다 - 관련성이 아니라 결정론적 임의 순서다.

## 2. 기존 코퍼스로 답할 수 있는 질문만 (7개)

A 는 정답 문서가 원 코퍼스(2문서)에 없는 질문에서 구조적으로 못 맞춘다. 이 표는 같은 범위의 질문만 비교해 '검색 방식'의 차이를 본다.

| 방식 | hit@1 | hit@3 | MRR |
|---|---:|---:|---:|
| A 기존 키워드(2문서) | 0.714 | 1.0 | 0.857 |
| B 키워드(확장 코퍼스) | 0.714 | 1.0 | 0.857 |
| C RAG(bge-m3) | 1.0 | 1.0 | 1.0 |
| D RAG 폴백(키워드) | 0.571 | 1.0 | 0.786 |

## 3. 비용

| 항목 | 값 |
|---|---|
| 인덱스 크기 | 217,126 bytes (52청크 x 1024차원 float32 + 해시) |
| 콜드 구축 | 3.147 s (임베딩 52건) |
| 웜 재구축(캐시) | 0.003 s (임베딩 0건) |
| 질의 임베딩 p50 / p95 | 51.99 / 60.32 ms |
| 첫 임베딩 호출(모델 로드 포함) | 1421.7 ms |
| 키워드 방식 지연(p50) | 2.67 ms (기존 함수 반복 호출) / 0.15 ms (RAG 폴백 구현), 임베딩·외부 의존 없음 |
| RAG 질의 지연(p50) | 53.75 ms (질의 임베딩 포함) |

## 4. 장애 시 동작 (Ollama 접속 불가)

- 접속 불가 임베더로 검색: method=`keyword`, 첫 질의 6.2 ms (연결 거부를 기다린 시간), 이후 질의 0.2 ms (실패 후 60초 동안 재시도하지 않아 대기 없음). 사유: 임베딩 사용 불가(ConnectError: [Errno 61] Connection refused) - 키워드 검색으로 대체했습니다
- 폴백(키워드)의 정확도는 위 표 D: hit@1 0.444, hit@3 0.756.

## 5. 실패 사례

### RAG(C) hit@3 실패 2건

- Q08 「쿨다운은 왜 하는 거야」 정답 ['hr-recovery'] / 상위3 ['dehydration-heat-fever', 'sensor-gaps-and-wear', 'stress-and-sleep-deprivation']
- Q44 「심박 센써가 계속 끊겨여」 정답 ['sensor-gaps-and-wear'] / 상위3 ['heart-rate-ranges', 'when-to-see-doctor', 'stress-and-sleep-deprivation']

### 키워드(B) hit@3 실패 11건

- Q08 「쿨다운은 왜 하는 거야」 정답 ['hr-recovery'] / 상위3 ['bradycardia-overview', 'dehydration-heat-fever', 'emergency-response']
- Q13 「빈맥 원인이 뭔가요」 정답 ['tachycardia-overview'] / 상위3 ['heart-rate-ranges', 'hrv-and-trends', 'red-flags']
- Q15 「스마트워치 심박 측정은 얼마나 정확해?」 정답 ['wearable-measurement-limits'] / 상위3 ['hr-recovery', 'hrv-and-trends', 'resting-hr-basics']
- Q17 「달릴 때 워치 심박이 이상하게 튀어요」 정답 ['wearable-measurement-limits'] / 상위3 ['heart-rate-ranges', 'hr-recovery', 'red-flags']
- Q21 「어제 술 마셨더니 안정시 심박이 높네요」 정답 ['stimulants-caffeine-alcohol'] / 상위3 ['dehydration-heat-fever', 'heart-rate-ranges', 'red-flags']
- Q23 「잠을 못 잤더니 심박이 높아요」 정답 ['stress-and-sleep-deprivation', 'sleep-hr'] / 상위3 ['dehydration-heat-fever', 'heart-rate-ranges', 'hr-recovery']
- Q28 「언제 병원에 가야 하나요? 심박이 계속 높을 때」 정답 ['when-to-see-doctor'] / 상위3 ['dehydration-heat-fever', 'service-detection-rules', 'emergency-response']
- Q29 「병원 가기 전에 뭘 준비해서 가면 돼요」 정답 ['when-to-see-doctor'] / 상위3 ['emergency-response', 'medications-and-hr', 'bradycardia-overview']
- Q35 「운동 중엔 왜 알림이 안 와요」 정답 ['service-detection-rules', 'hr-recovery'] / 상위3 ['emergency-response', 'exercise-hr-zones', 'heart-rate-ranges']
- Q42 「자다가 심박이 30대로 나왔는데 괜찮을까요」 정답 ['sleep-hr', 'bradycardia-overview'] / 상위3 ['heart-rate-ranges', 'hr-recovery', 'red-flags']
- Q44 「심박 센써가 계속 끊겨여」 정답 ['sensor-gaps-and-wear'] / 상위3 ['dehydration-heat-fever', 'hr-recovery', 'bradycardia-overview']

## 6. 트레이드오프와 한계

- **정확도**: 임베딩 검색은 조사·어미·동의어·오타에 강하다(표 1, 5). 키워드는 질문 어절이 문서에 그대로 있어야 맞춘다.
- **지연**: 질의마다 임베딩 호출(로컬 Ollama)이 들어가 키워드(2.67 ms)보다 약 20배 느리다(53.75 ms). 그래도 LLM 생성(수 초)에 비하면 작다. 서버가 응답하지 않고 멈춘 경우는 연결 거부와 달리 질의 임베딩 타임아웃(EMBED_TIMEOUT_S, 기본 3초)만큼 기다린 뒤 키워드로 내려간다(표 4는 즉시 거부된 경우).
- **A 와 B 의 차이는 검색 방식이 아니라 코퍼스 확장 효과**다: 같은 함수를 2문서와 19문서에 적용했을 뿐이다. 검색 방식의 효과는 표 2(같은 범위 질문)와 B-C 비교로 본다.
- **의존성**: Ollama + bge-m3(1.2 GB) 가 필요하다. 꺼져 있으면 키워드 폴백으로 내려가며 정확도가 떨어진다(표 D). 폴백은 서비스 중단을 막는 용도다.
- **비용 구조**: 인덱스는 코퍼스 변경 시 바뀐 청크만 다시 임베딩하고(웜 0건), 질의 임베딩은 캐시하지 않는다.
- **평가의 한계**: 질문 45개는 작고, 작성자 편향이 있으며, 정답이 여러 문서에 걸치는 질문은 복수 정답으로 관대하게 채점했다. 검색 결과가 실제 답변 품질(환각 감소)로 이어지는지는 이 평가가 재지 않는다.
- **코퍼스**: 출처는 '자체 작성한 일반 상식 요약(특정 기관 문서를 인용·대조하지 않음)'이며 임상 검토를 받지 않았다. 일부 수치는 대략적인 값이고 검증하지 않았다(각 문서 끝 `문서 정보` 참고).