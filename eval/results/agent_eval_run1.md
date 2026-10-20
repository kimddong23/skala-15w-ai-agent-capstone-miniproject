# 에이전트 종단 평가 (langsmith)

- 실행: 2026-10-20T03:10:32, 예제 20건, LLM ON
- 모델: fast=qwen2.5:3b, quality=qwen3:8b
- LangSmith 실험(비공개)

## 지표 (평균)

| 분류 | n | route_accuracy | expected_types_recall | escalation_correct | validator_passed | validator_first_try | no_diagnosis_assertion | latency_s |
|---|---|---|---|---|---|---|---|---|
| 전체 | 20 | 0.9 | 1.0 | 0.95 | 0.95 | 0.95 | 1.0 | 17.65 |
| analysis | 11 | 0.909 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 23.9 |
| followup | 3 | 0.667 | 1.0 | 0.667 | 0.667 | 0.667 | 1.0 | 16.667 |
| simple | 6 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 6.683 |

## 예제별

| # | 분류 | 질의(마지막 턴) | route | 기대유형 | 탐지유형 | 에스컬 | 검증 | 지연(s) | 실패 지표 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | analysis | U07 최근 한 달 심박 감지된 이상 있어? | analysis | - | sensor_gap | 0/0 | 통과(시도 1) | 34.4 | - |
| 1 | analysis | U01 최근 7일 심박 점검해줘 | analysis | - | - | 0/0 | 통과(시도 1) | 21.4 | - |
| 2 | followup | 방금 결과를 한 줄로 요약해줘 | analysis | - | - | 0/1 | 실패/폴백(시도 2) | 23.7 | route_accuracy, escalation_correct, validator_passed, validator_first_try |
| 3 | simple | 안녕하세요 | simple | - | - | 0/0 | 통과(시도 0) | 2.8 | - |
| 4 | simple | 운동 중에 심박이 올라가는 건 이상한 건가요? | simple | - | - | 0/0 | 통과(시도 0) | 2.2 | - |
| 5 | analysis | U06 최근 30일 이상 징후 분석해줘 | analysis | personal_deviation,sensor_gap | personal_deviation,sensor_gap | 0/0 | 통과(시도 1) | 29.0 | - |
| 6 | analysis | U13 최근 30일 심박 괜찮은지 봐줘 | analysis | elevated_resting_trend,sensor_gap | elevated_resting_trend,resting_tachycardia,sensor_gap | 0/0 | 통과(시도 1) | 27.6 | - |
| 7 | analysis | U05 최근 30일 심박 이상 점검해줘 | analysis | resting_tachycardia | resting_tachycardia,sensor_gap | 1/1 | 통과(시도 1) | 24.1 | - |
| 8 | followup | 그럼 운동 중엔 괜찮은 거야? | simple | - | - | 1/1 | 통과(시도 0) | 10.5 | - |
| 9 | analysis | U05 최근 일주일 심박 이상 점검해줘 | analysis | resting_tachycardia | resting_tachycardia | 0/0 | 통과(시도 1) | 26.4 | - |
| 10 | analysis | U09 최근 2주 이상 있는지 확인해줘 | analysis | - | sensor_gap | 0/0 | 통과(시도 1) | 22.4 | - |
| 11 | simple | 이 서비스는 어떤 데이터를 쓰나요? | simple | - | - | 0/0 | 통과(시도 0) | 10.2 | - |
| 12 | analysis | U12 최근 4주 심박 점검해줘 | analysis | resting_tachycardia | resting_tachycardia | 0/0 | 통과(시도 1) | 25.0 | - |
| 13 | simple | 휴식기 심박이 100을 넘으면 위험한가요? | simple | - | - | 0/0 | 통과(시도 0) | 9.3 | - |
| 14 | analysis | U08 최근 3주 동안 심박이 평소와 다른지 확인해줘 | analysis | personal_deviation | personal_deviation,sensor_gap | 0/0 | 통과(시도 1) | 26.3 | - |
| 15 | followup | 그럼 지난 3일은 어때? | analysis | - | - | 0/0 | 통과(시도 1) | 15.8 | - |
| 16 | simple | 서맥이 뭐예요? | simple | - | - | 0/0 | 통과(시도 0) | 9.0 | - |
| 17 | analysis | U03 최근 30일 이상 있었는지 점검해줘 | analysis | bradycardia | bradycardia,sensor_gap | 1/1 | 통과(시도 1) | 22.7 | - |
| 18 | simple | 심박수 정상 범위가 어떻게 돼? | simple | - | - | 0/0 | 통과(시도 0) | 6.6 | - |
| 19 | analysis | U15 심박 괜찮은지 봐줘 | simple | - | - | 0/0 | 통과(시도 0) | 3.6 | route_accuracy |
