# 허용 목록 필터의 비용 (오프라인 측정)

이상이 있는 리포트에서는 LLM 본문이 *사실 서술*(시각·지속·수치·유형·맥락·데이터 한계)만 남고 조언·평가·안심·진료 언급은 코드가 쓰는 고정 문구로 대신한다. 이 표는 그 대가로 LLM 본문 글자가 얼마나 살아남는지를 저장된 v2 리포트 원문에 `narration_only` 를 적용해 잰 것이다 (LLM 호출 없음).

- 대상: 33건 (`before_after.json, before_after_fast3b_analysis8b.json` 의 v2 analysis 리포트 중 템플릿 대체가 아니고 '이상 없음' 문장이 없는 것). 이상 목록이 저장돼 있지 않아 문장으로 판별한 근사다.
- 주의: 실제 서비스는 닫힌 어휘 검사에 이번 분석의 FACTS 단어(동적 어휘)를 함께 쓰지만, 저장된 결과에는 FACTS 가 없어 이 측정은 고정 어휘만 쓴다 - 생존율의 *하한*에 가깝다.
- 글자 생존율: 평균 0.47, 최소 0.12, 최대 0.77
- 사실 서술이 하나도 남지 않아 템플릿 본문이 필요해지는 리포트: 0건 / 33건
- 문장 생존율: 134 / 369 (0.36)

읽는 법: 사라지는 것은 주로 [해석]·[권고]의 조언·평가·원인 추정 문장이다(예: '\~놓친 것으로 보여요', '일시적일 수 있으나 주의가 필요합니다'). 안전을 얻는 대신 이상이 있는 리포트의 해석 깊이가 줄어든다는 트레이드오프이다.

| 출처 | 케이스 | 본문 글자 | 남은 글자 | 생존율 | 문장 | 남은 문장 | 비어 있음 |
|---|---|---|---|---|---|---|---|
| before_after | D07 | 639 | 412 | 0.64 | 15 | 7 | 아니오 |
| before_after | D09 | 298 | 159 | 0.53 | 10 | 3 | 아니오 |
| before_after | D10 | 518 | 219 | 0.42 | 12 | 4 | 아니오 |
| before_after | D11 | 489 | 265 | 0.54 | 11 | 4 | 아니오 |
| before_after | D12 | 535 | 389 | 0.73 | 15 | 8 | 아니오 |
| before_after | D13 | 312 | 203 | 0.65 | 9 | 4 | 아니오 |
| before_after | D14 | 318 | 96 | 0.30 | 10 | 2 | 아니오 |
| before_after | D15 | 246 | 45 | 0.18 | 9 | 2 | 아니오 |
| before_after | D20 | 249 | 108 | 0.43 | 9 | 3 | 아니오 |
| before_after | D22 | 421 | 171 | 0.41 | 11 | 3 | 아니오 |
| before_after | D26 | 476 | 275 | 0.58 | 12 | 5 | 아니오 |
| before_after | H09 | 246 | 83 | 0.34 | 10 | 3 | 아니오 |
| before_after | H14 | 315 | 104 | 0.33 | 9 | 2 | 아니오 |
| before_after | H15 | 316 | 126 | 0.40 | 10 | 3 | 아니오 |
| before_after | H16 | 345 | 163 | 0.47 | 11 | 4 | 아니오 |
| before_after_fast3b_analysis8b | D07 | 628 | 355 | 0.57 | 15 | 6 | 아니오 |
| before_after_fast3b_analysis8b | D08 | 549 | 400 | 0.73 | 14 | 7 | 아니오 |
| before_after_fast3b_analysis8b | D09 | 378 | 158 | 0.42 | 10 | 3 | 아니오 |
| before_after_fast3b_analysis8b | D10 | 477 | 221 | 0.46 | 12 | 4 | 아니오 |
| before_after_fast3b_analysis8b | D11 | 422 | 283 | 0.67 | 12 | 5 | 아니오 |
| before_after_fast3b_analysis8b | D12 | 636 | 461 | 0.72 | 17 | 10 | 아니오 |
| before_after_fast3b_analysis8b | D13 | 314 | 203 | 0.65 | 10 | 4 | 아니오 |
| before_after_fast3b_analysis8b | D14 | 387 | 104 | 0.27 | 11 | 2 | 아니오 |
| before_after_fast3b_analysis8b | D15 | 266 | 83 | 0.31 | 9 | 3 | 아니오 |
| before_after_fast3b_analysis8b | D16 | 387 | 243 | 0.63 | 11 | 4 | 아니오 |
| before_after_fast3b_analysis8b | D20 | 259 | 94 | 0.36 | 9 | 3 | 아니오 |
| before_after_fast3b_analysis8b | D22 | 304 | 164 | 0.54 | 10 | 3 | 아니오 |
| before_after_fast3b_analysis8b | D26 | 477 | 218 | 0.46 | 11 | 4 | 아니오 |
| before_after_fast3b_analysis8b | H09 | 238 | 83 | 0.35 | 10 | 3 | 아니오 |
| before_after_fast3b_analysis8b | H11 | 529 | 407 | 0.77 | 13 | 8 | 아니오 |
| before_after_fast3b_analysis8b | H14 | 428 | 165 | 0.39 | 12 | 4 | 아니오 |
| before_after_fast3b_analysis8b | H15 | 352 | 44 | 0.12 | 10 | 2 | 아니오 |
| before_after_fast3b_analysis8b | H16 | 369 | 98 | 0.27 | 10 | 2 | 아니오 |
