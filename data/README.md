# data/

| 폴더 | git | 내용 |
|---|---|---|
| `processed/` | **포함** | 분 단위 가공 데이터(`vitals_minute.parquet`, `vitals_minute_clean.parquet`), 평가 라벨(`labels.json`), 가명 매핑(`id_map.json`), 검색 인덱스(`rag/`). 약 8MB |
| `raw/` | 제외 | Kaggle 원본 CSV(약 588MB). 필요할 때만 받는다 |
| `external/` | 제외 | 외부 연구 데이터(Stanford COVID-19). 라이선스 표기가 없어 로컬 전용 |

가공본이 저장소에 들어 있어서 저장소를 받으면 Kaggle 계정 없이 바로 실행된다(`bash sh/setup_venv.sh` → `bash sh/run_agent_fake.sh`).

## 출처와 라이선스

제3자 자료 전체의 출처와 조건은 루트의 [`NOTICE.md`](../NOTICE.md)에 있다. 아래는 이 폴더의 데이터에 관한 요약이다.

- **원본 출처**: Furberg, R., Brinton, J., Keating, M., Ortiz, A. (2016). *Crowd-sourced Fitbit datasets 03.12.2016-05.12.2016*. Zenodo. [doi:10.5281/zenodo.53894](https://doi.org/10.5281/zenodo.53894). 라이선스 **CC BY 4.0** (저작자 표시).
- **재배포처**: Kaggle [`arashnic/fitbit`](https://www.kaggle.com/datasets/arashnic/fitbit) (Fitbit Fitness Tracker Data). Kaggle의 표기는 원본과 다르며 우리는 여기서 내려받았다.
- Kaggle의 표기가 원본과 다르므로 **원본의 CC BY 4.0을 따른다**: 원저자를 밝히고 변경 사항을 명시한다. 이 저장소의 가공본은 원본에서 다음을 바꾼 것이다: 초 단위 심박을 분 단위로 집계, 걸음·수면 결합과 중복 제거, 사용자 ID 가명화(`U01`\~), 평가용 합성 이상 주입과 정답 라벨(`labels.json`). 주입이 없는 `vitals_minute_clean.parquet`도 함께 둔다.
- 저장소 루트의 MIT 라이선스([`LICENSE`](../LICENSE))는 코드와 문서에 적용되고 이 폴더의 가공본에는 적용되지 않는다. 가공본은 CC BY 4.0의 표시 조건을 유지한다.
- 그래서 가공본을 이 저장소에 그대로 포함해 재배포한다. 컬럼 사전과 전처리 단계는 [`docs/data.md`](../docs/data.md)에 있다.

### 외부 연구 데이터 (`external/`, git 제외)

Stanford COVID-19 웨어러블(Mishra 외, 2020, *Nat. Biomed. Eng.*, [doi:10.1038/s41551-020-00640-6](https://doi.org/10.1038/s41551-020-00640-6))은 데이터 라이선스 표기가 없다. 공개 다운로드 주소는 `https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip`이고 원자료의 날짜는 출처가 개인정보 보호를 위해 이동시킨 값이다. 그래서 원시 파일과 분 단위 변환본은 `data/external/stanford/`에만 두고 커밋하지 않으며 다운로드·변환 스크립트(`eval/external/`)와 집계 지표(`eval/results/external_stanford.md`)만 공개한다.

## 가공 데이터를 처음부터 다시 만들기 (선택)

```bash
kaggle datasets download -d arashnic/fitbit --unzip -p data/raw      # Kaggle API 자격 증명 필요(~/.kaggle/kaggle.json)
.venv/bin/python -m app.data.prepare                                 # seed 고정이라 같은 결과가 나온다. --no-inject 면 합성 이상을 넣지 않는다
```

`prepare` 는 원본 두 기간을 합치고 중복을 없앤 뒤, 사용자 ID 를 `U01`…로 가명화하고 평가용 합성 이상을 실제 데이터 위에 주입한다(라벨은 `labels.json`).
