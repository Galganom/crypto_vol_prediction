# 기존 원안의 형식을 유지한 수정 논문

`paper_old/main.tex`의 학술적 서술 방식과 장·절 제목 37개의 순서를 유지하면서, 현재 코드와 결과에 맞춰 논문을 갱신했습니다. 기존 제목·저자·소속, 본문의 표·그림 구성, 가로형 윈도별 부록도 따릅니다. 데이터가 없었던 분석은 별도로 재실행해 채웠습니다.

## 주요 파일

- `main.tex`, `main.pdf`: 논문 진입 파일과 컴파일된 PDF.
- `sections/`: 원안의 목차에 대응하는 본문 소스.
- `refs.bib`: 원안의 참고문헌에 실제 사용한 `ProsusAI/finbert`의 Araci (2019) 문헌을 추가한 파일.
- `REVISION_NOTES.md`: 변경 근거, 재실험 범위, 원안과 결과의 차이.
- `generated/`: 본문 표, 중간 데이터, 검증 보고서.
- `figures/`: 체크포인트와 예측으로 재생성한 사례·분포 그림 등.
- `../results/ablations/paper_revision/`: 추가로 학습한 BTC 4h 비교 실험의 예측·체크포인트·학습 이력.
- `../results/analysis/paper_revision/`: 기사 범주·감성·반감기·사례·Z-score·보정 DM 검정 데이터.

`paper_old/`, 원래 모델 코드, `data/`, `results/main/`은 보존합니다. 새 실험과 분석은 위의 별도 경로에 기록합니다.

## 표와 그림 재생성

프로젝트 루트에서 실행합니다.

```bash
.venv/bin/python paper/build_artifacts.py
.venv/bin/python paper/build_restored_analysis.py --require-complete
```

첫 스크립트는 기존 주요 실험의 지표와 기사별 파라미터를 추출합니다. 두 번째 스크립트는 원안 형식의 본문 표, 보정 DM 검정, signed-return Z-score decile, 감성·범주 통계, 사례·분포 그림, 새 비교 실험 표를 만듭니다. `--require-complete`는 기사 분류와 72개 P2 학습 결과가 모두 있는지 확인하며, 미완성 결과를 완성된 표로 출력하지 않습니다.

가격 프레임 캐시가 없을 때는 아래 검증을 먼저 실행합니다. 원시 가격에서 특징·분할·정규화를 재구성하고, 저장된 24개 자산·윈도·예측기간 조합의 목표값, TS 예측, P1 검증 손실과 비교합니다. 계산에 수 분 이상 걸릴 수 있습니다.

```bash
.venv/bin/python paper/verify_price_pipeline.py
.venv/bin/python paper/verify_main_forecasts.py
```

임시 가격 프레임은 `/tmp/crypto-paper-btc-bars.pkl`과 `/tmp/crypto-paper-eth-bars.pkl`에 저장됩니다. `build_restored_analysis.py`는 BTC 캐시를 사용합니다. `verify_main_forecasts.py`는 이 프레임과 원래 뉴스 임베딩·체크포인트로 주요 NED 예측 전체를 다시 계산해 저장된 값과 비교합니다.

## 누락됐던 비교 실험 재학습

CUDA가 보이는 환경에서 다음 명령을 사용합니다.

```bash
.venv/bin/python paper/run_missing_experiments.py
```

BTC 4h의 여섯 윈도 각각에 12가지 뉴스 분기를 학습합니다. 현재 주요 실험의 선택된 P1 기준모형을 재사용하며, P2의 전체 학습 데이터·최대 120 epoch·patience 20을 적용합니다. 완료된 `metrics.json`이 있는 조합은 건너뛰므로 재시작할 수 있습니다. 처음부터 다시 학습하려면 `--out`으로 다른 출력 디렉터리를 지정하십시오.

비교 항목은 constant/power/linear/exponential 감쇠, 뉴스 수·recency sentiment·당일 sentiment·768차원 평균 임베딩, sentiment decay·PCA 256 decay, 6h/24h lookback입니다. 12h full-dimensional exponential은 세 비교표의 공통 기준입니다. 이 기준도 동일한 추가 실험 코드로 P2를 다시 학습한 값이며, 주요 결과표의 기존 체크포인트와 동일한 수치를 강제하지 않습니다.

CUDA graphs와 fused AdamW를 사용해 실행을 가속합니다. 데이터나 epoch 상한을 줄이지 않습니다. 첫 실행에서 사용하는 전체 설정은 `results/ablations/paper_revision/run_config.json`에 기록됩니다.

저장된 추가 실험의 목표값·TS 가중치·선택 기준·지표와 독립적인 CPU 예측 재구성을 확인하려면 다음을 실행합니다.

```bash
.venv/bin/python paper/verify_restored_experiments.py --require-complete
```

## 기사 범주 분석 재생성

이 분석은 BTC 4h 테스트 예측에 실제로 진입한 7,845개 고유 기사에만 적용합니다. 원래 범주 분석 코드의 12개 분류 체계·뉴스레터 제목 규칙·NLI hypothesis·0.25 임계값을 사용합니다. 필요 모델을 준비한 다음 GPU에서 실행합니다.

```bash
.venv/bin/python - <<'PY'
import os
from pathlib import Path
os.environ.setdefault('HF_HOME', '/tmp/crypto-paper-revision/huggingface')
from huggingface_hub import snapshot_download
snapshot_download(
    'MoritzLaurer/ModernBERT-large-zeroshot-v2.0',
    revision='a51e07b524299e309dd2b88d48b0cfa2bd9ec598',
    cache_dir=Path(os.environ['HF_HOME']) / 'hub',
    allow_patterns=['*.json', '*.safetensors', 'tokenizer*', 'special_tokens_map.json'],
)
PY
.venv/bin/python paper/classify_news_categories.py
.venv/bin/python paper/build_restored_analysis.py --require-complete
```

분류 스크립트는 로컬 캐시만 읽으며, 중간 CSV를 통해 재개할 수 있습니다. `HF_HOME`을 지정하면 캐시 위치를 변경할 수 있습니다. 범주 라벨은 사후 해석에만 쓰이며 예측모형 학습에는 들어가지 않습니다. 이미 저장된 감성 확률과 CryptoBERT 임베딩은 다시 추출하지 않습니다.

## PDF 빌드

표준 TeX 환경에서는 다음을 실행합니다.

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Tectonic도 사용할 수 있습니다.

```bash
tectonic -X compile paper/main.tex
```

이 환경에서는 `/tmp/crypto-paper-tools/tectonic`과 `/tmp/crypto-paper-cache`에 임시 도구와 TeX 패키지를 준비했습니다. 다른 컴퓨터에서는 TeX를 별도로 설치해야 합니다. Python은 프로젝트 `.venv`를 사용했습니다.

표를 다시 생성한 뒤에는 본문에서 직접 서술한 숫자와 해석도 확인하고 PDF를 다시 빌드해야 합니다. 요약 지표는 여섯 윈도의 산술평균이며, 전체 DM 검정은 연결된 관측별 손실 차이를 사용합니다. `DA`는 현재 코드의 테스트 중앙값 기준 분류 정확도이고, Z-score는 향후 4h 수익률을 TS 예측 변동성으로 나눈 값입니다. 논문에는 이 정의를 명시했습니다.
