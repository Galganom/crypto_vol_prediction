# 원안 형식 복원과 현재 결과 반영

## 수정 기준

기준 원안은 `paper_old/main.tex`입니다. 제목·저자·소속과 영어 학술 논문의 서술 방식을 유지하고, 37개 장·절 제목 및 순서를 그대로 따랐습니다. 정확한 대조 결과는 `generated/outline_comparison.json`에 있습니다. 본문은 `sections/`로 분리했으며 `main.tex`에서 원안 순서대로 읽습니다.

원안의 분석을 결과 파일 부족 때문에 삭제하지 않았습니다. 필요한 추가 실험은 현재 모델의 기준모형·데이터·분할을 사용해 실행하고, 기술통계와 그림은 현재 체크포인트와 예측에서 다시 계산했습니다. `paper_old`, 원래 모델 코드, 원래 `results/main` 및 입력 데이터는 변경하지 않았습니다.

## 원안에서 복원한 구성

| 원안의 위치 | 이번에 반영한 자료 |
|---|---|
| Methodology | 같은 하위 목차, 특징표, 모델 구조도, 기사 감쇠 개념도, P1/P2 설정표 |
| Main Results | 1h/4h 두 패널의 주요 결과표, BTC와 ETH 결과 및 윈도별 해석 |
| Z-score analysis | 기존 분석 스크립트의 signed forward-return 정의로 다시 계산한 10분위 표 |
| Timescale | 같은 2024-03-05 사례, 상위 기사 세 건, 현재 파라미터의 공동·주변 분포 |
| News category and impact heterogeneity | 원래 12개 분류 체계를 재실행한 기사별 범주와 범주별 통계표 |
| Sentiment versus event content | 저장된 FinBERT 확률과 현재 파라미터를 결합한 감성군 표·검정 |
| Alternative decay specifications | constant, power-law, exponential, linear의 P2 재학습 |
| Alternative news representations | 뉴스 수, 두 가지 sentiment 집계, 768차원 평균 임베딩, sentiment/PCA/full embedding decay 비교 |
| Appendix | 원안처럼 1h/4h를 나란히 배치한 가로형 윈도별 표; ETH 표 추가 |
| Sensitivity to lookback horizon | 원안에서 제목만 있던 부분을 6h/12h/24h 재학습 비교로 채움 |

## 현재 코드와 일치시킨 설명

- 주요 실행 경로는 `expanding_window_btc.py` / `expanding_window_eth.py` → `main_model_cryptobert.py`입니다. 주요 뉴스 입력은 PCA가 없는 768차원 CryptoBERT 임베딩입니다.
- 뉴스 head는 LayerNorm → 768 → 128 → 32 → 2입니다. 출력은 양의 크기 `s`와 분당 `0.001 + 0.499 sigmoid(.)`의 감쇠율입니다.
- 예측은 `TS × exp(clip(a × S + b, -2, 2))`입니다. `s` 자체를 분산의 가산적 인과효과로 설명하지 않고, `a`, `b`, clipping을 함께 명시했습니다.
- P1은 TS를 학습하고 P2는 그 가중치를 고정한 채 뉴스를 학습합니다. P3는 없습니다. 주요 모델의 auxiliary loss는 0이고 `ts_scale`은 1로 고정됩니다. 고정 TS의 dropout은 P2 학습 중에는 켜져 있습니다.
- 손실은 `delta=0.5`의 Huber-on-log이며 평가 log-MSE와 구별합니다. AdamW, weight decay, 학습률, epoch 상한과 patience를 실제 설정으로 수정했습니다.
- HAR는 네 개 원시 RV 특징으로 log 목표를 회귀합니다. EGARCH는 고정 파라미터로 필터링한 한 bar 분산에 H를 곱하는 현재 비교모형입니다. EGARCH+NED에는 별도의 auxiliary weight 0.1이 있습니다.
- 12h 기사 조회는 저장 bar 시각보다 엄격히 이전인 기사만 포함합니다. 현재 bar의 완료된 가격 특징과 bar 초반의 뉴스 cutoff를 구별했습니다.
- `DA`는 원안 제목을 유지하되 현재 코드의 실제 정의인 테스트 실제 RV 중앙값 기준 분류 정확도로 설명했습니다.
- 주요 지표는 여섯 윈도의 산술평균입니다. 원안 제목에 맞춰 HLN 소표본 보정 DM 검정을 관측별 저장 예측에서 추가 계산했습니다. 원래 파이프라인의 normal-reference 검정도 보존합니다.
- forward label의 validation 경계 중첩, 종료일 00:00 포함 상한, 누락 bar의 horizon 해석을 실험 한계에 반영했습니다.

## 재계산·재학습의 범위

주요 결과표는 기존 `results/main/{btc,eth}_cryptobert`의 예측을 사용합니다. 24개 자산·윈도·horizon 조합의 목표값과 전처리·정규화·TS 예측·P1 검증 손실을 대조했습니다. 5개 모델을 포함하는 주요 지표는 120행입니다.

추가 비교 실험은 `paper/run_missing_experiments.py`가 수행합니다. BTC 4h, 여섯 윈도, 윈도당 12가지 P2 조합으로 총 72회입니다. 기존 P1 체크포인트를 공통으로 쓰고 전체 학습 데이터를 적용합니다. 데이터 축소나 epoch 상한 축소는 하지 않습니다. 고차원 기사 head의 크기, Huber loss, 최대 120 epoch/patience 20 등 현재 주요 모델의 설정에 맞췄습니다.

기존 보조 스크립트 중에는 이전 결과 디렉터리, 다른 head 크기·epoch 상한 등을 참조하는 것이 있어, 비교 원리는 유지하면서 현재 설정에 맞는 별도 실행 스크립트를 만들었습니다. 감쇠 함수는 `ablations/decay_spec_ablation.py`, 뉴스 표현 방식은 `ablations/news_representation_compare.py`의 설계를 기준으로 했습니다. 후자의 실제 구현처럼 집계 뉴스도 고정 TS 위의 residual head에 넣습니다. 원안 본문의 “TS 입력에 직접 추가”라는 설명과는 구별했습니다.

추가 실험에서는 exponential 기준도 같은 GPU 실행 방식으로 다시 학습합니다. 따라서 세 비교표의 full-dimensional/12h exponential은 서로 같은 값이지만 주요 결과표의 기존 NED 값과는 조금 다를 수 있습니다. NumPy의 고정 shuffle 순서와 CUDA graphs/fused AdamW를 사용하며, 그 사실과 단일 seed의 한계를 본문에 밝혔습니다. 일반 학습과 GPU graph 학습의 12단계 가중치 갱신 일치도도 별도로 확인했습니다.

범주 분석은 BTC 4h 테스트 예측에 실제로 쓰인 7,845개 고유 기사만 대상으로 합니다. `ablations/relabel_news_categories.py`의 제목 규칙·분류 정의·hypothesis·0.25 임계값을 그대로 사용하고, ModernBERT 체크포인트 revision을 고정했습니다. 범주와 감성 통계는 기사당 동일 가중치입니다. 재학습한 비교 모델의 파라미터를 주요 해석 분석에 섞지 않았습니다.

## 해석에서 달라진 주요 사항

- BTC의 TS-only 대비 QLIKE 감소는 1h 7.38%, 4h 10.83%입니다. log-MSE 감소는 각각 0.99%, 3.11%입니다.
- ETH의 QLIKE 감소는 1h 1.32%, 4h 2.04%이고 log-MSE 차이는 유의하지 않습니다. 2025년 두 윈도에서는 QLIKE도 악화됩니다.
- EGARCH+NED는 log-MSE를 낮추지만 QLIKE는 악화시킵니다. 모든 손실과 자산에서 뉴스가 일관되게 우월하다는 주장은 유지하지 않았습니다.
- 기존 Z-score 스크립트는 표준화된 log RV가 아니라 **향후 4h 수익률 / TS 예측 변동성**을 계산합니다. 이 정의로 복원한 결과에서 양쪽 꼬리는 개선되지만 중간 decile의 log-MSE는 악화됩니다.
- 기사 반감기의 중앙값은 42.7분, 사분위 범위는 16.8–88.8분입니다. 원안의 기사 수와 예전 반감기 값을 새 표·그림에 재사용하지 않았습니다.
- 사례 그림은 원안과 같은 날짜·기준 시점을 사용하되 현재 모델에서 남은 신호가 큰 세 기사를 선택했습니다. 기사와 순위·반감기가 달라졌습니다.
- negative–positive 감성 차이는 재현되지만 ambiguous–decisive 차이는 임계값에 민감합니다. 금융 leverage 효과를 식별했다는 강한 주장은 사용하지 않았습니다.
- 범주 분석에서는 Macro의 평균 크기 8.96과 중앙 반감기 78.1분이 가장 큽니다. Geopolitical은 평균 크기 8.08로 두 번째지만 중앙 반감기는 47.1분으로, 지속시간 순위까지 원안과 같지는 않습니다.
- 새 커널 비교에서 constant도 QLIKE를 7.16% 낮춥니다. Linear는 QLIKE 13.96%, log-MSE 3.58% 감소로 가장 낮은 평균 손실을 보였습니다. 재학습 exponential 기준의 감소율은 각각 10.57%, 2.48%입니다.
- Sentiment decay는 QLIKE 10.68%, log-MSE 2.34% 감소로 full embedding과 비슷합니다. PCA 256 decay는 각각 12.55%, 3.02% 감소로 이번 실행에서는 full embedding보다 좋았습니다. 당일 sentiment 집계도 9.03%, 2.01% 감소했습니다.
- 6h/12h/24h lookback 모두 TS-only보다 평균 손실이 낮고, 6h가 가장 낮았습니다. 단일 seed의 재학습 결과이므로 최적 설정을 확정했다는 결론은 쓰지 않았습니다.
- 실제 사용한 감성 모델은 `ProsusAI/finbert`이므로 Araci (2019)를 추가했습니다. 기존 참고문헌 항목은 삭제하지 않았습니다.

새 비교 실험의 우열과 결론은 `results/ablations/paper_revision/metrics_mean.csv`와 본문 Robustness 절에서 확인할 수 있습니다. 구 원안의 “감쇠 없으면 개선 전무”, “sentiment가 거의 전부 회복” 등의 문장은 새 결과가 뒷받침하는 범위에서만 서술합니다.

## 재현 자료

실행 명령은 `README.md`에 있습니다. 원래 입력과 주요 결과의 SHA-256은 `generated/source_sha256.json`, 새 실행의 설정은 각 결과 디렉터리의 `run_config.json` 및 `category_config.json`에 기록됩니다. 모든 비교 모델에는 선택된 체크포인트, 관측별 예측, 검증 손실 이력, 성능 지표가 저장됩니다. PCA 및 집계 특징 정규화가 필요한 경우 해당 학습 파라미터도 저장합니다.
