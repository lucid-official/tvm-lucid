# tvm-lucid

**Terminal Velocity Matching** (Zhou, Parger, Haque & Song, ICLR 2026, [arXiv:2511.19797](https://arxiv.org/abs/2511.19797))
를 [Lucid](https://github.com/lucid-official/lucid)로 처음부터 학습하는 연구용 프로젝트입니다.
대상은 **CIFAR-10 class-conditional, 픽셀 공간(32×32)** 이고, 기본 백본은 S/2 크기(약 33M)입니다.

방법 자체(네트워크·목적함수·샘플러)는 다시 구현하지 않습니다. 전부 `lucid.models.generative.tvm` 를 그대로 씁니다.
이 저장소에 있는 것은 실험 장치입니다: 결정적 데이터 파이프라인, 학습 루프, 학습 동역학 로깅, epoch 별 평가, 체크포인트, 플롯.

## 구성

```
configs/            cifar10_s2.toml (본 실험), debug.toml (수 분짜리 점검)
tvm_lucid/
  config.py         TOML + --set 덮어쓰기
  data.py           CIFAR-10 다운로드·md5 검증, (seed, epoch, batch) 결정적 배치
  model.py          lucid.models.tvm_imagenet256_gen 을 CIFAR-10 크기로 override
  dynamics.py       그룹별 grad/param norm, update ratio, EMA drift, activation RMS
  evaluate.py       샘플 그리드, flow-map gap, 시간별 FM probe, FID/IS
  inception.py      Lucid inception_v3_cls 특징 / 확률
  stats.py          FID·IS 수치 계산 (numpy, 테스트 포함)
  run.py            run 디렉토리, env 기록, JSONL 로거, 원자적 체크포인트
  train.py          학습 진입점
scripts/
  plot_dynamics.py  run 의 JSONL → plots/*.png
  sample.py         체크포인트 → NFE × guidance 그리드
tests/              stats·data·config (Lucid 불필요) + test_tvm_method.py (방법 자체 검증, Lucid 필요)
```

## 환경 (Mac Studio)

Lucid 체크아웃(`~/Desktop/Programming/lucid`)의 가상환경을 그대로 씁니다. 엔진을 다시 빌드할 필요가 없습니다.

```bash
LUCID_PY=~/Desktop/Programming/lucid/.venv/bin/python
uv pip install --python $LUCID_PY tqdm pillow matplotlib        # 한 번만
$LUCID_PY -c "import lucid, lucid.models as M; print(lucid.__version__, hasattr(M, 'tvm_imagenet256_gen'))"
```

마지막 줄이 `True` 를 출력해야 합니다(Lucid `main` 이 TVM 을 포함한 `8739862` 이후여야 함).

## 실행

```bash
cd ~/Desktop/Programming/tvm-lucid

# 0) 파이프라인 전체 점검 (수 분; 첫 실행은 CIFAR-10 다운로드와 FID 기준 통계 계산이 추가됨)
$LUCID_PY -m tvm_lucid.train --config configs/debug.toml

# 1) 속도·메모리 측정 → batch / 예산 결정
$LUCID_PY -m tvm_lucid.train --config configs/cifar10_s2.toml --benchmark

# 2) 본 학습 (ssh 를 끊어도 계속 돌도록)
nohup caffeinate -dimsu $LUCID_PY -m tvm_lucid.train --config configs/cifar10_s2.toml \
    > train.out 2>&1 &
tail -f train.out

# 3) 중단 후 재개 (예산 등은 --set 으로 바꿀 수 있음)
$LUCID_PY -m tvm_lucid.train --resume runs/cifar10_s2-YYYYmmdd-HHMMSS --set train.hours=24

# 4) 플롯 / 샘플링 (학습 중에도 가능)
$LUCID_PY scripts/plot_dynamics.py runs/cifar10_s2-YYYYmmdd-HHMMSS
$LUCID_PY scripts/sample.py runs/cifar10_s2-YYYYmmdd-HHMMSS --ckpt best_fid --nfe 1 2 4 8 --w 1.5 2 2.5
```

어떤 설정이든 `--set section.key=value` 로 덮어쓸 수 있습니다. 값은 TOML 로 읽습니다
(`--set train.batch_size=32 --set eval.nfe=[1,2,4,8] --set model.depth=8`).
`[model]` 에는 `TVMConfig` 의 모든 필드를 쓸 수 있습니다(`time_sampler`, `scaled_parameterization`, `ratio_t_equals_s` …).

## run 디렉토리

```
runs/<name>-<timestamp>/
  config.json      해석된 설정 (재개 시 기준)
  env.json         호스트, Lucid 버전·커밋, 디바이스, 시작 시각
  metrics.jsonl    kind=step     : loss, terminal, fm, img/s, step_s, mem/peak GiB
                   kind=dynamics : grad_norm/param_norm (embed, blocks.XX, final, total),
                                   update_ratio, ema_drift{target,eval},
                                   activation_rms{time_embed, adaln_last, adaln_final}
  epoch.jsonl      epoch 별 평가 (아래)
  events.jsonl     start / stop(사유) / 재개 / non-finite skip / best_fid
  samples/         epoch_XXXX_nfeK.png  (행 = 클래스, 고정 노이즈)
  ckpt/            last.lucid, best_fid.lucid, epoch_XXXX.lucid
  plots/           plot_dynamics.py 출력
```

## 측정 항목

| 항목 | 주기 | 의미 |
|---|---|---|
| `loss`, `terminal`, `fm` | 매 step (기록은 `log_every`) | Eq. 15 와 두 항. terminal 은 t=s 샘플에서 0 |
| `grad_norm.*`, `param_norm.*` | `dynamics_every` | 블록별 기울기 붕괴/폭발 조기 발견 |
| `update_ratio` | `dynamics_every` | ‖Δθ‖/‖θ‖ — Adam 이 실제로 적용하는 유효 학습률 |
| `ema_drift.target/eval` | `dynamics_every` | θ*, θ** 가 live 가중치에서 얼마나 떨어져 있는지 |
| `activation_rms.*` | `dynamics_every` | 논문 Figure 4 의 양: 시간 임베딩과 마지막 adaLN modulation 의 크기 |
| 샘플 그리드 | 매 epoch | 고정 노이즈 → epoch 간 같은 칸 비교 가능 |
| `flowmap_gap_KvN` | 매 epoch | 같은 노이즈의 K-step 과 N-step 샘플 MSE — one-step 사상이 ODE 경로를 얼마나 따라잡았는지 (Eq. 5 의 displacement error 대리 지표) |
| `probe_fm_{live,ema}` | 매 epoch | 테스트 이미지에서 s 구간별 무조건부 FM 오차 — terminal 항이 맞추려는 속도장이 어느 시간대에서 아직 틀린지 |
| `fid_nfeK`, `is_nfeK` | `fid_every_epochs` | 5k 샘플, 학습 split 기준 |

## 주의

- **FID 는 논문 표와 비교할 수 없습니다.** Lucid 의 ImageNet `inception_v3_cls` 를 쓰기 때문에(TF-Inception 아님) 이 프로젝트의 run 끼리 비교하는 용도입니다.
- **`lucid.compile` 은 쓰지 않습니다.** TVM 의 손실은 JVP 안에서 역전파를 돌리므로 컴파일된 학습 스텝은 eager 로 fallback 됩니다.
- **재개의 결정성.** 데이터 순서는 중단 여부와 무관하게 정확히 같습니다. 모델 내부 난수(t, s, 노이즈, label drop)는 Metal 생성기 상태를 저장할 수 없어 `(seed, step)` 으로 재시드합니다 — 결정적이지만 중단 없는 run 과 같은 스트림은 아닙니다.
- **EMA.** θ*(0.99)는 목적함수의 타깃, θ**(기본 0.999)는 평가·샘플링용입니다. 둘 다 `state_dict` 밖에 있어 체크포인트에 따로 저장됩니다. run 이 10만 step 을 넘으면 `model.eval_ema_decay=0.9999` 를 고려하세요.
- **출력 위치.** `runs/` 는 git 에서 제외됩니다. Lucid 저장소 안에 두지 마세요(맥북과 동기화되어 GB 단위 체크포인트가 복사됩니다).

## 테스트

```bash
$LUCID_PY -m pytest -q tests      # pytest 가 없으면: uv pip install --python $LUCID_PY pytest scipy
```

`test_tvm_method.py` 는 방법 자체를 검증합니다 — 종단 속도가 변위의 s-미분(유한차분과 비교)인지,
기울기가 JVP 를 통과하는지, modulation norm 이 modulation 크기에 불변인지, scaled
parameterisation 이 정확히 w 배인지, 두 EMA 가 Algorithm 1 을 따르는지, 샘플러가 Figure 3 인지.
Lucid 라이브러리의 테스트 스위트는 TVM 을 zoo 공통 계약(학습 스텝·출력 형식·생성·동기화 횟수)으로만
다루므로, 이런 방법 특화 검증은 이 저장소에 둡니다. Lucid 를 업데이트한 뒤 실험 전에 한 번 돌리세요.
