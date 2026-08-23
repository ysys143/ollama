# FreeToken vs Ollama 분석 리포트

- FreeToken 소스: `/tmp/freetoken` (github.com/FlashML-org/FreeToken, Apache-2.0, 단일 커밋 `0ab982f`, 485 파일 / Python ~71k LOC)
- Ollama 소스: 이 저장소 (HEAD `fb30760`, llama.cpp `b10488` 핀, MLX `27fec90` 핀)
- llama.cpp 소스: `/tmp/lcpp` (`b10488` 태그, sparse checkout — 핀된 버전의 실제 플래그 확인용)
- 논문: arXiv 2608.16157 "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution"
  (Berkeley Sky/BAIR 중심 + MIT 공저 — Yang, Fan, Pan, Xi, Wang, Sun, Keutzer, Han, Zaharia, Xu, Stoica)

---

## 0. 한 줄 결론

같은 범주의 제품이 아니다. Ollama는 **모든 하드웨어 × 모든 모델**을 다루는 로컬 LLM 플랫폼이고,
FreeToken은 **"VRAM에 안 들어가는 MoE 모델을 NVIDIA GPU 한 장으로 돌리는"** 단일 시나리오에
전부를 건 서빙 엔진이다. 2배 격차는 그 단일 시나리오 안에서만 성립한다.

## 1. 정체성 비교

| | Ollama | FreeToken |
|---|---|---|
| 목적 | 로컬 LLM 플랫폼(모델 배포·관리·실행·에이전트) | MoE 전용 엣지 서빙 엔진 |
| 언어 | Go + vendored C/C++ (llama.cpp) + MLX | Python + CUDA/Triton + C++ 커널 |
| 실행 경로 | `llama-server` 서브프로세스 (`llm/llama_server.go`), macOS는 Go MLX 러너 (`x/mlxrunner`, 21k LOC) | in-process Python 엔진 (`engine/`, `scheduler/`) |
| HW | CUDA 12/13, ROCm 7.1/7.2, Vulkan, Metal(MLX v3/v4), Jetson, CPU | **Linux x86_64 + NVIDIA only** (driver r580+, CUDA 13, RTX 30/40/50) |
| 모델 포맷 | GGUF(전 quant) + safetensors import/convert/quantize | HF safetensors, FTW, 일부 GGUF(Gemma-4) |
| Quant | GGUF 전체 K-quant 계열 | BF16 / FP8-block / MXFP4 / NVFP4 / ds_fp4 / Q4_0 |
| 모델 수 | 수천 개 (레지스트리) | 문서상 10개 아키텍처 계열, 논문 기준 20+ MoE |
| 기능 범위 | pull/Modelfile, 멀티모달, 임베딩, 이미지 생성(`x/imagegen`), 에이전트(`agent/`), TUI, 데스크톱 앱, OpenAI+Anthropic API | 서빙 + `ft shell`/`ft launch`, OpenAI+Anthropic+Responses API, 데스크톱 앱은 별도 배포(비공개) |
| 외부 의존 | 벤더링된 llama.cpp/MLX 소스 | `flashlib==0.3.0` 핀 (device LRU admission 커널) — **저장소에 없음**, torch 2.11 고정 |

## 2. 승부가 갈리는 지점: MoE 오프로딩 입도

### Ollama
- `llm/llama_server.go:403-410` — `-ngl` 만 전달, 기본은 llama-server의 `-ngl auto`.
- **레이어 단위** 정적 배치. 로드 시점에 결정되고 런타임에 안 바뀐다.
- expert 단위 플래그를 **노출하지 않음**(전 트리 grep 결과 0건). §10-B 참조 — 업스트림에는 있다.
- 즉 MoE 라우팅이 토큰마다 바뀐다는 사실을 시스템이 전혀 모른다.

### FreeToken
- 캐시 단위가 **(layer, expert) 슬롯** — `moe/offload_cache.py`, 디바이스측 LRU admission 커널(`lru_ensure`)로 매 스텝 갱신.
- 백엔드 4종 (`docs/models.md`):
  - `fused` — 전문가 전체 GPU 상주 (dense 모델은 항상 여기로)
  - `offload` — 호스트 RAM에 상주, miss는 PCIe 스트리밍
  - `cpu` — miss를 CPU에서 계산
  - `hybrid` — miss를 PCIe fetch / CPU compute로 **측정 대역폭 비율대로 분할**
- `ft bench bw`(`moe/benchbw.py`)가 CPU MoE GEMV 실측 대역폭과 PCIe gather 실측 대역폭을 재고,
  겹쳐 돌렸을 때의 값으로 `load_hybrid_fetch_fraction`(논문의 q* 정책)을 정한다.
  규칙: CPU 커널 대역폭 > 2× PCIe면 `hybrid`, 아니면 `offload`.
- 이 분할이 **닫힌 형태 비율**이라 CUDA Graph 안에 캡처된다(`cudaLaunchHostFunc` submit/sync 노드).
  논문은 KTransformers/llama.cpp의 호스트측 휴리스틱이 그래프 캡처를 못 하는 점을 명시적으로 공격한다.

### 실측 근거(논문 §5.3)
RTX 5090 서빙 용량(Qwen3.6 expert pool의 37%, DSV4-Flash의 11%)에서 decode 시점 expert read miss rate:

| 정책 | Qwen3.6 | DSV4-Flash |
|---|---|---|
| FreeToken global LRU | **16%** | **39%** |
| KTransformers (prefill 시점 갱신) | 41% | 59% |
| llama.cpp (라우팅 무시 정적 분할) | 62% | 89% |

Ollama는 llama.cpp 계열이므로 마지막 행과 같은 구조다. 2배 격차의 근본 원인이 여기다.

## 3. FreeToken 고유 메커니즘

1. **Prefill full-layer double buffering** — 8192토큰 청크당 1.19–1.22초 = 64.4GB expert pool을
   52.7GB/s로 한 번 흘리는 시간 = PCIe 5.0 ×16 실효 상한. 계산이 전송 뒤에 완전히 숨는다.
   두 번째 버퍼를 끄면 4k에서 -19%, 8k -25%, 16k -26%.
2. **Semantic anchor / tool-call anchor** — `core.py:62 toolcall_anchor_len`,
   `scheduler/cache.py:144 snapshot_toolcall_anchor`. 툴콜 지점에서 GDN(선형 어텐션) 상태와
   SWA 윈도우를 얼려둔다. 에이전트가 클라이언트에서 툴 결과/thinking 블록을 고쳐 다시 보내도
   앵커 이후만 fork되므로 수천 토큰 재prefill을 피한다.
3. **Elastic VRAM** — `POST /v1/cache/rebuild`, `ft ctl cache rebuild`, `engine/cache_budget.py`.
   엔진 재시작·가중치 재로드 없이 expert 캐시 ↔ KV 캐시 비율을 런타임 재조정.
   Ollama는 `server/sched.go`에서 컨텍스트를 줄여 **모델을 다시 로드**하는 방식
   (`reduceAutoNumCtxForLoadOOM`).
4. **FTW 포맷** (`checkpoint/ftw.py`) — 모든 텐서를 4096 정렬한 단일 논리 바이트 영역 + 8GiB 샤드.
   O_DIRECT 멀티스레드 preadv로 페이지 캐시 우회. `host_banks.py`의 **pin-after-fill**:
   lazy mmap을 채운 뒤에 `cudaHostRegister` — DSV4 137GiB에서 zero-fill 47초를 통째로 제거.
   Ollama는 GGUF mmap 또는 `--load-mode dio`로 llama-server에 위임.

## 4. Ollama가 앞서는 부분

- **하드웨어 커버리지**: Apple Silicon(MLX + Metal), AMD, Intel/Vulkan, Jetson, CPU-only.
  FreeToken은 전부 미지원 (`pyproject.toml` classifier에 Linux + NVIDIA CUDA만).
- **Speculative decoding**: `--spec-draft-model` 등(`llm/llama_server.go:815-821`),
  MLX 러너에는 `speculate.go`/`mtp.go`/`speculate_depth.go`. FreeToken에는 범용 speculative 경로 없음.
- **멀티모델 스케줄링**: `server/sched.go`가 VRAM fit 예측·언로드 타이머·모델 간 전환을 관리.
  FreeToken은 프로세스당 모델 1개(`daemon/serve_manager.py`가 전환을 감독).
- **동시성 기본값**: Ollama `OLLAMA_NUM_PARALLEL`, FreeToken `--max-running-requests` 기본 **4**.
- **생태계**: 레지스트리, Modelfile, 변환/양자화 파이프라인, 데스크톱 앱, 임베딩, 이미지 생성, 에이전트 런타임.
- **dense 모델**: FreeToken은 dense를 무조건 `fused`로 보내므로 오프로딩 이점이 0.
  VRAM에 다 올라가는 모델이면 두 엔진의 격차 근거 자체가 사라진다.

## 5. "Ollama보다 2–4배" 주장 검증

저장소에는 **Ollama 비교 벤치마크가 없다** (`benchmarks/`는 백엔드 간 비교 3종뿐, 전 트리에서
"ollama" 문자열은 `launch.py:652` 주석 1건). 수치는 전부 논문에 있다.

논문이 실제로 말하는 것:
- RTX 5090, 4개 에이전트 워크로드: Qwen3.6-35B-A3B(BF16)에서 77–83 tok/s로
  **"각 워크로드에서 가장 강한 베이스라인의 1.8–2.3×"**, DSV4-Flash(MXFP4)에서 22–25 tok/s로 **1.5–1.9×**.
- 하드웨어별(W2 코딩 에이전트): 3090/4090 **1.3×**, 5090 서버 **1.9×**, 5090 데스크톱 **2.1×**, 4060 랩톱 **1.8×**.
- RTX PRO 6000 + GLM-5.2(753B-A40B NVFP4): llama.cpp 7.3 → FreeToken 14.9 tok/s (**2.0×**). Ollama는 미실행.
- Tail TTFT: FreeToken 최악 44초 미만, Ollama 179초, llama.cpp 232초, KTransformers 946초.

즉:
1. "2–4배"는 논문 표현이 아니다. 논문 최대치는 **2.3×**이고, 그것도 "Ollama 대비"가 아니라
   **"그 셀에서 가장 강한 베이스라인 대비"**다. 베이스라인은 llama.cpp / Ollama / KTransformers / MoE-Infinity.
2. Ollama는 DeepSeek-V4-Flash를 **아예 서빙 못 해서** 절반의 셀에서 × 표시다. 배수를 계산할 수도 없다.
3. Ollama 단독 대비 배수는 논문에 명시되지 않았다. Ollama는 대체로 llama.cpp보다 느리므로
   실제로는 2.3×보다 클 수 있지만, **"2–4배"는 논문에 근거가 없는 2차 서술이다.**
4. 조건: NVIDIA GPU + Linux + MoE 모델 + VRAM 부족 + 에이전트형 다중 턴. 하나라도 어긋나면 무의미.

---

## 6. 기법 계보 — 이식 / 선행연구 / 신규

### 6.1 계층 1: 대규모 서빙 스택에서 그대로 이식

논문 §4 첫 문단이 직접 인정한다: *"FreeToken follows the GPU-centric serving architecture
established by systems such as SGLang and vLLM."* README도 mini-sglang 포크에서 출발했다고 밝힌다.

| 기법 | 출처 | 코드 근거 |
|---|---|---|
| Paged KV + radix prefix tree | SGLang | `kvcache/radix_cache.py` |
| Chunked prefill / continuous batching | vLLM·SGLang | `scheduler/prefill.py` |
| CUDA Graph 캡처 decode | vLLM·SGLang | `engine/graph.py` |
| Attention 커널 | FlashInfer, TRT-LLM, FA | `attention/fi.py`, `trtllm.py` |
| 선형 어텐션 커널 | flash-linear-attention | `kernel/fla/` |
| Tool-call / reasoning 파서 | LightLLM, vLLM | `server/function_call_parser.py:1` ("Adapted from LightLLM") |
| OpenAI / Anthropic / Responses API | vLLM | `server/anthropic_api.py:3` ("Adapted from vLLM's AnthropicServingMessages") |

### 6.2 계층 2: MoE 오프로딩 문헌에 이미 있던 것

대규모 서빙 스택에는 없지만 **FreeToken이 처음도 아닌** 것들. 논문 §6이 선행연구를 정직하게 나열한다.

| 기법 | 선행 연구 |
|---|---|
| host에 expert pool + GPU에 부분 캐시 | **EdgeMoE (2023)** 가 확립한 구조 |
| **LRU expert 캐시** | **Mixtral-offloading (2023)** 이 이미 LRU + speculative prefetch |
| miss expert를 CPU에서 계산 | **Fiddler (2024)** 가 최초, KTransformers가 AMX 커널로 실용화 |
| CPU/GPU 작업 분배 | HybriMoE (2025) 스케줄 시뮬레이션, SMoE (2025) greedy two-pointer |
| 레이어 단위 weight 스트리밍 + 더블 버퍼링 | FlexGen, DeepSpeed-Inference (dense 모델 대상) |
| KV ↔ expert 메모리 탄력 재분배 | eLLM (2025), WiSP (2026), FluxMoE (2026) |
| 세션 간 prefix 재사용 계층화 | SGLang HiCache |

**"LRU expert 캐시"가 FreeToken의 신규 개념이라는 서술은 틀리다.** 2023년부터 있었다.
새로운 건 그걸 *어디에* 넣었느냐(6.3-②)다.

### 6.3 계층 3: FreeToken이 실제로 새로 제안한 것

논문 §6 마지막 문장이 자기 기여를 이렇게 한정한다: *"Our contribution is the integrated runtime
and the measured-bandwidth model that coordinates caching, transfer, and CPU execution within it."*

**① q* 정책** — §7에서 상술. 가장 명확한 신규 기여.

**② CUDA Graph 안에 상주하는 동적 expert 캐시** (논문 §4.1)
- dedup → residency 분류 → `q` 계산 → victim 선정 → logical ID를 physical slot ID(또는 CPU 플래그)로
  재작성까지 **MoE 레이어당 GPU 커널 하나**
- victim 선정은 **single-pass로 K개 LRU 후보**를 뽑고 앞의 `q ≤ K`만 소비 — 축출 슬롯마다 전체
  스캔하는 고전적 LRU 함정 회피. 실현된 miss 수와 무관하게 항상 1패스
- 모든 expert bank가 동일한 logical→slot 매핑을 공유해 **device-resident 인덱스 리스트 하나로
  전 bank 복사를 fused launch** (`moe/offload_cache.py`의 `_FUSED_COPY`)
- CPU 브랜치까지 `cudaLaunchHostFunc` submit/sync 노드로 **같은 그래프에 캡처**
- 기존 오프로딩 시스템은 라우팅 의존 제어가 호스트에 있어 매 MoE 레이어에서 device sync가 걸린다.
  논문: *"llama.cpp likewise cannot maintain graph execution in its hybrid mode."*

**③ Semantic anchor state checkpoint** — 개념적으로 가장 독창적
- recurrent 레이어(GDN, KDA, SWA)는 prefix 전체를 하나의 상태로 압축해 **부분 재사용이 불가능**.
  체크포인트를 몇 개만 들 수 있고, **어디 두느냐가 전부**다.
- 관찰: 에이전트 하네스는 **특수 토큰 경계에서만** 컨텍스트를 편집한다.
  OpenClaw(최신 턴 제외 thinking 제거), OpenCode(오래된 tool output을 placeholder로 치환),
  SWE-agent(마지막 n개 observation만 유지).
- 그래서 `</think>`, `</tool_call>`, `</tool_output>`, 턴 경계에 앵커링하면 편집 후 생존 확률이 높다.
- radix 트리 자체는 SGLang 것이지만 **노드에 recurrent-state 스냅샷을 매달고 그 예산을 의미
  경계에 쓴다**는 건 선행 연구가 없다.

### 6.4 부분적 신규

| 기법 | 새로운 부분 |
|---|---|
| Prefill full-layer 더블 버퍼링 | 더블 버퍼링 자체는 FlexGen/DeepSpeed. 새로운 건 **라우팅을 알기 전에 레이어 전체를 통째로 보낸다**는 것 + **prefill/decode가 슬롯 풀을 공유**해 별도 prefill 캐시도 phase handoff도 없고 prefill 잔여가 decode를 seed |
| 런타임 캐시 재구성 | eLLM/WiSP와 목표는 겹침. 새로운 건 **scheduler safe point에서 캡처된 실행 경로를 재수립** — 논문 표현으로 그들은 "passive bytes"만 옮김 |
| FTW + pin-after-fill | 순수 엔지니어링. 알려진 트릭이지만 시스템 레벨에서 명시적으로 다룬 사례는 드묾 |
| 에이전트 워크로드 평가 | **방법론 기여.** 기존 MoE 오프로딩 논문들의 single-shot·short-prompt 평가를 지적하고 실제 하네스 4종으로 다중 턴 측정 → "single-stream 벤치가 베이스라인을 과대평가한다" |

### 6.5 의도적으로 쓰지 않은 것

신규성의 위치를 확정해 준다.

- **expert prefetch / 라우팅 예측** (ProMoE, ExpertFlow, FineMoE, MoE-Infinity) — 안 씀.
  논문 논리: *"예측이 아무리 정확해도 모든 miss는 결국 PCIe 전송이라 decode 지연이 링크에 묶이고,
  호스트 연산 능력은 놀고 있다."* 즉 **"miss를 얼마나 잘 예측하나"를 "남은 miss를 어떻게
  서빙하나"로 문제를 바꾼 것**이 진짜 기여점이다.
- **정확도 트레이드** (HOBBIT 저정밀 fetch, SiDA/SMoE expert skip, Pre-gated MoE 라우터 파인튜닝)
  — 안 씀. 라우팅된 연산은 exact, 모델은 무수정. GPU/CPU 부분합을 정확히 병합.
- **Speculative decoding** — 아예 없음. Ollama에는 있다.

---

## 7. q* 정책 원리

### 문제

토큰 하나, MoE 레이어 하나. 라우터가 전문가 12개를 고르고 8개는 GPU 캐시에 있다.
남은 **miss 4개를 어떻게 처리할 것인가.**

- (A) PCIe로 가져와 GPU에서 계산 — 느린 링크. 대신 캐시에 남아 다음 토큰에 재사용
- (B) 호스트 RAM에서 CPU가 직접 계산 — 전송 없음. 대신 느리고 캐시에 아무것도 안 남음

기존 답: llama.cpp/Ollama는 결정 자체를 안 함(로드 시점 고정). KTransformers는 항상 (B)라서
**PCIe가 놀아도 CPU만 씀**. 순수 offload는 항상 (A)라서 **CPU 코어와 잔여 호스트 대역폭이 놂**.

### 통찰

**PCIe 전송과 CPU 계산은 둘 다 같은 호스트 DRAM을 읽는다.** DMA가 링크를 포화시키면
호스트 대역폭 중 `B_P`가 이미 소비된 상태다. CPU가 쓸 수 있는 건 `B_H`가 아니라 **`B_H − B_P`**.
기존 하이브리드 연구들이 놓친 지점이며, CPU 몫을 `B_H`로 잡으면 항상 과대평가하게 된다.

### 유도 (논문 §3.2, 식 2–4)

전문가 하나 크기 `S`, miss `m`개 중 `q`개를 PCIe로:

```
T_fill = q·S / B_P                    (PCIe 브랜치)
T_cpu  = (m−q)·S / (B_H − B_P)        (CPU 브랜치, 잔여 대역폭)
```

두 브랜치는 **동시 실행**되므로 레이어 지연은 느린 쪽. 최적점은 둘이 같아지는 지점:

```
    q        B_P                               B_P
  ───── = ─────────       →        q* ≈ m · ─────
  m − q   B_H − B_P                           B_H
```

**`q*/m = B_P/B_H`** — 두 대역폭의 비율 하나가 전부다.
`B_H → B_P`면 `q* → m`으로 순수 fetch에 수렴하므로 별도 분기 없이 한 식이 모든 하드웨어를 덮는다.
FreeToken은 `q*`를 정수로 반올림하고 **최소 1개는 항상 fetch**해서 캐시가 계속 데워지게 한다.

### 논문 Table 1 하드웨어에 대입

| 머신 | B_P (GB/s) | B_H (GB/s) | q*/m | 해석 |
|---|---|---|---|---|
| 5090 desktop | 49.0 | 53.8 | **0.91** | 91% PCIe — CPU가 보탤 게 거의 없음 |
| 5090 server | 52.7 | 77.3 | **0.68** | 68% PCIe, 32% CPU |
| 4090 | 25.1 | 63.2 | **0.40** | 링크가 느려 CPU가 절반 이상 |
| 3090 | 25.3 | 56.7 | **0.45** | 〃 |
| PRO 6000 | 51.5 | 178 | **0.29** | 8채널 DDR5 → CPU가 주력 |
| 4060 laptop | 11.8 | 47.5 | **0.25** | PCIe ×8 → 75%를 CPU가 |

**같은 RTX 5090인데 desktop 0.91, server 0.68.** GPU 실리콘은 동일하고 호스트만 다르다
(듀얼채널 소비자용 DDR5 vs 다채널 서버). 스펙시트로는 못 뽑는 값이라 `ft bench bw`가
배포 텐서 shape 그대로 실측한다. 논문 §5.3에서 llama.cpp가 desktop으로 옮길 때 성능의 20%를
잃고 FreeToken은 4%만 잃는 이유가 이 표에 있다.

---

## 8. Ollama 흡수 가능성

### 8.1 진단: CUDA 경로에서 실행 루프를 소유하지 않는다

```
find ml -type f
  → ml/backend.go, ml/device.go, ml/path.go (+ 떠도는 .cu 파일 하나)
```

자체 GGML Go 엔진이 이 트리에 없다. `model/`에도 `parsers/`, `renderers/`만 남았다. 추론 경로는 둘:

- **CUDA/ROCm/Vulkan** → `llama-server` **서브프로세스** (`llm/llama_server.go:348 startLlamaServer`)
- **Apple** → `x/mlxrunner` (in-process Go 엔진, 21k LOC)

NVIDIA 경로에서 Ollama가 가진 통제 수단은 **CLI 플래그뿐**이다. FreeToken의 신규 기법은 전부
MoE 레이어 내부·스케줄러 내부·CUDA Graph 캡처 내부에 산다. 프로세스 경계 밖에서 손댈 수 없다.

아이러니: Ollama가 실행 루프를 소유하는 유일한 경로(MLX/Apple)는 **통합 메모리라 PCIe 오프로딩
문제 자체가 거의 없다.** 통제권이 있는 곳엔 문제가 없고, 문제가 있는 곳엔 통제권이 없다.

### 8.2 티어별 판정

**T0 — Semantic anchor: 기계는 이미 있고 정책만 바꾸면 됨 (MLX 한정)**

Ollama MLX 러너에 recurrent state 스냅샷 인프라가 이미 완성돼 있다:

```
x/mlxrunner/cache/cache.go       Snapshot / PrepareSnapshots(offsets) / Restore / Merge / Split
x/mlxrunner/cache/recurrent.go   RecurrentCache — conv state + delta state 스냅샷
x/mlxrunner/prefix_cache.go:324  스냅샷을 prefix 트라이 노드에 부착
x/models/nn/recurrent.go:52      WithSnapshotSplits — 지정 오프셋에서 스캔을 분할
```

`PrepareSnapshots(offsets []int)` — **임의 오프셋에 체크포인트 예약 가능**. FreeToken의
semantic anchor가 필요로 하는 기계 그대로다. 빠진 건 **배치 정책**뿐:

```go
// x/mlxrunner/pipeline.go:139
const snapshotInterval = 8192
for offset := snapshotInterval; offset < len(inputs); offset += snapshotInterval {
    snapshotOffsets = append(snapshotOffsets, offset)
}
const preThinking = 4
if end := len(inputs) - preThinking; end > 0 {
    snapshotOffsets = append(snapshotOffsets, end)
}
```

**8192 토큰 고정 주기 + 프롬프트 끝.** 논문이 공격하는 게 정확히 이것 —
*"checkpoints are sparse... any checkpoint taken after the modified position becomes invalid."*
에이전트가 30k 지점의 thinking 블록을 지우면 24576 체크포인트만 살아남아 5천 토큰 이상을 재prefill한다.

필요한 변경은 이 함수 하나 + 모델별 앵커 토큰 ID 테이블(특수 토큰은 `x/tokenizer`,
`model/parsers`에 이미 있음). 주기적 스냅샷은 백스톱으로 유지.
**난이도 작음. 아키텍처 변형 0.** GGUF/CUDA 경로엔 해당 없음.

**T1 — expert 단위 오프로딩 플래그: 배관만**
- `api.Options` 필드 추가 + `llm/llama_server.go` 파라미터 조립 + **`server/sched.go`/`fs/ggml`
  메모리 추정기가 "MoE 텐서는 CPU에 남는다"를 인지** (이게 실제 작업량. 안 하면 OOM/과소적재)
- 얻는 것: 레이어 → 텐서 단위 정적 배치. 동적 LRU는 못 얻음
- 난이도 작음~중간. 아키텍처 변형 없음

**T2 — LRU expert 캐시 + q*: Ollama가 결정할 수 없는 영역**
- ggml은 텐서→디바이스 배치를 **그래프 빌드 시점에 확정**(`ggml_backend_sched`).
  "매 토큰 슬롯 재배치"와 근본적으로 충돌
- `ggml_mul_mat_id`에 슬롯 캐시·residency 테이블·victim 선정을 넣어야 함
- `llama/compat/001-llama-cpp-hooks.patch` 패치 레이어가 있지만 이 규모를 패치로 유지하는 건
  비현실적(`LLAMA_CPP_VERSION` 핀 갱신마다 리베이스)
- **업스트림 llama.cpp 작업이지 Ollama 작업이 아님**

**T3 — CUDA Graph 상주 동적 캐시: 아키텍처 방향을 되돌려야 함**
- 목적이 "MoE 레이어마다의 host sync 제거"인데 그러려면 forward 루프·그래프 캡처·커널을 소유해야 함
- Ollama는 정확히 반대 방향으로 이동했다(자체 GGML 엔진 폐기 → llama-server 위임)
- 사실상 vLLM/SGLang급 엔진 재구축

### 8.3 대안: 세 번째 러너로 붙이기

아키텍처적으로는 가장 깔끔하다. 추상화 지점이 이미 있다:

```
llm/server.go:59              type LlamaServer interface { ... 18개 메서드 }
x/mlxrunner/client.go:495     var _ llm.LlamaServer = (*Client)(nil)      ← 선례
server/sched.go:588           llama, err = mlxrunner.NewClient(...)        ← 분기점
```

`ft serve`를 스폰하고 HTTP로 말하면 된다. FreeToken이 OpenAI/Anthropic 호환 서버를 제공하므로
`Completion`/`Chat`/`Embedding` 매핑은 얕고, `Load`/`MemorySize`/`VRAMByGPU`는
`/health`·`/v1/stats`·`/v1/cache/status`로 채울 수 있으며 런타임 캐시 재조정까지 덤으로 따라온다.

**막는 건 아키텍처가 아니라 배포다:**

| 충돌 | 내용 |
|---|---|
| 의존성 | Python 3.10+, torch 2.11 고정, CUDA 13 툴체인(nvcc JIT), flashlib/flashinfer/sglang-kernel. **단일 바이너리 + 런타임 백엔드 dlopen** 철학과 정면 충돌 |
| 모델 포맷 | Ollama 생태계 전체가 GGUF 전제 — `convert/`, `x/create/`, `server/quantization.go`, 레지스트리, Modelfile |
| 하드웨어 | Linux + NVIDIA 전용. macOS/AMD/Windows 사용자에겐 무의미 |
| 정체성 | 결과물이 "Ollama"가 아니라 **"Ollama가 실행하는 별개 스택"** |

`CONTRIBUTING.md`의 *"Issues that may not be accepted: changes that add significant friction to
the user experience / create a large future maintenance burden"* 에 정면으로 걸린다.

### 8.4 요약

| 기법 | 흡수 가능? | 필요한 변형 |
|---|---|---|
| **Semantic anchor** | **예 (MLX)** | `pipeline.go:139` 정책 교체. **기계는 이미 있음** |
| expert 단위 정적 배치 | 예 | 플래그 배관 + 메모리 추정기 보정 |
| LRU expert 캐시 | 사실상 아니오 | ggml 배치 모델 수술 — 업스트림 문제 |
| q* 하이브리드 분할 | 사실상 아니오 | 상동 + 대역폭 프로파일링 인프라 |
| CUDA Graph 상주 캐시 | 아니오 | 실행 루프 소유권 회복 = 엔진 재구축 |
| prefill 더블 버퍼링 | 아니오 | 상동 |
| 런타임 VRAM 재분배 | 아니오 | llama-server가 재시작 없는 재구성을 지원해야 함 |

**한 줄:** 정책 하나(semantic anchor)는 거의 공짜로 흡수 가능하고 — Ollama가 이미 스냅샷 인프라를
다 만들어놨기 때문이다 — 나머지는 전부 **Ollama가 llama-server에 위임하기로 한 결정의 반대편**에
있다. 심각한 변형이 필요한 게 아니라 **그 위임 결정 자체를 되돌려야** 한다.

---

## 9. 확인된 사실 / 미검증 사항

### 확인됨 — 업스트림 llama.cpp에는 expert 오프로딩 플래그가 이미 있다

핀된 `b10488`을 직접 받아 확인:

```
/tmp/lcpp/common/arg.cpp:2715   {"-ot",    "--override-tensor"}
/tmp/lcpp/common/arg.cpp:2721   {"-cmoe",  "--cpu-moe"}
/tmp/lcpp/common/arg.cpp:2728   {"-ncmoe", "--n-cpu-moe"}, "N"
```

**즉 §2의 갭은 llama.cpp 문제가 아니라 순수하게 Ollama 배관 문제다.**
Ollama가 핀한 바로 그 버전에 기능이 있는데 `llm/llama_server.go`가 전달하지 않는다.

### 미검증 — 성능 수치는 아무도 재현하지 않았다

이 리포트의 모든 배수·tok/s·miss rate는 **논문에서 읽은 값**이다. 이 세션에서 벤치마크를
돌린 적이 없다. 다음이 남아 있다:

1. 동일 GGUF·동일 머신에서 `ollama serve` vs `llama-server -ncmoe N` 직접 비교
2. Ollama 단독 대비 FreeToken 배수 (논문에 명시 없음)
3. `-ncmoe`를 노출했을 때 miss rate가 실제로 얼마나 개선되는지 (정적 배치의 질 개선일 뿐
   동적 LRU는 아니므로 개선폭이 제한적일 것으로 예상되나 미측정)

**측정 없이 이 수치를 인용해 업스트림에 제안하지 말 것.**

---

## 10. 후속 조치 권고

### A. 먼저 측정

같은 모델·같은 하드웨어에서 `ollama serve` vs `llama-server -ncmoe N`. 이게 있어야 그 다음이 선다.

### B. Ollama — Performance 이슈로, 프레이밍 주의

`CONTRIBUTING.md`가 꼽는 ideal issues에 `Performance`가 있다. 단:

- ❌ "`--n-cpu-moe` 플래그를 노출해주세요" — Ollama는 노브 추가를 싫어한다
  (*"new features add surface area"*). `-ngl auto` 자동화가 그들의 철학
- ✅ **"MoE 모델이 VRAM을 초과할 때 Ollama가 llama-server 직접 실행보다 느리다"** — 문제 리포트로.
  해법(`-ncmoe` 자동 계산, 메모리 추정기가 MoE 텐서 인지)은 제안으로만

`CONTRIBUTING.md` proposal tips: *"Explain the problem you are trying to solve, not what you are
trying to do."*

### C. Ollama MLX 스냅샷 — 이슈보다 PR

위치가 특정됐고(`x/mlxrunner/pipeline.go:139`) 변경이 작다. non-trivial 판단이 애매하니
이슈를 먼저 열고 PR을 붙이는 게 안전하다. GDN을 쓰는 `qwen3_5_moe`부터 효과가 크다.

### D. llama.cpp — 이슈 내지 말 것

1. 내려던 것(`-ncmoe` 계열)은 **이미 있다** (§9). "이미 있음"으로 닫힌다
2. 남은 건 `ggml_backend_sched` 수술이라 이슈 한 장으로 될 규모가 아니고,
   `CONTRIBUTING.md:34` *"Features must begin with an issue... let interest accumulate"* 라 해도
   구현 의지 없는 논문 소개는 stale로 죽는다
3. **`CONTRIBUTING.md:25` — AI로 이슈/PR 본문을 작성하는 것이 명시적으로 금지돼 있다.**
   낸다면 반드시 본인이 직접 작성해야 한다

### E. 이해상충 주의

FreeToken은 flashml.ai에서 데스크톱 앱을 배포하는 팀의 논문이다. 경쟁 프로젝트 이슈 트래커에
"이 팀 기술을 도입하라"를 올리면 홍보로 읽힐 소지가 있다. **본인 측정으로 프레이밍해야 한다.**
