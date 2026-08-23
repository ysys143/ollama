# FreeToken vs Ollama 비교 분석

- FreeToken 소스: `/tmp/freetoken` (github.com/FlashML-org/FreeToken, Apache-2.0, 단일 커밋 `0ab982f`, 485 파일 / Python ~71k LOC)
- Ollama 소스: `/home/user/ollama` (HEAD `fb30760`, llama.cpp `b10488` 핀, MLX `27fec90` 핀)
- 논문: arXiv 2608.16157 "FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution"
  (Berkeley Sky/BAIR 중심 + MIT Song Han 공저 — Stoica, Zaharia, Keutzer, Xu 등)

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
- `--n-cpu-moe` / `--override-tensor` 같은 expert 단위 플래그는 **노출하지 않음**(전 트리 grep 결과 0건).
  llama.cpp가 지원해도 Ollama 사용자는 못 쓴다.
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

## 3. 그 외 FreeToken 고유 메커니즘

1. **Prefill full-layer double buffering** — 8192토큰 청크당 1.19–1.22초 = 64.4GB expert pool을
   52.7GB/s로 한 번 흘리는 시간 = PCIe 5.0 ×16 실효 상한. 계산이 전송 뒤에 완전히 숨는다.
   두 번째 버퍼를 끄면 4k에서 -19%, 8k -25%, 16k -26%.
2. **Semantic anchor / tool-call anchor** — `core.py:62 toolcall_anchor_len`,
   `scheduler/cache.py:144 snapshot_toolcall_anchor`. 툴콜 지점에서 GDN(선형 어텐션) 상태와
   SWA 윈도우를 얼려둔다. 에이전트가 클라이언트에서 툴 결과/thinking 블록을 고쳐 다시 보내도
   앵커 이후만 fork되므로 수천 토큰 재prefill을 피한다. **Ollama/llama.cpp에는 대응물이 없다**
   (llama-server 슬롯 prompt cache는 공통 prefix까지만).
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

## 6. Ollama 트리에 주는 시사점

1. **expert 단위 오프로딩 노출이 없다.** 현재 트리는 `-ngl`만 넘긴다.
   llama.cpp의 `--n-cpu-moe`/`--override-tensor`만 옵션으로 뚫어도 MoE 오프로딩 시나리오는 개선된다
   (동적 LRU는 아니지만 정적 배치의 질이 올라간다).
2. **에이전트 컨텍스트 편집에 대한 대응이 없다.** 툴콜 턴마다 prefix가 깨지는 비용은
   `x/mlxrunner/prefix_cache.go`가 있는 MLX 경로 외에는 llama-server 슬롯에 의존한다.
   FreeToken의 tool-call anchor는 값싸게 흉내낼 여지가 있는 아이디어다.
3. **런타임 메모리 재분배가 없다.** 컨텍스트 압박 시 모델 재로드가 유일한 수단이다.
4. 반대로 FreeToken이 Ollama를 못 따라가는 축(멀티 백엔드, 모델 배포, speculative decoding,
   멀티모델 스케줄링)은 구조적이며 단기간에 좁혀지지 않는다.
