# 실험 시도 이력 (attempts)

이 디렉토리는 이번 실험(성향 데이터 조합에 따른 R&R 매칭 품질 검증)의 과거 실행 시도를 보존합니다.
현재 진행 중인 실험 결과는 상위 디렉토리(`backend_gemini/`, `backend_gemma4_api/`)에 저장됩니다.

## 시도별 요약

### attempt1_no_judge_gemini_only
- **실행 시각**: 2026-04-21 19:38~19:48
- **구성**: gemini + gemma4-api, iter=3, max_rounds=1, **judge 없음**
- **결과**:
  - Gemini 6 run: 전부 정상 (planning ≈ 0.36~0.42)
  - Gemma 6 run: 전부 실패 (`max_tokens=65000 > max_model_len=8192`로 400 에러)
- **교훈**: Gemma 서버의 context 상한이 너무 작았음 → attempt2에서 32k로 증가

### attempt2_gemma_context_400
- **실행 시각**: 2026-04-21 20:58 (1 run만에 실패 확인)
- **구성**: Gemma만 재실행 (server.max_model_len=32768), iter=3
- **결과**: Gemma 6 run 전부 실패 (여전히 400 에러)
- **원인**: `max_model_len=32768`이어도 `max_tokens=65000 > max_model_len`이라 vLLM이 거부
- **교훈**: 서버 컨텍스트가 아니라 **요청 max_tokens 자체**가 문제 → attempt3에서 프록시로 캡

### attempt3_gemma_proxy_no_judge
- **실행 시각**: 2026-04-21 21:05~22:13 (5/6 진행 후 중지)
- **구성**: Gemma만, proxy(cap=8000) 경유, judge 없음
- **결과**: Gemma 5 run 정상 (planning ≈ 0.29~0.32)
- **중단 사유**: 프로토콜의 LLM-as-a-Judge(Structure/Assignment/Debate) 누락 확인
- **교훈**: runner가 judge를 호출하지 않음 + final_wbs/debate_log 스냅샷을 저장하지 않아 사후 judge 불가 → attempt4에서 runner에 judge + snapshot 저장 추가

### attempt4 (현재 진행 중)
- **위치**: 상위 디렉토리 (`backend_gemini/`, `backend_gemma4_api/`)
- **구성**: gemini + gemma4-api, iter=3, max_rounds=1, **judge(Gemini 단독) 포함**
- runner가 각 run마다
  1. `compute_all_metrics()` — 14종 지표
  2. `evaluate_wbs()` — Structure/Assignment/Debate 3차원 judge
  3. `snapshots/*.json` — final_wbs + debate_log 원본 저장
- 완료 예상: 약 95분 (Gemini ~10분 + Gemma ~85분)
