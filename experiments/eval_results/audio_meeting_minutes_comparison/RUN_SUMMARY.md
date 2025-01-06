# STT 파일럿 실행 결과

실행일: 2026-04-25

## 실행 범위

공개 AMI Meeting Corpus의 `edinburghcstr/ami` Hugging Face dataset에서 `test/ihm` split을 사용했다. 너무 짧은 발화는 WER가 불안정하므로 5초 이상 발화 3개만 골라 파일럿으로 실행했다.

실행 조건:

| 조건 | 값 |
|---|---|
| dataset | AMI |
| split/config | `test` / `ihm` |
| sample filter | `min_duration_sec=5`, `limit=3` |
| baseline pipeline | WhisperX ASR |
| target model | `large-v3` |
| pilot model | `tiny.en` |
| compute type | `int8` |
| batch size | 4 |

## 결과 요약

### WhisperX segmented ASR pilot

| 모델 | n | mean WER | mean CER | mean MER | mean WIL | mean RTF | mean wall-clock | peak RAM | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WhisperX `large-v3` | 3 | 0.1671 | 0.1429 | 0.1671 | 0.1671 | 0.1439 | 0.7310s | 4.48GB | 0 |
| WhisperX `tiny.en` | 3 | 0.1823 | 0.1429 | 0.1803 | 0.2067 | 0.0433 | 0.2213s | 2.01GB | 0 |

해석:

- `large-v3`가 `tiny.en`보다 WER 기준 약간 좋았다: 0.1671 vs 0.1823.
- `tiny.en`은 속도가 훨씬 빨랐다: RTF 0.0433 vs 0.1439.
- 두 조건 모두 RTF가 1보다 작아, 이 작은 발화 샘플에서는 실시간보다 빠르게 처리됐다.
- 이번 subset에는 숫자/고유명사가 없어 `numeric_accuracy`, `entity_recall`은 `null`이다.
- 이번 실행은 ASR segment 평가만 수행했다. DER/JER/cpWER는 full meeting audio와 reference diarization RTTM을 붙인 다음 계산해야 한다.

### Gemma4 E2B-it audio pilot

Gemma4 오디오 입력은 Google 문서의 방식에 맞춰 `transformers` main 브랜치(`5.7.0.dev0`)로 올린 뒤 실행했다. PyPI `transformers 4.57.6`에는 `gemma4` processor/model 코드가 없어 `AutoProcessor`가 실패했다.

AMI 5초 이상 발화 3개 기준 평균 비교:

| 모델 | n | mean WER | mean CER | mean RTF | mean wall-clock | peak VRAM | peak RAM | failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WhisperX `large-v3` | 3 | 0.1671 | 0.1429 | 0.1439 | 0.7310s | 0.086GB | 4.48GB | 0 |
| WhisperX `tiny.en` | 3 | 0.1823 | 0.1429 | 0.0433 | 0.2213s | 0.086GB | 2.01GB | 0 |
| Gemma4 E2B-it full | 3 | 0.1391 | 0.0904 | 6.4997 | 32.9725s | 5.65GB | 11.66GB | 0 |
| Gemma4 E2B-it 4bit | 1 | N/A | N/A | N/A | N/A | N/A | N/A | 1 |
| Gemma4 E2B-it 8bit | 1 | N/A | N/A | N/A | N/A | N/A | N/A | 1 |

동일 AMI 발화 1개 기준 상세 비교:

| 조건 | WER | CER | RTF | wall-clock | peak VRAM | peak RAM | failure | 해석 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| WhisperX `large-v3` | 0.2105 | 0.1875 | 0.1839 | 0.9455s | 0.086GB | 4.48GB | false | 빠름. 기준 STT 파이프라인 |
| Gemma4 E2B-it full | 0.1053 | 0.0729 | 10.6335 | 54.6560s | 6.15GB | 11.63GB | false | 정확도는 좋지만 CPU offload로 매우 느림 |
| Gemma4 E2B-it 4bit | N/A | N/A | N/A | N/A | N/A | N/A | true | RTX 2080 8GB에서 quantized model 일부가 CPU/disk로 dispatch되어 로드 실패 |
| Gemma4 E2B-it 8bit | N/A | N/A | N/A | N/A | N/A | N/A | true | `bitsandbytes`/`transformers main` 조합에서 `Int8Params` 인자 호환성 오류 |

Gemma full 출력:

| reference | hypothesis |
|---|---|
| `YOU OPEN THE WINDOW YOU READ THROUGH IT YOU MIGHT CLICK ON YOU KNOW CLOSE IT AGAIN STRAIGHT AWAY` | `you open the window you read through it you might click on you know close it against straight away yeah` |

해석:

- 3개 segment 평균에서도 Gemma4 E2B-it full이 WhisperX large-v3보다 WER/CER가 낮았다.
- 하지만 속도는 Gemma full이 훨씬 느렸다. 평균 RTF 6.50은 실시간 처리에 부적합하다.
- 4bit/8bit 양자화는 현재 RTX 2080 환경에서 성공하지 못했다. 추가 재시도에서 4bit는 강제 단일 GPU 배치로 모델 로딩까지는 성공했지만 generate 단계에서 dtype 처리 오류가 났고, 8bit는 강제 단일 GPU 배치 시 CUDA OOM이 발생했다.
- 따라서 현재 환경의 실용적 기준선은 WhisperX large-v3이고, Gemma full은 정확도 가능성은 있으나 운영성에서 불리하다.

## 산출물

| 파일 | 설명 |
|---|---|
| `stt_eval_results_largev3_segments.csv` | `large-v3` 요약 지표 CSV |
| `stt_eval_results_largev3_segments.jsonl` | `large-v3` reference/hypothesis 포함 상세 결과 |
| `stt_eval_results_largev3_segments.summary.json` | `large-v3` 평균 지표 |
| `stt_eval_results_tiny_segments.csv` | `tiny.en` 요약 지표 CSV |
| `stt_eval_results_tiny_segments.jsonl` | `tiny.en` reference/hypothesis 포함 상세 결과 |
| `stt_eval_results_tiny_segments.summary.json` | `tiny.en` 평균 지표 |
| `stt_eval_results_gemma4_e2b_full.csv` | Gemma4 E2B-it full 1개 샘플 결과 |
| `stt_eval_results_gemma4_e2b_full_segments.csv` | Gemma4 E2B-it full 3개 샘플 결과 |
| `stt_eval_results_gemma4_e2b_4bit.csv` | Gemma4 E2B-it 4bit 실행 실패 기록 |
| `stt_eval_results_gemma4_e2b_8bit.csv` | Gemma4 E2B-it 8bit 실행 실패 기록 |
| `stt_eval_results_gemma4_e2b_4bit_retry.csv` | 강제 단일 GPU 배치 4bit 재시도 실패 기록 |
| `stt_eval_results_gemma4_e2b_8bit_retry.csv` | 강제 단일 GPU 배치 8bit 재시도 실패 기록 |
| `stt_eval_results_gemma_vs_whisperx_pilot.csv` | 동일 샘플에서 WhisperX large-v3와 Gemma 조건을 모은 비교표 |
| `stt_eval_summary_pilot.csv` | 파일럿 평균 지표 비교표 |

## 현재 한계

1. Gemma4 E2B-it full과 WhisperX는 3개 segment만 평가했다. 샘플 수가 작아 통계적 결론으로 쓰기에는 부족하다.
2. Gemma4 E2B-it 4bit/8bit 양자화는 현재 환경에서 실패했다.
3. Pyannote diarization 평가(DER/JER)는 아직 수행하지 않았다. 이번 Hugging Face AMI row는 segmented utterance 기준이라 full-meeting diarization benchmark에는 부족하다.
4. `whisperx` 설치 과정에서 `torch`, `transformers`, `numpy`가 교체됐고, Gemma4 지원을 위해 `transformers` main을 설치하면서 `huggingface-hub`가 `whisperx` 요구사항과 충돌한다는 경고가 생겼다. `whisperx` import는 확인했지만, 장기적으로는 별도 conda/venv로 분리하는 편이 안전하다.

## 다음 실행 단위

보고서용 결과로 만들려면 다음을 추가 실행한다.

1. AMI/ICSI full meeting 3개를 내려받아 WhisperX+Pyannote diarization을 수행한다.
2. reference RTTM/XML을 변환해 DER/JER/cpWER를 계산한다.
3. 가능하면 VRAM 16GB 이상 GPU에서 4bit 양자화를 재실행한다.
4. 같은 CSV 스키마에 Gemma 결과를 추가하고, `WER/CER/DER/RTF/VRAM/failure` 기준으로 비교한다.
