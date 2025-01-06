# Gemma4 E2B-it 오디오 입력 STT vs STT+Pyannote 비교

작성일: 2026-04-25

## 1. 비교 목적

비교 대상은 두 파이프라인이다.

| 구분 | 파이프라인 | 기대 산출물 |
|---|---|---|
| A. Gemma4 E2B-it 직접 오디오 | 오디오 입력 -> 모델이 transcript/화자별 발화 추출 | STT transcript, 가능하면 화자 라벨 |
| B. 기존 STT+Pyannote | WhisperX STT -> alignment -> Pyannote diarization -> 화자별 transcript | 화자 라벨 포함 transcript |

현재 비교의 1차 목적은 회의록 품질이 아니라 STT 성능이다. 즉 모델이 정답 transcript를 얼마나 정확히 맞췄는지, 화자를 얼마나 잘 구분했는지, 숫자/이름/결정사항 같은 핵심 사실을 얼마나 보존했는지, 그리고 같은 오디오를 처리하는 데 걸린 시간과 자원 사용량이 어떤지를 본다.

현재 저장소에서 확인된 실제 구현은 B에 가깝다. `STT_DONGK.py`는 WhisperX `large-v3`로 STT를 수행하고, WhisperX alignment 후 `DiarizationPipeline`으로 화자를 분리해 `[SPEAKER] : text` 형식의 transcript를 만든다. Gemma 요약 단계는 주석 처리되어 있고, 프론트엔드의 음성 STT UI도 실제 업로드 API와 연결되어 있지 않다.

따라서 이 문서는 실제 오디오 실험 결과가 아니라, 공개 데이터셋과 현재 구현 상태를 기준으로 한 STT 중심 비교 설계다. 실제 Gemma 오디오 transcript와 STT+Pyannote transcript가 들어오면 아래 지표표에 그대로 채점하면 된다.

## 2. 사용 데이터

### 2.1 저장소 내 예비 데이터

현재 저장소에서 바로 사용 가능한 기준 데이터:

| 파일 | 용도 | 통계 |
|---|---|---|
| `sample_data/sample_meeting_transcript.txt` | 정답에 가까운 기준 회의 transcript | 20개 발화, 6명 화자, 5,702자, 숫자/일정/성과 목표 27개 |
| `sample_data/sample_meeting_no_schedule.txt` | 일정/숫자 정보가 제거된 난이도 변형 | 20개 발화, 6명 화자, 5,396자, 숫자 정보 1개 |

`sample_meeting_transcript.txt`는 회의록 평가에 적합하다. 발화자별 역할, 정량 수치, 결정사항, 담당자, 다음 액션이 모두 포함되어 있다. 특히 회의록 품질을 가르는 핵심 정보는 다음과 같다.

| 분류 | 기준 정보 |
|---|---|
| 회의 주제 | P마켓 수지점 매출 하락 원인 분석 및 P-Reset 2026 추진 |
| 핵심 문제 | 매출 하락, 기존 고객 이탈, 신규 주민 미유입, 낡은 오프라인 진열, 배송 기준, 앱/DB 미연동 |
| 주요 수치 | 2024년 195억 원 -> 작년 161억 원, 17% 이상 하락, 유입 인구 35만 명, 20% 증가, 신규 주민 7만 명, 배송 불만 45%, LightGBM F1 0.83 이상 |
| 담당자 | 이동헌: 데이터/타겟팅, 박선민: 프로모션/리뉴얼, 이주성: 운영 안정화, 장선열: IT/DB 통합, 윤수빈: 모니터링/문서화 |
| 결정사항 | 프로젝트명 `P-Reset 2026`, 내일 오전 10시까지 세부 Task 보고 |

### 2.2 웹 공개 데이터셋 사용안

보고서/논문 형태를 생각하면 저장소 내 합성 샘플만 쓰기보다 공개 벤치마크를 함께 쓰는 편이 낫다. 권장 조합은 `AMI/ICSI + QMSum`이다.

| 데이터셋 | 사용 목적 | 장점 | 주의점 |
|---|---|---|---|
| AMI Meeting Corpus | 오디오 입력, STT, 화자분리, 회의록 작성 end-to-end 평가 | 100시간 규모 회의 녹음, close-talk/far-field 등 다양한 신호, orthographic transcription 및 annotation 제공. CC BY 4.0 공개 | 영어 회의 데이터. 다운로드 용량이 큼 |
| ICSI Meeting Corpus | 자연 회의 오디오 기반 robustness 평가 | 약 70시간 회의 녹음, orthographic transcription, dialogue acts, 일부 summarization/topic annotation 제공. CC BY 4.0 공개 | 회의당 headset mix 약 120MB, 전체 다운로드는 큼 |
| QMSum | STT 이후 회의록/요약 downstream 참고 평가 | 232개 회의, 1,808개 query-summary pair, transcript에 speaker 정보 포함, MIT license GitHub 공개 | 원본 오디오 평가는 불가. STT 단계 성능 비교에는 직접 사용하지 않음 |

출처:

- AMI/ICSI corpora: https://groups.inf.ed.ac.uk/ami/
- AMI Corpus: https://groups.inf.ed.ac.uk/ami/corpus/
- ICSI download: https://groups.inf.ed.ac.uk/ami/icsi/download/
- QMSum: https://github.com/Yale-LILY/QMSum

실험 구성은 다음처럼 잡는 것이 가장 깔끔하다.

| 평가 층위 | 데이터셋 | 비교 조건 | 산출 지표 |
|---|---|---|---|
| 오디오 -> transcript | AMI 또는 ICSI 소량 subset | Gemma 직접 오디오 STT vs WhisperX STT | WER, CER, MER, WIL, numeric/entity recall |
| 오디오 -> 화자별 transcript | AMI 또는 ICSI 소량 subset | Gemma 화자 라벨 vs Pyannote diarization | DER, JER, speaker count error, speaker-attributed WER |
| 속도/자원 | AMI 또는 ICSI 동일 subset | 같은 오디오 길이에서 양쪽 처리 | RTF, wall-clock latency, GPU VRAM, CPU/RAM, 실패율 |
| downstream 참고 | QMSum 또는 로컬 샘플 | STT 결과를 요약/WBS에 넣었을 때 | summary factuality, action item recall, WBS faithfulness |

실제 다운로드는 전체가 아니라 subset으로 시작한다. 예를 들어 AMI 3개 회의 또는 ICSI 3개 회의만 내려받아도 데모/보고서용으로 충분하고, 이후 전체 실험으로 확장할 수 있다.

## 3. STT 평가 지표

### 3.1 정답 transcript 대비 정확도

| 지표 | 정의 | 해석 |
|---|---|---|
| WER | Word Error Rate = substitutions, deletions, insertions / reference words | 영어 AMI/ICSI의 기본 STT 정확도 지표. 낮을수록 좋음 |
| CER | Character Error Rate | 한국어/숫자/고유명사처럼 띄어쓰기 영향이 큰 경우 보조 지표로 사용 |
| MER | Match Error Rate | WER보다 insertion/deletion 편향을 줄여 보는 보조 지표 |
| WIL | Word Information Lost | 의미 정보 손실 정도를 보는 보조 지표 |
| Segment WER | 발화 segment 단위 평균 WER | 긴 회의 전체 평균이 특정 구간 오류를 숨기는 문제를 완화 |
| Keyword/Entity Recall | 사람 이름, 조직명, 제품명, 일정, 금액, 수치 중 정답 대비 맞춘 비율 | 회의록/WBS에 중요한 사실 보존 평가 |
| Numeric Accuracy | 숫자 표현이 정답과 같은 비율 | `195억`, `17%`, 날짜/시간 같은 회의 핵심 수치 왜곡 확인 |

보고서 메인 표에는 `WER`, `CER`, `Numeric Accuracy`, `Entity Recall`을 넣고, 부록에는 `MER`, `WIL`, `Segment WER`를 두면 충분하다.

### 3.2 화자분리 및 화자 귀속

| 지표 | 정의 | 해석 |
|---|---|---|
| DER | Diarization Error Rate = missed speech + false alarm + speaker confusion | 화자분리 표준 지표. 낮을수록 좋음 |
| JER | Jaccard Error Rate | 화자별 overlap 품질을 보는 보조 지표 |
| Speaker Count Error | 예측 화자 수 - 실제 화자 수 | 회의 참가자 수 추정 실패 확인 |
| Speaker Attribution Accuracy | 발화 텍스트가 올바른 speaker에 붙은 비율 | R&R 배정과 회의록 책임 소재에 직접 중요 |
| cpWER | concatenated minimum-permutation WER | 화자 이름 permutation을 고려한 화자별 transcript WER |
| Overlap Robustness | 겹침 발화 구간에서의 DER/WER | 실제 회의 환경 robustness 확인 |

Gemma 직접 오디오가 화자 라벨을 안정적으로 내지 못하면, 화자분리 지표는 `N/A`가 아니라 실패로 따로 기록한다. 회의록/R&R 용도에서는 화자 정보 부재 자체가 품질 결함이기 때문이다.

### 3.3 속도, 자원, 운영성

| 지표 | 정의 | 해석 |
|---|---|---|
| RTF | Real Time Factor = 처리 시간 / 오디오 길이 | 1보다 작으면 실시간보다 빠름 |
| Wall-clock Latency | 파일 업로드 시작부터 transcript 완료까지 초 단위 시간 | 사용자 체감 속도 |
| Time to First Text | 첫 transcript chunk가 나오기까지 걸린 시간 | 스트리밍/인터랙티브 UX에 중요 |
| Peak GPU VRAM | 처리 중 최대 GPU 메모리 | 로컬 배포 가능성 판단 |
| Peak RAM | 처리 중 최대 시스템 메모리 | 서버 운영 비용 판단 |
| Failure Rate | 전체 파일 중 실패/timeout 비율 | 운영 안정성 |
| Setup Friction | 토큰, 모델 다운로드, 라이선스 동의, 의존성 수 | 재현성과 배포 난이도 |
| Cost per Audio Hour | API 비용 또는 GPU 사용 시간 환산 비용 | 실서비스/보고서 Pareto 비교 |

속도 비교는 반드시 동일 오디오, 동일 하드웨어, 동일 batch 조건에서 수행한다. `WhisperX large-v3 + int8 + batch_size=4`처럼 현재 코드의 조건을 명시하고, Gemma 쪽도 모델 ID, quantization, endpoint, max token, temperature를 기록한다.

### 3.4 종합 점수

STT 성능 종합 점수는 다음처럼 분리 산출한다.

| 점수 | 구성 |
|---|---|
| `asr_score` | `1 - WER` 50%, `1 - CER` 20%, numeric accuracy 15%, entity recall 15% |
| `speaker_score` | `1 - DER` 40%, `1 - JER` 20%, speaker attribution 30%, speaker count score 10% |
| `ops_score` | RTF 40%, latency 20%, peak VRAM 20%, failure rate 20% |
| `overall_stt_score` | asr 50%, speaker 30%, ops 20% |

논문/보고서 표에서는 종합 점수보다 원지표를 우선 제시한다. 종합 점수는 모델 선택을 위한 보조 판단으로만 쓴다.

## 4. 현재 기준 예비 비교

| 항목 | Gemma4 E2B-it 직접 오디오 | 기존 STT+Pyannote |
|---|---|---|
| 장점 | 오디오를 바로 넣어 transcript/요약을 얻을 수 있으면 파이프라인이 단순함 | transcript, timestamp, 화자 라벨이 남아 검증 가능. 숫자, 근거 추적, WBS/RAG 입력으로 재사용 가능 |
| 약점 | 중간 transcript가 없으면 오류 원인 분석이 어렵다. 화자 귀속/숫자 보존을 검증하기 어렵다 | STT, alignment, diarization, 요약까지 단계가 길고 HF 토큰/모델 다운로드/GPU 의존성이 큼 |
| 회의록 정확성 리스크 | 작은 모델이 직접 오디오에서 요약까지 수행하면 숫자/담당자 누락 가능성이 큼 | STT 오류는 남지만 transcript를 기준으로 후처리/검수 가능 |
| 화자 분리 | 모델 출력에 화자 구조를 강제해야 함 | Pyannote로 명시적 화자 라벨 생성 |
| 운영 비용 | 모델/엔드포인트가 오디오를 직접 지원하면 호출은 단순하나, 실패 디버깅 비용이 큼 | GPU 메모리와 처리 시간이 더 들지만 관찰 가능성이 높음 |
| 추천 용도 | 빠른 초안, 화자/수치 정확도가 덜 중요한 내부 확인 | 공식 transcript, 회의록 공식 기록, R&R 배정, WBS 생성, 근거 기반 평가 |

현 저장소의 목적이 WBS 생성과 R&R 배정까지 이어지는 것이라면, 기본값은 STT+Pyannote를 유지하는 편이 낫다. 이유는 회의록이 단순 요약물이 아니라 `rag_meeting_logs`로 들어가 Supervisor Agent의 담당자 배정 근거가 되기 때문이다. 이 경우 누가 어떤 문제를 제기했는지와 숫자/기한이 남아야 한다.

Gemma4 E2B-it 직접 오디오는 보조 경로로 두는 것이 적절하다. 빠른 회의 요약 초안을 만들고, 공식 입력은 STT+Pyannote transcript 기반 요약으로 확정하는 구조가 가장 안전하다.

## 5. 실제 실험 프로토콜

### 데이터셋

| 세트 | 구성 |
|---|---|
| D1 public-clean | AMI close-talk 또는 ICSI headset mix 3개 회의 |
| D2 public-noisy | AMI far-field 또는 room microphone 3개 회의 |
| D3 summary-benchmark | QMSum test subset 20~50개 query-summary pair, STT downstream 참고용 |
| D4 local-control | 저장소의 P-Reset 샘플 회의록 및 no-schedule 변형 |

각 세트마다 정답 transcript와 정답 회의록 JSON을 만든다.

```json
{
  "topic": "",
  "decisions": [],
  "action_items": [
    {"owner": "", "task": "", "deliverable": "", "deadline": ""}
  ],
  "numeric_facts": [],
  "risks": [],
  "open_questions": []
}
```

### 실행 조건

| 조건 | 설명 |
|---|---|
| A1 | Gemma4 E2B-it 직접 오디오 -> transcript |
| A2 | Gemma4 E2B-it 직접 오디오 -> 화자별 transcript |
| B1 | WhisperX+Pyannote -> transcript만 평가 |
| B2 | WhisperX+Pyannote -> 화자별 transcript 평가 |
| B3 | WhisperX+Pyannote transcript -> 현재 WBS 파이프라인 입력 |
| C1 | QMSum transcript -> Gemma4 E2B-it query/general summary, downstream 참고 |

### 합격 기준

| 항목 | 최소 기준 |
|---|---:|
| WER | 0.20 이하 |
| CER | 0.12 이하 |
| DER | 0.20 이하 |
| Numeric Accuracy | 0.90 이상 |
| Entity Recall | 0.90 이상 |
| Speaker Attribution Accuracy | 0.80 이상 |
| RTF | 1.0 이하 |
| Failure Rate | 0.05 이하 |
| Hallucinated Critical Fact | 0건 |
| WBS downstream faithfulness | 기존 `metrics.py` 기준 0.90 이상 |

### 실행 준비

필요 패키지:

```bash
pip install datasets jiwer rouge-score bert-score soundfile librosa
```

QMSum은 GitHub에서 바로 받을 수 있다.

```bash
git clone https://github.com/Yale-LILY/QMSum.git external_datasets/QMSum
```

AMI/ICSI는 공식 다운로드 페이지에서 회의 ID를 선택해 필요한 audio stream만 받는 방식이 낫다. 전체 corpus는 크므로 처음에는 3개 회의만 받는다.

권장 1차 평가 순서:

1. AMI 또는 ICSI 3개 회의의 audio/transcript를 받아 audio -> transcript WER/CER를 먼저 측정한다.
2. 같은 subset에서 DER/JER/cpWER/speaker attribution을 측정한다.
3. 동일 환경에서 RTF, wall-clock latency, peak VRAM/RAM, failure rate를 기록한다.
4. QMSum은 STT 자체가 아니라 transcript가 회의록/요약 품질에 미치는 downstream 참고 평가로 사용한다.
5. 최종적으로 STT+Pyannote transcript를 현재 WBS 파이프라인의 `meeting_text`로 넣어 downstream `faithfulness`, `assignment`, `autoscore` 변화를 확인한다.

결과 기록용 기본 스키마는 `eval_results/audio_meeting_minutes_comparison/stt_eval_schema.csv`에 둔다.

## 6. 구현상 확인된 보완점

1. `STT_DONGK.py`의 HF 토큰이 코드 주석에 남아 있다. 실제 토큰이면 즉시 폐기하고 `.env`로 옮겨야 한다.
2. `STT_DONGK.py`의 Gemma 요약 단계는 주석 처리되어 있어 현재는 transcript만 반환한다.
3. `frontend/index.html`에는 STT 업로드 UI가 있지만 `api.py`에 `/api/stt` 또는 파일 업로드 엔드포인트가 없다.
4. 프론트엔드 payload 생성 로직에서 `S.ragMode === 'stt'`일 때 업로드 파일을 전송하지 않고 `meeting_text`를 비운다.
5. 회의록 평가 결과를 WBS 평가(`faithfulness`, `assignment`, `autoscore`)와 연결하려면 STT 산출물을 `meeting_text`로 주입하는 실험 조건을 추가해야 한다.

## 7. 결론

현재 프로젝트 기준의 결론은 다음과 같다.

| 결론 | 판단 |
|---|---|
| 공식 transcript/RAG 입력 | STT+Pyannote 기반이 우세 |
| 빠른 transcript 초안 | Gemma4 E2B-it 직접 오디오가 유리할 수 있음 |
| 평가 가능성/디버깅 | STT+Pyannote가 우세 |
| 담당자 배정/WBS downstream 품질 | 화자별 transcript가 있는 STT+Pyannote가 더 적합 |
| 권장 구조 | Gemma 직접 오디오는 초안, STT+Pyannote는 기준 STT 파이프라인 |

실제 오디오와 Gemma 직접 오디오 transcript가 확보되면, 이 리포트의 `WER`, `CER`, `DER`, `cpWER`, `Numeric Accuracy`, `Entity Recall`, `RTF`, `Peak VRAM`, `Failure Rate`를 중심으로 재채점하면 된다.
