# 점수 구간별 LLM-Judge 정성 샘플

실측 스냅샷에서 LLM-Judge Overall 점수에 따라 대표 샘플을 모았습니다. 각 샘플은 **WBS 발췌·Judge 점수·Judge reason**을 함께 보여줍니다.

샘플 선정 원칙:
- Low (Overall < 0.55): Gemma-4B (Gemini-3.1-Pro로 rejudge한 풀버전 reason)
- Mid (0.62 ≤ Overall ≤ 0.70): Gemma-4-26B 스냅샷 자체의 풀버전 reason
- High (Overall ≥ 0.74): 같은 Gemma-4-26B 풀버전 reason

판정 모델(Judge)은 모든 샘플에서 Gemini-3.1-Pro-Preview 동일.


## Index

| Bucket | Backbone | Cond | Run | Overall | S | A | D |
|---|---|---|---|---|---|---|---|
| Low | Gemma-4B | C4_with_disc | r1 | 0.437 | 0.53 | 0.00 | 0.90 |
| Low | Gemma-4B | C0_llm_only | r3 | 0.470 | 0.47 | N/A | N/A |
| Mid | Gemma-4-26B | C0_llm_only | r1 | 0.630 | 0.63 | N/A | N/A |
| Mid | Gemma-4-26B | C0_llm_only | r2 | 0.630 | 0.63 | N/A | N/A |
| High | Gemma-4-26B | C3_3rounds | r1 | 0.787 | 0.70 | 0.77 | 0.95 |
| High | Gemma-4-26B | C3_3rounds | r2 | 0.801 | 0.77 | 0.73 | 0.95 |

## [Low] Gemma-4B · C4_with_disc · r1 · Overall **0.44**

- 점수: S=0.53  A=0.00  D=0.90  Overall=0.44

**WBS 발췌**
- 총 task 수: **66** (L1 6 / L2 15 / L3 45)

```text
[L1] L1-01 프로젝트 정의 및 요구사항 분석  (role=PM, est=44.0d)
  [L2] L2-01-01 비즈니스 프로세스 정의  (role=Business Analyst)
    [L3] L3-01-01-01 핵심 CS 상담 시나리오 정의
  [L2] L2-01-02 기술 스택 및 아키텍처 설계  (role=Architect)
    [L3] L3-01-02-01 MSA/모놀리스 구조 결정 및 분할 설계
[L1] L1-02 개발 환경 구축 및 기반 인프라 설정  (role=DevOps, est=44.0d)
  [L2] L2-02-01 데이터베이스 및 보안 설정  (role=Data Engineer)
    [L3] L3-02-01-01 PostgreSQL 스키마 구현 및 마이그레이션 스크립트 작성
  [L2] L2-02-02 API 게이트웨이 및 인증 시스템 구축  (role=Backend Developer)
    [L3] L3-02-02-01 API Gateway 설정 및 라우팅 규칙 정의
[L1] L1-03 핵심 AI 및 백엔드 로직 개발  (role=Backend Developer, est=45.0d)
  [L2] L2-03-01 AI 챗봇 인터페이스 구현  (role=Backend Developer)
    [L3] L3-03-01-01 OpenAI API 연동 및 프롬프트 엔지니어링
  [L2] L2-03-02 CRM 및 데이터 동기화 모듈 개발  (role=Backend Developer)
    [L3] L3-03-02-01 Salesforce 데이터 조회(Read) REST API 구현
```

**Structure reason (Judge)**
> A=0.2 B=0.4 C=1.0; WBS is severely truncated leaving most L2 tasks without L3 children; L3 durations are within 1-1

**Assignment reason (Judge)**
> A=0.0 B=0.0 C=0.0; key issues: Complete mismatch between team profile IDs and assigned IDs in the WBS. All actual team members are completely idle (0 tasks assigned

**Debate reason (Judge)**
> A=1.0 B=1.0 C=0.8 D=0.8; key issues: 5 distinct roles participate with highly concrete, task-ID-referenced technical analysis. Minor

_Snapshot:_ `gemma_ablation/snapshots/wbs_snapshot_C4_with_disc_r1_gemma4-api_piai_20260421_023659.json`

---

## [Low] Gemma-4B · C0_llm_only · r3 · Overall **0.47**

- 점수: S=0.47  A=N/A  D=N/A  Overall=0.47

**WBS 발췌**
- 총 task 수: **77** (L1 6 / L2 18 / L3 53)

```text
[L1] L1-01 프로젝트 정의 및 요구사항 분석  (role=Planner, est=20.0d)
  [L2] L2-01-01 비즈니스 프로세스 분석  (role=Planner)
    [L3] L3-01-01-01 현행 CS 프로세스 AS-IS 매핑 및 병목 지점 식별
  [L2] L2-01-02 기술 스택 및 아키텍처 설계  (role=Architect)
    [L3] L3-01-02-01 MSA 서비스 경계 및 통신 프로토콜 정의
[L1] L1-02 개발 환경 구축 및 기반 인프라 설정  (role=DevOps, est=30.0d)
  [L2] L2-02-01 백엔드 핵심 서비스 API 개발 (Core Logic)  (role=Backend Developer)
    [L3] L3-02-01-01 핵심 비즈니스 로직 API 명세 및 설계
  [L2] L2-02-02 외부 시스템 연동 모듈 개발  (role=Backend Developer)
    [L3] L3-02-02-01 Salesforce API 연동을 위한 인증 및 연결 모듈 구현
[L1] L1-03 프론트엔드 UI/UX 구현  (role=Frontend Developer, est=27.0d)
  [L2] L2-03-01 챗봇 대화 인터페이스 구현  (role=Frontend Developer)
    [L3] L3-03-01-01 챗봇 UI 컴포넌트 설계 및 기본 레이아웃 구현
  [L2] L2-03-02 실시간 분석 대시보드 구현  (role=Frontend Developer)
    [L3] L3-03-02-01 핵심 지표(KPI) 시각화 컴포넌트 개발
```

**Structure reason (Judge)**
> A=0.2 B=0.4 C=0.8; A: 6 L1s and 18 L2s exist, but 17 out of 18 L2

_Snapshot:_ `gemma_ablation/snapshots/wbs_snapshot_C0_llm_only_r3_gemma4-api_piai_20260421_000826.json`

---

## [Mid] Gemma-4-26B · C0_llm_only · r1 · Overall **0.63**

- 점수: S=0.63  A=N/A  D=N/A  Overall=0.63

**WBS 발췌**
- 총 task 수: **34** (L1 3 / L2 8 / L3 23)

```text
[L1] L1-01 요구사항 분석 및 시스템 설계  (role=PM/Planner, est=34.0d)
  [L2] L2-01-01 비즈니스 요구사항 및 프로세스 정의  (role=PM/Planner)
    [L3] L3-01-01-01 고객사 CS 워크플로우 분석
  [L2] L2-01-02 UI/UX 프로토타이핑  (role=Designer)
    [L3] L3-01-02-01 챗봇 대화 인터페이스 설계
[L1] L1-02 인프라 구축 및 데이터 환경 설정  (role=DevOps, est=27.0d)
  [L2] L2-02-01 AWS 클라우드 환경 구축  (role=DevOps)
    [L3] L3-02-01-01 네트워크 및 보안 그룹 설정
  [L2] L2-02-02 데이터베이스 및 스토리지 구축  (role=Data Engineer)
    [L3] L3-02-02-01 RDS 인스턴스 프로비저닝
[L1] L1-03 핵심 비즈니스 로직 및 API 개발  (role=Backend Developer, est=19.0d)
  [L2] L2-03-01 AI 챗봇 엔진 개발  (role=Backend Developer)
    [L3] L3-03-01-01 OpenAI API 연동 모듈 구현
  [L2] L2-03-02 CRM 시스템 연동 개발  (role=Backend Developer)
    [L3] L3-03-02-01 Salesforce API 인증 모듈 구현
```

**Structure reason (Judge)**
> A=0.5 B=0.4 C=1.0; 3 L1s with good L3 coverage, all estimates 1-10d but no buffer, excellent domain-specific titles.

_Snapshot:_ `gemma26_ablation/snapshots/wbs_snapshot_C0_llm_only_r1_qwen-api_gemma26_ablation_20260423_135834.json`

---

## [Mid] Gemma-4-26B · C0_llm_only · r2 · Overall **0.63**

- 점수: S=0.63  A=N/A  D=N/A  Overall=0.63

**WBS 발췌**
- 총 task 수: **34** (L1 3 / L2 8 / L3 23)

```text
[L1] L1-01 요구사항 분석 및 시스템 설계  (role=PM/Planner, est=30.0d)
  [L2] L2-01-01 비즈니스 요구사항 및 프로세스 정의  (role=Planner)
    [L3] L3-01-01-01 고객사 CS 워크플로우 분석
  [L2] L2-01-02 UI/UX 디자인 설계  (role=Designer)
    [L3] L3-01-02-01 대시보드 와이어프레임 작성
[L1] L1-02 백엔드 및 인프라 구축  (role=Backend Developer, est=45.0d)
  [L2] L2-02-01 인프라 및 DB 환경 설정  (role=DevOps)
    [L3] L3-02-01-01 AWS 클라우드 리소스 구축
  [L2] L2-02-02 핵심 비즈니스 로직 개발  (role=Backend Developer)
    [L3] L3-02-02-01 OpenAI API 연동 모듈 개발
[L1] L1-03 프론트엔드 구현  (role=Frontend Developer, est=19.0d)
  [L2] L2-03-01 상담사 대시보드 구현  (role=Frontend Developer)
    [L3] L3-03-01-01 실시간 데이터 차트 컴포넌트
  [L2] L2-03-02 AI 챗봇 인터페이스 구현  (role=Frontend Developer)
    [L3] L3-03-02-01 실시간 메시지 스트리밍 UI
```

**Structure reason (Judge)**
> A=0.5 B=0.4 C=1.0; 3 L1s with good L3 coverage, L3 durations 1-10 days but no buffer, excellent domain-specific tasks.

_Snapshot:_ `gemma26_ablation/snapshots/wbs_snapshot_C0_llm_only_r2_qwen-api_gemma26_ablation_20260423_140046.json`

---

## [High] Gemma-4-26B · C3_3rounds · r1 · Overall **0.79**

- 점수: S=0.70  A=0.77  D=0.95  Overall=0.79

**WBS 발췌**
- 총 task 수: **36** (L1 3 / L2 8 / L3 25)

```text
[L1] L1-01 요구사항 분석 및 시스템 설계  (role=PM/Planner, est=38.0d)
  [L2] L2-01-01 비즈니스 요구사항 및 UX 설계  (role=Designer/Planner)
    [L3] L3-01-01-01 사용자 페르소나 및 시나리오 정의
  [L2] L2-01-02 기술 아키텍처 및 데이터 설계  (role=Backend Developer)
    [L3] L3-01-02-01 ERD 설계 및 DB 스키마 확정
[L1] L1-02 백엔드 및 핵심 로직 개발  (role=Backend Developer, est=42.0d)
  [L2] L2-02-01 AI 챗봇 엔진 개발  (role=Backend Developer)
    [L3] L3-02-01-01 OpenAI API 연동 모듈 개발
  [L2] L2-02-02 외부 시스템(CRM) 연동 개발  (role=Backend Developer)
    [L3] L3-02-02-01 Salesforce REST API 인증 구현
[L1] L1-03 프론트엔드 구현  (role=Frontend Developer, est=19.0d)
  [L2] L2-03-01 챗봇 인터페이스 개발  (role=Frontend Developer)
    [L3] L3-03-01-01 실시간 채팅 UI 컴포넌트 구현
  [L2] L2-03-02 실시간 분석 대시보드 개발  (role=Frontend Developer)
    [L3] L3-03-02-01 데이터 시각화 차트 구현
```

**Structure reason (Judge)**
> A=0.5 B=0.6 C=1.0; 3 L1 phases with one L2 lacking L3s, buffer is under 10% (~9%), but task titles are highly specific.

**Assignment reason (Judge)**
> A=0.9 B=0.4 C=1.0; High skill fit except QA assigned dev task. Severe workload imbalance (26d vs 4.5d). All utilized.

**Debate reason (Judge)**
> A=1.0 B=1.0 C=0.8 D=1.0; 4 roles active, concrete task IDs cited, minor name overlap, clear buffer adjustment.

_Snapshot:_ `gemma26_ablation/snapshots/wbs_snapshot_C3_3rounds_r1_qwen-api_gemma26_ablation_20260423_143800.json`

---

## [High] Gemma-4-26B · C3_3rounds · r2 · Overall **0.80**

- 점수: S=0.77  A=0.73  D=0.95  Overall=0.80

**WBS 발췌**
- 총 task 수: **36** (L1 3 / L2 8 / L3 25)

```text
[L1] L1-01 요구사항 분석 및 시스템 설계  (role=PM/Planner, est=31.0d)
  [L2] L2-01-01 비즈니스 요구사항 및 프로세스 정의  (role=PM/Planner)
    [L3] L3-01-01-01 고객사 CS 워크플로우 분석
  [L2] L2-01-02 UI/UX 및 시스템 아키텍처 설계  (role=Designer/Architect)
    [L3] L3-01-02-01 Figma 기반 와이어프레임 제작
[L1] L1-02 백엔드 및 데이터 인프라 구축  (role=Backend Developer, est=40.0d)
  [L2] L2-02-01 데이터베이스 및 서버 환경 구축  (role=DevOps/Backend)
    [L3] L3-02-01-01 PostgreSQL DB 인스턴스 설정
  [L2] L2-02-02 외부 시스템 연동 모듈 개발  (role=Backend Developer)
    [L3] L3-02-02-01 Salesforce API 인증 구현
[L1] L1-03 핵심 비즈니스 로직 개발  (role=Backend Developer, est=32.0d)
  [L2] L2-03-01 AI 챗봇 엔진 개발  (role=Backend Developer)
    [L3] L3-03-01-01 OpenAI API 연동 모듈 구현
  [L2] L2-03-02 실시간 분석 및 리포트 엔진  (role=Data Engineer)
    [L3] L3-03-02-01 실시간 통계 집계 쿼리 최적화
```

**Structure reason (Judge)**
> A=0.5 B=0.8 C=1.0; 3 L1s limits hierarchy score despite perfect L3 depth. Buffer is 12.6%. Tasks are highly specific.

**Assignment reason (Judge)**
> A=0.8 B=0.4 C=1.0; Good skill fit overall but Data Engineer handles many Backend tasks. Severe workload imbalance (max/min 4.6x).

**Debate reason (Judge)**
> A=1.0 B=1.0 C=0.8 D=1.0; 4 roles active, concrete analysis with task IDs, minor name/role slip, clear convergence.

_Snapshot:_ `gemma26_ablation/snapshots/wbs_snapshot_C3_3rounds_r2_qwen-api_gemma26_ablation_20260423_144446.json`

---

## 점수대별 패턴 요약

| Bucket | 자주 보이는 Judge 지적 | 산출물 형태 |
|---|---|---|
| **Low (<0.55)** | WBS truncation, L2가 L3 자식 없음, L1 phase가 너무 적거나 균형 부족 | L1·L2 골격은 보이지만 본문이 잘려 L3 거의 없음, 또는 task 수 과대(70~80개)이지만 깊이 부족 |
| **Mid (0.62~0.70)** | 깊이는 합격 / 일부 L2 child 누락 / Assignment skill-fit OK지만 workload 편향 / Debate 인용 약함 | 30~35 task, 3-level 구조 충족, 일부 phase에서 child 누락 |
| **High (≥0.74)** | 4 roles 모두 활성, task ID 명시 인용, buffer 명확, skill fit·utilization 양쪽 충족 | 33~36 task, 모든 L1 아래 L2/L3 균등, role 활성도 ≥4 |

핵심 신호:

- **Structure 점수가 0.5 이하** → 거의 항상 **L3 누락 / WBS 본문 truncation** 문제. task 수가 적어서가 아니라 L1·L2만 있고 L3가 없는 형태.
- **Assignment 점수가 0.5 이하** → 직군 라벨만 있고 멤버 ID 매핑 실패, 또는 1명에게 26d 같은 큰 편향 (workload Gini ↑).
- **Debate 점수가 0.6 이하** → 토론 메시지에 task ID 인용 없음, buffer 조정 근거 모호, 일부 role(QA·Designer)이 침묵.

_원본 스냅샷·CSV는 `qualitative_by_score/` 와 `gemma_ablation/`, `gemma26_ablation/` 하위에 그대로 보존._
