"""
RAG 전략 모듈
4가지 검색 전략을 제공합니다:
  - Vanilla RAG   : FAISS Dense Vector 유사도 검색 (기준선)
  - Hybrid Search : BM25 + Dense Vector + Reciprocal Rank Fusion
  - Graph RAG     : 엔티티 공출현 그래프 기반 이웃 탐색
  - Agentic RAG   : 커버리지를 평가하며 반복적으로 쿼리를 확장하는 멀티홉 검색
"""
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────
# 전략 메타데이터 (Streamlit UI 표시용)
# ────────────────────────────────────────────────────────────────

STRATEGY_INFO: Dict[str, Dict[str, Any]] = {
    "vanilla": {
        "name": "Vanilla RAG",
        "icon": "🔵",
        "description": "FAISS Dense Vector 유사도 검색",
        "detail": (
            "문장 임베딩(all-MiniLM-L6-v2)으로 코사인 유사도를 계산합니다. "
            "전통적인 RAG 기준선으로, 의미(semantics) 기반 매칭에 강합니다."
        ),
        "pros": ["의미 기반 매칭 (동의어, 패러프레이즈 대응)", "언어 무관", "구현 단순"],
        "cons": ["희귀 키워드 검색 약점 (lexical gap)", "그래프 관계 미고려", "Out-of-domain 임베딩 성능 저하"],
    },
    "hybrid": {
        "name": "Hybrid Search",
        "icon": "🟡",
        "description": "BM25 + Dense Vector + RRF",
        "detail": (
            "키워드 빈도 기반 BM25(lexical)와 Dense Vector(semantic)를 결합하여 "
            "Reciprocal Rank Fusion(RRF)으로 최종 순위를 통합합니다."
        ),
        "pros": ["키워드 + 의미 모두 커버", "RRF로 안정적인 앙상블 순위", "Out-of-domain 강인성"],
        "cons": ["BM25 계산 비용 추가", "언어별 토크나이저 품질 차이", "파라미터 튜닝(k1, b) 필요"],
    },
    "graph": {
        "name": "Graph RAG",
        "icon": "🟢",
        "description": "지식 그래프 기반 엔티티 탐색",
        "detail": (
            "문서 내 엔티티(기술명, 역할, 일정 단위 등)를 추출하고 공출현 그래프를 구성합니다. "
            "쿼리 관련 시드 문서에서 시작해 그래프 이웃으로 탐색 범위를 확장합니다."
        ),
        "pros": ["문서 간 관계 기반 검색", "명시적 연결이 없는 연관 문서 발견", "설명 가능한 경로"],
        "cons": ["엔티티 추출 품질 의존", "그래프 구축 비용", "도메인별 패턴 튜닝 필요"],
    },
    "agentic": {
        "name": "Agentic RAG",
        "icon": "🔴",
        "description": "반복적 멀티홉 검색 (커버리지 평가 기반)",
        "detail": (
            "초기 검색 후 쿼리 커버리지를 평가하고, 부족하면 새 쿼리를 생성해 최대 3홉 반복합니다. "
            "LLM 연결 시 LLM이 다음 쿼리를 생성하고, Mock 모드에서는 경험적 규칙으로 확장합니다."
        ),
        "pros": ["동적 쿼리 확장", "높은 리콜", "부족한 컨텍스트 자동 보완"],
        "cons": ["홉수만큼 지연 증가", "LLM 호출 비용 누적", "루프 종료 조건 설계 필요"],
    },
}


# ────────────────────────────────────────────────────────────────
# 공통 기반 클래스
# ────────────────────────────────────────────────────────────────

class BaseRAGStrategy(ABC):
    """RAG 전략 추상 기반 클래스"""

    strategy_key: str = "base"

    @abstractmethod
    def retrieve(
        self,
        query: str,
        doc_type: Optional[str],
        k: int,
        documents: List[Dict[str, Any]],
        vectorstore=None,
        embeddings=None,
        llm=None,
    ) -> List[Dict[str, Any]]:
        """
        문서 검색.
        반환: List[Dict] — 각 항목에 content, metadata, _rag_score, _rag_info 포함
        """

    def get_info(self) -> Dict[str, Any]:
        return STRATEGY_INFO.get(self.strategy_key, {})

    # ── 공통 헬퍼 ────────────────────────────────────────────────

    def _filter_by_type(
        self, documents: List[Dict], doc_type: Optional[str]
    ) -> List[Dict]:
        if doc_type is None:
            return documents
        return [d for d in documents if d["metadata"].get("doc_type") == doc_type]

    def _keyword_search(
        self, query: str, documents: List[Dict], k: int
    ) -> List[Dict[str, Any]]:
        """키워드 오버랩 기반 폴백 검색"""
        query_words = set(query.lower().split())

        def score(doc: Dict) -> int:
            return len(query_words & set(doc["content"].lower().split()))

        sorted_docs = sorted(documents, key=score, reverse=True)
        results = []
        for doc in sorted_docs[:k]:
            result = dict(doc)
            result["_rag_score"] = float(score(doc))
            result["_rag_info"] = {"method": "keyword_fallback", "overlap": score(doc)}
            results.append(result)
        return results

    def _faiss_search_with_score(
        self,
        query: str,
        vectorstore,
        doc_type: Optional[str],
        k: int,
        documents: List[Dict],
    ) -> List[Dict[str, Any]]:
        """FAISS similarity_search_with_score 래퍼 (실패 시 빈 리스트 반환)"""
        if vectorstore is None:
            return []
        try:
            filter_dict = {"doc_type": doc_type} if doc_type else None
            raw = vectorstore.similarity_search_with_score(
                query, k=min(k * 3, max(len(documents), 1)), filter=filter_dict
            )
            results = []
            for doc, dist in raw[:k]:
                results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "_rag_score": float(1.0 / (1.0 + dist)),
                        "_rag_info": {"method": "faiss_dense", "l2_distance": float(dist)},
                    }
                )
            return results
        except Exception:
            return []


# ────────────────────────────────────────────────────────────────
# 1. Vanilla RAG
# ────────────────────────────────────────────────────────────────

class VanillaRAGStrategy(BaseRAGStrategy):
    """기존 구현과 동일: FAISS Dense Vector 유사도 검색"""

    strategy_key = "vanilla"

    def retrieve(self, query, doc_type, k, documents, vectorstore=None, embeddings=None, llm=None):
        filtered = self._filter_by_type(documents, doc_type)
        results = self._faiss_search_with_score(query, vectorstore, doc_type, k, filtered)
        if results:
            return results
        return self._keyword_search(query, filtered, k)


# ────────────────────────────────────────────────────────────────
# 2. Hybrid RAG (BM25 + Dense + RRF)
# ────────────────────────────────────────────────────────────────

class HybridRAGStrategy(BaseRAGStrategy):
    """BM25 + Dense Vector + Reciprocal Rank Fusion"""

    strategy_key = "hybrid"
    RRF_K = 60  # RRF 하이퍼파라미터

    def retrieve(self, query, doc_type, k, documents, vectorstore=None, embeddings=None, llm=None):
        filtered = self._filter_by_type(documents, doc_type)
        if not filtered:
            return []

        bm25_results = self._bm25_search(query, filtered, k * 2)
        dense_results = self._faiss_search_with_score(query, vectorstore, doc_type, k * 2, filtered)
        if not dense_results:
            dense_results = self._keyword_search(query, filtered, k * 2)

        return self._rrf_combine(bm25_results, dense_results, k)

    def _bm25_search(self, query: str, documents: List[Dict], k: int) -> List[Dict]:
        try:
            from rank_bm25 import BM25Okapi

            corpus = [doc["content"].lower().split() for doc in documents]
            bm25 = BM25Okapi(corpus)
            query_tokens = query.lower().split()
            scores = bm25.get_scores(query_tokens)

            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            results = []
            for bm25_rank, idx in enumerate(ranked[:k]):
                result = dict(documents[idx])
                result["_rag_score"] = float(scores[idx])
                result["_rag_info"] = {"method": "bm25", "bm25_score": float(scores[idx])}
                result["_bm25_rank"] = bm25_rank
                results.append(result)
            return results
        except ImportError:
            # rank_bm25 미설치 → 키워드 폴백
            return self._keyword_search(query, documents, k)

    def _rrf_combine(
        self,
        bm25_results: List[Dict],
        dense_results: List[Dict],
        k: int,
    ) -> List[Dict]:
        """Reciprocal Rank Fusion으로 두 결과 목록 통합"""
        key_to_doc: Dict[str, Dict] = {}
        key_to_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)

        for rank, doc in enumerate(bm25_results):
            key = doc["content"][:120]
            key_to_doc[key] = doc
            key_to_ranks[key]["bm25"] = rank

        for rank, doc in enumerate(dense_results):
            key = doc["content"][:120]
            if key not in key_to_doc:
                key_to_doc[key] = doc
            key_to_ranks[key]["dense"] = rank

        rrf_scores: Dict[str, float] = {}
        for key, ranks in key_to_ranks.items():
            rrf_scores[key] = sum(
                1.0 / (self.RRF_K + r + 1) for r in ranks.values()
            )

        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for key in sorted_keys[:k]:
            doc = dict(key_to_doc[key])
            ranks = key_to_ranks[key]
            doc["_rag_score"] = rrf_scores[key]
            doc["_rag_info"] = {
                "method": "rrf",
                "rrf_score": round(rrf_scores[key], 5),
                "bm25_rank": ranks.get("bm25", "—"),
                "dense_rank": ranks.get("dense", "—"),
            }
            results.append(doc)
        return results


# ────────────────────────────────────────────────────────────────
# 3. Graph RAG
# ────────────────────────────────────────────────────────────────

class GraphRAGStrategy(BaseRAGStrategy):
    """엔티티 공출현 그래프 기반 이웃 탐색"""

    strategy_key = "graph"

    # WBS/프로젝트 문서 도메인 엔티티 패턴
    ENTITY_PATTERNS = [
        r'(?:Phase|단계|페이즈)\s*\d+',
        r'\d+\s*주',
        r'\d+\s*일',
        r'(?:백엔드|프론트엔드|QA|디자인|기획|PM|개발자|데이터)',
        r'(?:Python|FastAPI|React|Vue|Django|Spring|AWS|GCP|Docker|Kubernetes|PostgreSQL|MySQL|Redis)',
        r'(?:API|REST|GraphQL|OAuth|JWT|CI[/\-]?CD)',
        r'(?:버퍼|마일스톤|스프린트|릴리즈|배포|테스트)',
        r'(?:연동|통합|마이그레이션|리팩토링)',
        r'(?:지연|초과|리스크|완료|교훈)',
    ]

    def retrieve(self, query, doc_type, k, documents, vectorstore=None, embeddings=None, llm=None):
        filtered = self._filter_by_type(documents, doc_type)
        if not filtered:
            return []

        # 1. 엔티티 인덱스 구성
        entity_to_docs, doc_to_entities = self._build_entity_index(filtered)

        # 2. 시드 문서 탐색 (키워드 기반)
        seeds = self._find_seeds(query, filtered, vectorstore, doc_type, k=2)
        seed_indices = [s["_doc_idx"] for s in seeds if s.get("_doc_idx", -1) >= 0]

        # 3. 그래프 이웃 확장
        neighbor_indices = self._expand_neighbors(seed_indices, entity_to_docs, doc_to_entities)

        # 4. 시드 + 이웃 순서로 top-k 선택
        candidate_indices = list(dict.fromkeys(seed_indices + neighbor_indices))

        results = []
        for rank, idx in enumerate(candidate_indices[:k]):
            doc = filtered[idx]
            is_seed = idx in seed_indices
            entities = doc_to_entities.get(idx, [])
            result = dict(doc)
            result["_rag_score"] = 1.0 / (1.0 + rank)
            result["_rag_info"] = {
                "method": "graph_rag",
                "node_type": "seed" if is_seed else "neighbor",
                "entities": entities[:6],
                "graph_path": (
                    f"{'[시드]' if is_seed else '[이웃]'} "
                    f"← {', '.join(entities[:3]) if entities else '공통 엔티티 없음'}"
                ),
            }
            results.append(result)
        return results[:k]

    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        for pattern in self.ENTITY_PATTERNS:
            entities.extend(re.findall(pattern, text, re.IGNORECASE))
        return list({e.strip().lower() for e in entities if e.strip()})

    def _build_entity_index(
        self, documents: List[Dict]
    ) -> Tuple[Dict[str, set], Dict[int, List[str]]]:
        entity_to_docs: Dict[str, set] = defaultdict(set)
        doc_to_entities: Dict[int, List[str]] = {}
        for idx, doc in enumerate(documents):
            entities = self._extract_entities(doc["content"])
            doc_to_entities[idx] = entities
            for e in entities:
                entity_to_docs[e].add(idx)
        return entity_to_docs, doc_to_entities

    def _find_seeds(
        self,
        query: str,
        documents: List[Dict],
        vectorstore,
        doc_type: str,
        k: int,
    ) -> List[Dict]:
        faiss_seeds = self._faiss_search_with_score(query, vectorstore, doc_type, k, documents)
        if faiss_seeds:
            for i, s in enumerate(faiss_seeds):
                # content로 원본 인덱스 역추적
                for idx, d in enumerate(documents):
                    if d["content"] == s["content"]:
                        faiss_seeds[i]["_doc_idx"] = idx
                        break
                else:
                    faiss_seeds[i]["_doc_idx"] = -1
            return faiss_seeds

        query_words = set(query.lower().split())
        scored = sorted(
            enumerate(documents),
            key=lambda x: len(query_words & set(x[1]["content"].lower().split())),
            reverse=True,
        )
        results = []
        for idx, doc in scored[:k]:
            result = dict(doc)
            result["_doc_idx"] = idx
            result["_rag_score"] = len(query_words & set(doc["content"].lower().split()))
            results.append(result)
        return results

    def _expand_neighbors(
        self,
        seed_indices: List[int],
        entity_to_docs: Dict[str, set],
        doc_to_entities: Dict[int, List[str]],
    ) -> List[int]:
        seed_entities = set()
        for idx in seed_indices:
            seed_entities.update(doc_to_entities.get(idx, []))

        neighbor_scores: Dict[int, int] = defaultdict(int)
        for entity in seed_entities:
            for doc_idx in entity_to_docs.get(entity, set()):
                if doc_idx not in seed_indices:
                    neighbor_scores[doc_idx] += 1

        return sorted(neighbor_scores, key=lambda x: neighbor_scores[x], reverse=True)


# ────────────────────────────────────────────────────────────────
# 4. Agentic RAG
# ────────────────────────────────────────────────────────────────

class AgenticRAGStrategy(BaseRAGStrategy):
    """커버리지를 평가하며 반복적으로 쿼리를 확장하는 멀티홉 검색"""

    strategy_key = "agentic"
    MAX_HOPS = 3
    COVERAGE_THRESHOLD = 0.72

    def retrieve(self, query, doc_type, k, documents, vectorstore=None, embeddings=None, llm=None):
        filtered = self._filter_by_type(documents, doc_type)
        if not filtered:
            return []

        accumulated: List[Dict] = []
        seen: set = set()
        hop_log: List[Dict] = []
        current_query = query

        for hop in range(self.MAX_HOPS):
            hop_results = self._single_hop(current_query, filtered, vectorstore, doc_type, k)

            new_results = []
            for r in hop_results:
                key = r["content"][:80]
                if key not in seen:
                    seen.add(key)
                    new_results.append(r)

            accumulated.extend(new_results)
            coverage = self._coverage(query, accumulated)

            hop_log.append(
                {
                    "hop": hop + 1,
                    "query": current_query,
                    "new_docs": len(new_results),
                    "coverage": round(coverage, 2),
                    "stopped": coverage >= self.COVERAGE_THRESHOLD,
                }
            )

            if coverage >= self.COVERAGE_THRESHOLD:
                break

            next_query = self._next_query(query, current_query, accumulated, llm)
            if next_query == current_query:
                break
            current_query = next_query

        results = []
        for rank, doc in enumerate(accumulated[:k]):
            result = dict(doc)
            result["_rag_info"] = {
                "method": "agentic",
                "hops": hop_log,
                "final_rank": rank + 1,
                "total_retrieved": len(accumulated),
                "coverage": hop_log[-1]["coverage"] if hop_log else 0,
            }
            results.append(result)
        return results[:k]

    def _single_hop(
        self, query: str, documents: List[Dict], vectorstore, doc_type, k: int
    ) -> List[Dict]:
        results = self._faiss_search_with_score(query, vectorstore, doc_type, k, documents)
        return results if results else self._keyword_search(query, documents, k)

    def _coverage(self, query: str, results: List[Dict]) -> float:
        if not results:
            return 0.0
        query_words = {w for w in query.lower().split() if len(w) > 1}
        if not query_words:
            return 1.0
        combined = " ".join(r["content"] for r in results).lower()
        covered = sum(1 for w in query_words if w in combined)
        return covered / len(query_words)

    def _next_query(
        self,
        original: str,
        current: str,
        results: List[Dict],
        llm=None,
    ) -> str:
        # LLM 사용 가능하고 Mock이 아닌 경우
        if llm is not None and "Mock" not in type(llm).__name__:
            try:
                ctx = " ".join(r["content"][:80] for r in results[:2])
                prompt = (
                    f"검색 목표: {original}\n"
                    f"현재까지 수집된 내용 요약: {ctx[:250]}\n\n"
                    f"아직 부족한 정보를 채울 짧은 검색 쿼리를 한 줄로 작성하세요 (이전 쿼리: {current}):\n"
                )
                response = llm.invoke([{"role": "user", "content": prompt}])
                new_q = str(response.content).strip().split("\n")[0][:100]
                if new_q and new_q != current:
                    return new_q
            except Exception:
                pass

        # 경험적 확장 (도메인 키워드 기반)
        expansions = [
            ("WBS", "단계별 일정 산정 근거"),
            ("일정", "마일스톤 완료 기준 스프린트"),
            ("버퍼", "지연 원인 리스크 완화"),
            ("교훈", "실패 사례 과거 프로젝트"),
            ("API", "연동 통합 인터페이스"),
        ]
        for keyword, expansion in expansions:
            if keyword in original and expansion not in current:
                return f"{original} {expansion}"

        # 그래도 없으면 약간 변형
        if current == original:
            return f"{original} 세부 일정 교훈"
        return current  # 더 이상 확장 불가 → 루프 종료


# ────────────────────────────────────────────────────────────────
# 5. LLM Reranker (Vanilla 상위 후보 재정렬)
# ────────────────────────────────────────────────────────────────

class LLMRerankerStrategy(BaseRAGStrategy):
    """
    1차: Vanilla dense 검색으로 k*3 후보 확보
    2차: LLM에게 "이 쿼리에 가장 관련있는 문서 top-k"를 물어 재정렬
    LLM 호출 실패 시 1차 결과를 그대로 반환 (fail-safe)
    """

    strategy_key = "llm_rerank"
    CANDIDATE_MULT = 3

    def retrieve(self, query, doc_type, k, documents, vectorstore=None, embeddings=None, llm=None):
        filtered = self._filter_by_type(documents, doc_type)
        if not filtered:
            return []
        candidates = self._faiss_search_with_score(query, vectorstore, doc_type, k * self.CANDIDATE_MULT, filtered)
        if not candidates:
            candidates = self._keyword_search(query, filtered, k * self.CANDIDATE_MULT)
        if len(candidates) <= k or llm is None:
            return candidates[:k]
        try:
            reranked = self._llm_rerank(query, candidates, k, llm)
            return reranked if reranked else candidates[:k]
        except Exception:
            return candidates[:k]

    def _llm_rerank(self, query: str, candidates: List[Dict], k: int, llm) -> List[Dict]:
        import json, re
        preview = []
        for idx, doc in enumerate(candidates):
            snippet = (doc.get("content") or "")[:300].replace("\n", " ")
            preview.append(f"[{idx}] {snippet}")
        prompt = (
            "아래 쿼리에 가장 관련 높은 문서의 인덱스를 관련도 내림차순으로 "
            f"정확히 {k}개 골라 JSON 배열로만 답하시오. 설명/주석 금지.\n\n"
            f"쿼리: {query}\n\n문서 후보:\n" + "\n".join(preview) +
            f"\n\n출력 예: [3, 0, 7]"
        )
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", str(resp))
        m = re.search(r"\[\s*[0-9,\s]+\]", text)
        if not m:
            return candidates[:k]
        picks = json.loads(m.group(0))
        selected = []
        seen = set()
        for i in picks:
            if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen:
                doc = dict(candidates[i])
                doc["_rag_info"] = {**doc.get("_rag_info", {}), "rerank_rank": len(selected) + 1}
                selected.append(doc)
                seen.add(i)
                if len(selected) >= k:
                    break
        # 부족분은 원본 순서로 채움
        if len(selected) < k:
            for doc in candidates:
                if doc not in selected:
                    selected.append(doc)
                if len(selected) >= k:
                    break
        return selected


# ────────────────────────────────────────────────────────────────
# 팩토리
# ────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, BaseRAGStrategy] = {
    "vanilla": VanillaRAGStrategy(),
    "hybrid": HybridRAGStrategy(),
    "graph": GraphRAGStrategy(),
    "agentic": AgenticRAGStrategy(),
    "llm_rerank": LLMRerankerStrategy(),
}


def get_strategy(strategy_key: str) -> BaseRAGStrategy:
    """전략 키로 인스턴스 반환. 알 수 없는 키는 vanilla 반환."""
    return _REGISTRY.get(strategy_key, _REGISTRY["vanilla"])


def list_strategy_keys() -> List[str]:
    return list(_REGISTRY.keys())
