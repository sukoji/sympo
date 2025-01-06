"""
FAISS 기반 벡터 데이터베이스
PRD, 팀원 프로필, 참고 WBS, 회의록을 단일 Vector DB에 인덱싱합니다.
"""
import os
from typing import List, Dict, Any

from schemas.prd_schema import PRDInput
from schemas.member_schema import MemberProfile


class WBSVectorStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self._documents: List[Dict[str, Any]] = []
        self._vectorstore = None
        self._use_faiss = False
        self._embeddings = None
        self._try_init_faiss()

    def _try_init_faiss(self):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._use_faiss = True
        except Exception as e:
            print(f"[WARN] FAISS 초기화 실패, In-Memory 모드로 동작: {e}")
            self._use_faiss = False

    def add_prd(self, prd: PRDInput) -> None:
        content = f"""[PRD] {prd.project_name}
목표: {prd.project_goal}
범위: {prd.scope}
핵심 기능: {', '.join(prd.key_features)}
기술 스택: {', '.join(prd.tech_stack_requirements)}
제약사항: {', '.join(prd.special_constraints)}
"""
        self._add_document(content, {"doc_type": "prd", "project_name": prd.project_name})

    def add_member(self, member: MemberProfile) -> None:
        role = member.role.value if member.role else "Unknown"
        content = f"""[팀원] {member.name} ({role})
기술스택: {', '.join(member.tech_stack)}
강점: {', '.join(member.strengths)}
약점: {', '.join(member.weaknesses)}
경력: {member.years_of_experience}년
"""
        self._add_document(content, {
            "doc_type": "member",
            "member_id": member.member_id,
            "member_name": member.name,
            "role": role,
        })

    def add_reference_wbs(self, wbs_text: str, project_name: str = "참고 프로젝트") -> None:
        for i, chunk in enumerate(self._chunk_text(wbs_text)):
            self._add_document(chunk, {"doc_type": "reference_wbs", "project_name": project_name, "chunk_index": i})

    def add_disc_profile(self, profile: Any) -> None:
        content = profile.to_rag_text()
        self._add_document(content, {
            "doc_type": "disc_profile",
            "member_name": profile.name,
            "type_code": getattr(profile, "type_code", ""),
            "primary_type": getattr(profile, "primary_type", ""),
        })

    def add_meeting_log(self, meeting_text: str, meeting_date: str = "") -> None:
        for i, chunk in enumerate(self._chunk_text(meeting_text, chunk_size=400)):
            self._add_document(chunk, {"doc_type": "meeting_log", "meeting_date": meeting_date, "chunk_index": i})

    def retrieve_by_type(self, query: str, doc_type: str, k: int = 3) -> List[Dict[str, Any]]:
        if self._use_faiss and self._vectorstore:
            try:
                docs = self._vectorstore.similarity_search(query, k=k * 3, filter={"doc_type": doc_type})
                return [{"content": d.page_content, "metadata": d.metadata} for d in docs[:k]]
            except Exception:
                pass
        filtered = [d for d in self._documents if d["metadata"].get("doc_type") == doc_type]
        return self._simple_search(query, filtered, k)

    def retrieve_all_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self._use_faiss and self._vectorstore:
            try:
                docs = self._vectorstore.similarity_search(query, k=k)
                return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
            except Exception:
                pass
        return self._simple_search(query, self._documents, k)

    def _add_document(self, content: str, metadata: Dict[str, Any]) -> None:
        self._documents.append({"content": content, "metadata": metadata})
        if self._use_faiss and self._embeddings:
            try:
                from langchain_core.documents import Document
                from langchain_community.vectorstores import FAISS

                doc = Document(page_content=content, metadata=metadata)
                if self._vectorstore is None:
                    self._vectorstore = FAISS.from_documents([doc], self._embeddings)
                else:
                    self._vectorstore.add_documents([doc])
            except Exception as e:
                print(f"[WARN] FAISS 추가 실패: {e}")

    def _simple_search(self, query: str, documents: List[Dict], k: int) -> List[Dict[str, Any]]:
        query_words = set(query.lower().split())

        def score(doc):
            return len(query_words & set(doc["content"].lower().split()))

        return sorted(documents, key=score, reverse=True)[:k]

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500) -> List[str]:
        words = text.split()
        chunks = []
        step = max(chunk_size // 5, 1)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + step])
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]

    def get_stats(self) -> Dict[str, int]:
        from collections import Counter

        return dict(Counter(d["metadata"].get("doc_type", "unknown") for d in self._documents))
