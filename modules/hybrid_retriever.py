from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
import faiss
from nltk.tokenize import word_tokenize
import nltk
from sklearn.preprocessing import MinMaxScaler

@dataclass
class SearchResult:
    """검색 결과를 저장하는 데이터 클래스"""
    content: str
    metadata: Dict
    score: float
    category: str

class HybridRetriever:
    def __init__(self, documents: List[Dict], embeddings: np.ndarray, categories: List[str]):
        """
        BM25와 FAISS를 결합한 하이브리드 검색 시스템
        
        Args:
            documents: 문서 리스트
            embeddings: 문서 임베딩 배열
            categories: 카테고리 리스트
        """
        self.documents = documents
        self.categories = categories
        
        # BM25 초기화
        nltk.download('punkt')
        self.tokenized_corpus = [word_tokenize(doc['content'].lower()) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # FAISS 초기화
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        # 정규화를 위한 스케일러
        self.scaler = MinMaxScaler()
        
    def search(self, 
              query: str, 
              query_embedding: np.ndarray, 
              category: Optional[str] = None,
              k: int = 5, 
              alpha: float = 0.5) -> List[SearchResult]:
        """
        하이브리드 검색 수행
        
        Args:
            query: 검색 쿼리
            query_embedding: 쿼리 임베딩
            category: 검색할 카테고리 (선택사항)
            k: 반환할 결과 수
            alpha: BM25와 FAISS 점수 가중치 (0: BM25만 사용, 1: FAISS만 사용)
            
        Returns:
            검색 결과 리스트
        """
        # BM25 검색
        tokenized_query = word_tokenize(query.lower())
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_scores = self.scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        
        # FAISS 검색
        faiss_distances, _ = self.index.search(query_embedding.reshape(1, -1), len(self.documents))
        faiss_scores = 1 / (1 + faiss_distances.flatten())  # 거리를 유사도 점수로 변환
        
        # 앙상블 점수 계산
        ensemble_scores = alpha * faiss_scores + (1 - alpha) * bm25_scores
        
        # 카테고리 필터링
        if category:
            category_mask = [doc['metadata']['category'] == category for doc in self.documents]
            ensemble_scores = ensemble_scores * category_mask
        
        # 상위 k개 결과 선택
        top_indices = np.argsort(ensemble_scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if ensemble_scores[idx] > 0:  # 0점 이상인 결과만 포함
                results.append(SearchResult(
                    content=self.documents[idx]['content'],
                    metadata=self.documents[idx]['metadata'],
                    score=float(ensemble_scores[idx]),
                    category=self.documents[idx]['metadata']['category']
                ))
        
        return results

class CategoryManager:
    """카테고리별 검색 관리 클래스"""
    def __init__(self, categories: List[str]):
        self.categories = categories
        self.category_descriptions = {
            "상품 정보": "상품의 상세 정보, 가격, 원산지 등을 검색합니다",
            "주문/배송": "주문 현황과 배송 정보를 확인합니다",
            "교환/환불": "교환 및 환불 관련 정보를 찾습니다",
            "FAQ": "자주 묻는 질문과 답변을 검색합니다"
        }
    
    def get_category_description(self, category: str) -> str:
        """카테고리 설명을 반환합니다"""
        return self.category_descriptions.get(category, "")
    
    def filter_results_by_category(self, 
                                 results: List[SearchResult], 
                                 category: str) -> List[SearchResult]:
        """특정 카테고리의 검색 결과만 필터링합니다"""
        return [r for r in results if r.category == category]

class QueryProcessor:
    """쿼리 전처리 및 임베딩 생성 클래스"""
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        
    def process_query(self, query: str) -> tuple:
        """
        쿼리를 전처리하고 임베딩을 생성합니다
        
        Returns:
            (전처리된 쿼리, 쿼리 임베딩)
        """
        import openai
        openai.api_key = self.openai_api_key
        
        # 쿼리 전처리
        processed_query = query.strip()
        
        # OpenAI 임베딩 생성
        response = openai.Embedding.create(
            input= processed_query,
            model="text-embedding-3-small" #최신 임베딩 모델로 변환 openai 최 우선으로 진행
        )
        query_embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
        
        return processed_query, query_embedding