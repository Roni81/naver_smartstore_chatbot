import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from openai import OpenAI
from dataclasses import dataclass
import pickle
from pathlib import Path

@dataclass
class Document:
    """문서를 표현하는 데이터 클래스"""
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

class DataProcessor:
    def __init__(self, openai_api_key: str, data_dir: str = "data"):
        """
        데이터 처리 및 저장을 위한 클래스 초기화
        """
        self.client = OpenAI(api_key=openai_api_key)  # OpenAI 클라이언트 초기화
        self.data_dir = Path(data_dir)
        self.documents = []
        self.embedding_dim = 1536

        
    def load_csv_files(self) -> None:
        """data 디렉토리의 모든 CSV 파일을 로드하고 처리합니다."""
        print("Loading CSV files...")
        csv_files = list(self.data_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                print(f"Processing {csv_file.name}...")
                
                # CSV 파일명으로 카테고리 결정
                category = self._determine_category(csv_file.stem)
                
                for _, row in df.iterrows():
                    # 메타데이터 생성
                    metadata = {
                        'source': csv_file.stem,
                        'category': category,
                        'row_idx': row.name
                    }
                    
                    # 문서 내용 생성 (모든 컬럼을 문자열로 결합)
                    content = " ".join(
                        f"{col}: {str(val)}" for col, val in row.items()
                        if pd.notna(val)  # NaN 값 제외
                    )
                    
                    # Document 객체 생성
                    doc = Document(
                        content=content,
                        metadata=metadata
                    )
                    self.documents.append(doc)
                    
            except Exception as e:
                print(f"Error processing {csv_file.name}: {str(e)}")
                
        print(f"Total documents loaded: {len(self.documents)}")
    
    def _determine_category(self, filename: str) -> str:
        """파일명을 기반으로 카테고리 결정"""
        category_mapping = {
            'product': '상품 정보',
            'order': '주문/배송 정보',
            'refund': '교환/환불 정보',
            'customer': '상담원 연결'
        }
        
        for key, value in category_mapping.items():
            if key in filename.lower():
                return value
        return '기타'
    
    def create_embeddings(self, batch_size: int = 100) -> None:
        """문서들의 임베딩을 생성합니다."""
        print("Creating embeddings...")
        
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            batch_texts = [doc.content for doc in batch]
            
            try:
                # 새로운 API 방식으로 임베딩 생성
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model="text-embedding-ada-002"
                )
                
                # 임베딩 할당
                for j, embedding_data in enumerate(response.data):
                    embedding = np.array(embedding_data.embedding, dtype=np.float32)
                    self.documents[i + j].embedding = embedding
                
                print(f"Processed {i + len(batch)}/{len(self.documents)} documents")
                
            except Exception as e:
                print(f"Error creating embeddings: {str(e)}")
                # 임시 임베딩으로 대체 (테스트용)
                for doc in batch:
                    if doc.embedding is None:
                        doc.embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        
    def get_embeddings(self) -> np.ndarray:
        """모든 문서의 임베딩을 반환합니다."""
        if not self.documents:
            raise ValueError("문서가 없습니다.")
        
        # 임베딩이 없는 문서 확인
        docs_without_embeddings = [i for i, doc in enumerate(self.documents) if doc.embedding is None]
        
        if docs_without_embeddings:
            print(f"Creating embeddings for {len(docs_without_embeddings)} documents...")
            self.create_embeddings()
        
        return np.vstack([doc.embedding for doc in self.documents])
    
    def save_processed_data(self, save_dir: str = "processed_data") -> None:
        """처리된 데이터를 저장합니다."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        with open(save_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"Data saved to {save_dir}")
    
    def load_processed_data(self, load_dir: str = "processed_data") -> None:
        """저장된 데이터를 불러옵니다."""
        load_path = Path(load_dir)
        
        if (load_path / "documents.pkl").exists():
            with open(load_path / "documents.pkl", 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Data loaded from {load_dir}")
        else:
            raise FileNotFoundError(f"No processed data found in {load_dir}")

# 테스트 코드
if __name__ == "__main__":
    # OpenAI API 키 설정
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    
    # 데이터 프로세서 테스트
    processor = DataProcessor(api_key)
    
    # 이미 처리된 데이터가 있는지 확인
    if os.path.exists("processed_data/documents.pkl"):
        processor.load_processed_data()
    else:
        processor.load_csv_files()
        processor.create_embeddings()
        processor.save_processed_data()
    
    print(f"Loaded {len(processor.documents)} documents")