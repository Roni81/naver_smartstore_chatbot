from typing import List, Dict, Optional
from datetime import datetime
import json
from dataclasses import dataclass, asdict
import uuid

@dataclass
class Message:
    """대화 메시지를 표현하는 데이터 클래스"""
    id: str
    role: str  # 'user' 또는 'assistant'
    content: str
    category: str
    timestamp: str
    metadata: Optional[Dict] = None

class ConversationManager:
    def __init__(self, max_history: int = 20):
        """대화 관리를 위한 클래스 초기화"""
        self.conversations = {}
        self.max_history = max_history
        self.current_session_id = None
        # 초기 세션 생성
        self.current_session_id = self.initialize_session()
    
    def initialize_session(self) -> str:
        """새로운 대화 세션 초기화"""
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = []  # 새 세션의 대화 기록 초기화
        return session_id
    
    def add_message(self, 
               role: str, 
               content: str, 
               category: str,
               session_id: Optional[str] = None,
               metadata: Optional[Dict] = None) -> Message:
        """새로운 메시지를 대화 기록에 추가"""
        # 세션 ID가 None이거나 conversations에 없는 경우 새 세션 생성
        if session_id is None or session_id not in self.conversations:
            session_id = self.initialize_session()
            self.current_session_id = session_id
        
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            category=category,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        # 해당 세션의 메시지 리스트가 없으면 생성
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append(message)
        
        # 최대 기록 수 제한
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
        
        return message
        
    def get_conversation_history(self, 
                               session_id: Optional[str] = None, 
                               last_n: Optional[int] = None) -> List[Message]:
        """대화 기록 조회"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id not in self.conversations:
            # 세션이 없으면 새로 생성
            self.current_session_id = self.initialize_session()
            session_id = self.current_session_id
        
        history = self.conversations[session_id]
        if last_n is not None:
            history = history[-last_n:]
        
        return history
    
    def get_category_history(self, 
                           category: str, 
                           session_id: Optional[str] = None) -> List[Message]:
        """특정 카테고리의 대화 기록만 반환합니다"""
        history = self.get_conversation_history(session_id)
        return [msg for msg in history if msg.category == category]
    
    def get_last_message(self, session_id: Optional[str] = None) -> Optional[Message]:
        """마지막 메시지를 반환합니다"""
        history = self.get_conversation_history(session_id)
        return history[-1] if history else None
    
    def clear_history(self, session_id: Optional[str] = None):
        """대화 기록을 초기화합니다"""
        if session_id is None:
            session_id = self.current_session_id
        
        if session_id in self.conversations:
            self.conversations[session_id] = []
    
    def save_conversation(self, 
                         filepath: str, 
                         session_id: Optional[str] = None):
        """대화 기록을 파일로 저장합니다"""
        if session_id is None:
            session_id = self.current_session_id
            
        history = self.get_conversation_history(session_id)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'session_id': session_id,
                'messages': [asdict(msg) for msg in history]
            }, f, ensure_ascii=False, indent=2)
    
    def load_conversation(self, filepath: str) -> str:
        """
        저장된 대화 기록을 불러옵니다
        
        Returns:
            불러온 대화의 session_id
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        session_id = data['session_id']
        self.conversations[session_id] = [
            Message(**msg) for msg in data['messages']
        ]
        return session_id
    
    def build_context(self, 
                     session_id: Optional[str] = None, 
                     max_context: int = 5) -> str:
        """
        RAG 모델을 위한 컨텍스트를 생성합니다
        
        Args:
            session_id: 세션 ID
            max_context: 포함할 최근 메시지 수
            
        Returns:
            컨텍스트 문자열
        """
        history = self.get_conversation_history(session_id, last_n=max_context)
        context = []
        
        for msg in history:
            role = "사용자" if msg.role == "user" else "assistant"
            context.append(f"{role}: {msg.content}")
            
        return "\n".join(context)
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict:
        """세션 요약 정보를 반환합니다"""
        history = self.get_conversation_history(session_id)
        
        return {
            'total_messages': len(history),
            'user_messages': sum(1 for msg in history if msg.role == 'user'),
            'assistant_messages': sum(1 for msg in history if msg.role == 'assistant'),
            'categories': list(set(msg.category for msg in history)),
            'start_time': history[0].timestamp if history else None,
            'last_time': history[-1].timestamp if history else None
        }
    def get_context_window(self, 
                      session_id: Optional[str] = None, 
                      window_size: int = 5) -> str:
        """
        최근 대화 컨텍스트를 문자열로 반환
        
        Args:
            session_id: 세션 ID
            window_size: 포함할 최근 메시지 수
            
        Returns:
            최근 대화 내용을 포함한 문자열
        """
        # 최근 메시지 가져오기
        history = self.get_conversation_history(session_id, last_n=window_size)
        
        # 대화 내용 포매팅
        context_messages = []
        for msg in history:
            role_text = "사용자" if msg.role == "user" else "assistant"
            context_messages.append(f"{role_text}: {msg.content}")
        
        return "\n".join(context_messages)