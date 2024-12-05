import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from typing import Optional, List, Dict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.data_processor import DataProcessor
from modules.hybrid_retriever1 import HybridRetriever
from modules.conversation_manager import ConversationManager

class VegetableBot:
    def __init__(self, openai_api_key: str):
        """챗봇 초기화"""
        self.openai_api_key = openai_api_key
        
        # 컴포넌트 초기화
        self.initialize_components()
        
        # 카테고리 설정
        self.categories = ['상품 정보', '주문/배송 정보', '교환/환불 정보', '상담원 연결']
        
    def initialize_components(self):
        """컴포넌트 초기화"""
        print("Initializing components...")
        
        # 데이터 프로세서 초기화
        self.data_processor = DataProcessor(self.openai_api_key)
        
        try:
            # 저장된 데이터가 있는지 확인
            if os.path.exists("processed_data/documents.pkl"):
                print("Loading processed data...")
                self.data_processor.load_processed_data()
            else:
                print("Processing new data...")
                self.data_processor.load_csv_files()
                self.data_processor.create_embeddings()
                os.makedirs("processed_data", exist_ok=True)
                self.data_processor.save_processed_data()
        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            if not os.path.exists("data"):
                os.makedirs("data", exist_ok=True)
            self.data_processor.load_csv_files()
            self.data_processor.create_embeddings()
        
        # 검색기 초기화
        embeddings = self.data_processor.get_embeddings()
        self.retriever = HybridRetriever(
            documents=self.data_processor.documents,
            embeddings=embeddings
        )
        
        # 대화 관리자 초기화
        self.conversation_manager = ConversationManager()
    
    def get_response(self, user_input: str, category: str) -> str:
        """사용자 입력에 대한 응답 생성"""
        try:
            # ID가 필요한 카테고리 체크
            if category in ['주문/배송 정보', '교환/환불 정보']:
                user_id = st.session_state.get('user_id')
                if not user_id:
                    return "주문 조회를 위해 ID를 입력해주세요."
            
            if category == "상담원 연결":
                return "상담원과 연결을 도와드리겠습니다.\n\n📞 상담원 전화번호: 010-1234-5678"
            
            query_embedding = self._create_embedding(user_input)
            search_results = self.retriever.search(
                query=user_input,
                query_embedding=query_embedding,
                category=category,
                k=3
            )
            
            return self._generate_gpt_response(
                query=user_input, 
                search_results=search_results, 
                category=category
            )
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다. 다시 시도해 주세요."
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """텍스트의 임베딩 생성"""
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _generate_gpt_response(self, query: str, search_results: list, category: str) -> str:
        """GPT를 사용하여 응답 생성"""
        client = OpenAI(api_key=self.openai_api_key)
        
        # 검색 결과로부터 컨텍스트 구성
        context = "\n".join([
            f"- {result.content}" for result in search_results
        ])
        
        # 대화 이력 가져오기
        conversation_history = self.conversation_manager.get_context_window(window_size=3)
        
        # 카테고리별 프롬프트 설정
        category_prompts = {
            "공통 필수 사항" :"""
            - 당신은 이 스토어의 상품 전문가 입니다.
            - 이전 채팅 기록을 기반으로 답변을 합니다
            - 밝은 태도와 목소리로 답변합니다.
            - 명확한 정보만 답변합니다
            """,

            "상품 정보": """
            if 상품의 구매에 대한 문의 
            - 상품 가격
            - 상품 상세 정보
            - 상품의 구매 링크 제공
            elif 상품의 상식적인 정보(알러지, 보관법)
            - 상식적인 정보 제공
            elif 상품의 활용에 대한 문의(활용법, 레시피 등)
            - 활용법에 대한 정보 제공
            else
            - 상품의 상세 내용 제공
            - 상품의 일반적인 내용 제공

            참조데이터:[/Users/sungminhong/Documents/naver1/ver1/data/product_info.csv, 
                    /Users/sungminhong/Documents/naver1/ver1/data/review.csv,
                    /Users/sungminhong/Documents/naver1/ver1/data/policy_info.csv,
                    /Users/sungminhong/Documents/naver1/ver1/data/QnA.csv, 
                    /Users/sungminhong/Documents/naver1/ver1/data/FAQ.csv]
            """,
                 
            "주문/배송 정보": """
            다음 정보를 확인하여 답변해주세요:
            - 상품의 이름
            - 배송 예정일과 현재 상태
            - 배송 관련 요청사항 처리
            - 배송 조회 방법 안내
            
            참조데이터:[/Users/sungminhong/Documents/naver1/ver1/data/product_info.csv,
                     /Users/sungminhong/Documents/naver1/ver1/data/order_info.csv, 
                     /Users/sungminhong/Documents/naver1/ver1/data/policy_info.csv]

            필수 사항
            - 당신은 이 스토어의 상품 전문가 입니다.
            - 채팅 히스토리를 기반으로 한 답변을 합니다
            - 밝은 태도와 목소리로 답변합니다.
            - 명확한 정보만 답변합니다
            """,
            
            
            "교환/환불 정보": """
            다음 사항을 포함하여 안내해주세요:
            - 상품 사진 업데이트(스토어 링크)
            - 교환/환불 가능 여부
            - 처리 절차와 방법
            - 소요 시간과 주의사항
            - 교환 비용(/Users/sungminhong/Documents/naver1/ver1/data/policy_info.csv['exchange_cost'])
            - 환불 비용(/Users/sungminhong/Documents/naver1/ver1/data/policy_info.csv['return_cost'])

            "참조데이터" : [/Users/sungminhong/Documents/naver1/ver1/data/product_info.csv,
                         /Users/sungminhong/Documents/naver1/ver1/data/order_info.csv,
                         /Users/sungminhong/Documents/naver1/ver1/data/policy_info.csv]
            """,


            "상담원 연결": """
            다음 사항을 포함하여 안내해 주세요:
            - 상담원 전화번호
            
            참조데이터 : [.data/product_info.csv]
            """
        }
        
        category_prompt = category_prompts.get(category, "")
        
        # 시스템 프롬프트
        system_prompt = f"""당신은 채소 쇼핑몰 '채소애'의 친절한 고객 서비스 assistant입니다.
    
다음 지침을 따라주세요:
1. 이전 대화 내용을 참고하여 일관성 있게 답변하세요
2. 현재 {category} 카테고리의 문의를 처리하고 있습니다
3. 항상 공손하고 친절한 어투를 사용하세요
4. 정확한 정보만을 제공하세요
5. 이전 문의와 관련된 추가 정보가 있다면 언급해주세요

{category_prompts.get(category, "")}"""
        
        # 사용자 프롬프트
        user_prompt = f"""
이전 대화 내역:
{conversation_history}

관련 정보:
{context}

사용자 질문: {query}

위 정보를 바탕으로:
1. 이전 대화 맥락을 유지하며 답변해주세요
2. 새로운 정보가 있다면 이전 답변과 연결하여 설명해주세요
3. 답변 마지막에는 추가 문의사항이 있는지 확인해주세요
"""

        
        # API 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7  # 적절한 창의성 유지
        )
        
        return response.choices[0].message.content

def get_welcome_message(category: str, user_id: Optional[str] = None) -> str:
    """카테고리별 웰컴 메시지와 주문 정보 반환"""
    
    # 주문 정보 조회 함수 (예시)
    def get_order_info(user_id: str) -> str:
        """CSV 파일에서 주문 정보를 조회하는 함수"""
        try:
            # 주문 정보 CSV 파일 읽기
            order_df = pd.read_csv('/Users/sungminhong/Documents/naver1/ver1/data/order_info.csv')
            policy_df = pd.read_csv('/Users/sungminhong/Documents/naver1/ver1/data/policy_info.csv')
            
            # user_id로 주문 필터링
            user_orders = order_df[order_df['user_id'] == user_id]
            
            if user_orders.empty:
                return "주문 내역이 없습니다."
            
            # 주문 정보 포매팅
            order_info = "📦 주문 내역:\n"
            for idx, order in user_orders.iterrows():
                order_info += f"""
    {idx+1}. 주문번호: {order['order_id']}
    - 상품명 : {order['title']}
    - 주문일자: {order['start_date']}
    - 배송예정일: {order['expected_destination']}
    - 배송일: {order['delivery_date']}
    - 배송상태: {order['delivery_status']}
    - 송장번호: {order['invoice']}
    - 택배사 : {policy_df['delivery_com'],policy_df['delivery_com_url']}
    """
            return order_info
        
        except Exception as e:
            print(f"Error fetching order info: {str(e)}")
            return "주문 정보 조회 중 오류가 발생했습니다."

    base_messages = {
        "상품 정보": """
안녕하세요! 채소애의 상품정보 메뉴를 선택하셨습니다. 🥬

다음과 같은 정보를 알려드릴 수 있습니다:
- 상품 정보(가격, 원산지 등)
- 상품 보관법, 영양정보
- 상품 추천
- 상품 활용법
""",
        "주문/배송 정보": f"""
안녕하세요! 채소애의 주문/배송 정보 메뉴를 선택하셨습니다. 👩‍🍳

고객님의 ID: {user_id}
{get_order_info(user_id) if user_id else '주문 조회를 위해 ID를 입력해주세요.'}

주문이나 배송에 대해 궁금하신 점을 물어보세요!
""",
        "교환/환불 정보": f"""
안녕하세요! 채소애의 교환/환불 정보 메뉴를 선택하셨습니다. 🥗

고객님의 ID: {user_id}
{get_order_info(user_id) if user_id else '주문 조회를 위해 ID를 입력해주세요.'}

교환이나 환불에 대해 궁금하신 점을 물어보세요!
""",
        "상담원 연결": """
상담원에게 직접 연결해드리겠습니다.

📞 상담원 전화번호: [010-1234-5678](010-1234-5678)
"""
    }
    
    return base_messages.get(category, "안녕하세요! 무엇을 도와드릴까요?")

def create_streamlit_ui():
    """Streamlit UI 생성"""
    st.title("채소애 AI Assistant 🥬")
    
    # OpenAI API 키 확인
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다!")
        return
    
    # 챗봇 초기화
    if 'bot' not in st.session_state:
        st.session_state.bot = VegetableBot(openai_api_key)
    
    # 세션 상태 초기화
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_category' not in st.session_state:
        st.session_state.current_category = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    # 사이드바
    # 사이드바
    with st.sidebar:
        st.title("카테고리 선택")
        
        # ID 입력 필드
        user_id = st.text_input("주문 조회를 위한 ID를 입력하세요:", key="id_input")
        if user_id:
            if 'user_id' not in st.session_state or st.session_state.user_id != user_id:
                st.session_state.user_id = user_id
                st.success(f"ID가 입력되었습니다: {user_id}")
                
                # 현재 선택된 카테고리가 주문 관련이면 자동으로 정보 업데이트
                if st.session_state.get('current_category') in ['주문/배송 정보', '교환/환불 정보']:
                    welcome_msg = get_welcome_message(
                        st.session_state.current_category, 
                        user_id
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": welcome_msg
                    })
                    st.rerun()
        
        # 카테고리 버튼들
        for category in st.session_state.bot.categories:
            if st.button(category):
                st.session_state.current_category = category
                
                # 주문 관련 카테고리 선택 시 ID 체크
                if category in ['주문/배송 정보', '교환/환불 정보']:
                    welcome_msg = get_welcome_message(category, st.session_state.user_id)
                else:
                    welcome_msg = get_welcome_message(category)
                
                # 웰컴 메시지 추가
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": welcome_msg
                })
                st.session_state.bot.conversation_manager.add_message(
                    role="assistant",
                    content=welcome_msg,
                    category=category
                )
                st.rerun()
        
        # 대화 내역 지우기 버튼
        if st.button("대화 내역 지우기"):
            st.session_state.messages = []
            st.session_state.bot.conversation_manager.clear_history()
            st.rerun()
    
    # 채팅 화면
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요"):
        if not st.session_state.current_category:
            st.warning("먼저 카테고리를 선택해주세요!")
            return
        
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.bot.conversation_manager.add_message(
            role="user",
            content=prompt,
            category=st.session_state.current_category
        )
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 응답 생성
        with st.spinner('응답을 생성하고 있습니다...'):
            response = st.session_state.bot.get_response(
                prompt, 
                st.session_state.current_category
            )
        
        # 응답 추가
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.bot.conversation_manager.add_message(
            role="assistant",
            content=response,
            category=st.session_state.current_category
        )
        
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    create_streamlit_ui()