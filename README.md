# Naver Smartstore 챗봇 SaaS

네이버 스마트 스토어 챗봇 제작 PoC 프로젝트
  
## 기술스택
- OpenAI API
- GPTs
- FAISS Venctor db
- FAISS + RM25 Ensemble Retriever

## Project Structure
```python
project/
│
├── data/
│       └── FAQ.py
│       └── order_info.py
│       └── policy_info.py
│       └── product_info.py
│       └── review.py
│       └── QnA.py
│
├── modules/
│       └── __init__.py
│       └── conversation_manager.py
│       └── data_processor.py
│       └── hybrid_retriever.py
│
├── processed_data/
│   └── documents.py
│
└── README.md
└── main.py
└── requirements.txt
```
