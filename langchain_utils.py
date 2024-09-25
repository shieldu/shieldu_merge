import os
from dotenv import load_dotenv  # .env 파일에 저장된 환경변수를 불러오기 위한 패키지
from langchain_community.document_loaders import PyPDFLoader  # PDF 파일을 불러오는 모듈
from langchain.text_splitter import CharacterTextSplitter  # 긴 텍스트를 여러 조각으로 나누는 모듈
from langchain_community.embeddings import HuggingFaceEmbeddings  # 텍스트를 벡터로 변환하는 임베딩 모듈
from langchain_community.vectorstores import Chroma  # 벡터 데이터를 저장하고 검색하는 데이터베이스
from langchain_community.chat_models import ChatOpenAI  # OpenAI GPT 모델을 활용한 챗봇 API
from langchain.chains import ConversationalRetrievalChain  # 대화형 정보 검색 체인

# 환경 변수를 로드하여 API 키와 같은 중요한 정보를 관리합니다.
load_dotenv()

class LangChainHelper:
    def __init__(self):
        """LangChainHelper 클래스의 생성자.
        
        - vector_store: 벡터 스토어를 저장할 변수, PDF에서 추출된 텍스트의 임베딩이 저장됨.
        - embeddings: Hugging Face 임베딩 모델을 사용해 텍스트를 벡터로 변환하는 역할.
        - openai_api_key: OpenAI API 키를 환경 변수에서 불러옴.
        - chat_history: 대화 기록을 저장하는 리스트.
        """
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.chat_history = []  # 대화 기록 저장을 위한 리스트

    def process_pdf(self, pdf_file):
        """PDF 파일을 처리하는 함수.
        
        - 파일을 서버에 저장한 후 PyPDFLoader로 불러옴.
        - PDF의 내용을 텍스트로 변환하고, 텍스트가 너무 길 경우 적절한 크기로 분할함.
        - 텍스트를 임베딩하여 벡터 스토어(Chroma DB)에 저장.
        
        Args:
            pdf_file: 업로드된 PDF 파일 객체.
        """
        # 'uploads' 폴더에 PDF 파일을 저장
        file_path = os.path.join('uploads', pdf_file.filename) # PDF 파일을 저장할 경로를 만든다.
        pdf_file.save(file_path) # 해당 경로에 PDF 파일을 실제로 저장하는 명령
        
        # LangChain 커뮤니티에서 제공하는 PDF 파일을 불러오는 도구인 PyPDFLoader를 사용해 PDF 파일 로드
        loader = PyPDFLoader(file_path)
        data = loader.load() # PDF 파일의 각 페이지를 data라는 변수에 저장하며, 페이지별로 나누어 텍스트를 처리할 수 있습니다.

        # 각 페이지에서 텍스트 내용을 추출하여, 리스트로 만든 이후 이를 하나의 긴 문자열로 결합
        text = "".join([doc.page_content for doc in data])

        # 텍스트를 1000자 단위로 나누고, 각 조각에서 200자씩 중복되도록 설정
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        # chunks 리스트에 들어있는 각각의 텍스트 조각을 임베딩 모델을 사용하여 벡터로 변환하고, 이를 Chroma 데이터베이스에 저장
        self.vector_store = Chroma.from_texts(chunks, self.embeddings)

    def get_answer(self, question):
        """사용자의 질문에 대한 답변을 생성하는 함수.
        
        - 벡터 스토어(DB)에서 유사한 텍스트를 검색한 후 GPT 모델을 사용해 답변을 생성.
        - 대화 기록을 함께 전달하여 대화 맥락을 유지함.
        
        Args:
            question: 사용자가 입력한 질문.
        
        Returns:
            GPT 모델이 생성한 답변.
        """
        # 벡터 스토어가 없을 경우 PDF 파일 업로드 요청 메시지 반환
        if self.vector_store is None:
            return "먼저 PDF 파일을 업로드하세요."

        # 유사한 텍스트를 벡터 스토어에서 검색 (질문의 임베딩 벡터와 가장 유사한 벡터를 검색해 반환)
        docs = self.vector_store.similarity_search(question)

        # OpenAI GPT-3.5-turbo 모델을 사용해 답변 생성
        llm = ChatOpenAI(api_key=self.openai_api_key, model="gpt-3.5-turbo", temperature=0.1) # temperature는 생성되는 텍스트의 창의성과 무작위성 제어하기 위한 것으로 2에 가까울수록 창의적이다. 
        # 벡터 스토어에서 검색된 결과를 바탕으로 GPT 모델이 답변을 생성할 수 있도록 대화형 정보 검색 체인을 만든다.
        chain = ConversationalRetrievalChain.from_llm(llm, self.vector_store.as_retriever())

        # 사용자의 질문과 대화 기록을 함께 전달하여 GPT 모델이 더 일관성 있는 답변을 생성
        inputs = {
            "question": question,
            "chat_history": self.chat_history # self.chat_history는 이전 질문과 답변이 저장된 리스트로, 대화의 맥락을 유지하는 데 사용
        }

        # ConversationalRetrievalChain을 실행하여 입력된 질문과 대화 기록을 바탕으로 GPT 모델이 답변을 생성
        result = chain.run(inputs)

        # 새로운 질문과 그에 대한 답변을 대화 기록에 추가
        self.chat_history.append((question, result))

        return result
