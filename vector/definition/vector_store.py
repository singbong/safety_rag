import os
from langchain_core.embeddings import Embeddings
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
import faiss
import numpy as np
import pickle
import glob
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_community import VertexAIRank
from langchain_community.docstore.in_memory import InMemoryDocstore

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["PROJECT_ID"] = os.getenv("PROJECT_ID")
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

class VertexEmbeddings:
    """
    Google Vertex AI를 이용한 LangChain Embeddings 래퍼 클래스
    """
    def __init__(self, project_id: Optional[str] = None, region: str = "us-central1", model: str = "gemini-embedding-001"):
        if project_id is None:
            project_id = os.getenv("PROJECT_ID")
        if not project_id:
            raise ValueError("PROJECT_ID가 설정되지 않았습니다.")
        
        self.model = model
        self.project_id = project_id
        self.region = region
        self.output_dir = "../data/vector_store"
        os.makedirs(self.output_dir, exist_ok=True)

        # Vertex AI 초기화
        try:
            vertexai.init(location=self.region)
            self.embedding_model = TextEmbeddingModel.from_pretrained(model)
            print("Vertex AI 초기화 성공")
        except Exception as e:
            print(f"Vertex AI 초기화 실패: {e}")
            print("서비스 계정 키가 올바르게 설정되었는지 확인하세요.")
            raise

    def embedding_document(self, content: str) -> List[float]:
        inputs = [TextEmbeddingInput(text=content)]
        result = self.embedding_model.get_embeddings(inputs)
        return result[0].values
    def embedding_query(self, content: str) -> List[float]:
        inputs = [TextEmbeddingInput(text=content)]
        result = self.embedding_model.get_embeddings(inputs)
        return result[0].values

class store_vector_db:
    """
    FAISS 벡터 데이터베이스 관리 클래스
    """
    def __init__(self, embedding_dimension: int = 3072):
        self.embedding_model = VertexEmbeddings()
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatL2(embedding_dimension)
        self.documents = []
        self.metadata = []
        self.ensemble_retriever = None
        self.reranker = None
    
    def _format_text_for_embedding(self, document: Dict[str, Any]) -> str:
        context_enrichment = document.get('context_enrichment', '')
        page_content = document.get('page_content', '')
        return f"{context_enrichment} : {page_content}" if context_enrichment else page_content
    
    def create_vector_store(self, save_path: Optional[str] = "../data/vector_store/", pkl_directory: str = "../data/contextual_content_docs/"):
        documents = self._load_all_pkl_files(pkl_directory)
        
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.documents = []
        self.metadata = []
        
        embeddings = []
        for i, doc in enumerate(documents):
            try:
                formatted_text = self._format_text_for_embedding(doc)
                embedding = self.embedding_model.embedding_document(formatted_text)
                embeddings.append(embedding)
                self.documents.append(formatted_text)
                self.metadata.append(doc.get('metadata', {}))
                
                if (i + 1) % 100 == 0:
                    print(f"진행률: {i + 1}/{len(documents)}")
            except Exception as e:
                print(f"문서 {i} 실패: {e}")
                continue
        
        if embeddings:
            self.index.add(np.array(embeddings, dtype=np.float32))
            print(f"생성 완료 - 벡터 수: {self.index.ntotal}")
        
        if save_path:
            self.save_vector_store(save_path)
        
        self._initialize_retrievers()
        return {"total_vectors": self.index.ntotal, "documents_count": len(self.documents)}
    
    def save_vector_store(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(self.index, f"{save_path}.index")
        with open(f"{save_path}.pkl", 'wb') as f:
            pickle.dump({'documents': self.documents, 'metadata': self.metadata, 'embedding_dimension': self.embedding_dimension}, f)
        print(f"저장 완료: {save_path}")
    
    def load_vector_store(self, load_path: str):
        self.index = faiss.read_index(f"{load_path}.index")
        with open(f"{load_path}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_dimension = data['embedding_dimension']
        
        print(f"로드 완료: 벡터 수 {self.index.ntotal}, 차원 {self.embedding_dimension}")
        self._initialize_retrievers()
        return {"total_vectors": self.index.ntotal, "documents_count": len(self.documents)}
    
    def load_documents_from_pkl(self, pkl_path: str):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_all_pkl_files(self, pkl_directory: str):
        pkl_files = glob.glob(os.path.join(pkl_directory, "*.pkl"))
        all_documents = []
        for pkl_file in pkl_files:
            try:
                documents = self.load_documents_from_pkl(pkl_file)
                all_documents.extend(documents)
            except Exception as e:
                print(f"파일 {pkl_file} 로드 실패: {e}")
        print(f"총 문서 수: {len(all_documents)}")
        return all_documents
    
    def search(self, query: str, k: int = 40):
        """
        앙상블 검색 (의미적 + 키워드 검색, 가중치 7:3)
        """
        if not self.ensemble_retriever:
            return []
        
        retrieved_docs = self.ensemble_retriever.invoke(query)

        return [{"document": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs[:k]]
    
    def _create_langchain_documents(self):
        return [Document(page_content=doc_text, metadata=metadata) for doc_text, metadata in zip(self.documents, self.metadata)]
    
    def _create_vertex_embeddings_wrapper(self):
        class VertexAIEmbeddingsWrapper(Embeddings):
            def __init__(self, vertex_embeddings):
                self.vertex_embeddings = vertex_embeddings
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [self.vertex_embeddings.embedding_document(text) for text in texts]
            
            def embed_query(self, text: str) -> List[float]:
                return self.vertex_embeddings.embedding_query(text)
        
        return VertexAIEmbeddingsWrapper(self.embedding_model)
    
    def _initialize_retrievers(self):
        try:
            langchain_docs = self._create_langchain_documents()
            if not langchain_docs:
                return
            
            vector_store = FAISS(
                embedding_function=self._create_vertex_embeddings_wrapper(),
                index=self.index,
                docstore=self._create_docstore(langchain_docs),
                index_to_docstore_id={i: str(i) for i in range(len(langchain_docs))}
            )
            
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_store.as_retriever(), BM25Retriever.from_documents(langchain_docs)],
                weights=[0.7, 0.3]
            )
            
            # 리랭커도 함께 초기화
            self.reranker = VertexAIRank(
                project_id=self.embedding_model.project_id,
                location_id="global",
                ranking_config="default_ranking_config",
                top_n=20,
                model="semantic-ranker-default-004",
            )
            print("✅ 검색기 및 리랭커 초기화 완료")
            
        except Exception as e:
            print(f"검색기 초기화 실패: {e}")
            self.ensemble_retriever = None
            self.reranker = None
    
    def _create_docstore(self, langchain_docs):
        docstore = InMemoryDocstore()
        for i, doc in enumerate(langchain_docs):
            docstore.add({str(i): doc})
        return docstore

    def document_rerank(self, query: str, documents: List[Dict[str, Any]], k: int = 20):
        """
        검색된 문서들을 Vertex AI Rank로 리랭킹
        
        Args:
            query: 검색 쿼리
            documents: search 함수에서 반환된 문서 리스트
            k: 반환할 문서 수
            
        Returns:
            리랭킹된 문서 리스트 (search 함수와 동일한 형식)
        """

        for doc in documents:
            doc['metadata'] = {
                'pdf_filename': doc['metadata']['pdf_filename'],
                'page_numbers': str(doc['metadata']['page_numbers'])
            }

        # 검색 결과를 Document 객체로 변환
        docs_for_rerank = [
            Document(page_content=doc_result["document"], metadata=doc_result["metadata"])
            for doc_result in documents
        ]
        reranker = VertexAIRank(
                project_id=self.embedding_model.project_id,
                location_id="us-central1",
                ranking_config="default_ranking_config",
                top_n= k,
                model="semantic-ranker-default-004",
            )
        try:
            # Vertex AI Rank로 리랭킹 (올바른 매개변수 순서: documents, query)
            reranked_docs = self.reranker.compress_documents(docs_for_rerank, query)
            result = []

            for doc in reranked_docs[:k]:
                result.append(docs_for_rerank[int(doc.metadata['id'])])

            def clean_documents(documents: list[Document]) -> list[Document]:
                def clean_text(text: str) -> str:
                    if not isinstance(text, str):
                        return ""
                    cleaned_text = ''.join(char for char in text if char.isprintable())
                    return cleaned_text
                cleaned_documents = []
                for doc in documents:
                    cleaned_content = clean_text(doc.page_content)
                    
                    # 정제된 내용으로 새로운 Document 객체를 생성합니다.
                    cleaned_doc = Document(
                        page_content=cleaned_content,
                        metadata=doc.metadata
                    )
                    cleaned_documents.append(cleaned_doc)
                return cleaned_documents

            return clean_documents(result)
            # 결과를 search 함수와 동일한 형식으로 변환
            # return [
            #     {"document": doc.page_content, "metadata": doc.metadata}
            #     for doc in result[:k]
            # ]
            return result
            
        except Exception as e:
            print(f"리랭킹 실패: {e}")
            return docs_for_rerank[:k]
