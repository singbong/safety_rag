from vector.definition.vector_store import store_vector_db

if __name__ == "__main__":
    print("FAISS 벡터 데이터베이스 생성을 시작합니다...")
    vector_db = store_vector_db()
    result = vector_db.create_vector_store(
        save_path="../data/vector_store/faiss_vector_db",
        pkl_directory="../data/contextual_content_docs/"
    )
    print("FAISS 벡터 데이터베이스 생성이 완료되었습니다.")
    print(f"결과: {result}")
