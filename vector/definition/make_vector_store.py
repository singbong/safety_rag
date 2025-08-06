from case_vector_store import store_vector_db


vector_store = store_vector_db()

vector_store.create_vector_store(pkl_directory="../data/case_data/original_data/case_festival.pkl", file_name="case_festival")

vector_store.create_vector_store(pkl_directory="../data/case_data/original_data/case_climate.pkl", file_name="case_climate")