from semantic_split_vertex import PDFSemanticChunker
from tqdm import tqdm
import glob
import os
import pickle



pdf_paths = glob.glob("../data/original_pdf/*.pdf")

os.makedirs("../data/chunked_docs", exist_ok=True)

print(pdf_paths)
for i in tqdm(pdf_paths):
    chunker = PDFSemanticChunker()
    docs = chunker.create_semantic_chunks(i)
    with open(f"../data/chunked_docs/{i.split('/')[-1].split('.')[0]}.pkl", "wb") as f:
        pickle.dump(docs, f)
