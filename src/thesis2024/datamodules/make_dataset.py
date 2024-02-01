"""Create the dataset for the thesis.

Making the dataset for the thesis, by loading datasources (currently PDFs)
and creating a persistent Chroma vector store from them.
"""

import os
import openai
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings



def create_persistent_chroma_store(openai_embedding, chunk_size, chunk_overlap):
    """Create Vector Store.

    Create a persistent Chroma vector store from PDF documents in a directory.
    :param directory: Path to the directory containing PDF documents.
    :param store_path: Path to save the persistent Chroma vector store.
    :param chunk_size: Size of text chunks for each document.
    :param chunk_overlap: Overlap between consecutive text chunks.
    :return: None
    """
    data_dir = os.path.join(os.getcwd(), 'data/raw/DLAI_CWYDcourse')
    persist_dir = os.path.join(os.getcwd(), 'data/processed/chroma')

    # Process each PDF file in the directory
    all_docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(data_dir, filename)

            # Load PDF document
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
            )
            docs = text_splitter.split_documents(documents)
            print(f"{len(docs)} chunks created from {filename}")
            all_docs.extend(docs)

    if openai_embedding:
        embedding_func = OpenAIEmbeddings()
    else:
        embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Save the Chroma vector store persistently
    Chroma.from_documents(all_docs, embedding_func, persist_directory=persist_dir)
    return print(f"Chroma vector store created and saved at {persist_dir}")


def load_peristent_chroma_store(openai_embedding):
    """Load a persistent Chroma vector store.

    :param persist_path: Path to the persistent Chroma vector store.
    :return: Chroma vector store.
    """
    ## Initiate embedding function
    if openai_embedding:
        embedding_func = OpenAIEmbeddings()
    else:
        embedding_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    persist_dir = os.path.join(os.getcwd(), 'data/processed/chroma')

    return Chroma(persist_directory=persist_dir, embedding_function=embedding_func)


def add_files_to_chroma_store(new_files_dir):
    """Add new files to the Chroma vector store.

    :param new_files_dir: Path to the directory containing new files.
    :return: None
    """
    pass


if __name__ == '__main__':

    ## Retrieve OpenAI API key from environment variable
    openai.api_key  = os.environ['OPENAI_API_KEY']

    # Get the data and process it
    create_persistent_chroma_store(openai_embedding=True,
                                   chunk_size=1000,
                                   chunk_overlap=150)
    pass











# def merge_hyphenated_words(text: str) -> str:
#     """Merge hyphenated words."""
#     return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

# def fix_newlines(text: str) -> str:
#     """Fix newlines."""
#     return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

# def remove_multiple_newlines(text: str) -> str:
#     """Remove multiple newlines."""
#     return re.sub(r"\n{2,}", "\n", text)


# def clean_text(pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]) -> List[Tuple[int, str]]:
#     """Clean text.

#     :param pages: List of tuples of page number and text.
#     :param cleaning_functions: List of cleaning functions.
#     :return: List of tuples of page number and text.
#     """
#     cleaned_pages = []
#     for page_num, text in pages:
#         for cleaning_function in cleaning_functions:
#             text = cleaning_function(text)
#         cleaned_pages.append((page_num, text))
#     return cleaned_pages
