from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


def custom_prompt():
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer."""

    template = "You are a bot that answers questions from a patients electronic health ecord or charts.\n\
                If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\
                {context}\n\
                Question: {question}"

    custom_rag_prompt = PromptTemplate.from_template(
        template=template,
        # input_variables=["context", "question"]
    )
    return custom_rag_prompt

def prepare_rag_llm(
    token, llm_model, vector_store_path, temperature, max_length, question
) -> list[Document]:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=token)
    vector_store = Chroma(
    collection_name="ehr_collection",
    embedding_function=embeddings,
    persist_directory=vector_store_path,
)
    name = question.lower().split("name:")[-1].split("\n")[0].split("dos:")[0].strip()
    dos = question.lower().split("name:")[-1].split("\n")[0].split("dos:")[-1].split("\n")[0].strip()
    if name:
        if dos:
            filter = {
            "$and": [
                {
                    "patient_name": {
                        "$eq": "alexis sparks"
                    }
                },
                {
                    "date_of_service": {
                        "$eq": "20250215"
                    }
                }
            ]
        }
        else:
            filter = {
                    "patient_name": {
                        "$eq": name
                    }
                }

    else:
        results = [Document(
        page_content="NoNameUsed",
        metadata={
            "patient_name": name,
            "date_of_service":dos,
        },
        )]
        return results

    results = vector_store.similarity_search(
        question,
        k=2,
        filter=filter,
    )
    # for res in results:
    #     print(f"* {res.page_content} [{res.metadata}]")
    return results


def generate_answer(question, token, vector_store_path, llm_model):
    answer = "An error has occurred"

    if token == "":
        answer = "Insert the open api key"
        doc_source = ["no source"]
    else:
        retrieved_docs = prepare_rag_llm(
        token, llm_model, vector_store_path, 0.7, 300, question
        )
        if len(retrieved_docs) == 1 and retrieved_docs[0].page_content == "NoNameUsed":
            answer = "Please ask again with a name of patient prefixed with name:"
            doc_source = ["no source found in knowledge base"]
        else:
            formatted_question = question.lower().split("name:")[0]
            # response = st.session_state.conversation({"question": question})
            docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
            doc_source = [doc.page_content for doc in retrieved_docs]
            # answer = response.get("answer").split("Helpful Answer:")[-1].strip()
            custom_rag_prompt = custom_prompt()
            llm = ChatOpenAI(model_name=llm_model, temperature=0)
            prompt = custom_rag_prompt.invoke({"question": formatted_question, "context": docs_content})
            llm_response = llm.invoke(prompt)
            answer = llm_response.content

    return answer, doc_source
    