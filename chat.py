from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import os
import shutil
from dotenv import load_dotenv


load_dotenv()

OPENAI_KEY = os.getenv("API_KEY")

embeddings = OpenAIEmbeddings(
    openai_api_key = OPENAI_KEY
)



chat_template = """You are a helpful AI assistant for the bank named "Smart Bank" that can answer user questions based on the chat history and the given context.

<CONTEXT>:
{context}
</CONTEXT>

CHAT HISTORY:
{chat_history}

User Question: {question}

AI Assistant:"""

chat_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=chat_template
)

def format_docs(docs):
    return "\n\n".join(doc[0].page_content for doc in docs)



vector_db = FAISS.load_local("local_vector_store/faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", ai_prefix="AI Assistant")

chat_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt,
    verbose=False
)


def chat(message):
    if not os.path.exists("local_vector_store/faiss_index"):
        return "Error: No local vector store found in /local_vector_store"
    context = vector_db.similarity_search_with_score(message, k=4)
    context = format_docs(context)
    chat_history = memory.load_memory_variables({})['chat_history']
    res = chat_chain.invoke({"question": message, "context": context, "chat_history": chat_history})['text']
    memory.save_context({"input": message}, {"output": res})
    return res


def main():
    print("\n\n\n\n")
    while True:
        user_input = input("\n\nQuestion: ").strip()

        if not user_input:
            print("Error: Input is empty")
            continue

        if user_input.lower() == "exit":
            print("Exiting the chat.")
            break

        response = chat(user_input)

        print("\nResponse: ", response)

        

if __name__ == "__main__":
    main()


