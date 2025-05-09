import os,re
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq
from openai import OpenAI
import google.generativeai as genai
import PIL
import streamlit as st

WORKING_DIR = "rag-data"
DOCS_DIR = os.path.join(WORKING_DIR, "images")
EMBS_DIR = os.path.join(WORKING_DIR, "embeddings")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
if not os.path.exists(EMBS_DIR):
    os.mkdir(EMBS_DIR)
# Groq 
groq_api_key = st.secrets.groq_api_key #os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)
#deepseek
deepseek_api_key = st.secrets.deepseek_api_key #os.getenv("GROQ_API_KEY")
deepseek_client = OpenAI(api_key=deepseek_api_key,base_url="https://api.deepseek.com")
#openai
openai_api_key = st.secrets.openai_api_key #os.getenv("GROQ_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)
#gemini
gemini_api_key = st.secrets.google_api_key #os.getenv('GOOGLE_API_KEY') 
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-2.0-flash")
#openai
openai_api_key =  st.secrets.openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

ai_provider = st.secrets.ai_provider

if ai_provider == 'openai':
    emb_model = None
else:
    emb_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

describe_image_prompt = """
Ihre Aufgabe ist es, den Inhalt eines Bildes umfassend und prägnant zu beschreiben.
Die Beschreibung muss für die Verwendung in einem RAG-System optimiert sein.
Hier ist das Bild:
"""

def embedding_func(texts: list[str]) -> np.ndarray:
    if emb_model:
        embeddings = emb_model.encode(texts, convert_to_numpy=True)
        return embeddings
    return None

def openai_embedding(texts: list[str]) -> np.ndarray:
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [dat.embedding for dat in response.data]

def llm_model_func_google(prompt,images=[]) -> str:
    
    
    combined_prompt = [prompt]
    combined_prompt += [PIL.Image.open(image) for image in images] 
    # Call the Gemini model
    response = model.generate_content(combined_prompt)
    
    return response.text



#gen_llm = llm_model_func

#if ai_provider == 'openai':
#    gen_llm = llm_model_func_openai#gpt_4o_mini_complete
#    embedding_model = openai_embed
#elif ai_provider == 'google':
#    gen_llm = llm_model_func_google
#elif ai_provider == 'deepseek':
#    gen_llm = llm_model_func_deepseek


def load_embs():
    embeddings_path = os.path.join(EMBS_DIR,'embeddings.npy')
    if os.path.exists(embeddings_path):
        e = np.load(embeddings_path)
        return e
    return []

def save_embs(embs):
    embeddings_path = os.path.join(EMBS_DIR,'embeddings.npy')
    np.save(embeddings_path,embs)


def load_images():
    index_path = os.path.join(WORKING_DIR,'index.txt')
    if os.path.exists(index_path):
        with open(index_path,"r") as f:
            index = [doc.strip() for doc in f.readlines()]
        return index
    return []

def run_new_indexing():
    documents = os.listdir(DOCS_DIR)
    docs_data = []
    
    for doc in documents:
        #dochash = hashlib.md5(data.encode()).hexdigest()
        if doc in st.session_state.index:
            continue
        doc_path = os.path.join(DOCS_DIR,doc)
        image_description = llm_model_func_google(prompt=describe_image_prompt,images=[doc_path])
        print("image:",doc,image_description)
        #citations.append(doc)
        st.session_state.index.append(doc)
        docs_data.append(image_description)
        #if len(docs_data) > 100:
        #    embeddings = openai_embedding(docs_data) 
        #    if len(st.session_state.index_embs) > 0:
        #        st.session_state.index_embs = np.vstack([st.session_state.index_embs,embeddings])         
        #    else:
        #        st.session_state.index_embs = embeddings
        #    docs_data = []
    
    embeddings = openai_embedding(docs_data)
    embeddings = embedding_func(docs_data)
    if len(st.session_state.index_embs) > 0:
        st.session_state.index_embs = np.vstack([st.session_state.index_embs,embeddings])         
    else:
        st.session_state.index_embs = embeddings
    save_embs(st.session_state.index_embs)
    with open(os.path.join(WORKING_DIR,'index.txt'),"w") as f:
        f.write("\n".join(st.session_state.index))
    with open(os.path.join(WORKING_DIR,'images_descriptions.txt'),"w") as f:
        f.write("\n".join([f"{i}:{j}" for i, j in zip(st.session_state.index, docs_data)]))
    
    print("images indexing DONE!!")


def save_docs(docs):
    
    for doc in docs:
        file_path = os.path.join(DOCS_DIR, doc.name)
        with open(file_path ,'wb') as fp:
            fp.write(doc.getbuffer())

def rag(question,database,thredshold=0.4):

    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    #query = openai_embedding(question)
    query = embedding_func(question)
    similarities = [cosine_similarity(query, sentence_embedding) for sentence_embedding in database]
    print(similarities)
    scores =  sorted([score for score in similarities if score > thredshold],reverse=True)
    if len(scores) == 0:
        scores = [max(similarities)]
    idxs = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:len(scores)]
    # validate results:
    images = [st.session_state.index[i] for i in idxs][:5]
    with open(os.path.join(WORKING_DIR,'images_descriptions.txt'),"r") as fp:
        images_descriptions = [line for line in fp.readlines() if line.split(":")[0].strip() in images]
    print(images)
    prompt = f"""
        Ihre Aufgabe ist es, die Relevanz der RAG (Retrieval-Augmented Generation) Ergebnisse zu validieren.  
        Gegeben ist eine Liste von Bildnamen mit den entsprechenden Beschreibungen und einer Frage.
        Identifizieren und geben Sie die Namen der Bilder zurück, deren Beschreibungen hochgradig relevant für die Frage sind.
        Bilder, die nicht direkt oder eng mit der Frage verbunden sind, sollen verworfen werden.  

        Frage: {question}

        Eingabe: {"\n".join(images_descriptions)}  

        Ausgabeformat:  
        - Die Liste der ausgewählten Bildnamen muss:  
          1. Durch ein Komma (`,`) getrennt sein.  
          2. Zwischen `<IMAGES>` und `</IMAGES>` eingeschlossen sein.  

        Wichtiger Hinweis:  
        - Stellen Sie sicher, dass nur hochrelevante Bilder in die Ausgabe aufgenommen werden.  
        - Keine zusätzlichen Texte oder Kommentare sollen in der Ausgabe enthalten sein.

    """ 
    print(prompt)
    response = llm_model_func_google(prompt,[])
    print(response)
    response = re.search(r"<IMAGES>(.*?)</IMAGES>", response)
    if response:
        images = response.group(1).split(",")

    return images


def main():
    st.title("RAG AI Agent (demo)")

    if "index" not in st.session_state:
        st.session_state.index_embs = load_embs()
        st.session_state.index = load_images()
    print(st.session_state.index)
    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message("user"): 
                st.markdown(msg['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg['content'])    
        
    with st.sidebar:
        new_docs = st.file_uploader("Upload PDF files here and update index.", accept_multiple_files=True)
        
        all_documents = os.listdir(DOCS_DIR)
        st.markdown(
            """
            <style>
            .custom-scroll-area {
                max-height: 6cm;
                overflow-y: auto;
                background-color: rgba(200, 200, 200, 0.3);  /* light gray with more opacity */
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .custom-scroll-area >p{
                margin-bottom:0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        docs_area = "<div class='custom-scroll-area'>"
        docs_area += f"<h3>Index content:</h3>"
        for doc in all_documents:
            docs_area += f"<p>📄 {doc}</p>"
        docs_area += "</div>"
        st.markdown(docs_area, unsafe_allow_html=True)
    
        
        if st.button("Update Index"):
            save_docs(new_docs)
            run_new_indexing()
        if st.button("Exit"):
            st.write("Good Bye...!")
            st.stop()      
    # Chat input for the user
    user_input = st.chat_input("What do you want to know?")

    if user_input:
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({'role':'user','content':user_input})
        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Create a placeholder for the streaming text
            message_placeholder = st.empty()
            full_response = ""
            
            # Properly consume the async generator with async for
            
            images = rag(user_input,st.session_state.index_embs) 
            try:
                for i in images:
                    st.markdown(i)
                cols = st.columns(len(images)) 
                for i in range(len(images)):
                    cols[i].image(os.path.join(DOCS_DIR,images[i]), use_container_width=True)
            except:
                st.markdown("Sorry, No images were found!")
            
if __name__ == "__main__":
    main()