import os,re
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq
from openai import OpenAI
import google.generativeai as genai
import PIL
import streamlit as st
import streamlit.components.v1 as components
import base64
from io import BytesIO

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

#openai
openai_api_key = st.secrets.openai_api_key #os.getenv("GROQ_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

#gemini
gemini_api_key = st.secrets.google_api_key #os.getenv('GOOGLE_API_KEY') 
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

ai_provider = st.secrets.ai_provider

if ai_provider == 'openai':
    emb_model = None
else:
    emb_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    #emb_model = SentenceTransformer("mixedbread-ai/deepset-mxbai-embed-de-large-v1", truncate_dim=1024)

describe_image_prompt = """
Ihre Aufgabe ist es, den Inhalt eines Bildes umfassend und prägnant zu beschreiben.
Die Beschreibung muss für die Verwendung in einem RAG-System optimiert sein.
Hier ist das Bild:
"""

def sentence_transformers_embs(texts: list[str]) -> np.ndarray:
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

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

def llm_model_func_groq(prompt,images=[]) -> str:
    #combined_prompt = [prompt]
    
    #combined_prompt += [image_to_base64( PIL.Image.open(image) ) for image in images] 
    #combined_prompt = "\n".join(combined_prompt)
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # or another Groq-supported model like llama3-70b
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(images[0])}",
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

def llm_model_func_openai(prompt,images=[]) -> str:
    
    combined_prompt = [prompt]
    
    #combined_prompt += [image_to_base64( PIL.Image.open(image) ) for image in images] 
    #combined_prompt = "\n".join(combined_prompt)
    response = openai_client.responses.create(
        model="gpt-4.1",
        input=[
                {
                    "role": "user",
                    "content": [
                        { "type": "input_text", "text": prompt },
                        {
                          "type": "input_image",
                          "image_url": f"data:image/jpeg;base64,{encode_image(images[0])}",
                        },
                    ]
                }
            ],
        )
    return response.output_text

def llm_model_func_google(prompt,images=[]) -> str:
    
    
    combined_prompt = [prompt]
    combined_prompt += [PIL.Image.open(image) for image in images] 
    # Call the Gemini model
    response = model.generate_content(combined_prompt)
    
    return response.text



gen_llm = llm_model_func_groq
embedding_func = sentence_transformers_embs

if ai_provider == 'openai':
    gen_llm = llm_model_func_openai
    embedding_func = openai_embedding
elif ai_provider == 'google':
    gen_llm = llm_model_func_google
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

def load_file(filename):
    index_path = os.path.join(WORKING_DIR,filename)
    if os.path.exists(index_path):
        with open(index_path,"r") as f:
            index = [doc.strip() for doc in f.readlines() if doc.strip()]
        return index
    return []

def run_new_indexing():
    documents = os.listdir(DOCS_DIR)
    print(documents)
    docs_data = []
    
    for doc in documents:
        #dochash = hashlib.md5(data.encode()).hexdigest()
        if doc in st.session_state.index:
            continue
        doc_path = os.path.join(DOCS_DIR,doc)
        image_description = gen_llm(prompt=describe_image_prompt,images=[doc_path])
        print("image:",doc,image_description)
        #citations.append(doc)
        st.session_state.index.append(doc)
        st.session_state.descriptions.append(image_description)
        docs_data.append(image_description)
        #if len(docs_data) > 100:
        #    embeddings = openai_embedding(docs_data) 
        #    if len(st.session_state.index_embs) > 0:
        #        st.session_state.index_embs = np.vstack([st.session_state.index_embs,embeddings])         
        #    else:
        #        st.session_state.index_embs = embeddings
        #    docs_data = []
    if len(docs_data):
        embeddings = embedding_func(docs_data)
        if len(st.session_state.index_embs) > 0:
            st.session_state.index_embs = np.vstack([st.session_state.index_embs,embeddings])         
        else:
            st.session_state.index_embs = embeddings
        
        save_embs(st.session_state.index_embs)
        
        with open(os.path.join(WORKING_DIR,'index.txt'),"w") as f:
            f.write("\n".join(st.session_state.index))
        with open(os.path.join(WORKING_DIR,'images_descriptions.txt'),"w") as f:
            f.write("\n".join([f"{i}:{j}" for i, j in zip(st.session_state.index, st.session_state.descriptions)]))
        st.session_state['index_changed'] = True
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
    response = gen_llm(prompt,[])
    print(response)
    response = re.search(r"<IMAGES>(.*?)</IMAGES>", response)
    if response:
        images = response.group(1).split(",")

    return images


import numpy as np

def remove_file_data(file_to_remove):
    try:
        # Read the filenames from index.txt
        index_file = os.path.join(WORKING_DIR,'index.txt')
        with open(index_file, 'r') as file:
            filenames = file.readlines()
        
        # Strip newline characters and clean the list
        filenames = [filename.strip() for filename in filenames]
        
        if file_to_remove not in filenames:
            print(f"'{file_to_remove}' not found in {index_file}.")
            #return
        
        # Find the index of the file to remove
        remove_index = filenames.index(file_to_remove)
        
        # Remove the filename from the list
        filenames.pop(remove_index)
        print(f"Removed '{file_to_remove}' from {index_file}.")
        
        # Save the updated filenames back to index.txt
        with open(index_file, 'w') as file:
            for filename in filenames:
                file.write(filename + '\n')
        
        # Step 3: Remove the corresponding embedding
        embeddings_file = os.path.join(EMBS_DIR,'embeddings.npy')
        embeddings = np.load(embeddings_file)
        
        if remove_index >= len(embeddings):
            print("Error: Index out of range for embeddings.")
            return
        
        # Remove the embedding at the specified index
        embeddings = np.delete(embeddings, remove_index, axis=0)
        print(f"Removed embedding at index {remove_index} from {embeddings_file}.")
        
        # Save the updated embeddings back to embeddings.npy
        np.save(embeddings_file, embeddings)
        
        # Remove the corresponding description
        desc_file = os.path.join(WORKING_DIR,'images_descriptions.txt')
        with open(desc_file, 'r') as file:
            descriptions = file.readlines()
        
        if remove_index >= len(descriptions):
            print("Error: Index out of range for descriptions.")
            return
        
        # Remove the description at the specified index
        descriptions.pop(remove_index)
        print(f"Removed description at index {remove_index} from {desc_file}.")
        
        # Save the updated descriptions back to files_desc.txt
        with open(desc_file, 'w') as file:
            for description in descriptions:
                file.write(description)

        file_path = os.path.join(DOCS_DIR,file_to_remove)
        
        if os.path.exists(file_path):
            os.remove(file_path)
        print("All associated data has been removed successfully.")
        load_index()
        
        st.rerun()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_index():
    st.session_state.index_embs = load_embs()
    st.session_state.index = load_file('index.txt')
    st.session_state.descriptions = [desc.split(':',1)[1].strip() for desc in load_file('images_descriptions.txt')]
        
def main():
    st.title("RAG AI Agent (demo)")

    if "index" not in st.session_state:

        load_index()

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.

    def display_images(images):
        try:
            for i in images:
                st.markdown(i)
            cols = st.columns(len(images)) 
            for i in range(len(images)):
                cols[i].image(os.path.join(DOCS_DIR,images[i].strip()), use_container_width=False)
        except:
            st.markdown("Es wurden leider keine Bilder gefunden!")
            
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message("user"): 
                st.markdown(msg['content'])
        else:
            with st.chat_message("assistant"):
                display_images(msg['content'])    
        
    with st.sidebar:
        new_docs = st.file_uploader("Laden Sie hier PDF-Dateien hoch und aktualisieren Sie den Index.", accept_multiple_files=True)
        
        all_documents = os.listdir(DOCS_DIR)
        
        
        
        st.title("Index content:")
        st.markdown("""
            <style>
                .stVerticalBlock {
                    gap:0;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        with st.container(height=200):
            for filename in all_documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"🖼️ {filename}")
                with col2:
                    if st.button("✖", type='tertiary',key=f'btn-{filename}'):
                        remove_file_data(filename)


        st.button("",type='tertiary',key='s1')
        if st.button("Update Index"):
            save_docs(new_docs)
            run_new_indexing()
        st.button("",type='tertiary',key='s2')
        if st.button("Exit"):
            st.write("Good Bye...!")
            st.stop()      
    # Chat input for the user
    user_input = st.chat_input("Suche nach Bildern")

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
            display_images(images)
            st.session_state.messages.append({'role':'assistant','content':images})            

if __name__ == "__main__":
    main()