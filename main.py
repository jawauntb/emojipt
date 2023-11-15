from flask import Flask, request, jsonify
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.document_loaders import LocalFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain.schema import StrOutputParser, RunnablePassthrough
from flask_cors import CORS
import os
import requests
import base64
import glob


app = Flask(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY")
CORS(app)

# Initialize LLMChain
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
prompt_template = "Give me 20 emojis related to {input}"
llm_chain = LLMChain(llm=llm,
                     prompt=PromptTemplate.from_template(prompt_template))


@app.route('/')
def index():
  return 'Hello from Flask!'


@app.route('/emojis', methods=['POST'])
def generate_emojis():
  data = request.get_json()
  input_data = data.get('input', '')
  # Generate result from LLMChain
  result = llm_chain.run(input_data)

  # Debugging print statements
  print("Result: ", result)
  print("input: ", input_data)

  emojis = result

  return jsonify({'emojis': emojis})


@app.route('/promptly', methods=['POST'])
def generate_response():
  data = request.get_json()
  payload = data.get('payload', '')
  authkey = 'Bearer ' + openai_api_key
  headers = {'Authorization': authkey, 'Content-Type': 'application/json'}

  response = requests.post('https://api.openai.com/v1/chat/completions',
                           headers=headers,
                           json=payload)
  return jsonify(response.json())


@app.route('/sitesee', methods=['POST'])
def get_sitesee():
  data = request.get_json()
  payload = data.get('payload', '')
  authkey = 'Bearer ' + openai_api_key
  headers = {'Authorization': authkey, 'Content-Type': 'application/json'}

  response = requests.post('https://api.openai.com/v1/chat/completions',
                           headers=headers,
                           json=payload)
  return jsonify(response.json())


@app.route('/gen_image', methods=['POST'])
def generate_image():
  data = request.get_json()
  payload = data.get('payload', '')

  authkey = 'Bearer ' + openai_api_key
  headers = {'Authorization': authkey, 'Content-Type': 'application/json'}

  response = requests.post('https://api.openai.com/v1/images/generations',
                           headers=headers,
                           json=payload)
  return jsonify(response.json())


@app.route('/analyze_uploaded_image', methods=['POST'])
def analyze_uploaded_image():
  # Check if the post request has the file part
  if 'image' not in request.files:
    return jsonify({'error': 'No image part in the request'}), 400

  file = request.files['image']

  if file.filename == '':
    return jsonify({'error': 'No image selected for uploading'}), 400

  # Convert the image file to a base64 encoded string
  image_encoded = base64.b64encode(file.read()).decode('utf-8')

  # Prepare the payload for OpenAI's GPT-4 Vision API
  payload = {
    "model":
    "gpt-4-vision-preview",
    "messages": [{
      "role":
      "user",
      "content": [{
        "type":
        "text",
        "text":
        "Give me a detailed description of this UI and the various components in it, so that i can reproduce it. If its not a UI, interpret the image as though it is a novel UI"
      }, {
        "type": "image_url",
        "image_url": {
          "url": f"data:image/jpeg;base64,{image_encoded}"
        }
      }]
    }],
    "max_tokens":
    4000,
    "detail": "high"
  }

  headers = {
    'Authorization': 'Bearer ' + openai_api_key,
    'Content-Type': 'application/json'
  }

  # Send the request to OpenAI's API
  response = requests.post('https://api.openai.com/v1/chat/completions',
                           headers=headers,
                           json=payload)

  if response.status_code != 200:
    return jsonify({'error':
                    'Error processing the image'}), response.status_code

  # Return the response from the OpenAI API
  return jsonify(response.json())


@app.route('/rag_qa', methods=['POST'])
def rag_qa():
    # Load and split documents
    loader = LocalFileLoader(file_paths=glob.glob('slowgpt_docs/*.txt'))
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed and store document splits
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Retrieve and generate response
    question = request.json.get('question')
    formatted_docs = "\n\n".join(doc.page_content for doc in retriever.get_relevant_documents(question))
    prompt = hub.pull("rlm/rag-prompt")

    # Set up the RAG chain
    rag_chain = (
        {"context": formatted_docs, "question": question}
        | prompt
        | ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
        | StrOutputParser()
    )

    # Generate the response
    response = rag_chain.invoke(question)

    # Clean up
    vectorstore.delete_collection()

    return jsonify({'answer': response})

if __name__ == "__main__":
  app.run()
