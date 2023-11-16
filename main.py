from flask import Flask, request, jsonify
from langchain import PromptTemplate, OpenAI, LLMChain
# from langchain.document_loaders import LocalFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS
import os
import requests
import base64
import json

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


def load_documents_from_directory(directory_path):
  documents = []
  for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
      with open(os.path.join(directory_path, filename), 'r') as file:
        content = file.read()
        documents.append(content)  # Each document's content is a string
  return documents


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
    "detail":
    "high"
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


def load_data_from_ipfs(cid):
    url = f'https://ipfs.io/ipfs/{cid}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Process your data (e.g., load into global variables)
        return data
    else:
        print(f'Failed to fetch data from IPFS: {response.status_code}')
        return None

# Global variables to hold embeddings and document splits
global_embeddings = None
global_splits = None

def load_embeddings_and_splits():
  global global_embeddings, global_splits
  if global_embeddings is None or global_splits is None:
    with open('./slow_embeddings.json', 'r') as file:
      data = json.load(file)
    global_embeddings = np.array(data['embeddings'])
    global_splits = data['splits']


def find_relevant_splits(question_embedding, top_n=3):
  global global_embeddings, global_splits
  if global_embeddings is None or global_splits is None:
    load_embeddings_and_splits()
  similarities = cosine_similarity([question_embedding], global_embeddings)[0]
  top_indices = np.argsort(similarities)[-top_n:]
  return [global_splits[i] for i in top_indices]


@app.route('/rag_qa', methods=['POST'])
def rag_qa():
  question = request.json.get('question')
  question_embedding = OpenAIEmbeddings().get_embeddings(question)

  relevant_splits = find_relevant_splits(question_embedding)
  formatted_docs = "\n\n".join(relevant_splits)

  prompt = f"You are an assistant... \nQuestion: {question}\nContext: {formatted_docs}\nAnswer:"
  response = ChatOpenAI(model_name="gpt-4-1106-preview",
                        temperature=0).invoke(prompt)

  return jsonify({'answer': response})


if __name__ == "__main__":
  app.run()
