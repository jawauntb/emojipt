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


# Function to load embeddings and their associated document splits
def load_embeddings_and_splits():
  print('load_embeddings_and_splits')
  with open('slow_embeddings.json', 'r') as file:
    data = json.load(file)
  embeddings = np.array(data['embeddings'])
  splits = data['splits']
  return embeddings, splits


# Function to find the most relevant document splits
def find_relevant_splits(question_embedding, embeddings, splits, top_n=3):
  print('find_relevant_splits')
  similarities = cosine_similarity([question_embedding], embeddings)[0]
  top_indices = np.argsort(similarities)[-top_n:]
  return [splits[i] for i in top_indices]


@app.route('/rag_qa', methods=['POST'])
def rag_qa():
  print('running')
  # Load embeddings and splits
  embeddings, splits = load_embeddings_and_splits()

  # Retrieve the question from the request and generate its embedding
  question = request.json.get('question')
  question_embedding = OpenAIEmbeddings().get_embeddings(question)
  print('getting question')
  # Find the most relevant document splits
  relevant_splits = find_relevant_splits(question_embedding, embeddings,
                                         splits)

  # Construct the context from relevant splits
  formatted_docs = "\n\n".join(relevant_splits)
  print('sending')
  # Construct the prompt for the language model
  prompt = f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question}\nContext: {formatted_docs}\nAnswer:"

  # Generate the response using the language model
  response = ChatOpenAI(model_name="gpt-4-1106-preview",
                        temperature=0).invoke(prompt)

  # Return the generated response
  return jsonify({'answer': response})


if __name__ == "__main__":
  print('hi')
  app.run()
