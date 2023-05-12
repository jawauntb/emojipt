from flask import Flask, request, jsonify
from langchain import PromptTemplate, OpenAI, LLMChain
from flask_cors import CORS
import os

app = Flask(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY")
CORS(app)

# Initialize LLMChain
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
prompt_template = "Give me some emojis related to {input}"
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

  # Depending on your result you may need to adjust the following lines
  # Check if input is emoji or not

  # If it's not emoji, generate 6 related emojis
  emojis = result

  return jsonify({'emojis': emojis})


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000)
