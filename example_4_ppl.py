"""@ https://example.person.repl.co/poast"""

from flask import Flask, request, jsonify
from langchain import PromptTemplate, OpenAI, LLMChain
from flask_cors import CORS
import os

# Setup your environment
app = Flask(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY")
CORS(app)

# Initialize LLMChain
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
prompt_template = "do something with this {input}"
prompt = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# default route
@app.route('/')
def index():
  return llm_chain.run("hello there!")

# do something with user input here
@app.route('/poast', methods=['POST'])
def poast():
  data = request.get_json()
  input_data = data.get('input', '')
  # Generate result from LLMChain
  result = llm_chain.run(input_data)

  # Debugging print statements
  print("Result: ", result)

  return jsonify({'response': result})
  # end example
if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8080)
  
