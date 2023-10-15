from flask import Flask, request, make_response
# from langchain import PromptTemplate, OpenAI, LLMChain
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY")
CORS(app)

# # Initialize LLMChain
# llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
# prompt_template = "Give me 20 emojis related to {input}"
# llm_chain = LLMChain(llm=llm,
#                      prompt=PromptTemplate.from_template(prompt_template))


@app.route('/')
def index():
  return 'Hello from Flask!'


# @app.route('/emojis', methods=['POST'])
# def generate_emojis():
#   data = request.get_json()
#   input_data = data.get('input', '')
#   # Generate result from LLMChain
#   result = llm_chain.run(input_data)

#   # Debugging print statements
#   print("Result: ", result)
#   print("input: ", input_data)

#   emojis = result

#   return jsonify({'emojis': emojis})


@app.route('/promptly', methods=['POST'])
def generate_response():
  data = request.get_json()
  payload = data.get('payload', '')
  authkey = 'Bearer ' + openai_api_key
  headers = {'Authorization': authkey, 'Content-Type': 'application/json'}

  # Make a request to the OpenAI API
  response = requests.post('https://api.openai.com/v1/chat/completions',
                           headers=headers,
                           json=payload)

  # Convert the requests.Response object to a Flask Response object
  flask_response = make_response((response.content, response.status_code))

  # Copy headers from the original response (optional, based on your needs)
  for header in response.headers:
    flask_response.headers[header] = response.headers[header]

  return flask_response


if __name__ == "__main__":
  app.run()
