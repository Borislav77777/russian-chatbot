from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch
import os

app = Flask(__name__)

# Загрузка модели и токенизатора
try:
    tokenizer = AutoTokenizer.from_pretrained("t-bank-ai/ruDialoGPT-medium", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("t-bank-ai/ruDialoGPT-medium", trust_remote_code=True)
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.json['message']
    
    # Подготовка входных данных
    inputs = tokenizer(user_message, return_tensors="pt")
    
    # Генерация ответа
    with torch.no_grad():
        generated_token_ids = model.generate(
            inputs['input_ids'],
            max_length=100,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
    
    # Декодирование ответа
    bot_response = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)
    
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
