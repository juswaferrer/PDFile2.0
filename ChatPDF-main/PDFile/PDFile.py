from flask import Flask, request, render_template, jsonify
import io
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

app = Flask(__name__)

# Load QA model and tokenizer
model_name = "distilbert/distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

pdf_conversations = {}

# Function to extract text and title from PDF
def extract_text_and_title_from_pdf(pdf_content):
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        title = pdf_reader.metadata.title if pdf_reader.metadata.title else None
        for page in pdf_reader.pages:
            text += page.extract_text()
        if not title:
            title = text[:30] + "..." if len(text) > 30 else text
        return title, text
    except Exception as e:
        raise Exception("Error extracting text from PDF.")

# Function to answer the question
def answer_question(question, context):
    try:
        answer = qa_pipeline(question=question, context=context)
        answer_text = answer["answer"]
        return answer_text
    except Exception as e:
        raise Exception("Error answering the question.")

@app.route('/')
def index():
    return render_template('PDFile.html')

@app.route('/chat', methods=['POST'])
def chat():
    pdf_file = request.files['pdf_file']
    pdf_content = pdf_file.read()

    if not pdf_content:
        return jsonify({'error': "Please upload a PDF file."})
    
    user_question = request.form['user_query']
    if not user_question:
        return jsonify({'error': "Please provide a question."})

    try:
        title, pdf_text = extract_text_and_title_from_pdf(pdf_content)
        answer = answer_question(user_question, pdf_text)

        # Store conversation in pdf_conversations
        if title not in pdf_conversations:
            pdf_conversations[title] = []
        pdf_conversations[title].append({'question': user_question, 'answer': answer})

        return jsonify({'title': title, 'answer': answer})
    except Exception as e:
        return jsonify({'error': f"Error: {e}"})

@app.route('/conversations', methods=['GET'])
def get_conversations():
    return jsonify(pdf_conversations)

if __name__ == '__main__':
    app.run(debug=True)