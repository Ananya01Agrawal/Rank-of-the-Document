from flask import Flask, render_template, request, redirect, url_for
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    # Check if the file extension is allowed
    return '.' in filename

def get_uploaded_documents():
    # Read all uploaded documents and return their contents
    uploaded_documents = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'r') as file:
            uploaded_documents.append(file.read())
    return uploaded_documents

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('files[]')
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    return redirect(url_for('index'))

@app.route('/search', methods=['POST'])
def search():

    query = request.form['query']
    uploaded_documents = get_uploaded_documents()

    # Convert documents and query into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(uploaded_documents + [query])

    # Calculate cosine similarity between query vector and document vectors
    query_vector = vectors[-1]  # Last vector corresponds to the query
    similarities = cosine_similarity(query_vector, vectors[:-1])

    # Sort documents based on similarity scores
    sorted_indices = similarities.argsort()[0][::-1]


    # Create a list to store document rank and content
    ranked_documents = []

    # Display all documents with their ranks
    for rank, index in enumerate(sorted_indices):
        document_rank = rank + 1
        document_content = uploaded_documents[index]
        ranked_documents.append((document_rank, document_content))

    return render_template('result.html', ranked_documents=ranked_documents)

if __name__ == '__main__':
    app.run(debug=True)
