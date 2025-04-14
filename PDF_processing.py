from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json

pdf_paths = [
    "/Users/olivia/Documents/python/RCP0032 Intake 10 Student Internship Summary Reports.pdf",
    "/Users/olivia/Documents/python/Research Computing Platform Student Internship Handbook.pdf",
    "/Users/olivia/Documents/python/Student Projects Outline - Summer 2425.pdf"
]

# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path) # Create PDF reader
    pdf_docs = loader.load()

    text = "" # Initialise empty string for extracted text
    for page in pdf_docs: # Loop through each page in PDF
        text += page.page_content

    return text if text else ""  # Return empty string if no text extracted

# Function to generate keywords using TF-IDF
def generate_keywords_tfidf(content, top_n=5): 
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform([content])
    keywords = vectorizer.get_feature_names_out()
    return list(keywords)



# Scrape PDFs
pdf_data = [] # Initialise 

for pdf_path in pdf_paths: # Loop through each PDF file path
    pdf_text = extract_text_from_pdf(pdf_path)
    pdf_keywords = generate_keywords_tfidf(pdf_text)
    
    # Extract the file name (without extension) from the file path
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # os.path.basename(pdf_path) : extracts file name e.g. "example.pdf"
    # os.path.splitext(...)[0] : removes file extension e.g. "example"

    # Extract PDF name
    pdf_source = os.path.basename(pdf_path)
    
    pdf_data.append({
        "title": f"Content from {file_name}",
        "content": pdf_text,
        "topic": file_name,  # Use the file name as the topic
        "keywords": pdf_keywords,
        "source": pdf_source  # Do not save any source for PDFs
    })

# Check if everything works (comment out)
# for data in pdf_data:
#     print(f"Title: {data['title']}")
#     print(f"Content: {data['content'][:500]}")  # Print the first 500 characters of the content
#     print(f"Topic: {data['topic']}")
#     print(f"Keywords: {data['keywords']}")
#     print(f"Source: {data['source']}")

# Convert the data into a JSON file
with open("aggregated_data.json", "w") as f:
    json.dump(pdf_data, f, indent=4)