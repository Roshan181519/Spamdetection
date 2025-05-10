### 📌 **Spam Detection API**  

This project is a **Flask-based Spam Detection API** that classifies text messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** techniques and a **Naïve Bayes model** trained on TF-IDF features.  

🚀 **Live Demo (if deployed)**: *["in future i wull add "]*  

---

## 📂 **Project Structure**  

```
SpamDetection/
│── app.py                  # Flask API for spam detection  
│── model_training.ipynb     # Jupyter Notebook for training the spam classifier  
│── spam_model.pkl           # Trained Naïve Bayes model  
│── vectorizer.pkl           # TF-IDF vectorizer for text transformation  
│── static/                  # Folder for UI assets (if any)  
│── templates/               # HTML templates for the UI  
│── README.md                # Project documentation  
│── requirements.txt         # Required dependencies  
│── .gitignore               # Ignored files (including large datasets)  
```

---

## ⚡ **Features**  

✔️ **Spam Detection** – Predicts whether a message is spam or not  
✔️ **Preprocessing** – Cleans text using tokenization, stopword removal, and lemmatization  
✔️ **API Endpoint** – Accepts text input via a simple JSON API  
✔️ **Web UI** – Easy-to-use interface for testing messages (if implemented)  

---

## 🔧 **Installation & Setup**  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/Roshan181519/Spamdetection.git
cd Spamdetection
```

### 2️⃣ **Create a Virtual Environment** (Recommended)  
```bash
python -m venv venv  
source venv/bin/activate  # On macOS/Linux  
venv\Scripts\activate  # On Windows  
```

### 3️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 4️⃣ **Run the Flask App**  
```bash
python app.py
```
The API will be available at `http://127.0.0.1:5000/`.

---

## 📡 **API Usage**  

### **1️⃣ Send a POST Request**  
```json
POST http://127.0.0.1:5000/predict
Content-Type: application/json
{
  "message": "Congratulations! You won a free iPhone! Claim now."
}
```

### **2️⃣ API Response**  
```json
{
  "category": "spam"
}
```

---

## ⚠️ **Important Note**  

Some large files (like `tfidf_features.csv`) were removed before pushing the project to GitHub because they exceeded the 100MB file size limit. If you need to generate these files again, you can retrain the model using `model_training.ipynb`.

---

## 🤝 **Contributing**  

Contributions, issues, and feature requests are welcome! Feel free to submit a pull request.  

---

  
