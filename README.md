## 🎥 AI Media Studio  
### AI Media Studio is a **Streamlit-based application** that allows you to generate high-quality **images** and **videos** from text prompts.  

## ✨ Features
* 📂 Upload multiple images and manage them in-app  
* 💬 Chat with **Gemini** about your images or creative prompts  
* 🖌️ Generate AI images directly inside the chat  
* 🎥 Convert AI-generated images into short AI videos using **Replicate**  
* 🖥️ Watch videos directly in the app (inline player)  
* ⚡ Clean & responsive **Streamlit** UI with sidebar management

## 🛠️ Tech Stack
* **Frontend/UI** : Streamlit
* **AI Models** : Google Gemini 
* **Image Processing** : PIL (Pillow) 
* **Environment Config** : python-dotenv

## ⚙️ Setup & Installation
### 1 Create & Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2 Install Dependencies
```bash
pip install -r requirements.txt
pip install streamlit pillow python-dotenv replicate google-genai requests
```

### 3 Setup Environment Variables
### Create a .env file in the project root and add:
```bash
REPLICATE_API_TOKEN=your_replicate_api_token
GOOGLE_API_KEY=your_google_api_key  
```

### 4 Run the App
```bash
streamlit run app.py
```

## 📌 Use Cases  
* 🎨 Creative media generation (posters, anime art, concept art).  
* 🎥 Quick video creation from static images.  
* 🌆 Cityscapes, anime, or fantasy content generation.  
* 🐾 Fun & engaging media (e.g., anime cats 🐱).




