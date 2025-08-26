## MovieGenie🍿🪄🎥🎞️

*** “Your wish, your movie!”🪄🎥 ***

MovieGenie is an   AI-powered movie recommendation system   that helps you discover your next favorite movie or show.  
It currently supports   Hollywood & Anime  , with smart recommendations, “Surprise Me” picks, and mood-based suggestions. Future plans include adding   K-Drama   and expanding the dataset.

---

## 📌 Features

- 🔎 Search & Pick: Choose a title you like from Hollywood or Anime.  
- 🧠 AI Recommendations:Content-based filtering using TF-IDF and cosine similarity.  
- 🎲 Surprise Me: Get a random movie/show based on your selected mood.  
- 🎭 Mood-Based Picks: Action 💥, Romance ❤️, Comedy 😂, Thriller 😱, Sci-Fi 👽, Adventure 🗺️, Drama 🎭.  
- 🎬 Posters & Links: Fetches posters and TMDB links dynamically using TMDB API.  
- ⭐ Ratings:Displays average ratings from datasets.  
- 📝 Description:Shows synopsis or combined genre/tag features.  

---
🌟 Wow Features

🎲 “Surprise Me” Button: Instantly discover a random movie or show based on your mood.

🧞‍♂️ AI-Powered Recommendations: Finds movies similar to your favorite picks using advanced content-based filtering.

🎭 Mood-Based Picks: Pick your mood and get suggestions tailored to it (Action 💥, Romance ❤️, Comedy 😂, etc.).

🖼️ Dynamic Posters & Links: Fetches posters and clickable TMDB URLs in real-time.

🔍 Fuzzy Search: Handles typos and close matches for smoother movie selection.

📊 Ratings & Insights: Shows average ratings and combines multiple metadata for smarter recommendations.

----

## 🛠️ Tech Stack

-Frontend:Streamlit (Python)  
- Backend / ML: Python, Pandas, NumPy, Scikit-learn  
- Datasets:    
    -Hollywood dataset (MovieLens: ratings, tags, movies)  
    - Anime dataset (CSV curated)  
    - TMDB API for posters and movie metadata  

- Deployment: Streamlit Cloud / Local  

---

## ⚙️ Setup Instructions

1.   Clone the repo:    
  
   git clone https://github.com/your-username/moviegenie.git
   cd .\project\
2. Create a virtual environment:
python -m venv myenv
source myenv/bin/activate   # Linux/Mac
.\myenv\Scripts\activate      # Windows

3.Install dependencies:

pip install -r requirements.txt
Add your TMDB API key (optional, for posters & links) in .streamlit/secrets.toml:

TMDB_API_KEY="YOUR_API_KEY_HERE"

4.Run the app:
streamlit run .\index.py

📊 Dataset Sources
MovieLens (ml-latest-small): Hollywood movies, ratings, tags

Anime dataset (CSV curated): Top 10,000 anime shows

TMDB API: Posters, metadata, URLs


🚀 Future Roadmap
📺 Add K-Drama dataset for Korean dramas.

🎭 User mood profiles for better recommendations.

👤 User login & saved preferences.

🔄 Hybrid filtering: Combine content-based and collaborative filtering.

☁️ Deploy globally on Streamlit Cloud or Heroku.

🎧 Genre & mood-specific recommendations with smarter scoring.

✨ Enhanced UI/UX: Carousel posters, dark mode, and hover details.

👩‍💻 Contributors
Manvi 

📜 License
This project is licensed under MIT License – free to use and modify.  

