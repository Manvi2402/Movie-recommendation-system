
# ==============================
# index.py —MovieGenie 🍿🪄🎥🎞️(Hollywood + Anime)
# ==============================
import random
import requests
import streamlit as st
import pandas as pd
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MovieGenie Recommender🪄", layout="wide")

# -------------------------
# TMDB API setup
# -------------------------
TMDB_API_KEY = "5c0f39faa981a31212714021294c1f88"  # 🔑 replace with your TMDB key
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# -------------------------
# Fetch poster & description from TMDB
# -------------------------
def fetch_tmdb_artifacts(title, year=None):
    try:
        params = {"api_key": TMDB_API_KEY, "query": title}
        if year:
            params["year"] = year
        response = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params)
        response.raise_for_status()
        data = response.json()
        if data["results"]:
            movie = data["results"][0]
            poster = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None
            tmdb_url = f"https://www.themoviedb.org/movie/{movie['id']}"
            return poster, tmdb_url
    except Exception as e:
        print(f"TMDB fetch error for {title}: {e}")
    return None, None

# ------------------------
# Load Datasets
# ------------------------
@st.cache_data
def load_hollywood():
    movies = pd.read_csv("datasets/hollywood/movies.csv")
    ratings = pd.read_csv("datasets/hollywood/ratings.csv")
    tags = pd.read_csv("datasets/hollywood/tags.csv")

    movies["genres"] = movies["genres"].fillna("")
    avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
    avg_ratings.columns = ["movieId", "avg_rating"]
    movies = movies.merge(avg_ratings, on="movieId", how="left")

    tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
    movies = movies.merge(tags_grouped, on="movieId", how="left")

    movies["combined_features"] = (movies["genres"] + " ") * 3 + (movies["tag"].fillna("") + " ") * 2
    movies["year"] = None
    movies["source"] = "Hollywood"
    movies["poster_url"] = movies.get("poster_url", "")
    movies["title"] = movies["title"].astype(str).str.strip().str.lower()
    movies["description"] = movies["tag"].fillna("").replace("", pd.NA)
    movies["description"] = movies["description"].fillna(movies["genres"].replace("", pd.NA))
    movies["description"] = movies["description"].fillna("No description available.")
    movies["genres"] = movies["genres"].fillna("")
    return movies[["title", "year", "avg_rating", "combined_features", "source", "poster_url", "description", "genres"]]

@st.cache_data
def load_anime():
    anime = pd.read_csv("datasets/anime/anime_top10000.csv")
    anime = anime.rename(columns={
        "Anime_Name": "title",
        "Anime_Air_Years": "year",
        "Anime_Rating": "avg_rating",
        "Synopsis": "synopsis"
    })
    anime["title"] = anime["title"].astype(str).str.strip().str.lower()
    anime["synopsis"] = anime["synopsis"].fillna("")
    anime["combined_features"] = (anime["synopsis"] + " ") * 5 + " anime show"
    anime["source"] = "Anime"
    anime["poster_url"] = anime.get("poster_url", "")
    anime["year"] = pd.to_numeric(anime["year"], errors="coerce")
    anime["description"] = anime["synopsis"].replace("", "No description available.")
    anime["genres"] = anime.get("genres", "")
    return anime[["title", "year", "avg_rating", "combined_features", "source", "poster_url", "description", "genres"]]

@st.cache_data
def load_all():
    hollywood = load_hollywood()
    anime = load_anime()
    df = pd.concat([hollywood, anime], ignore_index=True)
    df = df.drop_duplicates(subset=["title", "source"])
    df["combined_features"] = df["combined_features"].fillna("")
    df["avg_rating"] = df.groupby("source")["avg_rating"].transform(lambda x: x.fillna(x.mean()))
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["poster_url"] = df["poster_url"].fillna("")
    df["description"] = df["description"].fillna("No description available.")
    return df.reset_index(drop=True)

all_movies = load_all()

# ------------------------
# TF-IDF Vectorization
# ------------------------
@st.cache_data
def compute_tfidf_matrix(data: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english")
    return tfidf.fit_transform(data["combined_features"])

tfidf_matrix = compute_tfidf_matrix(all_movies)
indices = pd.Series(all_movies.index, index=(all_movies["title"] + "_" + all_movies["source"])).drop_duplicates()

# ------------------------
# Helpers
# ------------------------
def get_best_match(title, titles_list):
    matches = get_close_matches(title.lower(), titles_list, n=1, cutoff=0.6)
    return matches[0] if matches else None

def safe_title(s: str) -> str:
    try:
        return s.title()
    except:
        return s

# ------------------------
# Recommendation function
# ------------------------
def recommend(title, source, top_k=5):
    key = title.strip().lower() + "_" + source
    actual_key = get_best_match(key, indices.index.tolist())
    if not actual_key:
        return pd.DataFrame()
    idx = indices.get(actual_key)
    if idx is None or idx >= tfidf_matrix.shape[0]:
        return pd.DataFrame()

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    candidates = all_movies.copy()
    candidates['sim_score'] = sim_scores
    candidates['hybrid_score'] = 0.7 * candidates['sim_score'] + 0.3 * (candidates['avg_rating'].fillna(0) / 10)
    candidates = candidates.drop(idx).sort_values('hybrid_score', ascending=False).head(top_k)

    candidates['poster_url'] = None
    candidates['tmdb_url'] = None
    for i in candidates.index:
        poster, tmdb_url = fetch_tmdb_artifacts(candidates.at[i, "title"], candidates.at[i, "year"])
        candidates.at[i, "poster_url"] = poster
        candidates.at[i, "tmdb_url"] = tmdb_url

    return candidates

# ------------------------
# UI
# ------------------------
st.title("MovieGenie🍿🪄🎥🎞️")

# Category select
source_selected = st.selectbox("Select category:", ["Hollywood", "Anime"])
filtered = all_movies[all_movies["source"] == source_selected]

# Mood selection (for Surprise Me)
moods = ["Action", "Romance", "Comedy", "Thriller", "Science Fiction", "Adventure", "Drama"]
mood_emojis = {
    "Action": "💥", "Romance": "❤️", "Comedy": "😂",
    "Thriller": "😱", "Science Fiction": "👽", "Adventure": "🗺️", "Drama": "🎭"
}
selected_mood = st.selectbox("Pick your mood 🎭 (for Surprise Me only)", moods)
st.caption(f"You picked {mood_emojis.get(selected_mood,'')} {selected_mood}")

# Base movie select
selected_movie = st.selectbox(
    f"Choose a {source_selected} title you like:",
    sorted(filtered["title"].dropna().unique())
)

col_left, col_right = st.columns(2)
with col_left:
    get_recs = st.button("Get Recommendations")
with col_right:
    surprise = st.button("🎲 Surprise Me")

def show_base_preview(row: pd.Series, header: str):
    st.subheader(header)
    left, right = st.columns([1, 2])
    with left:
        poster = row.get("poster_url") or fetch_tmdb_artifacts(row["title"], row.get("year", None))[0]
        if poster:
            st.image(poster, use_container_width=True)
    with right:
        st.markdown(f"**Title:** {safe_title(row['title'])}")
        st.markdown(f"**Source:** {row['source']}")
        rating_disp = round(row["avg_rating"], 2) if pd.notna(row["avg_rating"]) else "N/A"
        st.markdown(f"**Rating:** ⭐ {rating_disp}")
        st.write(row.get("description", "No description available."))

def show_recs_grid(recommendations: pd.DataFrame):
    st.success("You might also enjoy:")
    cols = st.columns(len(recommendations))
    for col, (_, r) in zip(cols, recommendations.iterrows()):
        with col:
            poster = r["poster_url"]
            if poster:
                if r.get("tmdb_url"):
                    st.markdown(f"[![poster]({poster})]({r['tmdb_url']})", unsafe_allow_html=True)
                else:
                    st.image(poster, use_container_width=True, caption=safe_title(r["title"]))
            else:
                st.markdown("📽️")
                st.markdown(f"**{safe_title(r['title'])}**")
            st.markdown(f"*{r['source']}*")
            rating_disp = round(r["avg_rating"], 2) if pd.notna(r["avg_rating"]) else "N/A"
            st.markdown(f"⭐ {rating_disp}")

# --- Handle Recommendation flow
if get_recs and selected_movie:
    base_row = filtered[filtered["title"] == selected_movie].iloc[0]
    show_base_preview(base_row, f"🎬 You liked: {safe_title(selected_movie)}")
    with st.spinner("Finding your next favorite..."):
        recs = recommend(selected_movie, source_selected, top_k=5)
    if not recs.empty:
        show_recs_grid(recs)
    else:
        st.error("No recommendations found. Try a different title or use 🎲 Surprise Me.")

# --- Surprise Me flow
elif surprise:
    chosen = None
    if "genres" in filtered.columns and not filtered.empty:
        exploded = filtered.assign(genre_split=filtered["genres"].fillna("").astype(str).str.split("|")).explode("genre_split")
        exploded["genre_split"] = exploded["genre_split"].str.strip().str.lower()
        mood_lower = selected_mood.lower()
        mood_hits = exploded[exploded["genre_split"] == mood_lower]
        if not mood_hits.empty:
            chosen = filtered[filtered["title"].isin(mood_hits["title"])].sample(1).iloc[0]
    if chosen is None:
        chosen = filtered.sample(1).iloc[0]
    if chosen is not None:
        show_base_preview(chosen, f"🎬 Surprise pick: {safe_title(chosen['title'])}")
        with st.spinner(f"Finding movies like '{safe_title(chosen['title'])}'..."):
            recs = recommend(chosen["title"], chosen["source"], top_k=5)
        if not recs.empty:
            show_recs_grid(recs)
        else:
            st.error("No recommendations found. Try again 🎲.")

# Footer tip
st.caption("Tip: Add your TMDB key to .streamlit/secrets.toml as TMDB_API_KEY to enable posters & links.")
