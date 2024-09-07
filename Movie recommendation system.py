import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv('/content/movies.csv')

# Fill NaN values in the genres column
movies['genres'] = movies['genres'].fillna('')

# Create a TF-IDF Vectorizer to convert genres into numerical format
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the genres column into a TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity between all movies based on their genres
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on title
def get_recommendations(title, cosine_sim=cosine_sim):
    # Search for the movie by title (case-insensitive)
    idx = movies[movies['title'].str.contains(title, case=False)].index

    # If no movie is found, return an error message
    if len(idx) == 0:
        return ["No movie found with that title. Try again."]

    # Get the similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx[0]]))

    # Sort the movies based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 10 most similar movies
    movie_indices = [i[0] for i in sim_scores[1:11]]

    # Return the titles of the top 10 most similar movies
    return movies['title'].iloc[movie_indices].tolist()

# Console-based interaction
def movie_recommendation_system():
    print("Welcome to the Movie Recommendation System!")
    while True:
        # Prompt the user for a movie title
        movie_title = input("\nEnter a movie title (or 'exit' to quit): ").strip()
        
        # Exit the loop if the user types 'exit'
        if movie_title.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Get recommendations based on the input title
        recommendations = get_recommendations(movie_title)
        
        # Display the recommendations
        print(f"\nRecommendations for '{movie_title}':")
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie}")

# Run the movie recommendation system
if __name__ == "__main__":
    movie_recommendation_system()
