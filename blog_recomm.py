import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation


data = pd.read_csv('medium_data.csv')

# Preprocessing the dataset
data = data.drop_duplicates(subset=['title', 'url', 'subtitle'], keep='first')  # removal of duplicate values

data['Subtitle'] = data['subtitle'].fillna('NA')                    # missing value handling
data['content'] = data['title'].fillna('NA') + ' ' + data['Subtitle'].fillna('NA')  #

#Text vectorization using TF-IDF
vector = TfidfVectorizer(stop_words = 'english', max_features=1000)
matrix = vector.fit_transform(data['content'])

# Combine with numerical features
data['claps'] = data['claps'].fillna(0)
data['responses'] = data['responses'].fillna(0)
data['reading_time'] = data['reading_time'].fillna(0)

# data.to_csv('preprocessed_data.csv')

ui_features = data[['claps', 'responses', 'reading_time']]
ui_normalized = (ui_features - ui_features.mean()) / ui_features.std()

# Combine TF_IDF text features with ui_features
combined_features = np.hstack((matrix.toarray(), ui_normalized.to_numpy()))

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(combined_features)

# Applying Categorical features
data['publication_encoded'] = pd.factorize(data['publication'])[0]

# Applying time-based features
data['publication_year'] = pd.to_datetime(data['date'], errors='coerce', dayfirst=True).dt.year
data['publication_month'] = pd.to_datetime(data['date'], errors='coerce', dayfirst=True).dt.month
data['publication_day'] = pd.to_datetime(data['date'], errors='coerce', dayfirst=True).dt.day
data['publication_weekday'] = pd.to_datetime(data['date'], errors='coerce', dayfirst=True).dt.weekday

data['is_weekend'] = data['publication_weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Topic Modeling using LDA
lda = LatentDirichletAllocation(n_components=1, random_state=42)
topic_matrix = lda.fit_transform(matrix)
data['topic_distribution'] = topic_matrix
data.to_csv('preprocessed_data.csv')

def blog_recommendation(blog_index, top_n=5):

    # Get simmilarity scores for the blog
    sim_score = list(enumerate(similarity_matrix[blog_index]))
    # Sort by similarity scores
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    # Exclude the blog itself (index 0) and return the top N recommendations
    sim_score = sim_score[1:top_n+1]
    # Get recommended blog indices
    blog_indices = [i[0] for i in sim_score]
    return data[['title', 'url']].iloc[blog_indices]

recommendations = blog_recommendation(0)
print(recommendations)
