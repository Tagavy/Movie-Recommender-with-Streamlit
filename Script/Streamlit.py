#pip install streamlit --upgrade
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import movieposters as mp
from PIL import Image


## Streamlit TITLE
st.title(":red[Movie Recommender] :movie_camera:")

# Add links to the different sections
st.markdown('<a href="#by-popularity">By popuilarity</a> / <a href="#by-movies-similarity">By movies similarity</a> / <a href="#by-users-similarity">By users similarity</a>.', unsafe_allow_html=True)

## Title picture
# Load image from file
image = Image.open(r'Data/Pictures/Best-Christmas-Movies-FB.jpg')

# Display image in Streamlit app
st.image(image, use_column_width=True)

## Subtitle
st.subheader(":blue[By popularity]")

## **Recommenders**

movies = pd.read_csv(r'Data/movies_movies.csv')
ratings = pd.read_csv(r'Data/movies_ratings.csv')

### Genres list

genres = movies['genres'].str.split("|", expand=True)
genre_list = pd.unique(genres[[0, 1, 2, 3, 4, 5, 6]].values.ravel('K'))
fin_genre_list = np.delete(genre_list, np.where(genre_list == '(no genres listed)') | (genre_list == None))


# *Popularity Recommender*

def pop_rec(genre, year, n_output):

    scaler = MinMaxScaler()

    rating_df = pd.DataFrame(ratings.groupby('movieId')['rating'].mean()) # group movies and get their avarage rating
    rating_df['rating_count'] = ratings.groupby('movieId')['rating'].count() # get rating count of each movie(how many times each movie was rated)
    scaled_df = pd.DataFrame(scaler.fit_transform(rating_df), index=rating_df.index, columns=rating_df.columns) # scale the ratings and rating counts
      #scaled_df
    scaled_df["hybrid"] = scaled_df['rating'] + scaled_df['rating_count'] # add up rating and rating count for each mivie
    sort_rate = pd.DataFrame(scaled_df["hybrid"].sort_values(ascending=False))
    pattern = '\((\d{4})\)'
    recommend1 = sort_rate.merge(rating_df.merge(movies, how='left', left_index=True, right_on="movieId"), how='left', left_index=True, right_index=True)
    recommend1['year'] = movies.title.str.extract(pattern, expand=False)
    recommend2 = recommend1.dropna()
    recommender3 = recommend2.loc[recommend2.genres.str.contains(genre)] 
    recommender3['year'] = recommender3['year'].astype(int)
    recommender4 = recommender3.loc[(recommender3["year"] >= year-5) & (recommender3["year"] <= year+5)]
    return pd.DataFrame(recommender4['title']).head(n_output)

    
# BUTTON_genrebox
genre_inp = st.selectbox(
    'What genre would you like to watch?',
    (fin_genre_list))

st.write('Genre:', genre_inp)


# BUTTON_yearbox
year_inp = st.slider('Give a year range', 1902, 2018, 2010, key=1)
st.write(year_inp)

# BUTTON_number of recommended movies
num_inp1 = st.slider('Give a number of recommendations(1-20)', 1, 20, 1, key=2)
st.write(num_inp1)

#st.dataframe(pop_rec(genre_inp1, year_inp, num_inp))

# Posters
from PIL import Image
#posters_list 
posters_list1 = []
titles_list1 = []

for i in pop_rec(genre_inp, year_inp, num_inp1).title:
    titles_list1.append(i)
    try:
        link = mp.get_poster(title = i)
        posters_list1.append(link)
        
    except:
        posters_list1.append('https://www.movienewz.com/img/films/poster-holder.jpg')
        continue
        
columns1 = st.columns(int(len(posters_list1)))
for a, i, j in zip (columns1, posters_list1, titles_list1):
    with a:
        st.image(i, j, 100)

######################################################################

# *Recomender by movie corrrelation*
st.subheader(":blue[By movies similarity]")

# BUTTON_pick_reference_movie
original_list = movies['title']
result1 = st.selectbox('Select a movie that you like', original_list)
st.write(f'You have chosen {result1}')

# BUTTON_number of recommended movies
num_inp2 = st.slider('Give a number of recommendations(1-20)', 1, 20, 1, key=3)
st.write(num_inp2)

movie_Id = movies.loc[movies['title'] == f'{result1}', 'movieId'].iloc[0]

rating_pivot = pd.pivot_table(ratings, values = 'rating', columns = 'movieId', index = 'userId')

def recom_m(n, m):
    rating_pivot = pd.pivot_table(ratings, values = 'rating', columns = 'movieId', index = 'userId')
    correl_df = rating_pivot.corrwith(rating_pivot[n]) #correlation between the target item and all other items
    correl_df = pd.DataFrame(correl_df.sort_values(ascending=False)) #convert the Series of the correlation to a dataframe and sort it to have top items with the highest correlations
    correl_df1 = correl_df.merge(pd.DataFrame(movies[["movieId", "title"]]), on="movieId", how="left") #get names of the items
    correl_df1.dropna(inplace=True) #drop nulls
    rating_count = pd.DataFrame(ratings.groupby('movieId')['rating'].mean()) #create a dataframe fore the count of the ratings
    rating_count['rating_count'] = ratings.groupby('movieId')['rating'].count() #get the count of the rating to exclude items that are rated only few times
    rating_count.drop(1, inplace=True) #drop the target item itself from the df
    movie_corr_summary = correl_df1.join(rating_count['rating_count']) #join raitings with rating count
    top = movie_corr_summary[movie_corr_summary['rating_count']>=10].sort_values(0, ascending=False) #dropping items that are rated less than 10 times
    return top.head(m)

# Posters
posters_list2 = []
titles_list2 = []
for i in recom_m(movie_Id, num_inp2).title:
    link = mp.get_poster(title = i)
    #title_postlink = i, link
    posters_list2.append(link)
    titles_list2.append(i)
    
#posters_list 
columns2 = st.columns(len(posters_list2))
for a, i, j in zip (columns2, posters_list2, titles_list2):
    with a:
        st.image(i, j, 100)
#############################################################

# *Recomender by user corrrelation*

## Subtitle
st.subheader(":blue[By users similarity]")


# BUTTON_pick_reference_user
original_list1 = ratings.userId.unique()
result2 = st.selectbox('Select an ID of the user that you resonate with', original_list1)
st.write(f'You have chosen {result2}')


# BUTTON_number of recommended movies
num_inp3 = st.slider('Give a number of recommendations(1-20)', 1, 20, 1, key=4)
st.write(num_inp3)


def user_based_rec(userId, num_rec_items):
    users_items = rating_pivot.copy()
    users_items.fillna(0, inplace=True) # filling/replacing NaNs with 0
    user_similarities = pd.DataFrame(cosine_similarity(users_items), # compute cosine similarities
                                 columns=users_items.index, 
                                 index=users_items.index)
    user_Id = userId
    weights = (user_similarities.query("userId!=@user_Id")[user_Id] / sum(user_similarities.query("userId!=@user_Id")[user_Id])) # compute users' weights
    not_watched_movies = users_items.loc[users_items.index!=user_Id, users_items.loc[user_Id,:]==0] # select movies that the target user has not watched
    weighted_averages = pd.DataFrame(not_watched_movies.T.dot(weights), columns=["predicted_rating"]) # dot product between the not-watched-movies and the weights
    recommendations = weighted_averages.merge(movies, left_index=True, right_on="movieId")
    return recommendations.sort_values("predicted_rating", ascending=False).head(num_rec_items)

# Posters
posters_list3 = []
titles_list3 = []
for i in user_based_rec(result2, num_inp3).title:
    link = mp.get_poster(title = i)
    #title_postlink = i, link
    posters_list3.append(link)
    titles_list3.append(i)
    
#posters_list 
columns3 = st.columns(len(posters_list3))
for a, i, j in zip (columns3, posters_list3, titles_list3):
    with a:
        st.image(i, j, 100)
