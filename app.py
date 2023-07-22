import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import string
import re
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models
from streamlit import components



@st.cache_data 
def load_data():
    df = pd.read_excel("data/preprocessed/merged_df.xlsx")
    return(df)


# Function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Tokenization
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def create_word_cloud(text_list):
    # Concatenate the text from the list into a single string
    text = " ".join(text_list)

    # Download the NLTK stopwords (if not already downloaded)
    stop_words = set(stopwords.words('english'))

    # Preprocess the text - remove punctuation and stopwords
    translator = str.maketrans('', '', string.punctuation)
    words = nltk.word_tokenize(text.lower())
    words = [word.translate(translator) for word in words if word.lower() not in stop_words]

    # Combine the words back into a single string
    preprocessed_text = " ".join(words)

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(preprocessed_text)

    # Display the word cloud using matplotlib
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return (fig)


def home(mode):   
    st.title(mode)
    st.subheader('Welcome')
    st.markdown("NLP for Comprehensive Understanding of World Leaders' Speeches")
    st.markdown("In today's world, understanding the speeches of world leaders has become more critical than ever. These addresses shape policies, influence global affairs, and impact millions of lives. Harnessing the power of Natural Language Processing (NLP), an advanced AI technique, offers a transformative use case in comprehending and extracting insights from world leaders' speeches. By employing NLP's capabilities, we can unravel the nuances of language, sentiments, and key themes, providing invaluable knowledge for diplomacy, policymaking, and international relations.")


def Leader(mode,df):
    df = df

    # List of speaker and their posts
    Country = df["Country"].unique()

    # Create a dropdown list
    selected_country = st.selectbox("Select a Country:", Country)
    Leader = df.loc[df["Country"] == selected_country,"Name of Person Speaking"].unique()
    selected_leader = st.selectbox("Select a Leader:", Leader,  index=0)

    # Submit button
    if st.button("Submit"):
        result = df.loc[df["Country"]== selected_country]
        result = result.loc[result["Name of Person Speaking"] == selected_leader]

        text_result = result['text'].tolist()

        st.pyplot(create_word_cloud(text_result))

        # Create a Gensim dictionary and corpus

        text_result = result['text'].apply(preprocess_text)

        dictionary = Dictionary(text_result)
        corpus = [dictionary.doc2bow(doc) for doc in text_result]


        # LSI (Latent Semantic Indexing) Model for Topic Modeling
        lda_model = LdaModel(corpus, id2word=dictionary, num_topics=5)

        # Extracting topics from the LSI model
        topics = lda_model.print_topics(num_topics=5)

        coherence_model = CoherenceModel(model=lda_model, texts=text_result, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()

        # Prepare data for topic visualization
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_result]
        vis_data = pyLDAvis.gensim_models.prepare(lda_model, doc_term_matrix, dictionary)
        html_string = pyLDAvis.prepared_data_to_html(vis_data)
        # Display coherence score
        st.subheader("Coherence Score:")
        st.write(coherence_score)

        # Display extracted topics
        st.subheader("Extracted Topics:")
        for topic in topics:
            st.write(topic)

        # Topic visualization
        st.subheader("Topic Visualization:")
        components.v1.html(html_string, width=800, height=800, scrolling=True)


def Topics(mode):
    pass

def Trend(mode):
    pass

def main():
    main_mode = st.sidebar.selectbox('Select Menu',
        ['Home', 'Leader', "Topics","Words Trend"]
    )
    df = load_data()

    if main_mode == 'Home':
        home(main_mode)
    elif main_mode == 'Leader':
        Leader(main_mode,df)

    elif main_mode == 'Topics':
        Topics(main_mode)

    elif main_mode == 'Words Trend':
        Trend(main_mode)
   
if __name__ == '__main__':
    main()