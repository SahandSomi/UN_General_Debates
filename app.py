import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from collections import Counter
import plotly.graph_objects as go

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
from copy import deepcopy


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


def Leader(mode,df_1):
    df = deepcopy(df_1)

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

def Trend(mode,df_1):

    df = deepcopy(df_1)

    st.title("Multiple Word Input and List Saving")
    

    # Initialize an empty list to store events
    events = []

    # Create a form to add events
    with st.form("event_form"):
        # Text input for multiple words
        user_input = st.text_area("Enter multiple keywords to see trend:", "")
        event_name = st.text_input("Enter the name of the event:")
        event_year = st.number_input("Enter the year of the event:", min_value=1900, max_value=2100)

        # If the "Add Event" button is clicked, add the event to the list
        if st.form_submit_button(label="submit"):
            # Add the event details to the list as a dictionary
            event = {'event': event_name, 'year': event_year}
            events.append(event)

            # Convert the user input to a list of words
            keyword_list = user_input.split("\n")

            for index, row in df.iterrows():
                key_dict = extract_keyword_frequency(row['text'],keyword_list)
                for key, value in key_dict.items():
                    df.at[index, key] = value

            # To generate a DataFrame for the results and eliminate unnecessary columns, we will create a new DataFrame by dropping the columns that are not required
            df = df.drop(['text','session','code','Country','Name of Person Speaking'], axis=1)
            # Group by each year
            group_by_year = df.groupby('year')
            # Getting sum of each keywords
            answer = group_by_year.sum()

            # Create a line plot with multiple lines using Plotly
            fig = go.Figure()
            for i in answer.columns:
                fig.add_trace(go.Scatter(x=answer.index, y=answer[i], name=i))

            # Automatically generate annotations for events by year
            annotations = []
            if events[0]['event'] != "":
                annotations = []
                for event in events:
                    y_data =[answer[i][event['year']] for i in answer.columns]
                    y_data.append(0)
                    y_data = max(y_data)
                    annotation = dict(
                        x=event['year'], y=y_data,
                        xref="x", yref="y",
                        text=f"{event['event']} ({y_data})",
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-100
                    )
                    annotations.append(annotation)

                # Customize the plot layout
                fig.update_layout(xaxis_title='Year',
                                yaxis_title='Sum for wordss',
                                showlegend=True,
                                annotations=annotations)
                
            else:
                # Customize the plot layout
                fig.update_layout(xaxis_title='Year',
                                yaxis_title='Sum for wordss',
                                showlegend=True)

            st.plotly_chart(fig)

def extract_keyword_frequency(text,keyword_list):
    tokens = nltk.word_tokenize(text)
    keyword_counts = Counter(tokens)
    keyword_frequency = {keyword: keyword_counts[keyword] for keyword in keyword_list if keyword in keyword_counts}
    return keyword_frequency

def main():
    main_mode = st.sidebar.selectbox('Select Menu',
        ['Home', 'Leader',"Words Trend"]
    )
    df = load_data()

    if main_mode == 'Home':
        home(main_mode)
    elif main_mode == 'Leader':
        Leader(main_mode,df)

    elif main_mode == 'Words Trend':
        Trend(main_mode, df)
   
if __name__ == '__main__':
    main()