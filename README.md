# Comprehensive Understanding of World Leaders' Speeches using Natural Language Processing (NLP)

In today's interconnected world, the speeches of world leaders play a pivotal role in shaping policies, influencing global affairs, and impacting the lives of millions. This project aims to harness the power of Natural Language Processing (NLP), an advanced AI technique, to comprehensively understand and extract valuable insights from the speeches of world leaders. By employing NLP's capabilities, we can unravel the nuances of language, sentiments, and key themes in these speeches, providing invaluable knowledge for diplomacy, policymaking, and international relations.

## Overview

The "Comprehensive Understanding of World Leaders' Speeches" is a Streamlit web application that leverages NLP algorithms and techniques to analyze and visualize the speeches given by prominent world leaders. This repository contains all the necessary code and resources to set up and deploy the application.

## Features

1. **Speech Text Analysis**: The application performs sophisticated NLP analysis on the provided speeches, including:

   - Sentiment Analysis: Identifying the overall sentiment (positive, negative, or neutral) of the speech.
   - Named Entity Recognition (NER): Extracting important entities such as names, organizations, locations, etc., mentioned in the speech.
   - Keyword Extraction: Identifying the most relevant keywords and phrases used throughout the speech.

2. **Speech Themes and Topics**: The application identifies the key themes and topics discussed in the speech, allowing users to gain a deeper understanding of the leader's message.

3. **Visualizations**: The application presents interactive visualizations, such as word clouds, sentiment distribution, and entity frequency, to facilitate better comprehension and exploration of the speeches.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7 or above
- Streamlit library
- Natural Language Processing (NLP) libraries (e.g., NLTK, wordcloud)

## Getting Started

1. Clone this repository to your local machine:

```bash
git clone https://github.com/SahandSomi/UN_General_Debates
cd UN_General_Debates
```

2. Set up a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

5. Access the application in your web browser by visiting `http://localhost:8501`.

## Usage

1. Upload Speech: Use the application's intuitive user interface to upload a speech transcript in text format (plain text, PDF, or Word documents).

2. Speech Analysis: Once the speech is uploaded, the application will automatically process it using NLP techniques and display the results.

3. Explore Insights: Interact with the visualizations and explore the sentiment, entities, and key themes extracted from the speech.

## Contributions and Issues

Contributions to this project are welcome! If you find any issues or have ideas to enhance the application's functionality, please create a GitHub issue and submit a pull request.

## Acknowledgments

We extend our gratitude to the open-source NLP community for developing powerful libraries and models that make this project possible.

---

With the "Comprehensive Understanding of World Leaders' Speeches" web application, we aim to enable policymakers, researchers, and the public to gain deeper insights into the speeches that shape our world. By combining the power of NLP and human understanding, we hope to contribute to a more informed and interconnected global community. Happy analyzing!
