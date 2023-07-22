# Pandas for handling the data
import pandas as pd

from src.pre_processing.config import Text_by_session_path, Speakers_by_session_path

df_Text_by_session = pd.read_csv(Text_by_session_path)
df_Speakers_by_session = pd.read_csv(Speakers_by_session_path)
