# test_level.py
import pandas as pd
vse_df = pd.read_csv('data/processed/vse_syllabi_processed.csv')
print(vse_df[vse_df['course_code'] == '4IT537']['study_level'].values)