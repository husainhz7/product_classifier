
from config.core import config
from processing.data_manager import load_dataset, save_pipeline

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB

training_pipeline = Pipeline([
    ('cat', OneHotEncoder, ['main_category']),
    ('text', TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range = (1,3)), 'name'),
    ('classifier', MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None))
])

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[0],  # predictors
        data[1],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    training_pipeline.fit(X_train, y_train)

    y_pred = training_pipeline.predict(X_test)
    print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist= training_pipeline)
    # printing the score



if __name__ == "__main__":
    run_training()