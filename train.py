
from config.core import config
from processing.data_manager import load_dataset, save_pipeline

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


training_pipeline = Pipeline([
    ('preprocessing',  TfidfVectorizer(use_idf=True, ngram_range=(1, 3))),
    ('classifier', MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None))
])

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.name,  # predictors
        data.main_category,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    # Pipeline fitting
    training_pipeline.fit(X_train, y_train)

    y_pred = training_pipeline.predict(X_test)
    print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist= training_pipeline)

if __name__ == "__main__":
    run_training()