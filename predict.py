
from typing import Union
import pandas as pd
import numpy as np

from processing.data_manager import load_pipeline, __version__
from config.core import config
from processing.data_manager import pre_pipeline_preparation

pipeline_file_name = f"{config.pipeline_save_file}{__version__}.pkl"
titanic_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    print(pd.DataFrame(input_data))
    validated_data = pre_pipeline_preparation(data_frame=pd.DataFrame(input_data))
    validated_data=validated_data.reindex(columns='name')
    #print(validated_data)
    results = {"predictions": None, "version": __version__}
    
    predictions = titanic_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": __version__}
    print(results)

    return results

if __name__ == "__main__":

    data_in={'name':["iPhone"]}
    
    make_prediction(input_data=data_in)