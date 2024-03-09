
from typing import Union
import pandas as pd

from processing.data_manager import load_pipeline, __version__
from config.core import config
from processing.data_manager import pre_pipeline_preparation

pipeline_file_name = f"{config.pipeline_save_file}{__version__}.pkl"
product_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data = pre_pipeline_preparation(data_frame=pd.DataFrame(input_data))
    #print(validated_data)
    results = {"predictions": None, "version": __version__, "errors": None}
    
    predictions = product_pipe.predict(validated_data)

    results = {"predictions": predictions[0],"version": __version__, "errors": None}
    print(results)

    return results

if __name__ == "__main__":

    data_in={'name':['iPhone']}
    
    make_prediction(input_data=data_in)