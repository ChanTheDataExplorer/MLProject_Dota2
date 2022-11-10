import numpy as np

import bentoml
from bentoml.io import JSON

from input_processor import input_processor

model_ref = bentoml.xgboost.get("dota2_predictor_model:khgulotaaw62vc2n")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("dota2_predictor_model", runners=[model_runner])


@svc.api(input=JSON(), output=JSON())
async def classify(application_data):
    processor_class = input_processor()
    new_data = processor_class.process_input(application_data)

    print(f'new_data {type(new_data)} \n {new_data} \n ')

    vector = dv.transform(new_data)
    prediction = await model_runner.predict.async_run(vector)

    print(f'prediction {type(prediction)} \n {prediction} \n ')

    result = prediction

    if result > 0.5:
        return {
            "status": "Radiant Win",
            "prob": result
        }
    else:
        return {
            "status": "Dire Win",
            "prob": result
        }