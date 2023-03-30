from typing import Union

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from response_generator import generate_response
from pydantic import BaseModel


class Item(BaseModel):
    request_st: str

    
app = FastAPI()


@app.post("/get_response")
async def get_model_response(item: Item):
    """Fast API endpoint to get model response

    Parameters
    ----------
    request_st : str
        the user input to be feed into the chatbot model

    Returns
    -------
    JSONResponse object
        returns a json object with status and model response, or an error message
    """
    try:
        model_response = generate_response(item.request_st)
        return JSONResponse(status_code=200, content={'message': model_response})
    except Exception as e:
        return JSONResponse(status_code=500, content={'message': f'There was error {e}'})
        
