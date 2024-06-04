from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from Scraping.LLMsProcessing import retrieve_answers, gpt3_chain, gpt4_chain
import json

app = FastAPI()

app.mount("/Scripts", StaticFiles(directory="Scripts"), name="Scripts")
app.mount("/Style", StaticFiles(directory="Style"), name="Style")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="Template")

@app.post('/process_input')
def process_input(user_input: str = Form(...)):
    print("Received user input:", user_input)
    gpt3_answer = retrieve_answers(user_input, gpt3_chain)
    gpt4_answer = retrieve_answers(user_input, gpt4_chain)
    return {"gpt3_answer": gpt3_answer['result'], "gpt4_answer": gpt4_answer['result']}


@app.get("/", response_class=HTMLResponse)
async def chatbot(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class ConnectionManager:
    active_connections: List[WebSocket] = []

    @classmethod
    async def connect(cls, websocket: WebSocket):
        await websocket.accept()
        cls.active_connections.append(websocket)

    @classmethod
    def disconnect(cls, websocket: WebSocket):
        cls.active_connections.remove(websocket)

    @classmethod
    async def send_personal_message(cls, message: str, websocket: WebSocket):
        await websocket.send_text(message)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ConnectionManager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received WebSocket message: {data}")
            gpt3_answer = retrieve_answers(data, gpt3_chain)
            gpt4_answer = retrieve_answers(data, gpt4_chain)
            response = {
                "gpt3_answer": gpt3_answer['result'],
                "gpt4_answer": gpt4_answer['result']
            }
            # Convert response to JSON before sending
            await ConnectionManager.send_personal_message(json.dumps(response), websocket)
    except WebSocketDisconnect:
        ConnectionManager.disconnect(websocket)

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
