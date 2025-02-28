from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from chat.chat_manager import handle_chat
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Define the request model
class ChatRequest(BaseModel):
    query:str

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.post("/chat")
async def chat(request:ChatRequest):
    try:
        user_query = request.query  # Extract user input from JSON request
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required")

        response = handle_chat(user_query)
        if not isinstance(response, str):  # ✅ Ensure response is always a string
            print("⚠️ Warning: Response is not a string. Converting to string...")
            response = str(response)  # Convert response to string if needed

        print("✔️ Final Response:", response)  # ✅ Debugging output before returning

        return {"message":response}
    except Exception as e:
        return {"error":str(e)}

# Todo: to run the project use this command on terminal=>  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
if __name__ == "__main__":
    # main()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
