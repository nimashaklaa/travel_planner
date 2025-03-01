from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.tool_agent.feedback_agent import generate_updated_plan
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

# Define request models
class FeedbackRequest(BaseModel):
    original_plan: str

class UpdateRequest(BaseModel):
    original_plan: str
    feedback: str
    choice: int

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

@app.post("/get_update_options")
async def get_update_options(request: FeedbackRequest):
    """Returns update options to frontend."""
    return {
        "message": "What do you want to update?",
        "options": [
            "1. Transportation",
            "2. Accommodation",
            "3. Attractions",
            "4. Restaurants",
            "5. All of the above"
        ]
    }

@app.post("/update_plan")
async def update_plan(request: UpdateRequest):
    try:
        user_query = request.feedback
        original_plan = request.original_plan
        choice = request.choice

        print("passed data",user_query,original_plan,choice)

        # Generate updated plan
        updated_plan, scratchpad, actions_log = generate_updated_plan(user_query, original_plan, choice)

        return {"updated_plan": updated_plan}

    except Exception as e:
        return {"error": str(e)}

# Todo: to run the project use this command on terminal=>  uvicorn main:app --host 127.0.0.1 --port 8000 --reload
if __name__ == "__main__":
    # main()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
