from fastapi import FastAPI
from pydantic import BaseModel
from ai_gateway import AIGateway

app = FastAPI()
gateway = AIGateway()

class CategorizeRequest(BaseModel):
    description: str
    amount: float = 0.0

class CategorizeResponse(BaseModel):
    category: str

@app.post("/api/categorize", response_model=CategorizeResponse)
async def categorize(req: CategorizeRequest):
    prompt = (
        f"You are a financial categorization assistant. "
        f"Given a transaction description, return a single word category "
        f"like Food, Transport, Airtime, Entertainment, Bills, Shopping, Health, etc.\n\n"
        f"Transaction: \"{req.description}\" Amount: KSh {req.amount}\n"
        f"Category:"
    )
    category = await gateway.query(prompt)
    category = category.strip().capitalize()
    if not category.isalpha():
        category = "Other"
    return CategorizeResponse(category=category)

class InsightRequest(BaseModel):
    monthlyIncome: float
    monthlyExpenses: float
    categoryBreakdown: dict[str, float]
    previousMonthIncome: float | None = None
    previousMonthExpenses: float | None = None

class InsightResponse(BaseModel):
    narrative: str

@app.post("/api/insights/money-story", response_model=InsightResponse)
async def money_story(req: InsightRequest):
    breakdown_str = ", ".join(f"{cat}: KSh {amt:.2f}" for cat, amt in req.categoryBreakdown.items())
    prompt = (
        "You are a personal finance assistant. Write a friendly, insightful summary of the user's monthly finances.\n"
        f"Income: KSh {req.monthlyIncome:.2f}\n"
        f"Expenses: KSh {req.monthlyExpenses:.2f}\n"
        f"Top spending categories: {breakdown_str}\n"
    )
    if req.previousMonthIncome and req.previousMonthExpenses:
        prompt += f"Last month income: KSh {req.previousMonthIncome:.2f}, expenses: KSh {req.previousMonthExpenses:.2f}\n"
    prompt += "Provide 3-4 sentences summary, mention if spending increased, and give one simple saving tip. Keep it positive and encouraging."
    narrative = await gateway.query(prompt)
    return InsightResponse(narrative=narrative.strip())