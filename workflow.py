
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation'

)
model=ChatHuggingFace(llm=llm)

class SentimentSchema(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='Sentiment of the review')
parser=PydanticOutputParser(pydantic_object=SentimentSchema)

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')
parser2=PydanticOutputParser(pydantic_object=DiagnosisSchema)

class ReviewState(TypedDict):
    review:str
    sentiment:Literal['positive','negative']
    response:str
    diagnosis:dict

def findSentiment(state:ReviewState):
    prompt=f'for the following review find out the sentiment \n {parser.get_format_instructions()} {state["review"]}'
    response=model.invoke(prompt)
    sentiment=parser.parse(response.content)
    return {'sentiment':sentiment.sentiment}

def check_sentiment(state:ReviewState):
    if state['sentiment']=='positive':
        return 'PositiveSentiment'
    else:
        return 'Diagnosis'

def run_diagnosis(state:ReviewState):
    prompt = f"""Diagnose this negative review: {state['review']} {parser2.get_format_instructions()}"
    "Return issue_type, tone, and urgency.
"""
    response=model.invoke(prompt)
    output=parser2.parse(response.content)
    return {'diagnosis':output.model_dump()}

def positive_response(state:ReviewState):
    prompt=f"""write a warm thankyou message in response to this review:\n 
    {state['review']} \n
    Also, kindly ask the user to leave the feedback on our website"""

    response= model.invoke(prompt).content
    return {'response':response}

def negative_response(state:ReviewState):
    diagnosis=state['diagnosis']
    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message."""

    response=model.invoke(prompt).content
    return {'response':response}

graph=StateGraph(ReviewState)

graph.add_node('FindSentiment',findSentiment)
graph.add_node('PositiveSentiment',positive_response)
graph.add_node('NegativeSentiment',negative_response)
graph.add_node('Diagnosis',run_diagnosis)

graph.add_edge(START,'FindSentiment')
graph.add_conditional_edges(
    'FindSentiment',
    check_sentiment,
    {
        'PositiveSentiment': 'PositiveSentiment',
        'Diagnosis': 'Diagnosis'
    }
)
graph.add_edge('PositiveSentiment',END)
graph.add_edge('Diagnosis','NegativeSentiment')
graph.add_edge('NegativeSentiment',END)
workflow=graph.compile()


intial_state={
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}
workflow.invoke(intial_state)