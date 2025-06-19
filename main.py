from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from BaseRAG import answerWithRAG

query = input("Ask a question: ")
print()

currentDocs = set()

currentOutput = ""

load_dotenv()
llm = ChatOpenAI(model_name="gpt-4")

def backgroundText():
    t = ""

    if len(currentDocs) > 0:
        t += "\nHere is the current background info available: "
    else:
        t += "\nNo background info has been retrieved."
    for doc in currentDocs:
        t += doc + "\n"

    return t

def createRetrievePrompt():
    retrievePrompt = "Here is the users question: " + query
    retrievePrompt += backgroundText()
    retrievePrompt += "\nHere is the generated response so far: " + currentOutput + "\nBased on this, would new information from a document on Trump's second term be necessary to generate the next segment of the response? Return 'Yes' if new information needs to be retrieved, 'Continue' if it can keep going with the current background given, and 'No' if the next segment can be generated without any background"
    return retrievePrompt

def createGenerationPrompt():
    prompt = "Here is the user's question: " + query
    prompt += backgroundText()

    if currentOutput == "":
        prompt += "\nPlease generate the first segment of the response."
    else:
        prompt += "\nHere is the generated response so far: " + currentOutput + ". Please generate the next segment of the response."

    return prompt


retrievePrompt = createRetrievePrompt()
# retrieveToken = llm.invoke([HumanMessage(content=retrievePrompt)])
# if retrieveToken.content == "Yes":
#     _, rag = answerWithRAG()
# print(retrieveToken)
firstSeg = createGenerationPrompt()


print(firstSeg)

