import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from BaseRAG import answerWithRAG

load_dotenv()

try:
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    RAG, _ = answerWithRAG()
except Exception as e:
    print(e)
    exit()


def createRAGPrompt(question, document_chunk, generated_so_far):
    prompt = ("You are an advanced AI assistant. Your task is to continue generating an answer to a user's query based on a provided document chunk, and then reflect on your own generation."
              "\nPlease perform the following two tasks:"
              "\nTask 1: Generate the next segment of the answer. Read the original query, the provided document chunk, and any text that has already been generated. Generate the *next* single, coherent segment of the answer. A segment should be one or two sentences."
              "\nTask 2: Provide reflection tokens in JSON format. After generating the segment, you must provide a series of 'reflection tokens' that critique both the provided document chunk and the segment you just wrote.")

    prompt += "\nUser Query: " + question

    if generated_so_far == "":
        prompt += "\nYou will write the first segment of the answer."
    else:
        prompt += "\nPrevious Generation: " + generated_so_far

    prompt += "\nProvided Chunk: " + document_chunk

    prompt += "\nYou must provide your response as a single, valid JSON object and nothing else. Crucially, all keys and string values in the JSON object must be enclosed in double quotes. Do not add any text or explanation before or after the JSON object. The JSON object must have the following structure:"
    prompt += "\n{{   'relevance': '...',   'support': '...',   'utility': '...',   'continue_decision': '...',   'generated_segment': '...' }}"
    prompt += "\n'relevance': Is the provided document chunk relevant to the original user query? Values: 'yes', 'no"
    prompt += "\n'support': Is the generated_segment you wrote fully supported by the provided document chunk? Values: 'fully supported', 'partially supported', 'no support'"
    prompt += "\n'utility': How useful is the generated_segment for answering the Original User Query? Values: 'very useful', 'somewhat useful', 'not useful'"
    prompt += "\n'continue_decision': Should there be more semgents after this? Values: 'yes', 'no'"
    prompt += "\n'generated_segment': Generated segment you just wrote for task 1."
    return prompt

def createNoRAGPrompt(question, generated_so_far):
    prompt = (
        "You are an advanced AI assistant. Your task is to continue generating an answer to a user's query, and then reflect on your own generation."
        "\nPlease perform the following two tasks:"
        "\nTask 1: Generate the next segment of the answer. Read the original query and any text that has already been generated. Generate the next single, coherent segment of the answer. A segment should be one or two sentences."
        "\nTask 2: Provide reflection tokens in JSON format. After generating the segment, you must provide a series of 'reflection tokens' that critique the segment you just wrote.")

    prompt += "\nUser Query: " + question

    if generated_so_far == "":
        prompt += "\nYou will write the first segment of the answer."
    else:
        prompt += "\nPrevious Generation: " + generated_so_far

    prompt += "\nYou must provide your response as a single, valid JSON object and nothing else. Crucially, all keys and string values in the JSON object must be enclosed in double quotes. Do not add any text or explanation before or after the JSON object. The JSON object must have the following structure:"
    prompt += "\n{{'utility': '...',   'continue_decision': '...',   'generated_segment': '...' }}"
    prompt += "\n'utility': How useful is the generated_segment for answering the Original User Query? Values: 'very useful', 'somewhat useful', 'not useful'"
    prompt += "\n'continue_decision': Should there be more semgents after this? Values: 'yes', 'no'"
    prompt += "\n'generated_segment': Generated segment you just wrote for task 1."
    return prompt

def createRetrievalPrompt(question, chunks, generated_so_far):
    prompt = "You're a helpful AI assistant that operates as a RAG system for Trump's second term, which he served. Your policy is to retrieve documents for any query that requires specific, factual information on Trump's second term. For creative queries or ones unrelated to Trump's second term, you can answer from your own knowledge.\nHere is the user's query: " + question

    if len(chunks) == 0:
        prompt += "\nNo chunks have been retrieved so far"
    else:
        prompt += "\nHere are the chunks that've been retrieved so far:"
        for chunk in chunks:
            prompt += "\n" + chunk

    if generated_so_far == "":
        prompt += "\nThe answer hasn't been generated yet."
    else:
        prompt += "\nHere's what's been generated so far: " + generated_so_far

    prompt += "\nBased on your policy as a RAG system, what is your first action? If you think it'd help to retrieve more chunks, respond 'yes'. If you think the next segment can be generated with the current chunks, respond 'continue'. If you think no external information is needed for the next segment, respond 'no'"

    return prompt

def selectBestCandidate(candidates):
    score_map = {
        "relevance": {"yes": 1, "no": 0},
        "support": {"fully supported": 2, "partially supported": 1, "no support": 0},
        "utility": {"very useful": 2, "somewhat useful": 1, "not useful": 0}
    }
    highest_score = 0
    best_candidate = None

    for i, candidate in enumerate(candidates):
        if not candidate:
            continue

        rel_score = score_map["relevance"].get(candidate.get("relevance"), 0)
        sup_score = score_map["support"].get(candidate.get("support"), 0)
        util_score = score_map["utility"].get(candidate.get("utility"), 0)

        if rel_score == 0 or sup_score == 0:
            total_score = 0
        else:
            total_score = rel_score + sup_score + util_score

        # print(
        #     f"Candidate {i + 1}: Score={total_score}, Support='{candidate.get('support')}', Utility='{candidate.get('utility')}'")

        if total_score > highest_score:
            highest_score = total_score
            best_candidate = candidate

    # Ensure we only return a high-quality candidate
    # This threshold means it must be at least relevant, somewhat supported, and somewhat useful
    # if highest_score > 3:
    #     print(f"--- Selected Candidate with score: {highest_score} ---")
    #     return best_candidate
    # else:
    #     print("--- No high-quality candidate found. Ending generation. ---")
    #     return None
    return best_candidate

def answerQuestion(question):
    responseSoFar = ""
    currDocs = set()
    while True:
        retrievePrompt = createRetrievalPrompt(question, currDocs, responseSoFar)
        print("Retrieve Prompt: " + retrievePrompt)
        retrieveResponse = llm.invoke(retrievePrompt)
        print("Retrieve Token: " + retrieveResponse.content)
        if retrieveResponse.content == "yes":
            retrievedDocs = RAG.invoke(question  + " Current Answer: " + responseSoFar)
            for doc in retrievedDocs:
                currDocs.add(doc.page_content)

        best = ""

        if retrieveResponse.content == "no":
            prompt_text = createNoRAGPrompt(question, responseSoFar)
            raw_response = llm.invoke(prompt_text)
            raw_content = raw_response.content
            if raw_content.startswith("{{") and raw_content.endswith("}}"):
                raw_content = raw_content[1:-1]
            print(raw_content)

            best = json.loads(raw_content)
        else:
            candidateResponses = []

            for doc in currDocs:
                prompt_text = createRAGPrompt(question, doc, responseSoFar)
                try:
                    raw_response = llm.invoke(prompt_text)
                    raw_content = raw_response.content
                    if raw_content.startswith("{{") and raw_content.endswith("}}"):
                        raw_content = raw_content[1:-1]

                    json_response = json.loads(raw_content)
                    candidateResponses.append(json_response)

                except (json.JSONDecodeError, TypeError) as e:
                    print("Error")
                    print(raw_response)
                    print(prompt_text)
                    candidateResponses.append(None)

            best = selectBestCandidate(candidateResponses)
        if best:
            responseSoFar = responseSoFar + best.get("generated_segment")
            if best.get("continue_decision") == "no":
                return responseSoFar
        else:
            return responseSoFar

print(answerQuestion("What executive orders did Trump sign in his second term?"))
