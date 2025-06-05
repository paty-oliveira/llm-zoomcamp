import minsearch
import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def parse_documents(filename):
    documents = []

    with open(filename, "rt") as file:
        raw_docs = json.load(file)

    for item in raw_docs:
        for document in item["documents"]:
            document["course"] = item["course"]
            documents.append(document)

    return documents


def build_search_engine(documents, text_fields=[], keyword_fields=[]):
    engine = minsearch.Index(text_fields=text_fields, keyword_fields=keyword_fields)
    engine.fit(documents)

    return engine


def search(engine, question, filter={}, boost={}, num_results=5):
    results = engine.search(
        query=question, filter_dict=filter, boost_dict=boost, num_results=num_results
    )

    return results


def generate_prompt(search_results, question):
    prompt_template = """
        You are a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT:
        {context}
    """.strip()

    context = ""

    for result in search_results:
        context += f"section: {result['section']}\nquestion: {result['question']}\nanswer: {result['text']}\n\n"

    return prompt_template.format(question=question, context=context).strip()


def call_llm(prompt, model="gpt-4o"):
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def rag(question, course, faq_filename):
    faq_docs = parse_documents(faq_filename)
    search_engine = build_search_engine(
        faq_docs, text_fields=["question", "text", "section"], keyword_fields=["course"]
    )
    search_results = search(
        engine=search_engine,
        question=question,
        filter={"course": course},
        boost={"question": 3.0, "section": 0.5},
    )
    prompt = generate_prompt(search_results, question)
    anwser = call_llm(prompt)

    return anwser


if __name__ == "__main__":
    question = "I just disovered the course. Can I still join it?"
    course_filter = "data-engineering-zoomcamp"
    faq_document = "documents.json"
    anwser = rag(question=question, course=course_filter, faq_filename=faq_document)
    print(anwser)
