import minsearch
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models


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


def create_qdrant_client():
    client = QdrantClient("http://localhost:6333")

    return client


def create_collection(
    qdrant_client,
    collection_name,
    vector_size,
    payload_filter,
    distance_method=models.Distance.COSINE,
):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=distance_method),
    )

    qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name=payload_filter,
        field_schema="keyword",
    )


def insert_points(
    qdrant_client,
    documents,
    collection_name,
    embedding_model="jinaai/jina-embeddings-v2-small-en",
):
    points = []

    for index, document in enumerate(documents):
        text = document["question"] + " " + document["text"]
        vector = models.Document(text=text, model=embedding_model)
        point = models.PointStruct(id=index, vector=vector, payload=document)
        points.append(point)

    qdrant_client.upsert(collection_name=collection_name, points=points)


def query_points(
    qdrant_client,
    collection_name,
    question,
    query_filter_key,
    query_filter_value,
    embedding_model="jinaai/jina-embeddings-v2-small-en",
):
    queried_points = qdrant_client.query_points(
        collection_name=collection_name,
        query=models.Document(text=question, model=embedding_model),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=query_filter_key,
                    match=models.MatchValue(value=query_filter_value),
                )
            ]
        ),
    )

    results = []

    for point in queried_points.points:
        response = point.payload
        results.append(response)

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


def rag_with_custom_search_engine(question, course, faq_filename):
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
    answer = call_llm(prompt)

    return answer


def rag_with_vector_search(question, course, faq_filename):
    faq_docs = parse_documents(faq_filename)
    qdrant_client = create_qdrant_client()
    create_collection(
        qdrant_client=qdrant_client,
        collection_name=course,
        vector_size=512,
        payload_filter="course",
    )
    insert_points(
        qdrant_client=qdrant_client, documents=faq_docs, collection_name=course
    )
    search_results = query_points(
        qdrant_client=qdrant_client,
        collection_name=course,
        question=question,
        query_filter_key="course",
        query_filter_value=course,
    )
    prompt = generate_prompt(search_results, question)
    answer = call_llm(prompt)

    return answer


if __name__ == "__main__":
    question = "I just disovered the course. Can I still join it?"
    course_filter = "data-engineering-zoomcamp"
    faq_document = "documents.json"
    answer_with_customer_search = rag_with_custom_search_engine(
        question=question, course=course_filter, faq_filename=faq_document
    )
    print(answer_with_customer_search)
    answer_with_vector_db = rag_with_vector_search(
        question=question, course=course_filter, faq_filename=faq_document
    )
    print(answer_with_vector_db)
