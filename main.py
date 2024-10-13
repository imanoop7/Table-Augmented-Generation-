import pandas as pd
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection parameters
db_params = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "root"
}

def get_db_connection():
    """
    Create and return a database connection using the specified parameters.
    """
    try:
        connection = psycopg2.connect(**db_params)
        connection.set_session(autocommit=True)
        return connection
    except (Exception, psycopg2.Error) as error:
        print(f"Error while connecting to PostgreSQL: {error}")
        return None

def execute_query(query, params=None):
    """
    Execute a SQL query and return the results as a list of dictionaries.
    """
    connection = get_db_connection()
    if connection is None:
        return None

    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            if cursor.description:
                return cursor.fetchall()
            return None
    except (Exception, psycopg2.Error) as error:
        print(f"Error executing query: {error}")
        return None
    finally:
        if connection:
            connection.close()

class TAGSystem:
    def __init__(self):
        self.db_connection = get_db_connection()
        self.llm = Ollama(model="tinyllama")
        self.embeddings = OllamaEmbeddings(model="nomic-text-embedding")

    def query(self, natural_language_query):
        parsed_query = self._parse_query(natural_language_query)
        table_data = self._retrieve_table_data(parsed_query)
        relevant_data = self._semantic_search(parsed_query, table_data)
        answer = self._generate_answer(parsed_query, relevant_data)
        return answer

    def _parse_query(self, query):
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Extract key information from this query: {query}"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(query=query)

    def _retrieve_table_data(self, parsed_query):
        query = "SELECT * FROM sales WHERE date >= CURRENT_DATE - INTERVAL '3 months'"
        return execute_query(query)

    def _generate_answer(self, parsed_query, relevant_data):
        table_context = self._format_table_data(relevant_data)
        prompt = PromptTemplate(
            input_variables=["table_context", "parsed_query"],
            template="""
            Given the following relevant table data:
            {table_context}

            Answer the question: {parsed_query}
            Provide a detailed explanation based on the data.
            """
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(table_context=table_context, parsed_query=parsed_query)

    def _format_table_data(self, table_data):
        return "\n".join([str(row) for row in table_data])

    def _semantic_search(self, query, table_data, top_k=5):
        query_embedding = self.embeddings.embed_query(query)
        table_embeddings = self.embeddings.embed_documents([str(row) for row in table_data])
        similarities = [np.dot(query_embedding, emb) for emb in table_embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [table_data[i] for i in top_indices]

# Usage
tag_system = TAGSystem()
result = tag_system.query("Why did sales drop last quarter?")
print(result)
