import sqlite3
import pandas as pd
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import numpy as np

class TAGSystem:
    def __init__(self, db_path):
        self.db_connection = sqlite3.connect(db_path)
        self.llm = Ollama(model="tinyllama")
        self.embeddings = OllamaEmbeddings(model="nomic-text-embedding")

    def query(self, natural_language_query):
        parsed_query = self._parse_query(natural_language_query)
        table_data = self._retrieve_table_data(parsed_query)
        answer = self._generate_answer(parsed_query, table_data)
        return answer

    def _parse_query(self, query):
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Extract key information from this query: {query}"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(query=query)

    def _retrieve_table_data(self, parsed_query):
        query = "SELECT * FROM sales WHERE date >= date('now', '-3 months')"
        df = pd.read_sql_query(query, self.db_connection)
        return df.to_dict(orient='records')

    def _generate_answer(self, parsed_query, table_data):
        table_context = self._format_table_data(table_data)
        prompt = PromptTemplate(
            input_variables=["table_context", "parsed_query"],
            template="""
            Given the following table data:
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
tag_system = TAGSystem("path/to/your/database.db")
result = tag_system.query("Why did sales drop last quarter?")
print(result)