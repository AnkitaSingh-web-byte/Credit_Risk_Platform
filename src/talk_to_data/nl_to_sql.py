"""
nl_to_sql.py

Convert natural language questions
into SQL using Gemini.
"""

import os

from dotenv import load_dotenv
import google.generativeai as genai

from src.talk_to_data.prompt_templates import (
    DATABASE_SCHEMA,
    SQL_GENERATION_PROMPT
)

load_dotenv()


class NLToSQL:

    def __init__(self):

        print("=" * 50)
        print("ENV KEY =", os.getenv("GEMINI_API_KEY"))
        print("=" * 50)

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found"
            )

        genai.configure(
            api_key=api_key
        )

        self.model = genai.GenerativeModel(
            "gemini-1.5-flash"
        )

    def generate_sql(self, question):

        prompt = SQL_GENERATION_PROMPT.format(
            schema=DATABASE_SCHEMA,
            question=question
        )

        response = self.model.generate_content(
            prompt
        )

        sql = response.text.strip()

        sql = (
            sql.replace("```sql", "")
               .replace("```", "")
               .strip()
        )

        return sql


if __name__ == "__main__":

    converter = NLToSQL()

    question = (
        "Show average income of customers "
        "who defaulted"
    )

    sql = converter.generate_sql(
        question
    )

    print("\nQuestion:")
    print(question)

    print("\nGenerated SQL:")
    print(sql)