"""
chat_with_data.py

End-to-end Talk-To-Data workflow.

User Question
      ↓
Gemini
      ↓
SQL
      ↓
SQLite
      ↓
Result
"""

from src.talk_to_data.nl_to_sql import NLToSQL
from src.talk_to_data.query_runner import QueryRunner


class DataChatAssistant:

    def __init__(self):

        self.sql_generator = NLToSQL()

        self.query_runner = QueryRunner()

    def ask(self, question):

        try:

            print("\nGenerating SQL...")

            sql_query = self.sql_generator.generate_sql(
                question
            )

            print("\nGenerated SQL:")
            print("-" * 50)
            print(sql_query)

            result = self.query_runner.run_query(
                sql_query
            )

            return {
                "question": question,
                "sql": sql_query,
                "result": result
            }

        except Exception as e:

            return {
                "question": question,
                "sql": None,
                "result": None,
                "error": str(e)
            }

    def close(self):

        self.query_runner.close()


def ask_question(question):

    assistant = DataChatAssistant()

    response = assistant.ask(question)

    assistant.close()

    return response


if __name__ == "__main__":

    assistant = DataChatAssistant()

    print("=" * 60)
    print("HOME CREDIT TALK-TO-DATA")
    print("=" * 60)

    while True:

        question = input(
            "\nAsk a question (type 'exit' to quit): "
        )

        if question.lower() == "exit":
            break

        response = assistant.ask(
            question
        )

        if response.get("error"):

            print(
                f"\nError: {response['error']}"
            )

            continue

        print("\nResult:")
        print("-" * 50)

        print(
            response["result"]
        )

    assistant.close()