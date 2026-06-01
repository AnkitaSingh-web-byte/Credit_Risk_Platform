"""
prompt_templates.py

Prompt templates used for converting
natural language into SQL queries.
"""

DATABASE_SCHEMA = """
Table: application_train

Important Columns:

SK_ID_CURR                -> Customer ID
TARGET                    -> Defaulted (1) / Repaid (0)

AMT_INCOME_TOTAL          -> Customer income
AMT_CREDIT                -> Credit amount
AMT_ANNUITY               -> Loan annuity
AMT_GOODS_PRICE           -> Goods price

CNT_CHILDREN              -> Number of children
CNT_FAM_MEMBERS           -> Family members

DAYS_BIRTH                -> Customer age (negative days)
DAYS_EMPLOYED             -> Employment days

NAME_CONTRACT_TYPE        -> Loan type
CODE_GENDER               -> Gender
FLAG_OWN_CAR              -> Owns car
FLAG_OWN_REALTY           -> Owns house

NAME_INCOME_TYPE          -> Income source
NAME_EDUCATION_TYPE       -> Education level
NAME_FAMILY_STATUS        -> Family status
NAME_HOUSING_TYPE         -> Housing type

OCCUPATION_TYPE           -> Occupation
ORGANIZATION_TYPE         -> Employer type
"""


SQL_GENERATION_PROMPT = """
You are an expert SQL analyst.

Your task is to convert a user's question into SQL.

Rules:
1. Use only columns from the schema.
2. Return ONLY SQL.
3. Do not explain anything.
4. Use application_train table.
5. Prefer SQLite compatible SQL.

Schema:
{schema}

User Question:
{question}
"""