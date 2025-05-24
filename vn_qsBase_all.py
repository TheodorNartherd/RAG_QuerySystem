from vanna.qdrant import Qdrant_VectorStore
from vanna.openai import OpenAI_Chat

from vanna.types import TrainingPlan
from dotenv import load_dotenv
import os

"""
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM
"""
import sqlite3

load_dotenv('.env')

class VN_QsBase(Qdrant_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        Qdrant_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

    """
        oaClient = openai.OpenAI(api_key=config.get("api_key", None))
        llm = OpenAI(oaClient)
        self.kw_model = KeyLLM(llm)
    """

    # Prompt-Modul
    ## Table Representation
    def convert_ddlToSchema(self,ddl:str) -> str:
        chrList = ['"',"'","`", '\t']
        for chr in chrList:
            ddl = ddl.replace(chr,' ')

        code_lines = ddl.split('\n')
        schema = code_lines[0].strip().removeprefix('CREATE TABLE')
        code_lines = code_lines[1:]
        for cl in code_lines:
            if cl.startswith(','):
                cl = cl.replace(',', '')
            cl = cl.strip()
            if cl.upper().startswith('PRIMARY KEY'):
                continue
            if ('FOREIGN KEY' in cl.upper()  or len(cl)==1): 
                schema +=cl.replace('( ', '(').replace(' )', ')') + ' '
            else:
                firstWord = cl.split(' ')[0]
                schema += firstWord + ', '

        if not(schema.strip().endswith(')')):
            schema = schema + ')'
        result = ')'.join(schema.rsplit(', )', 1)).strip()
    
        return result
    
    def get_exampleValues(self,tbl_name,db_name) -> str:
        self.connect_to_sqlite(os.getenv('dbLoc') + '/' + db_name + '/' + db_name + '.sqlite')
        sql = 'SELECT * FROM ' + tbl_name + ' ORDER BY RANDOM() LIMIT 2'
        example_dict = self.run_sql(sql).to_dict('list')
        return 'Value-Examples: ' + str(example_dict).replace('{', '').replace('}', '')

    def add_ddl(self, ddl: str, **kwargs) -> str:
        schema = self.convert_ddlToSchema(ddl)
        exampleValues = self.get_exampleValues(kwargs.get('tbl_name'),kwargs.get('db_name'))
        schema += ' \n' + exampleValues
        return super().add_ddl(schema)
    
    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
        **kwargs
    ) -> str:
        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(ddl,**kwargs)
        else: 
            super().train(question,sql,documentation,plan)

    
    ## Prompt Representation
    #   Override of VannaBase. Newly implemented according to (Gao et al., 2023)
    def get_sql_prompt(
        self,
        initial_prompt : str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if initial_prompt is None:
            initial_prompt = f"Complete {self.dialect} SQL query only and with no explanation"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        if self.static_documentation != "":
            doc_list.append(self.static_documentation)

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]

        if len(question_sql_list) > 0:
            message_log.append(self.system_message('Some example questions and corresponding SQL queries are provided based on similar problems:'))

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.system_message('Answer the following:'))
        message_log.append(self.user_message(question))

        return message_log
    
    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += '\n' + f"{self.dialect} SQL tables, with their properties:\n\n"

            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n"

        return initial_prompt
    """
    # Schema-Linking-Modul
    def get_related_ddl(self, question: str, **kwargs) -> list:
        keywordList = self.get_keywords(question)
        ddl_list = []
        for keyword in keywordList:
            results = self._client.query_points(
            self.ddl_collection_name,
            query=self.generate_embedding(keyword),
            limit=2,
            with_payload=True,
            ).points
            result_list = [result.payload["ddl"] for result in results]
            ddl_list.extend(result_list)

        return list(set(ddl_list))

    def get_keywords(self,question):
        keywords = self.kw_model.extract_keywords(question)
        res_list = [x[0] for x in keywords]
        return res_list
    """
    
    # Correction-Modul
    def generateAndCorrect_sql(self, question: str, **kwargs) -> str:
        sql = self.generate_sql(question, **kwargs)
        db_id = kwargs.get('db_id')
        db_path = os.getenv('dbLoc') + '/' + db_id + '/' + db_id + '.sqlite'
        executable, message = self.check_sql(sql,db_path)
        if executable:
            return sql
        else:
            self.log(title="SQL Correction needed", message=message)
            sqlCorrected = self.correct_sql(question, sql, message)
            return sqlCorrected

    def check_sql(self,predicted_sql,db_path):
        conn = sqlite3.connect(db_path)
        conn.text_factory = bytes
        cursor = conn.cursor()
        try:
            cursor.execute(predicted_sql)
        except Exception as e:
            return False, str(e)
    
        predicted_res = cursor.fetchall()
        if len(predicted_res) > 0:
            return True, None
        else:
            return False, "sql returns no value"
        
    def correct_sql(self, question, sql, message) -> str:
        correction_prompt = self.get_correction_prompt(question, sql, message)
        self.log(title="Correction Prompt", message=correction_prompt)
        corrected_llm_response = self.submit_prompt(correction_prompt)
        corrected_sql = self.extract_keyDict(corrected_llm_response)["corrected_SQL"]
        return self.extract_sql(corrected_sql)
    
    def get_correction_prompt(self, question, sql, message) -> str:
        initial_prompt = f"You are a {self.dialect} expert. There is a SQL query generated based on the following Database Schema to respond to the Question. Executing this SQL has resulted in an error and you need to fix it based on the error message. \n" #Return the corrected SQL only and with no explanation.
        
        ddl_list = self.get_related_ddl(question)
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt += '\n' + f"Question:\n" + question + ' \n'
        initial_prompt += '\n' + f"Executed SQL:\n" + sql + ' \n'
        initial_prompt += '\n' + f"Error Message:\n" + message + ' \n'

        initial_prompt += '\n Please respond with a Python-Dictionary storing the two keys "chain_of_thought_reasoning" and "corrected_SQL" written as a one-liner. \n'


        message_log = [self.system_message(initial_prompt)]

        return message_log
    
    def extract_keyDict(self,llm_response):
        start = "{"
        end = "}"

        # Find the index of the start substring
        idx1 = llm_response.find(start)

        # Find the index of the end substring, starting after the start substring
        idx2 = llm_response.find(end, idx1 + len(start))

        # Check if both delimiters are found and extract the substring between them
        if idx1 != -1 and idx2 != -1:
            res = llm_response[idx1 + len(start):idx2]
            return eval('{' + res + '}')

        else:
            print("Delimiters not found")
