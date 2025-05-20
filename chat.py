import os
import json
from dotenv import load_dotenv
#from langchain_openai import ChatOpenAI
#from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types
from google.genai.types import Tool, GoogleSearch
from textwrap import dedent
#from langchain.agents import AgentExecutor, create_tool_calling_agent
#from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Dict, Any #, Optional, Union
import requests
#from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel #, ValidationError
#from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import logging
import sys  # Added for console logging alongside file logging
from time import strftime, gmtime, sleep
import concurrent.futures
import re
#from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Depends #, Request
from fastapi.security.api_key import APIKeyHeader
# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File Handler
file_handler = logging.FileHandler("api_chat.log")
file_handler.setFormatter(log_formatter)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False  # Prevent duplicate logs if root logger also has handlers
from utils_altair import sanitize_filename,image_file_to_base64,build_chart_from_synthesized_answer,chart_oppourtunity_validation

# Load environment
load_dotenv("prod.env")
SQL_API_ENDPOINT = "http://35.200.156.175:8074/query"
SQL_BATCH_API_ENDPOINT = "http://35.200.156.175:8075/batch_query"
VECTOR_API_ENDPOINT = "http://35.200.156.175:8787/search-topN"
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
# API Key and Security
API_KEY = os.getenv("ACQ_API_KEY")
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


# API Key verification dependency
async def verify_api_key(api_key: str = Depends(api_key_header)):
    logging.info(f"Received API Key: {api_key[:4]}****")  # Mask API key for security
    if api_key !=API_KEY :
        logging.warning(f"Unauthorized API access attempt with key: {api_key[:4]}****")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

def vector_data_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""Consider the query provided below. Your goal is to decorate the query and rewrite it in a form that can be used for searching through unstructured data. You must limit yourself to a maximum of 20 words.

            Here is some brief context about aspects of the Indian economy:
                1. Agriculture: Contributes about 15-20% to GDP, agriculture employs nearly half of the workforce. Monsoon patterns significantly impact agricultural output and rural demand.
                2. Industry: Manufacturing and construction are crucial for GDP growth and employment. Government initiatives like "Make in India" aim to boost manufacturing.
                3. Services: The services sector, including IT, finance, and telecommunications, contributes over 50% to GDP. IT services, in particular, are a major export and growth driver.
                4. Government Policies: Fiscal policies, such as taxation and public spending, influence economic growth. Monetary policies by the Reserve Bank of India (RBI) manage inflation and interest rates.
                5. Inflation: Influenced by food prices, fuel costs, and global commodity prices. The RBI uses repo rates and cash reserve ratios to control inflation.
                6. Foreign Direct Investment (FDI): FDI inflows boost infrastructure, technology, and job creation. Government policies aim to attract FDI in sectors like manufacturing and services.
                7. Global Economic Conditions: Exports, remittances, and foreign investments are affected by global demand and economic stability.
                8. Demographics: A young and growing workforce can drive economic growth, but requires adequate education and employment opportunities.
                9. Infrastructure: Investments in transportation, energy, and digital infrastructure enhance productivity and economic growth.
                10. Technological Advancements: Innovation and digitalization improve efficiency and competitiveness across sectors.

            INSTRUCTIONS:
                1. Keep in mind that you are providing annotation for queries related to the Indian economy. Focus on topics such as CPI (Consumer Price Inflation), GDP (Gross Domestic Product), IIP (Industrial Production) and others.
                2. Make sure you include all the key entities from the provided query.
                3. Make sure any commodities in the query are retained in your rephrased query.
                4. Make sure any events (festivals, pandemics, etc.) are retained in the rephrased query.
                5. If specific dates are mentioned, retain them in the rephrased query. If no dates are given, then and only then use {curdate} as a reference.
                    EXAMPLES:
                        1. "GDP from May 2019 to May 2024" -> "GDP from May 2019 to May 2024"
                        2. "Impact of GDP on IIP from May 2019 to May 2024" -> "Impact of Gross Domestic Product on Industrial Production between May 2019 and May 2024"
                        3. "Impact of COVID on IIP" -> "Industrial Production data and commentary from Dec 2019 to Dec 2020"
            """),
            temperature=0.0,
            ),
        contents=query
    )
    logger.info("Vector retrieval query: " + response.text)
    sleep(0.5)
    return response.text

def structured_data_queries(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent("""Consider the query provided below. Your goal is to decorate the query and rewrite it in a form that can be used for searching through structured data. You must limit yourself to a maximum of 15 words for each query, and a STRICT MAXIMUM of 3 sub-queries.
            INSTRUCTIONS:
                1. Keep in mind that you are generating data retrieval requests related to the Indian economy. Focus on topics such as CPI (Consumer Price Inflation), GDP (Gross Domestic Product), IIP (Industrial Production) and others. 
                Important: If the query has nothing to do with the economy (for example, queries about general topics such as the weather or sport), then DO NOT force fit the query to the economy. 
                Briefly, here is some sample information contained in the available data. THIS IS NOT AN EXHAUSTIVE LIST, but is indicative.
                    a. CPI inflation: All India as well as different states, monthly data, combined/urban/rural sectors, different commodity classes such as food, clothing, etc. Also available is wholesale price information for different commodities and housing price inflation for different cities in India.
                    b. GDP: National numbers of gross domestic product, national income, etc. Also available are numbers for state level production, services, industry sectors, import/export, and expenditure. Most GDP numbers are either annual or quarterly.
                    c. IIP industrial production: monthly or annual numbers for different sectors of Indian industry.
                2. Make sure you include all the key entities in at least one query.
                3. Make sure any commodities in the query are retained in your rephrased query.
                4. Query date range instructions:
                    a. If a specific date range is mentioned in the received query (e.g. May 2017 to May 2021), then repeat the same in a single sub-query (May 2017 to May 2021).
                    b. If no dates are mentioned in the query, then restrict the query to approximately one year.
                    c. If the impact of specific events is mentioned, generate subqueries for a few months before and after the event.
                5. IMPORTANT: Query size limiting instructions:
                    a. If specific states in India are not mentioned, mention that the sub-query is for ALL INDIA.
                    b. If specific sub-categories of commodities are not mentioned, state that the sub-query is at GENERAL category level.
                6. Importantly, AVOID generating queries that require all fields in all columns to be queried (for example, all states and all categories).
                7. Your TASK is to ensure that each sub-query is compact and easy to retrieve for a text2sql agent, and should ideally not result in more than 20-30 rows of data.
                8. IMPORTANT: If the query asks for GROWTH, RATE OF CHANGE, or COMPARISON (like growth of GDP in 2024 or comparison between two years), resolve the query to a LOWER GRANULARITY if available. For example:
                 - If a query asks for **GDP growth in 2024**, rephrase it to fetch **quarterly GDP data for 2023 and 2024**.
                 - If a query asks for **comparison of inflation between two years**, resolve it to **monthly or quarterly inflation rates** if available.
                 - Mention explicitly in the rephrased query that **"quarter-level data" or "month-level data" should be used for more accurate comparison**.

            EXTREMELY IMPORTANT: 
                1. Your output format should be a LIST OF STRINGS such as ["Sub-Query 1","Sub-Query 2","Sub-Query 3"], with each string containing one sub-query. 
                2. DO NOT output more than 3 sub-queries and stick to the word limit of 15 words per sub-query. DO NOT user more sub-queries than necessary.
                3. DO NOT create multiple sub-queries asking for the same data over continuous date ranges. Create single, compact sub-queries for the full date range.
                
            EXAMPLES:
                1. "GDP from May 2019 to May 2024" -> ["GDP from May 2019 to May 2024"]
                2. "Impact of GDP on IIP from May 2019 to May 2024" -> ["GDP from May 2019 to May 2024", "IIP from May 2019 to May 2024"]
                3. "Impact of COVID on IIP" -> ["IIP from Dec 2019 to Dec 2020"]
                4. "CPI for all years" -> ["CPI for last 5 years, all India, general category"]
                5. "CPI for Telangana in the last two years" -> ["CPI for Telangana, general category, last two years"]
                6. "CPI for footwear in the last 5 years" -> ["CPI for footwear, all India, last 5 years"]
                7. "Which industrial sector had the highest growth in 2023" -> ["IIP for all sectors in 2022 and 2023"]
                8. "Latest commentary and expert opinions on GDP" -> ["GDP in the last year"]
            """),
            temperature=0.0,
            ),
        contents=query
    )
    logger.info("SQL retrieval queries: ")
    logger.info(response.text)
    sleep(0.5)
    return response.text

def clarify_query(user_query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""You are tasked with rephrasing the given query to make it easier for an SQL agent to pull the right data. Generate a rephrased natural language version of the query keeping in mind the following steps:
            1. Analyze the provided query for key entities. For example, if the query is about inflation in vegetables which is a type of food, then include the phrase "inflation in vegetables which is a type of food".
            2. Analyze the query for the existence of dates or time-related phrases.
                a. For example, if the query asks about GDP growth since December 2024, rephrase it as "GDP growth from [December 2024] to [{curdate}]".
                b. For example, if the query asks about IIP in the last six months, compute the date six months BEFORE [{curdate}] and add this information to the query.
                c. If the query asks about the last quarter, compute the last full quarter BEFORE [{curdate}] and add this information to the query.
                d. If the query asks about vague timelines such as "long term" without specifying dates, use the date five years ago from [{curdate}].
                e. If the query contains an event without a date (for example, "COVID" or "51st meeting" or "the last world cup"), then use the google_search_tool to attach a date to the event.
                f. If the query asks for the impact of a specific event on certain quantities, use a date range of at most two years.
            3. IMPORTANT: Analyze the provided query for the existence of multiple entities. For example, if the query asks about "contribution of Maharashtra to total GDP", rewrite it as "GDP of Maharashtra and GDP of India".
            4. Wherever applicable, list exact sub-categories of products, or states within India. If no particular products, states, or other sub-categories are mentioned, include this information explicitly using phrases such as "for all of India" or "for food at a category level". If no specific sub-categories or states are required, mention "use * for pulling data only at category level" or "use * for pulling data at the country level" in the query.
            4. Always remember that all queries are related to India. If the word "India" is not mentioned in the query, include this in the rephrased query.
            5. IMPORTANT: If there is a comparison in the query between two values, mention both separately in the rephrased query. For example, "IIP of A compared to B" should be written as "We want the ratio of IIP of A, to the IIP of B" in the rephrased query.
            6. IMPORTANT: If the query contains a statistical enquiry such as "average", "mean", "median", then mention this metric prominently in the rephrased query as "AVERAGE", "MEAN", "MEDIAN", etc. as applicable.
            7. Do not include your reasoning trace in the rephrased query. Your output should contain only the information that is required from the SQL database.
            8. Handle Comparisons and Superlatives (MANDATORY):  If the query uses words like "highest", "lowest", "maximum", "minimum", or contains any COMPARISON: DO NOT remove or ignore the comparison. Instead, request data for **relevant entity** and explicitly state that it is to find the "highest", "lowest", etc. Example: "Which industry had the highest GVA growth?"
            """),
            tools=[google_search_tool],
            temperature=0.01,
            ),
        contents=user_query
    )
    logger.info("Original query: " + user_query + "\n")
    logger.info("Annotated query: " + response.text + "\n")
    vector_q    = vector_data_query(response.text)
    sql_q       = structured_data_queries(response.text)
    sleep(0.5)
    return vector_q, sql_q

# API call functions
def call_sql_api(question: str) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "access_token": "api-12345"}
    params = {"user_query": question}
    try:
        response = requests.get(SQL_API_ENDPOINT, headers=headers, params=params, timeout=300)
        response.raise_for_status()
        try:
            data = response.json().get("result", [])
            if isinstance(data, dict) and data.get("status") == "error":
                return {"status": "error", "message": data.get("message", "API returned an error status."), "data": data}
            return {"status": "success", "data": data}
        except json.JSONDecodeError:
            error_message = f"API returned non-JSON response: {response.text[:500]}"
            logger.error(f"JSONDecodeError for question '{question}'. Response text: {response.text}")
            return {"status": "error", "message": error_message, "data": response.text}
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException calling SQL API for question '{question}': {e}")
        return {"status": "error", "message": f"Request failed: {str(e)}", "data": None}
    except Exception as e:
        logger.error(f"Unexpected error calling SQL API for question '{question}': {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}", "data": None}

def call_sql_batch_api(question: List[str]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "access_token": "api-12345"}
    data = {
    "queries": question
    }
    #params = {"queries": question}
    try:
        response = requests.post(SQL_BATCH_API_ENDPOINT, headers=headers, data=json.dumps(data), timeout=300)
        response.raise_for_status()
        try:
            response = response.json()
            data = response.get("responses", {})
            if isinstance(response, dict) and response.get("successful") == 0:
                #return {"status": "error", "message": data.get("message", "API returned an error status."), "data": data}
                return {"status": "error", "message": "We did not find any relevant information in structured data.", "data": data}
            return {"status": "success", "data": data}
        except json.JSONDecodeError:
            error_message = f"API returned non-JSON response: {response.text[:500]}"
            logger.error(f"JSONDecodeError for question '{question}'. Response text: {response.text}")
            return {"status": "error", "message": error_message, "data": response.text}
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException calling SQL API for question '{question}': {e}")
        return {"status": "error", "message": f"Request failed: {str(e)}", "data": None}
    except Exception as e:
        logger.error(f"Unexpected error calling SQL API for question '{question}': {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}", "data": None}

def call_vector_search_api(question: str) -> Dict[str, Any]:
    question = question[0].strip()
    headers = {"Content-Type": "application/json", "access_token": "api-12345"}
    payload = {"question": question}

    try:
        response = requests.post(VECTOR_API_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response = response.json()
        retrieved_data = response.get("retrieved_results", {})

        results = []
        for d in retrieved_data:
            ref = d.get("reference", "")
            url = None

            # RBI/MOSPI Reference Pattern Routing
            if re.match(r"Inflation Expectations Survey of Households \w+ \d{4}", ref):
                url = "https://website.rbi.org.in/web/rbi/statistics/survey?category=24927098&categoryName=Inflation%20Expectations%20Survey%20of%20House-holds%20-%20Bi-monthly"
            elif re.match(r"Monetary Policy Report \w+ \d{4}", ref):
                url = "https://website.rbi.org.in/web/rbi/publications/articles?category=24927873"
            elif re.match(r"Minutes of the Monetary Policy Committee Meeting \w+ \d{4}", ref):
                url = "https://website.rbi.org.in/web/rbi/press-releases?q=%22Minutes+of+the+Monetary+Policy+Committee+Meeting%22"
            elif re.match(r"CPI Press Release \w+ \d{4}", ref):
                url = "https://www.mospi.gov.in/archive/press-release?field_press_release_category_tid=120&date_filter%5Bmin%5D%5Bdate%5D=&date_filter%5Bmax%5D%5Bdate%5D="
            elif re.match(r"Economic Survey \d{4} ?- ?\d{4}", ref):
                url = "https://www.indiabudget.gov.in/economicsurvey/allpes.php"
            elif re.match(r"IIP Press Release \w+ \d{4}", ref):
                url = "https://www.mospi.gov.in/archive/press-release?field_press_release_category_tid=121&date_filter%5Bmin%5D%5Bdate%5D=&date_filter%5Bmax%5D%5Bdate%5D="
            elif re.match(r"Monthly Economic Report \w+ \d{4}", ref):
                url = "https://dea.gov.in/monthly-economic-report-table"
            elif re.match(r"IIP Manual \w+ \d{4}", ref):
                url = "https://www.mospi.gov.in/sites/default/files/publication_reports/manual_iip_23oct08_0.pdf"

            results.append({
                "content": d.get("content"),
                "reference": ref,
                "cross_score": d.get("cross_score"),
                "url": url
            })

        return {
            "status": "success",
            "data": results,
            "suggested_ans": response.get("suggested_ans", "")
        }

    except Exception as e:
        logger.error(f"Error calling Vector Search API for question '{question}': {e}")
        return {"status": "error", "message": str(e), "data": None, "suggested_ans": ""}

# Task-based execution helpers

def generate_initial_queries(question: str) -> List[Dict[str, List[str]]]:
    # First iteration: both SQL and Vector queries
    vector_q, sql_q = clarify_query(question)
    #print("Split:")
    #print(sql_q.split("\"")[1:-1:2])
    sql_q = sql_q.split("\"")[1:-1:2]
    #print(type(sql_q))
    return [
        {"tool": "external_sql_api", "question": sql_q},
        {"tool": "external_vector_search_api", "question": [vector_q]}
    ]

def format_sql_table(sql_result):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent("""Consider the structured table output provided below. Rewrite it in the form of a nicely formatted table in Markdown. Below the table, include a short summary of the trends observed (increases, decreases, sudden peaks or troughs in the values). Do not call out any missing data, focus on what conclusions can be drawn from the data that is available.
            IMPORTANT: In the output table, make sure you mention the states (or all India), and the various commodities or sectors being represented.
            MOST IMPORTANT: If the received contents are not present (empty JSON), NO NOT HALLUCINATE the answer. Return saying that no data was received.
            """),
            temperature=0.0,
            ),
        contents=sql_result,
    )
    logger.info("Analyzed SQL output: " + response.text)
    sleep(0.5)
    return response.text

class Vector_questions(BaseModel):
    questions: List[str]

def generate_followup_queries(responses: Dict[str, Any], original_question: str) -> List[Dict[str, str]]:
    followups = []
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    llm = model.with_structured_output(Vector_questions)

    # --- Vector Follow-up Generation ---
    vector_data = responses.get("vector") or []
    if vector_data:
        # Focus on the top result for a more specific follow-up
        top_vector_result = vector_data[0]
        content_snippet = top_vector_result.get("content", "")[:1000] # Limit snippet size
        reference = top_vector_result.get("reference", "the document")

        if content_snippet:
            prompt_vector = ChatPromptTemplate.from_template(
                """Original Question: {original_question}

                Context from Document ('{reference}'): {content_snippet}

                Based *only* on the provided context and the original question, generate 2 to 3 simple, concise follow-up questions that the user can ask the Vector Search API. Each question should explore a different aspect or detail hinted at in the context. Avoid combining multiple ideas in one question.

                Do not ask for information not hinted at in the context. Output only the question text.

                Example: Breakdown the information into smaller, more specific questions. For example, if the context is about a country's GDP, return as [GDP growth rate, sector-wise contribution, historical trends].
                """
            )
            chain_vector = prompt_vector | llm

            try:
                response_obj = chain_vector.invoke({
                    "original_question": original_question,
                    "reference": reference,
                    "content_snippet": content_snippet
                })

                for q in response_obj.questions:
                    followups.append({
                        "tool": "external_vector_search_api",
                        "question": q.strip()
                    })

            except Exception as e:
                logger.error(f"Error generating vector follow-up question: {e}")
                # Fallback to simpler follow-up if LLM fails
                followups.append({
                    "tool": "external_vector_search_api",
                    "question": f"Provide more details about '{reference}' relevant to '{original_question[:50]}...'."
                })


    # --- SQL Follow-up Generation ---
    sql_data = responses.get("sql") or []
    if sql_data and isinstance(sql_data, list) and len(sql_data) > 0:
        # Summarize SQL data structure/headers for the LLM
        sql_summary = f"SQL query returned data with columns: {list(sql_data[0].keys())}. First row sample: {sql_data[0]}" if sql_data else "SQL query returned no data."

        prompt_sql = ChatPromptTemplate.from_template(
             """Original Question: {original_question}

             Initial SQL Result Summary: {sql_summary}

             Based on the original question and the columns/data available in the initial SQL result, identify the single most relevant metric or dimension. Generate ONE concise follow-up question to ask the SQL API for a more detailed breakdown (e.g., year-wise, category-wise, top N) of that specific metric/dimension relevant to the original question. Output only the question text."""
        )
        chain_sql = prompt_sql | llm
        try:
            response_obj = chain_sql.invoke({
                "original_question": original_question,
                "sql_summary": sql_summary
                })

            for q in response_obj.questions:
                followups.append({
                    "tool": "external_sql_api",
                    "question": q.strip()
                })

        except Exception as e:
            logger.error(f"Error generating SQL follow-up question: {e}")
            # Fallback if LLM fails
            metric = next(iter(sql_data[0]), None) if sql_data and sql_data[0] else None
            if metric:
                 followups.append({
                    "tool": "external_sql_api",
                    "question": f"Show detailed breakdown for '{metric}' relevant to '{original_question[:50]}...'."
                })

    # Limit to max 2-3 followups total if needed
    return followups


def run_task_based_execution(question: str) -> Dict[str, Any]:
    # Initial iteration
    initial_queries = generate_initial_queries(question)
    responses = {}
    full_responses = {}
    logger.info("\n--- Running Initial Queries ---")

    # Run SQL and Vector queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_tool = {}
        for q in initial_queries:
            tool_name = q["tool"]
            query_text = q["question"]
            logger.info(f"Scheduling {tool_name} with: {query_text}")
            if tool_name == "external_sql_api":
                # Schedule SQL batch API
                future = executor.submit(call_sql_batch_api, query_text)
            else:
                # Schedule Vector search API
                future = executor.submit(call_vector_search_api, query_text)
            future_to_tool[future] = tool_name

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_tool):
            tool_name = future_to_tool[future]
            try:
                resp = future.result()
            except Exception as exc:
                logger.error(f"{tool_name} generated an exception: {exc}")
                resp = {"status": "error", "message": str(exc), "data": None}
            logger.info(f"Response Status ({tool_name}): {resp.get('status')}")
            if tool_name == "external_sql_api":
                responses['sql'] = resp.get('data')
                full_responses['sql'] = resp
            else:
                responses['vector'] = {
                    'data': resp.get('data'),
                    'suggested_ans': resp.get('suggested_ans', '')
                }
                full_responses['vector'] = resp

    # Follow-up iteration - Pass the original question here
    followup_responses = []
    if False:
        logger.info("\n--- Generating Follow-up Queries ---")
        vector_data_for_followup = responses.get('vector', {}).get('data') or []
        followup_queries = generate_followup_queries(
            {'sql': responses.get('sql'), 'vector': vector_data_for_followup},
            original_question=question # Pass original question
        )
        followup_responses = []
        logger.info(f"Generated {len(followup_queries)} follow-up queries.")
        if followup_queries:
            logger.info("\n--- Running Follow-up Queries ---")
            for fq in followup_queries:
                tool_name = fq["tool"]
                query_text = fq["question"]
                logger.info(f"Calling {tool_name} with: {query_text}")
                if tool_name == "external_sql_api":
                    resp = call_sql_api(query_text)
                else: # vector search
                    resp = call_vector_search_api(query_text)
                logger.info(f"Response Status ({tool_name}): {resp.get('status')}")
                followup_responses.append({
                    "tool": tool_name,
                    "question": query_text,
                    "data": resp.get('data'),
                    "status": resp.get('status'),
                    "message": resp.get('message'),
                    "suggested_ans": resp.get('suggested_ans', '')
                })

    # --- Synthesis Step ---
    logger.info("\n--- Synthesizing Final Answer ---")

    # Check if all SQL results failed and unstructured data is insufficient
    sql_results = full_responses.get("sql", {}).get("data", [])
    all_sql_failed = (
    isinstance(sql_results, list) and
    len(sql_results) > 0 and
    all(item.get("success") == False for item in sql_results if isinstance(item, dict))
    )  # <== Added

    
    all_vector_insufficient = responses.get('vector', {}).get('suggested_ans', '').strip() == "<insufficient-data>" # <== Added
    print(all_sql_failed)
    print(all_vector_insufficient)
    if all_sql_failed and all_vector_insufficient:  # <== Added
        logger.info("All SQL failed and unstructured data is insufficient — returning '<insufficient-data>'.")  # <== Added
        return {
            "initial": responses,
            "followups": followup_responses,
            "synthesis": "<insufficient-data>"  # <== Added
        }  # <== Added

    synthesis_context = f"Original Question: {question}\n\n"
    synthesis_context += "Initial Structured Table Data:\n"
    try:
        if not all_sql_failed:
            sql_data = json.dumps(responses.get('sql', 'No data'), indent=2)
            print("sql data",sql_data)

            
            formatted_sql_table=format_sql_table(sql_data)
            synthesis_context += formatted_sql_table + "\n\n"
            
    except Exception as e:
        logger.warning(f"Could not serialize structured table data for synthesis context: {e}")
        synthesis_context += str(responses.get('sql', 'Error formatting data')) + "\n\n"

    synthesis_context += "Initial Text-Based Commentary Data (Content and Reference):\n"
    initial_vector_info = responses.get('vector')
    if initial_vector_info:
        initial_vector_data = initial_vector_info.get('data')
        initial_suggested_ans = initial_vector_info.get('suggested_ans')
        if initial_suggested_ans:
            synthesis_context += f"  Suggested Answer: {initial_suggested_ans}\n"
        if isinstance(initial_vector_data, list) and initial_vector_data:
            for i, doc in enumerate(initial_vector_data[:3]):
                 if isinstance(doc, dict):
                     synthesis_context += f"Result {i+1}:\n"
                     synthesis_context += f"  Reference: {doc.get('reference', 'N/A')}\n"
                     synthesis_context += f"  URL: {doc.get('url', 'N/A')}\n"
                     synthesis_context += f"  Content: {doc.get('content', 'N/A')}\n\n"
        else:
            synthesis_context += "  No document data or data in unexpected format from initial text-based data search.\n\n"
    else:
        synthesis_context += "No initial text-based search information found.\n\n"

    if False:
        synthesis_context += "Follow-up Query Results:\n"
        if followup_responses:
            for i, res in enumerate(followup_responses):
                tool = res.get('tool', 'N/A')
                q = res.get('question', 'N/A')
                status = res.get('status', 'N/A')
                data = res.get('data', 'No data')
                message = res.get('message')
                suggested_ans_followup = res.get('suggested_ans')
    
                synthesis_context += f"Follow-up {i+1} ({tool} - Q: {q} - Status: {status}):\n"
                if suggested_ans_followup:
                    synthesis_context += f"  Suggested Answer: {suggested_ans_followup}\n"
    
                try:
                    if tool == "external_vector_search_api" and isinstance(data, list):
                         synthesis_context += "  Results:\n"
                         for j, doc in enumerate(data[:2]):
                             if isinstance(doc, dict):
                                 synthesis_context += f"    Result {j+1}:\n"
                                 synthesis_context += f"      Reference: {doc.get('reference', 'N/A')}\n"
                                 synthesis_context += f"      Content: {doc.get('content', 'N/A')[:500]}...\n"
                             else:
                                 synthesis_context += f"    Result {j+1}: {str(doc)}\n"
                         if not data:
                             synthesis_context += "    No data returned.\n"
                    else:
                        synthesis_context += f"  Data: {json.dumps(data, indent=2)}\n"
    
                except Exception as e:
                     logger.warning(f"Could not serialize follow-up data for synthesis context: {e}")
                     synthesis_context += f"  Data: {str(data)}\n"
    
                if status == 'error':
                     synthesis_context += f"  Error Message: {message or 'N/A'}\n"
                synthesis_context += "---\n"
        else:
            synthesis_context += "No follow-up queries executed.\n"

    max_context_len = 25000
    if len(synthesis_context) > max_context_len:
        logger.warning(f"Synthesis context truncated from {len(synthesis_context)} to {max_context_len} chars.")
        synthesis_context = synthesis_context[:max_context_len] + "\n... (context truncated)"

    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    
    synthesized_answer = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""Based on the original question {question} and the following collected data from various sources (structured table database, unstructured text search with references, and follow-up queries), synthesize a comprehensive and coherent answer. Integrate the information smoothly. For reference, today's date is {curdate}.
                                      
            Here is some brief context about aspects of the Indian economy:
                1. Agriculture: Contributes about 15-20% to GDP, agriculture employs nearly half of the workforce. Monsoon patterns significantly impact agricultural output and rural demand.
                2. Industry: Manufacturing and construction are crucial for GDP growth and employment. Government initiatives like "Make in India" aim to boost manufacturing.
                3. Services: The services sector, including IT, finance, and telecommunications, contributes over 50% to GDP. IT services, in particular, are a major export and growth driver.
                4. Government Policies: Fiscal policies, such as taxation and public spending, influence economic growth. Monetary policies by the Reserve Bank of India (RBI) manage inflation and interest rates.
                5. Inflation: Influenced by food prices, fuel costs, and global commodity prices. The RBI uses repo rates and cash reserve ratios to control inflation.
                6. Foreign Direct Investment (FDI): FDI inflows boost infrastructure, technology, and job creation. Government policies aim to attract FDI in sectors like manufacturing and services.
                7. Global Economic Conditions: Exports, remittances, and foreign investments are affected by global demand and economic stability.
                8. Demographics: A young and growing workforce can drive economic growth, but requires adequate education and employment opportunities.
                9. Infrastructure: Investments in transportation, energy, and digital infrastructure enhance productivity and economic growth.
                10. Technological Advancements: Innovation and digitalization improve efficiency and competitiveness across sectors.
                                      
            **Notes on synthesis**
            - If tabular data is present under "Initial Structured Table Data", REPEAT THIS TABLE EXACTLY WITHOUT HALLUCINATING ANY NEW INFORMATION.
            - Take special note of the summary and trends appended to the structured table data.
            - Also take special note of the Suggested Answer under "Initial Text-Based Commentary Data".
            - Attempt to draw joint conclusions from the table output (if available) and the unstructured text commentary (if available)
            - Ensure that your summarized commentary is comprehensive, and uses paragraphs and bullet points as appropriate.
            - If sufficient information is available in provided contents, do not keep your output too brief.
            - In your summary, focus on specific data values, prices, and percentages, IF they relate to the original query. Avoid returning purely qualitative observations.

            **Formatting Instructions:**
            - Begin the final answer with this header: `## Insights from Ingested Data`
            - Present structured data (like results from structured tables) in Markdown table format for clarity.
            - Keep tables concise, summarizing if necessary for very large datasets. Mention quantity units where applicable.
            - **When using any content from the 'Unstructured Text Data', cite the corresponding 'Reference' name and URL, but only if it is actually used.**
            - Indicate citations inline using square brackets like this: [1], [2], etc.
            - At the end of the answer, add a section titled `## References` that lists only the used references, numbered to match the inline citations.
            - Each citation must correspond to a source in a `## References` section at the end.
            - In the `## References` section:
                     - Only include references that were cited in the text.
                     - If no URL is available, skip the reference entirely.                 
            - Format the `## References` section exactly like this:
                                      
            ## References
            1. [Reference Name 1](https://example.com/study)
            2. [Reference Name 2](https://example.com/study)

            - If **no unstructured data is used**, do not generate a `## References` section.
            - If all data is structured, still provide useful insights but omit references.                        
            - **Do not include any references that were not cited in the synthesized answer.**
            - Avoid duplicate citations for the same source in the same paragraph — cite once per distinct point.
            - If data is conflicting or unavailable for parts of the question, acknowledge that.
            
            Synthesized Answer (use Markdown tables for structured data and cite vector search references ONLY when vector search content is used):"""),
            temperature=0.0,
            ),
        contents=synthesis_context
    )
    synthesized_answer = synthesized_answer.text

    if not all_sql_failed:
        logger.info(f" trying to building chart from formatted table {formatted_sql_table}")
        try:
            should_plot = chart_oppourtunity_validation(formatted_sql_table)
            if should_plot == "Yes":
                logger.info(f"Building chart from formatted table for question: {question}")
                logger.info(f"Building chart from formatted table: {formatted_sql_table}")
                chart = build_chart_from_synthesized_answer(formatted_sql_table)
                if chart != "some error occurred":
                    sanitized_question = sanitize_filename(question)
                    chart_file = f"{sanitized_question}.png"
                    chart.save(chart_file)
                    logger.info("Chart saved successfully.")
                
                    base64_image = image_file_to_base64(chart_file)
                    image_markdown = f"![Chart](data:image/png;base64,{base64_image})"

                    synthesized_answer = synthesized_answer.strip()

                    if image_markdown:
                        synthesized_answer += "\n\n## Visualization\n" + image_markdown
                    return {"initial": responses, "followups": followup_responses, "synthesis": synthesized_answer , "image": image_markdown}
                else:
                    logger.warning("Chart object was None. Skipping visualization.")
            else:
                logger.info("Chart opportunity not suitable, skipping chart rendering.")

        except Exception as chart_error:
            logger.error("Chart generation failed: %s", str(chart_error))


    
    

            
   
    # chart = build_chart_from_synthesized_answer(formatted_sql_table , question)

    
    # if chart:
        
    #     sanitized_question = sanitize_filename(question)
    #     chart.save(f"{sanitized_question}.png")
    #     logger.info("Chart saved .")
    #     base64_image = image_file_to_base64(f"{sanitized_question}.png")
    #     image_markdown = f"![Chart](data:image/png;base64,{base64_image})"

    #     synthesized_answer = synthesized_answer.strip()
    #     synthesized_answer += "\n\n## Visualization\n" + image_markdown


        
    #     # image_path = "/tmp/chart.png"
    #     # save(chart, image_path)  
        
    logger.info(synthesized_answer)
    return {"initial": responses, "followups": followup_responses, "synthesis": synthesized_answer , "image":"No image"}

# FastAPI app
app = FastAPI(
    title="Indian Economy Agent API",
    description="API for iterative querying of structured and unstructured data with follow-ups.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

@app.post("/execute_query/",
          summary="Execute a task-based query",
          response_description="The result of the iterative queries and synthesis",
          dependencies=[Depends(verify_api_key)])
async def execute_query(request: QueryRequest) -> JSONResponse:
    """
    Receives a question, runs the task-based execution pipeline
    (initial queries, follow-up queries, and synthesis),
    and returns the comprehensive result.
    """
    if not request.question.strip():
        return JSONResponse(status_code=400, content={"error": "Question cannot be empty."})
    try:
        result = run_task_based_execution(request.question)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in /execute_query endpoint: {e}")
        # Consider more specific error handling and logging
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred.", "details": str(e)})

if __name__ == "__main__":
    # This block will execute when the script is run directly.
    # To generate the OpenAPI schema to a file, you can run:
    # python api_chat.py > openapi.json
    #
    # To run the FastAPI server and access Swagger UI:
    # python api_chat.py
    # Then open your browser to http://127.0.0.1:8000/docs

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == ">": # Basic check if output is being redirected
        # When generating OpenAPI schema, avoid running the server or extensive logging
        # Temporarily disable file logging for schema generation to avoid creating empty log files
        # or interfering with stdout redirection.
        if file_handler in logger.handlers:
            logger.removeHandler(file_handler)

        open_api_schema = app.openapi()
        print(json.dumps(open_api_schema, indent=2))
    else:
        # Ensure file handler is re-added if it was removed (though typically not needed here as it's a separate run)
        if file_handler not in logger.handlers:
            logger.addHandler(file_handler)
        # If not redirecting output, run the Uvicorn server
        logger.info("Starting Uvicorn server.")
        uvicorn.run(app, host="0.0.0.0", port=7000)
