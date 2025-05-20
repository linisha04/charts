import altair as alt
import pandas as pd
import os
from google import genai
from google.genai import types
from textwrap import dedent
from time import strftime, gmtime, sleep

def build_chart_from_synthesized_answer(summary_with_table: str):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        
        config=types.GenerateContentConfig(
            
            system_instruction=dedent("""
                You are a professional data visualization assistant.
                You are given a markdown table containing unknown data types.
                Your task is to extract insights from it and generate **clean, aesthetic Altair charts** to show the most important **numeric trends over time or categories**.

                ## Step-by-step Instructions:
                
                Step 1 : Extract the relevant data from table and write Python code using Altair to visualize the most important **numeric trends** in the data.
                Step 2 : After step 1 , check which visualization should be done , to best represent the data.
                         Choose one from the following list=[Simple Bar Chart ,Simple Heatmap , Simple Histogram , Simple Line Chart,Bar Chart with Highlighted Bar ,
                         Bar Chart with Labels ,Bar Chart with Line at Mean , Bar Chart with Line on Dual Axis ,Bar Chart with Negative Values , Bar Chart with Rolling Mean ,Bar Chart with rounded edges ,Bar and Tick Chart
                         Bump Chart , Filled Step Chart , Line Chart with Confidence Interval Band , Line Chart with Cumulative Sum , Multi Series Line Chart ,Slope Graph ,Step Chart,
                         Donut Chart , Pacman Chart , Pie Chart ,Pie Chart with Labels , Brushing Scatter Plot to show data on a table ,Bubble Plot ,Scatter Plot with Href,
                         Scatter Plot with LOESS Lines , Scatter Plot with Minimap ,Parallel Coordinates Example ,Pyramid Pie Chart ,Ranged Dot Plot ,Ridgeline plot Example]

                Step 3 : After step 2 ,  write the correct altair code by follwoing instructions given below
                        * The chart should not only be accurate but also visually rich and user-friendly.  
                        * Ensure axis labels have meaningful units (e.g., "Inflation Rate (%)", "Growth (%)").
                        * Add a well-formatted title that reflects the insight (e.g., “IIP Growth Rate vs Index - Jan to Dec 2023”).
                        * Add gridlines to improve readability
                        * Ensure Y-axis has proper formatting (e.g., percentage or currency if relevant).
            
               * Prioritize columns that represent data or time-series trends, such as inflation rate, growth rate, or index values.
               * Choose appropriate chart types based on the table 
               * Make the chart clear, labeled, and use different colors for each line.
               * set appropriate width and height for the chart.
               * If applicable , sort the months or year properly on axis.
               * If multiple numeric columns are present and relevant, plot them  for comparison.
               * Title of the chart should be wrt to markdown table , chart should be labelled.
               * Use alt.Color('Category:N') instead of relying on automatic type inference.
               * After using transform_fold, **always specify the data type** explicitly in `color` and `tooltip` encodings for the folded columns.
               * Ensure all arrays  have the same length and produce valid pandas DataFrame without errors. 
               * Ensure the DataFrame is well-formed with no length mismatches.

               Example:As you are a data visualization assistant. Suppose  a markdown table is provided containing time-series economic data. Identify meaningful indicators such as index values and their growth rates over time (e.g., IIP Index and IIP Growth Rate , value , inflation rate , inflation index , growth rates). Create an Altair chart showing their trends across the available months. You may use dual axes if necessary. But your code should be correct altair code with proper labelling.

                       | Year | Month     | IIP Index | IIP Growth Rate |
                       |------|-----------|-----------|-----------------|
                       | 2023 | January   | 186.60    | 12.70           |
                       | 2024 | February  | 174.00    | 8.20            |

                       Please generate an Altair line chart with: - X-axis: Month-Year (e.g., Jan-2023, Feb-2023, …)
                                                                  - Left Y-axis: IIP Index
                                                                  - Right Y-axis (secondary): IIP Growth Rate (%)


                Output ONLY the Python code, and assign the chart to a variable named 'chart'.
                Do NOT include any explanation, markdown formatting, or comments — only valid Python code.
            """),
            temperature=0.0,

            
        ),
        contents=[summary_with_table]  
    )
   
    sleep(0.5)

    try:
        code = response.text
        if code.startswith("```python"):
            code = code[len("```python"):]
        if code.endswith("```"):
            code = code[:-3]
        code = code.strip()



        local_vars = {"alt": alt, "pd": pd}
        exec(code, {}, local_vars)
        return local_vars.get("chart")
    except Exception as e:
        print("Error executing Gemini generated code:", e)
        print("Gemini output was:\n", response.text)
        return "some error occurred"

import base64

def image_file_to_base64(path: str) -> str:
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

import re

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]', '_', name)

def chart_oppourtunity_validation(sql_result):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=dedent("""
                    You are a data visualization expert.
                    You will be given a structured SQL table output. 
                    Analyze the table and decide whether the data presents a meaningful opportunity for creating a chart or graph like line chart, bar chart, pie chart, etc.
                    Respond strictly with one of the following options (case-sensitive):['Yes', 'No']
                    Do not provide explanations, justifications, or any additional output. Only reply with 'Yes' or 'No'.
                """),
                temperature=0.0,
            ),
            contents=sql_result,
        )
        print("Chart opportunity response: %s", response.text)
        sleep(0.5)
        return response.text.strip()
    except Exception as e:
        print("Chart opportunity validation failed: %s", str(e))
        return "No" 

# def chart_type(summary_with_table:str)



