import streamlit as st
import requests
import base64
import re

st.set_page_config(page_title="Economic Insights", layout="wide")
st.title(" synthesized answer + chart")

API_TOKEN = "api-12345"  

question = st.text_input("Ask a question about the CPI dataset:")
if st.button("Submit") and question:
    with st.spinner("Fetching answer..."):
        try:
            response = requests.post(
                "http://localhost:8000/execute_query/", 
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                    "access_token": API_TOKEN
                },
                
                json={"question": question}
            )

            if response.status_code == 200:
                data = response.json()
                synthesis = data.get("synthesis", "")
                image_data = data.get("image", None)
                if image_data and image_data.startswith("data:image/png;base64,"):
                    img_base64 = image_data.split(",")[1]
                    img_bytes = base64.b64decode(img_base64)
                    st.image(img_bytes, caption="Generated Chart", use_container_width=True)


                # extract and show chart if it's a base64-encoded png image
                # chart_match = re.search(r'data:image/png;base64,([^)]*)', synthesis)
                # if chart_match:
                #     img_data = chart_match.group(1)
                #     img_bytes = base64.b64decode(img_data)
                #     st.markdown(synthesis.replace(chart_match.group(0), ""))
                #     st.image(img_bytes, caption="Generated Chart", use_column_width=True)
                # else:
                st.markdown(synthesis)
            else:
                st.error(f"failed to fetch data.   Status: {response.status_code}\n{response.text}")
        except Exception as e:
            st.error(f" some error => {e}")
