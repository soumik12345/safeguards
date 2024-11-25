import pandas as pd
import streamlit as st
import weave
from dotenv import load_dotenv

load_dotenv()
weave.init(project_name="guardrails-genie")

st.title(":material/monitoring: Evaluation")

if "start_evaluation" not in st.session_state:
    st.session_state.start_evaluation = False
if "ref" not in st.session_state:
    st.session_state.ref = None

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
dataset_name = st.sidebar.text_input("Dataset name", value="")
visualize_in_app = st.sidebar.toggle("Visualize in app", value=False)

if uploaded_file is not None:
    with st.expander("Dataset Preview"):
        dataframe = pd.read_csv(uploaded_file)
        data_list = dataframe.to_dict(orient="records")

        if dataset_name != "":
            dataset = weave.Dataset(name=dataset_name, rows=data_list)
            st.session_state.ref = weave.publish(dataset)
            st.write(
                f"Dataset published at https://wandb.ai/{st.session_state.ref.entity}/{st.session_state.ref.project}/weave/objects/{st.session_state.ref.name}/versions/{st.session_state.ref._digest}"
            )

            if visualize_in_app:
                st.dataframe(data_list)
                # dataset = weave.ref("weave:///geekyrakshit/guardrails-genie/object/sample-dataset:RvdLm7KZ5KXFGcXUHWMGoJBWRVmdxiH6VgWu4cpsDHM").get()

    run_evaluation_button = st.sidebar.button("Run Evaluation")
    st.session_state.start_evaluation = run_evaluation_button

    if st.session_state.start_evaluation:
        with st.expander("Evaluation Results"):
            st.write("Evaluation results will be displayed here.")
