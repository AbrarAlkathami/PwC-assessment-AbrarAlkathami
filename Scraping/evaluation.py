import mlflow
import os 
import pandas as pd 
import dagshub
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from Scraping.LLMsProcessing import retrieve_answers , gpt4_chain ,gpt3_chain

os.environ['OPENAI_API_KEY'] = "sk-proj-t3quEkktCcLSjpmFpUKNT3BlbkFJSyeBPFQno4NVMSw3P1BP"
os.environ['PINECONE_API_KEY'] = '085005dc-bd04-45c5-9636-09dfc1c13a79'

# dagshub.init(repo_owner='krishnaik06', repo_name='MLfLow', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/krishnaik06/MLfLow.mlflow")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print(embeddings)

index_name="data"
eval_data= PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings )



def evaluate(query):
        results = {}       

        with mlflow.start_run(run_name="Llama-2 Evaluation"):
            llama2_result = retrieve_answers(query, Llama2_chain)
            mlflow.log_param("query", query)
            mlflow.log_metric("result_length", len(llama2_result))
            mlflow.log_artifact(llama2_result, artifact_path="results/llama2")
            results[query] = {"Llama-2": llama2_result}

        with mlflow.start_run(run_name="Falcon-40b Evaluation"):
            falcon_result = retrieve_answers(query, Falcon_chain)
            mlflow.log_param("query", query)
            mlflow.log_metric("result_length", len(falcon_result))
            mlflow.log_artifact(falcon_result, artifact_path="results/falcon")
            results[query].update({"Falcon-40b": falcon_result})
    
        return results
        
# mlflow.set_experiment("LLM Evaluation")
# with mlflow.start_run() as run:
#     system_prompt = "Answer the following question in two sentences"
#     # Wrap "gpt-4" as an MLflow model.
#     logged_model_info = mlflow.openai.log_model(
#         model="gpt-4",
#         task=openai.chat.completions,
#         artifact_path="model",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": "{question}"},
#         ],
#     )


#     results=mlflow.evaluate(
#         logged_model_info.model_uri,
#         eval_data,
#         targets="ground_truth",
#         model_type="question-answering",
#         extra_metrics=[mlflow.metrics.toxicity(), mlflow.metrics.latency(),mlflow.metrics.genai.answer_similarity()]
#     )

#     print({results.metrics})
#     eval_table=results.tables["eval result table"]
#     print(eval_table)



import mlflow
import openai
import os
import pandas as pd
import dagshub

os.environ['OPENAI_API_KEY'] = "sk-proj-t3quEkktCcLSjpmFpUKNT3BlbkFJSyeBPFQno4NVMSw3P1BP"


dagshub.init(repo_owner='ILABRAR1', repo_name='my-first-repo', mlflow=True)
with mlflow.start_run():
  # Your training code here...
  mlflow.log_metric('accuracy', 42)
  mlflow.log_param('Param name', 'Value')

eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle. It was developed by Databricks, a company that specializes in big data and "
            "machine learning solutions. MLflow is designed to address the challenges that data "
            "scientists and machine learning engineers face when developing, training, and deploying "
            "machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics. It was developed in response to limitations of the Hadoop "
            "MapReduce computing model, offering improvements in speed and ease of use. Spark "
            "provides libraries for various tasks such as data ingestion, processing, and analysis "
            "through its components like Spark SQL for structured data, Spark Streaming for "
            "real-time data processing, and MLlib for machine learning tasks",
        ],
    }
)
mlflow.set_experiment("LLM Evaluation")
with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"
    # Wrap "gpt-4" as an MLflow model.
    logged_model_info = mlflow.openai.log_model(
        model="gpt-4",
        task=openai.chat.completions,
        artifact_path="model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"},
        ],
    )

    # Use predefined question-answering metrics to evaluate our model.
    results = mlflow.evaluate(
        logged_model_info.model_uri,
        eval_data,
        targets="ground_truth",
        model_type="question-answering",
        extra_metrics=[mlflow.metrics.toxicity(), mlflow.metrics.latency(),mlflow.metrics.genai.answer_similarity()]
    )
    print(f"See aggregated evaluation results below: \n{results.metrics}")

    # Evaluation result for each data record is available in `results.tables`.
    eval_table = results.tables["eval_results_table"]
    df=pd.DataFrame(eval_table)
    df.to_csv('eval.csv')
    print(f"See evaluation table below: \n{eval_table}")
