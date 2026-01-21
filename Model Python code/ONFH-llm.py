import os
import pandas as pd
import argparse
from tqdm import tqdm
from glob import glob
from langchain_community.document_loaders import JSONLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.chat_models import ChatZhipuAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import FAISS

## 调用本地知识库
def RAG_base_json(base_path):
    ## 加载json文件
    loader = JSONLoader(
        file_path=base_path,
        jq_schema=".[]",
        text_content=False
    )
    documents = loader.load()
    ## 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=500
                )
    sub_documents = text_splitter.split_documents(documents)
    ## 加载向量模型
    embeddings_model = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key="" # your own api key
    )
    ## 创建向量存储
    vector_store = InMemoryVectorStore(embeddings_model)
    
    for i in range(0, len(sub_documents), 64):
        if i + 64 < len(sub_documents):
            print('processing {} to {}'.format(i, i+64))
            _ = vector_store.add_documents(documents=sub_documents[i:i+64])
    else:
        print('processing {} to {}'.format(i, len(sub_documents)))
        _ = vector_store.add_documents(documents=sub_documents[i:])

    # 查询
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    return retriever
    
## 构造提示词模板
def prompt_template(task):
    ## task1
    if task == 'stage':
        template_zero_shot = """
            You are an orthopedic clinical expert focusing on the field of femoral head necrosis (OFHN), and able to answer questions raised by clinicians with rigor, the answer should be based on well acknowledged published clinical guidelines.
            Output format should be in JSON format, answer the question following this format:
                        
            Analysis: String, model generated chain of thought explanation.
            Recommended imaging modalities: String,
            Imaging finding: String,
            Source: String,
            Full Answer: String, please answer in combination with the above content.
            
            Gudieline context:\n{context}
            The question is:\n{question}

            """   
    # test_task2
    if task == 'image':
        template_zero_shot = """
        You are an orthopedic clinical expert focusing on the field of femoral head necrosis (OFHN), and able to answer questions raised by clinicians with rigor, the answer should be based on well acknowledged published clinical guidelines.
        Output format should be in JSON format, answer the question following this format:

        Analysis: String, model generated chain of thought explanation.
        Strength of Recommendation: <Expert Consensus/Expert Opinion/Limited/Moderate/Strong>,
        ACR Appropriateness Category: <May Be Appropriate/May Be Appropriate (Disagreement)/Usually Appropriate/Usually Not Appropriate>,
        Radiation Level: <Varies/0 mSv for both adult and pediatric/1-10 mSv for adult and 0.3-3 mSv for pediatric>,
        Study Quality: <Category 3/Category 3 and Category 4/Category 1 and Category 3/Category 2 and Category 4/Category 2 and Category 3 and Category 4/Category 1 and Category 2 and Category 4/Category 1 and Category 2 and Category 3 and Category 4/not specified>
        
        Gudieline context:\n{context}
        The question is:\n{question}

        """
    # test_task3
    if task == 'short_answer_exam':
        template_zero_shot = """
            You are an orthopedic clinical expert focusing on the field of femoral head necrosis (OFHN), and able to answer questions raised by clinicians with rigor, the answer should be based on well acknowledged published clinical guidelines.
            Please answer the question according to the source guideline given in the question. Output format should be in JSON format: 

            Analysis: String, model generated chain of thought explanation.
            Type of recommendation: String.
            Certainty of evidence: <Low/ Very low/There were no studies that either directly or indirectly answered our PICO question>.
            
            Gudieline context:\n{context}
            The question is:\n{question}

            """   
    # test_task4
    if task == 'guideline1':
        template_zero_shot = """
            You are an orthopedic clinical expert focusing on the field of femoral head necrosis (OFHN), and able to answer questions raised by clinicians with rigor, the answer should be based on well acknowledged published clinical guidelines.

            Output format should be in JSON format, answer the question following this format:
            Analysis:String, model generated chain of thought explanation,
            Source: String.
            Full Answer: String, please answer in combination with the above content.
           
            Gudieline context:\n{context}
            The question is:\n{question}

            """
    # test_task5
    if task == 'guideline2':
        template_zero_shot = """
            You are an orthopedic clinical expert focusing on the field of femoral head necrosis (OFHN), and able to answer questions raised by clinicians with rigor, the answer should be based on well acknowledged published clinical guidelines.
            Please answer the question according to the source guideline given in the question. Output format should be in JSON format:

            Analysis: String, model generated chain of thought explanation.
            Source : String, source of the guideline.
            Type of recommendation: <A clear recommendation cannot be made/It is weakly recommended or proposed to do it>.
            Strength of recommendation: <Not specified/Weakly>.
            Certainty of evidence: <Very weak,no confidence in the estimated effect/Limited confidence in the estimated effect/Moderate confidence in the estimated effect>.
            Full Answer: String, please answer in combination with the above content, don't list them item by item.
            
            Gudieline context:\n{context}
            The question is:\n{question}

            """
    # test_task6
    if task == 'guideline3':
        template_zero_shot = """
            You are an orthopedic clinical expert focusing on the field of femoral head necrosis (OFHN), and able to answer questions raised by clinicians with rigor, the answer should be based on well acknowledged published clinical guidelines.
            Please answer the question according to the source guideline given in the question. Output format should be in JSON format: 

            Analysis: String, model generated chain of thought explanation.
            Type of recommendation: string.
            Certainty of evidence: <Low/Very low/There were no studies that either directly or indirectly answered our PICO question>.
            
            Gudieline context:\n{context}
            The question is:\n{question}

            """
    # test_task7
    if task == 'option_exam':
        template_zero_shot = """
        You are an orthopedic clinical expert focusing on the field of femoral head necrosis (OFHN), and able to answer questions raised by clinicians with rigor, the answer should be based on well acknowledged published clinical guidelines.
        Please answer this choice question. If it is marked as a "multiple choice" in the question, then it is a multiple-choice question; otherwise, it is a single-choice question. Output format should be in JSON format:

        Analysis: String, model generated chain of thought explanation.
        Final Choice: String, (Directly provide options and answers, e.g. A.xx, B. xx, C. xx).
        Confidence Degree: Rate your own confidence in your answer based on a Likert scale that has the following grades: 1 = no confidence [stating it does not know]; 2 = little confidence [ie, maybe]; 3 = some confidence; 4 = confidence lie [likely]; 5 = high confidence [stating answer and explanation without doubt]).ence Degree: Rate your own confidence in your answer based on a Likert scale that has the followinggrades: 1 = no confidence [stating it does not know]; 2 = little confidence [ie, maybe]; 3 = some confidence; 4 = confidence lie [likely]; 5 = high confidence [stating answer and explanationwithout doubt])

        Gudieline context:\n{context}
        The question is:\n{question}

        """
    # test_task8
    if task == 'tf_exam':
        template_zero_shot = """
            You are an orthopedic clinical expert focusing on the field of femoral head necrosis (OFHN), and able to answer questions raised by clinicians with rigor, the answer should be based on well acknowledged published clinical guidelines.
            Output format should be in JSON format. Please judge whether this statement is correct, and answer using True or False:
            Analysis: String, model generated chain of thought explanation.
            Answer: <True/False>

            Gudieline context:\n{context}
            The question is:\n{question}

            """
    # test_task9
    if task == 'patient':
        template_zero_shot = """
            You are an orthopedic clinical expert focusing on the field of femoral head necrosis (OFHN), and able to answer questions raised by clinicians with rigor, the answer should be based on well acknowledged published clinical guidelines.
            Output format should be in JSON format. Please provide opinions based on the guidelines regarding whether the following treatment plans, and make a choice according to the prompts and don't provide other irrelevant content:

            Analysis: String, model generated chain of thought explanation.
            Unproven therapies: <Recommended/Not recommended>.
            Supportive therapy: <Recommended/Not recommended>.
            Core decompression: <Recommended/Not recommended>.
            Bone grafting: <Recommended/Not recommended>.
            Total hip arthroplasty: <Recommended/Not recommended>.
            
            Gudieline context:\n{context}
            The question is:\n{question}

            """
         
    prompt = PromptTemplate.from_template(template_zero_shot)
    return prompt

## 调用模型
def get_models(model_name, prompt, retriever):

    if model_name == 'deepseek':
        llm = ChatOpenAI(
        api_key="", # your api key
        base_url="",# the base url
        model="deepseek-v3",
        )

    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | JsonOutputParser()
    )

    return rag_chain

## 推理
def inference_and_process(df_data,  out_put_dir, task, llm, prompt):

    if task == "stage":
        for index, row in tqdm(df_data.iterrows()):
            myquestion = row['Specific_Question']
            answer = llm.invoke(myquestion)
            print(answer)
           
    if task == "image":
        for index, row in tqdm(df_data.iterrows()):
            myquestion = row['Guideline_Specific_Question']
            answer = llm.invoke(myquestion)
            print(answer)

    if task == "short_answer_exam":
        for index, row in tqdm(df_data.iterrows()):
            myquestion = row['Specific_Question']
            answer = llm.invoke(myquestion)
            print(answer)
                    
    if task == "guideline1":
        for index, row in tqdm(df_data.iterrows()):
            myquestion = row['Guideline_Specific_Question']
            answer = llm.invoke(myquestion)
            print(answer)
            
    if task == "guideline2":
        for index, row in tqdm(df_data.iterrows()):
            myquestion = row['Guideline_Specific_Question']
            answer = llm.invoke(myquestion)
            print(answer)
            
    if task == "guideline3":
        for index, row in tqdm(df_data.iterrows()):
            myquestion = row['Guideline_Specific_Question']
            answer = llm.invoke(myquestion)
            print(answer)
                    
    if task == "option_exam":
        for index, row in tqdm(df_data.iterrows()):
            question = row['Question']
            option = row['Option']
            myquestion = question+option
            answer = llm.invoke(myquestion)
            print(answer)
                   
    if task == "tf_exam":
        for index, row in tqdm(df_data.iterrows()):
            myquestion = row['Question']
            answer = llm.invoke(myquestion)
            print(answer)
            
    if task == "patient":
        for index, row in tqdm(df_data.iterrows()):
            myquestion = row['Question']
            answer = llm.invoke(myquestion)
            print(answer)

def main(args):
    # 1. 加载知识库
    retriever= RAG_base_json(args.base_path)
    # 2. 加载模板
    prompt = prompt_template(args.task)
    # 3. 加载模型
    llm = get_models(args.model_name, prompt, retriever)
    # 4. 推理
    df_data = pd.read_excel(args.ori_excel_path)
    inference_and_process(df_data, args.output_excel_path, args.task, llm, prompt) 
    ## 5. 结果保存
    print("done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='', required=False, help='knowledge base path')
    parser.add_argument('--task', type=str, default='stage', required=False, help='The task of inference')
    parser.add_argument('--ori_excel_path', type=str, default='', required=False, help='The origin excel file path')
    parser.add_argument('--output_excel_path', type=str, default='', required=False, help='The answer excel file path')
    opt = parser.parse_args()
    main(opt)