This repository contains a comprehensive benchmark dataset for evaluating Large Language Models (LLMs) on clinical knowledge of Osteonecrosis of the Femoral Head (ONFH). Curated by medical researchers, this collection includes： a general knowledge dataset (nine topics from UpToDate), a guideline item dataset (eight international clinical guidelines), a medical examination dataset (four types of examinations), and a real-case dataset. Among these, 207 items were used for the general knowledge question-answer (GKQA) task, 102 were multiple-choice questions for the medical exam QA (MEQA) task, 108 were guideline recommendations for the guideline item QA (GIQA) task, and 60 were clinical scenarios for the real-case QA (RCQA) task. A domain-specific structured ONFH knowledge base was constructed and integrated with a retrieval-augmented generation (RAG) system and optimized prompt engineering.

Using this repository, the performance of different LLMs on ONFH knowledge can be tested. This study first applies Retrieval-Augmented Generation (RAG) and prompting to establish MedAgent-ONFH, which can answer professional questions about ONFH based on our constructed knowledge base.

1.Datasets：In this study, four types of datasets were compiled: a general knowledge dataset, a guideline item dataset, a medical examination dataset, and a real-case dataset.

General knowledge dataset: The general knowledge dataset was compiled from UpToDate and includes all relevant topics related to ONFH. It includes epidemiology, pathogenesis, risk factors, clinical presentation and diagnosis, differential diagnosis, treatment strategies, preoperative assessment for total hip arthroplasty (THA), perioperative management, and postoperative rehabilitation.

Guideline dataset: The guideline dataset consists of English-language guidelines related to ONFH retrieved from the Embase, Medline/PubMed, and Cochrane Central Register of Controlled Trials (CENTRAL) databases. These guidelines cover various aspects of ONFH management, including the following: the American College of Rheumatology (ACR) Appropriateness Criteria for Osteonecrosis; the 2019 revised Association Research Circulation Osseous (ARCO) staging criteria and Steinberg staging for osteonecrosis of the femoral head; the ARCO classification criteria for alcohol- or glucocorticoid-associated osteonecrosis of the femoral head; the Japanese Orthopaedic Association (JOA) guidelines for osteonecrosis of the femoral head; the Chinese Medical Association Guidelines for the clinical diagnosis and treatment of osteonecrosis of the femoral head in adults; and the ACR and American Association of Hip and Knee Surgeons (AAHKS) Clinical Practice Guidelines for the Optimal Timing of the Elective Hip or Knee Arthroplasty for Patients with Symptomatic Moderate-to-Severe Osteoarthritis or Advanced Symptomatic Osteonecrosis with Secondary Arthritis for which Nonoperative Therapy is Ineffective.

Medical examination dataset: Through multiple rounds of screening and deduplication, 102 distinct questions were ultimately curated. Specifically, 31 questions were sourced from the CMLE contributed 31 questions, 30 from the CSRTE, and 41 from the orthopaedic attending/MRCS examinations. The collected questions were categorized by type into basic knowledge questions (n=56), scenario-based clinical questions (n=38), and multiple-answer multiple-choice questions (n=8). All the Chinese-language questions were translated into English and cross-verified by two native English-speaking attending physicians.

Real-case dataset: The real-case dataset comprises anonymized clinical data from 60 patients with ONFH, including patient age, sex, body mass index (BMI), past medical history, comorbidities, clinical symptoms/signs, physical examination findings, and imaging results.

2.Codes and prompts to test and set up LLMs and MedAgent-ONFH

The file 'ONFH-llm': Construction of the MedAgent-ONFH Model.

The file 'Prompt': Optimized prompt engineering.

Key features of this README:

1. Clinical Rigor: Highlights inclusion of major classification systems (ARCO, Steinberg) and international guidelines
2. Structured Knowledge: Organizes UpToDate content into 9 clinical domains for targeted evaluation
3. Ethical Compliance: Clear usage policies respecting copyrights of professional societies
4. Specialized Content: Covers unique aspects like arthroplasty timing and ONFH-specific rehabilitation
For any queries or feedback, please contact 202435958@mail.sdu.edu.cn
