## Smart Resume Screener

An AI-powered application that intelligently parses resumes, extracts skills and experience, and matches them against job descriptions using a fine-tuned LLM model.

## Objective

To build an intelligent system that:
- Extracts structured information from resumes (skills, experience, education)
- Matches candidate profiles with job descriptions using semantic similarity
- Generates a fit score (1–10) with justification for each candidate
- Displays shortlisted candidates through a simple web interface

## System Architecture

            ┌────────────────────────┐
            │   User Frontend (HTML) │
            │ Resume + Job Input Form│
            └──────────┬─────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │ FastAPI Backend (Python)│
            │ /score_resume endpoint  │
            └──────────┬─────────────┘
                       │
                       ▼
     ┌────────────────────────────────────┐
     │  LLM Model (Hugging Face / Local)  │
     │  - Fine-tuned on resume-job pairs   │
     │  - Predicts fit score + justification│
     └────────────────────────────────────┘
                       │
                       ▼
            ┌────────────────────────┐
            │ Database / Local Store │
            │ (Optional - store results)│
            └────────────────────────┘




## Tech Stack

Frontend | HTML, CSS (Simple Form UI) 
Backend | FastAPI (Python) 
Model Hosting | Hugging Face Hub / Local Model 
Database (optional) | SQLite / PostgreSQL 
Version Control | Git + GitHub 

**Example LLM Prompts**

**Fit Score Prompt**:
Compare the following resume with the given job description.
Rate the candidate’s fit on a scale of 1–10, and briefly justify the score.

Resume:
{{resume_text}}

Job Description:
{{job_description}}

Return your output in this format:
Score: <number between 1–10>


**Skill Extraction Prompt**
Extract the main technical and soft skills from the resume text.
Return a list of keywords.

**Example Output**

**Input:**
Resume: Skilled in Python, TensorFlow, and SQL. Worked as Data Scientist at ABC Corp.
Job Description: Looking for a Machine Learning Engineer skilled in Python and TensorFlow.

**Output:**
Score: 9
Justification: The candidate has strong overlap with required ML skills including Python and TensorFlow.

## Setup Instructions

### Clone the repository

git clone https://github.com/<your-username>/smart-resume-screener.git
cd smart-resume-screener

Install dependencies

pip install -r requirements.txt

Run the backend

uvicorn main:app --reload

Open the app

Visit http://127.0.0.1:8000 in your browser.

## Demo Video
https://drive.google.com/file/d/1PP2_wCVeHdOwCP-nvkCT0QrK7Ub_cjFg/view?usp=sharing


**Project Structure**

smart_resume_screener
|
main.py 
|
templates/ index.html         
|
model/ config.json       
|
requirements.txt
|
README.md
|
resume_dataset.csv    

