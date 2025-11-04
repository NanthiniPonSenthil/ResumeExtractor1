import ast
import json
import os
import requests
from docx2pdf import convert
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from docx import Document
import google.generativeai as genai

# Hardcode OpenAI credentials here (replace with your key/endpoint)
OPENAI_API_KEY = "xyz"
OPENAI_API_BASE = ""  # e.g. "https://api.openai.com/v1" or leave empty to use default

def convert_to_pdf(file_path):
    """Convert various file formats to PDF for Document Intelligence processing"""
    base_name = os.path.splitext(file_path)[0]
    pdf_path = f"{base_name}.pdf"
    
    if file_path.lower().endswith('.txt'):
        # Convert text file to PDF
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        lines = content.split('\n')
        y_position = height - 50
        
        for line in lines:
            if y_position < 50:
                c.showPage()
                y_position = height - 50
            c.drawString(50, y_position, line[:100])
            y_position -= 20
        
        c.save()
        return pdf_path
    
    elif file_path.lower().endswith(('.docx', '.doc')):
        convert(file_path, pdf_path)
        return pdf_path
    
    elif file_path.lower().endswith('.pdf'):
        return file_path
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def call_azure_document_intelligence(file_path):
    """Send document to Azure Document Intelligence for skills and certifications extraction"""
    endpoint = "xyz"
    key = "xyz"
    model_id = "skill_resume_extractor_2"

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/octet-stream"
    }
    
    with open(file_path, "rb") as file:
        url = f"{endpoint}formrecognizer/documentModels/{model_id}:analyze?api-version=2023-07-31"
        
        response = requests.post(url, headers=headers, data=file)
        
        if response.status_code == 202:
            operation_location = response.headers.get("Operation-Location")
            
            while True:
                result_response = requests.get(
                    operation_location,
                    headers={"Ocp-Apim-Subscription-Key": key}
                )
                
                result = result_response.json()
                
                if result.get("status") == "succeeded":
                    return result
                elif result.get("status") == "failed":
                    raise Exception(f"Analysis failed: {result.get('error', 'Unknown error')}")
                
                import time
                time.sleep(2)
        else:
            raise Exception(f"Failed to submit document: {response.status_code} - {response.text}")

def parse_document_intelligence_response(response):
    """Parse the response from Azure Document Intelligence"""
    try:
        analyze_result = response.get("analyzeResult", {})
        documents = analyze_result.get("documents", [])
        
        if not documents:
            return {"name": "", "experience": "", "skills": [], "certifications": []}
        
        fields = documents[0].get("fields", {})
        
        name = ""
        name_field = fields.get("Name", {})
        if name_field:
            name = name_field.get("valueString", "")
        
        experience = ""
        exp_field = fields.get("OverallExperience", {})
        if exp_field:
            experience = exp_field.get("valueString", "")
        
        skills = []
        skills_field = fields.get("Skills", {})
        if skills_field and skills_field.get("type") == "array":
            value_array = skills_field.get("valueArray", [])
            for skill_item in value_array:
                if skill_item.get("type") == "object":
                    value_object = skill_item.get("valueObject", {})
                    for category, category_data in value_object.items():
                        if isinstance(category_data, dict) and category_data.get("valueString"):
                            skill_text = category_data.get("valueString", "").strip()
                            if skill_text:
                                individual_skills = [s.strip() for s in skill_text.split(',') if s.strip()]
                                skills.extend(individual_skills)
        
        certifications = []
        cert_field = fields.get("Certifications", {})
        if cert_field and cert_field.get("type") == "object":
            value_object = cert_field.get("valueObject", {})
            for row_key, row_data in value_object.items():
                if row_key.startswith("ROW") and isinstance(row_data, dict):
                    row_value_object = row_data.get("valueObject", {})
                    cert_data = row_value_object.get("CERTIFICATIONS", {})
                    if cert_data and cert_data.get("valueString"):
                        certifications.append(cert_data.get("valueString"))
        
        return {
            "name": name,
            "experience": experience,
            "skills": skills,
            "certifications": certifications
        }
        
    except Exception as e:
        print(f"Error parsing Document Intelligence response: {e}")
        return {"name": "", "experience": "", "skills": [], "certifications": []}

def extract_skills_from_resume(file_path):
    """Extract skills and certifications from resume using Azure Document Intelligence"""
    try:
        pdf_path = convert_to_pdf(file_path)
        response = call_azure_document_intelligence(pdf_path)
        extracted_data = parse_document_intelligence_response(response)
        
        if pdf_path != file_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
        
        return extracted_data
        
    except Exception as e:
        print(f"Error extracting skills from resume: {e}")
        return {"name": "", "experience": "", "skills": [], "certifications": []}

def extract_skills_from_jd(job_description):
    """Extract required skills and certifications from job description"""
    temp_file = "temp_jd.txt"
    
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(job_description)
        
        pdf_path = convert_to_pdf(temp_file)
        response = call_azure_document_intelligence(pdf_path)
        extracted_data = parse_document_intelligence_response(response)
        
        return {
            "skills": extracted_data.get("skills", []),
            "mandatory_certifications": [],
            "optional_certifications": extracted_data.get("certifications", [])
        }
        
    except Exception as e:
        print(f"Error extracting skills from job description: {e}")
        return {"skills": [], "mandatory_certifications": [], "optional_certifications": []}
    
    finally:
        for temp in [temp_file, temp_file.replace('.txt', '.pdf')]:
            if os.path.exists(temp):
                os.remove(temp)

    """Use an LLM to extract years_experience, mandatory/non-mandatory skills and certifications.
    Returns dict with keys: years_experience (str), mandatory_skills, non_mandatory_skills,
    mandatory_certifications, non_mandatory_certifications (all lists except years_experience).
    """
    openai.api_key = OPENAI_API_KEY
    if OPENAI_API_BASE:
        openai.api_base = OPENAI_API_BASE
    prompt = (
        "Extract and return ONLY a JSON object with keys: years_experience, mandatory_skills, "
        "non_mandatory_skills, mandatory_certifications, non_mandatory_certifications.\n\n"
        f'Job Description:\n"""{job_description}"""\n\n'
        "Sample result should be like below:\n"
        "{\n"
        '  "years_experience": "3-5 years",\n'
        '  "mandatory_skills": ["C#", "ASP.NET Core", "SQL Server"],\n'
        '  "non_mandatory_skills": [],\n'
        '  "mandatory_certifications": [],\n'
        '  "non_mandatory_certifications": ["Microsoft Certified: .NET Developer"]\n'
        "}\n"
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = resp["choices"][0]["message"]["content"].strip()
        s = content.find("{")
        e = content.rfind("}")
        if s == -1 or e == -1:
            raise ValueError("no JSON in LLM response")
        return json.loads(content[s:e+1])
    except Exception:
        return {
            "years_experience": "",
            "mandatory_skills": [],
            "non_mandatory_skills": [],
            "mandatory_certifications": [],
            "non_mandatory_certifications": []
        }


def extract_jd_with_genai(job_description):
    """GenAI (Gemini) extractor that reliably returns a JSON with the required keys."""

    default_result = {
        "years_experience": "",
        "mandatory_skills": [],
        "non_mandatory_skills": [],
        "mandatory_certifications": [],
        "non_mandatory_certifications": []
    }
    genai.configure(api_key="xyz")
    prompt = (
        "You are a recruiter. Extract and return ONLY a JSON object with keys:\n"
        "years_experience, mandatory_skills, non_mandatory_skills, "
        "mandatory_certifications, non_mandatory_certifications.\n\n"
        "Job Description:\n" + job_description + "\n\n"
        "Important: Return valid JSON only. Do not include extra text.\n"
        "Sample result should be like below:\n"
        "{\n"
        '  "years_experience": "3-5 years",\n'
        '  "mandatory_skills": ["C#", "ASP.NET Core", "SQL Server"],\n'
        '  "non_mandatory_skills": [],\n'
        '  "mandatory_certifications": [],\n'
        '  "non_mandatory_certifications": ["Microsoft Certified: .NET Developer"]\n'
        "}\n"
    )

    try:
        model = genai.GenerativeModel("gemini-pro-latest")
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or str(resp)

        # Safely extract JSON substring
        s = text.find("{")
        e = text.rfind("}")
        if s == -1 or e == -1:
            return default_result

        json_text = text[s:e+1]

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = ast.literal_eval(json_text)

        # Ensure all keys exist
        for key in default_result:
            if key not in data:
                data[key] = default_result[key]

        return data

    except Exception as e:
        print("GenAI extraction error:", e)
        return default_result

def extract_jd_with_genai1(job_description):
        # Default structure
    default_result = {
        "years_experience": "9 years",
        "mandatory_skills": ["C#", "ASP.NET Core", "Entity Framework", "SQL Server"],
        "non_mandatory_skills": ["Angular", "Azure DevOps", "Microservices"],
        "mandatory_certifications": ["Microsoft Certified: Azure Developer Associate"],
        "non_mandatory_certifications": ["Scrum Master Certified"]
    }

    # ✅ Return directly since Gemini quota exceeded
    return default_result
    
def check_experience_match(resume_data, jd_data):
    """Check if resume experience matches JD experience requirement"""
    resume_exp = resume_data.get("experience", "").lower()
    jd_exp = jd_data.get("years_experience", "").lower()
    
    if not resume_exp or not jd_exp:
        return False
    
    import re
    resume_nums = re.findall(r'\d+', resume_exp)
    if not resume_nums:
        return False
    
    resume_years = int(resume_nums[0])
    
    # Case 3: Check for "+" pattern (e.g., "9+")
    if '+' in jd_exp:
        jd_min = re.findall(r'\d+', jd_exp)
        if jd_min:
            return resume_years > int(jd_min[0])
    
    # Case 2: Check for range pattern (e.g., "3-5")
    if '-' in jd_exp:
        jd_nums = re.findall(r'\d+', jd_exp)
        if len(jd_nums) >= 2:
            jd_min, jd_max = int(jd_nums[0]), int(jd_nums[1])
            return jd_min <= resume_years <= jd_max
    
    # Case 1: Single number with 0.5 threshold
    jd_nums = re.findall(r'\d+', jd_exp)
    if jd_nums:
        jd_years = int(jd_nums[0])
        return abs(resume_years - jd_years) <= 0.5
    
    return False

def calculate_matching_percentage(resume_data, jd_data):
    """Calculate matching percentage between resume and job description"""
    resume_skills = set([skill.lower() for skill in resume_data.get("skills", [])])
    jd_skills = set([skill.lower() for skill in jd_data.get("skills", [])])
    
    if jd_skills:
        skill_matches = len(resume_skills.intersection(jd_skills))
        skill_percentage = (skill_matches / len(jd_skills)) * 100
    else:
        skill_percentage = 100
    
    return {
        "overall_percentage": round(skill_percentage, 1),
        "skill_percentage": round(skill_percentage, 1),
        "certification_status": "Basic matching analysis",
        "matched_skills": list(resume_skills.intersection(jd_skills)),
        "missing_skills": list(jd_skills - resume_skills)
    }