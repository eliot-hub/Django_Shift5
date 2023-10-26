import os

import fitz
import requests
import spacy

from sentence_transformers import SentenceTransformer, util
from transformers import logging

logging.set_verbosity_error()


API_KEY = '3kO1RrqyZDXf8hUQLAnU6PIHT2BXN1sd_5EWg1pqOaM'
PATH = os.getcwd()


def pdf_to_text(fpath):
    """
    Convert a PDF file to text.

    Parameters
    ----------
    fpath (str): The file path to the PDF document.

    Returns
    -------
    str: The text extracted from the PDF, with line breaks replaced by spaces.
    """
    doc = fitz.open(os.path.join(PATH, fpath))
    txt = ""
    for page in doc:
        txt = txt + str(page.get_text()).replace("\n", " ")
    return txt



def geocode_address(address):
    """
    Geocode the provided address using the HERE Geocoding API.

    Parameters
    ----------
        address (str): The address to be geocoded.

    Returns
    -------
        dict: A dictionary containing geocoding information.
            - 'address' (str): The geocoded address.
            - 'city' (str): The city associated with the address.
            - 'zipcode' (str): The postal code associated with the address.
            - 'lat' (float): The latitude coordinate of the geocoded location.
            - 'lon' (float): The longitude coordinate of the geocoded location.
            - 'score' (str): The query score for the geocoding result.

    If an error occurs during the geocoding process, empty strings are returned for all fields.
    
    Note
    ----
    - This function is tailored to work with the HERE Geocoding API and may require appropriate API key and endpoint.
    - Error handling is in place to handle cases where geocoding fails.

    See HERE Geocoding API documentation for more details: https://developer.here.com/documentation/geocoding-search-api
    """
    params = {
        "q":address,
        "apiKey": API_KEY,
    }
    try:
        response = requests.get("https://geocode.search.hereapi.com/v1/geocode", params=params).json()
        address = response['items'][0]['address']['label']
        city = response['items'][0]['address']['city']
        zipcode = response['items'][0]['address']['postalCode']
        lat, lon = response['items'][0]['position']['lat'], response['items'][0]['position']['lng']
        score = response['items'][0]['scoring']['queryScore']
    except:
        address = ""
        city = ""
        zipcode = ""
        lat, lon = "", ""
        score = "" 
    # result = {
    #     "address" : address,
    #     "city" : city,
    #     "zipcode" : zipcode,
    #     "lat" : lat,
    #     "lon" : lon,
    #     "score" : score
    # }
    return {"lat":lat, "lon":lon}
    

def parse_document(text):
    """
    Extract information from a document (resume / job ad).

    This function uses a spaCy custom model to process a given resume and extracts information such as email addresses,
    diplomas, job titles, company names, and skills mentioned in the resume.

    Parameters
    ----------
    text (str): The text to be parsed.

    Returns
    -------
    dict: A dictionary containing extracted information, with the following keys:
        - 'EMAIL': List of email addresses found in the resume.
        - 'DIPLOMA': List of diplomas or educational qualifications mentioned in the resume.
        - 'JOB_TITLE': List of job titles or positions mentioned in the resume.
        - 'COMPANY': List of company names mentioned in the resume.
        - 'SKILL': List of skills or competencies mentioned in the resume.
        
    Note
    ----
    - The values in the returned dictionary are lists to accommodate multiple instances of each type of information.
    - Duplicate entries are removed from the lists, ensuring uniqueness.
    """
    # Load spaCy model
    nlp_model = spacy.load(os.path.join(PATH, "model-best-distilbert"))
    
    doc = nlp_model(text)
    parsed_info = {
        "JOB_TITLE" : [],
        "SKILL": [],
        "COMPANY" : [],
        "DIPLOMA" : [],
        "EMAIL" : [],
        "LINK": [],
    }
    for ent in doc.ents:
        parsed_info[ent.label_].append(ent.text)
    for token in doc:
        if token.like_email:
            parsed_info["EMAIL"].append(token)
        if token.like_url:
            parsed_info["LINK"].append(token)
        
    parsed_info["SKILL"] = list(set(parsed_info["SKILL"]))
    parsed_info["COMPANY"] = list(set(parsed_info["COMPANY"]))
    if len(parsed_info["JOB_TITLE"]):
        parsed_info["JOB_TITLE"] = parsed_info["JOB_TITLE"][0]
    else:
        parsed_info["JOB_TITLE"] = ""
        
    return parsed_info


def similarity_score(cv_desc, job_desc):
    """
    Calculate the similarity score between a CV description and a job description using a pre-trained model.

    Parameters
    ----------
    cv_desc (str): The CV description text.
    job_desc (str): The job description text.

    Returns
    -------
    float: The similarity score between the CV and job descriptions. Higher values indicate greater similarity.

    Example
    -------
    >>> cv_description = "Experienced software engineer skilled in Python and Java."
    >>> job_description = "We are looking for a software engineer proficient in Python and Java."
    >>> similarity_model = your_pretrained_model  # Replace with the actual model object.
    >>> similarity_score(cv_description, job_description, similarity_model)
    0.9123
    """
    modelPath = os.path.join(PATH, "model-all-MiniLM-L6-v2", "model")
    model = SentenceTransformer(modelPath)
    embeddings = model.encode([cv_desc, job_desc])

    return util.dot_score(embeddings[0], embeddings[1]).item()


def keywords_extraction(text):
    """
    Extracts keywords and relevant information from a given text document.

    Parameters
    ----------
    text (str): The text document to extract keywords and information from.

    Returns
    -------
    str: A formatted description that includes the job title, skills, and education information
    extracted from the input text. The format is as follows:
    "{job_title}. Skills: {comma-separated skills}. Education: {space-separated education}".

    Example
    -------
    >>> text = "John Doe, Software Engineer with a degree in Computer Science, is skilled in Python and Java."
    >>> keywords_extraction(text)
    "Software Engineer. Skills: Python, Java. Education: Computer Science."
    """
    doc_dict = parse_document(text)

    skills = ", ".join(doc_dict["SKILL"]) if len(doc_dict["SKILL"]) else ""
    education = " ".join(doc_dict["DIPLOMA"]) if len(doc_dict["DIPLOMA"]) else ""
    desc = f"{doc_dict['JOB_TITLE']}. Skills: {skills}. Education: {education}"
    return desc


def recommend_jobs(candidate_profile, job_ads, model, top_jobs=3):
    """
    Recommend top job opportunities to a candidate based on their profile and job advertisements.

    Parameters
    ----------
    candidate_profile (str): The candidate's profile or CV description.
    job_ads (list of str): A list of job advertisement descriptions.
    model (your_pretrained_model): The pre-trained model used for encoding text.
    top_jobs (int, optional): The number of top job recommendations to return. Default is 3.

    Returns
    -------
    list: A list of recommended job opportunities as tuples, each containing the job description and its similarity score
    with the candidate's profile. The list is sorted in descending order of similarity.

    Example
    -------
    >>> candidate_profile = "Experienced software engineer skilled in Python and Java."
    >>> job_ads = ["We are looking for a software engineer proficient in Python and Java.", "Data Scientist position available for a Python expert."]
    >>> recommendation_model = your_pretrained_model  # Replace with the actual model object.
    >>> recommended_jobs = recommend_jobs(candidate_profile, job_ads, recommendation_model, top_jobs=2)
    >>> for job, score in recommended_jobs:
    ...     print(f"Job: {job}\nSimilarity Score: {score}")
    Job: We are looking for a software engineer proficient in Python and Java.
    Similarity Score: 0.9123
    Job: Data Scientist position available for a Python expert.
    Similarity Score: 0.8456
    """
    candidate_desc = keywords_extraction(candidate_profile)
    
    jobs_desc = [keywords_extraction(job) for job in job_ads]
    scores = [similarity_score(candidate_desc, job_desc, model) for job_desc in jobs_desc]
    result = list(zip(job_ads, scores))
    result.sort(key = lambda x:x[1], reverse=True)
    return result[:top_jobs]