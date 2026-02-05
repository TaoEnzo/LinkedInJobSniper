import os
import glob  # ✅ 修复：添加了缺失的 glob 库
import smtplib
from typing import List, Optional
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# web clawing imports
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time
import random

import pandas as pd
from dotenv import load_dotenv
from jobspy import scrape_jobs

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# read pdf
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# =================================================================
# ✅ 1. 个性化配置 (根据你的简历修改)
# =================================================================

# 搜索关键词：针对你的 Marketing / CRM / Training / Luxury 背景
# 使用 OR 连接词可以扩大搜索范围
SEARCH_TERM = "Retail Training OR Clienteling Manager OR CRM Coordinator OR Luxury Sales Assistant"

# 地点：锁定巴黎
LOCATIONS = ["Paris, France"]

RESULT_LIMIT = 15
HOURS_OLD = 24

# 获取 Secrets
PROXY_URL = os.getenv("PROXY_URL", None)
RESUME = os.getenv("RESUME_CONTENT", "") # 注意：GitHub Secrets 名字要对应
if not RESUME:
    RESUME = os.getenv("RESUME_TEXT", "") # 兼容旧变量名

API_KEY = os.getenv("API_KEY") # 对应 GitHub Secret: API_KEY
BASE_URL = os.getenv("API_BASE") # 对应 GitHub Secret: API_BASE (DeepSeek 需要)

# =================================================================
# ✅ 2. AI 配置 (适配 DeepSeek 或 OpenAI)
# =================================================================

# Define the output data structure from AI
class JobEvaluation(BaseModel):
    """
    Structure for job evaluation output.
    """
    score: int = Field(description="A relevance score from 0 to 100 based on the resume match.")
    reason: str = Field(description="A concise, one-sentence reason for the score.")

# 自动判断模型名称
# 如果设置了 API_BASE (通常是 DeepSeek), 使用 deepseek-chat
# 否则默认使用 gpt-4o-mini (OpenAI 最具性价比模型)
model_name_to_use = "deepseek-chat" if BASE_URL else "gpt-4o-mini"

llm = ChatOpenAI(
    model_name=model_name_to_use, 
    temperature=0,
    api_key=API_KEY,
    base_url=BASE_URL,
)

# structure output
structured_llm = llm.with_structured_output(JobEvaluation)

# =================================================================
# ✅ 3. AI 评分标准 (根据你的简历重写)
# =================================================================

system_template = """
[Context]
You are an expert career coach in the Luxury & Retail industry. Your goal is to evaluate how well a job description matches a candidate's resume.

[Candidate Profile Summary]
- Education: Master in International Marketing (SKEMA), French Language background.
- Experience: Operations & Marketing at Qeelin (Kering), Client Marketing at Puig, Sales at Byredo.
- Key Skills: Clienteling, CRM (Salesforce, HubSpot), Retail Training, Event Coordination, Data Analysis (Power BI, Excel).
- Languages: Native Chinese, Professional French (C1), Professional English.

[Objectives]
Return a score (0-100) and a concise reason.

[Scoring Criteria]
1. **Industry Match (40%)**: Does the job belong to Luxury Retail, Fashion, or Beauty sectors? (High score for LVMH, Kering, Richemont brands).
2. **Role Relevance (40%)**: Is it related to Training, CRM, Clienteling, Marketing Coordination, or Retail Operations? 
3. **Skill & Language (20%)**: Does it require French/English/Chinese trilingual skills? Does it mention Salesforce/SAP/Excel?

[Negative Filter]
- If the job requires "Engineering", "Coding (Python/Java as main focus)", or "Senior Director level", give a LOW score (<30).
"""

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", """
    RESUME (Truncated):
    {resume}

    JOB TITLE: {title}
    JOB DESCRIPTION (Truncated):
    {description}

    Analyze the match based on the criteria. Be strict.
    """)
])

# Chain
evaluation_chain = prompt_template | structured_llm


# =================================================================
# 4. 功能函数 (保持原样，修复了 import glob)
# =================================================================

def load_resume_from_file():
    """
    read resume from 'resumes/'
    support .pdf .txt .md
    """
    resume_folder = "resumes"

    if not os.path.exists(resume_folder):
        os.makedirs(resume_folder)
        print("📁 Created 'resumes' folder.")
        return ""

    # ✅ 这里原本报错，现在修复了 (因为顶部加了 import glob)
    files = glob.glob(os.path.join(resume_folder,"*"))

    if not files:
        print("📁 No resume file found in 'resumes' folder.")
        return ""

    file_path = files[0]
    file_ext = os.path.splitext(file_path)[1].lower()
    content = ""

    print(f'📄 Loading resume from: {file_path}')

    try:
        if file_ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                content += page.extract_text() + "\n" or ""
        elif file_ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print("❌ Unsupported resume file format.")
            return ""
        return content
    except Exception as e:
        print(f"❌ Error reading resume file: {e}")
        return ""

# 如果环境变量没配置简历，尝试从本地读取
if not RESUME or len(RESUME) < 10:
    RESUME = load_resume_from_file()

def fetch_missing_description(url: str, proxies: dict = None) -> str:
    """
    if the jobspy cannot fetch description, try to fetch from job url directly.
    """
    print(f"   ⛑️  Attempting manual fetch for: {url}...")
    ua = UserAgent()
    headers = {
        "User-Agent": ua.random,
        "Accept-Language": "en-US,en;q=0.9",
        "Referrer": "https://www.google.com/"
    }

    try:
        time.sleep(random.uniform(2, 5))
        response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 针对不同网站的尝试
            description_div = soup.find("div", {"class": "show-more-less-html__markup"}) or \
                              soup.find("div", {"class": "description__text"}) or \
                              soup.find("div", {"class": "job-description"})
            if description_div:
                return description_div.get_text(separator="\n").strip()
            else:
                return soup.get_text()[:5000]
        else:
            return ""
    except Exception as e:
        return ""

def get_jobs_data(location: str) -> pd.DataFrame:
    proxies = [PROXY_URL] if PROXY_URL else None
    print(f"🕵️  CareerScout is searching for '{SEARCH_TERM}' in '{location}'...")
    
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            jobs = scrape_jobs(
                site_name=["linkedin"],
                search_term=SEARCH_TERM,
                location=location,
                result_wanted=RESULT_LIMIT,
                hours_old=HOURS_OLD,
                proxies=proxies
            )
            print(f"✅  Scraped {len(jobs)} jobs.")
            return jobs
        except Exception as e:
            print(f"   ❌ Error on attempt {attempt}: {str(e)}")
            if attempt < MAX_RETRIES:
                time.sleep(random.uniform(3, 6))
            else:
                print("All retry attempts failed.")
    return pd.DataFrame()

def evaluate_job(title: str, description: str) -> dict:
    if not description or len(str(description)) < 50:
        return {"score": 0, "reason": "Job description missing"}
    
    # 如果简历内容也没加载成功，给一个警告但继续运行
    resume_content = RESUME if RESUME else "Candidate has Luxury Retail and Marketing experience."

    try:
        result: JobEvaluation = evaluation_chain.invoke({
            "resume": resume_content[:3000], 
            "title": title,
            "description": description[:3000]
        })
        return {"score": result.score, "reason": result.reason}
    except Exception as e:
        print(f"⚠️  AI Evaluation Error: {e}")
        return {"score": 0, "reason": "AI Error"}

def send_email(top_jobs: List[dict]):
    if not top_jobs:
        print("📭  No matching jobs to send.")
        return

    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")
    
    if not sender or not password:
        print("❌ Email credentials missing. Check GitHub Secrets.")
        return

    subject = f"🚀 Job Report: Top {len(top_jobs)} Roles for {datetime.now().strftime('%Y-%m-%d')}"

    html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #2c3e50;">Daily Job Matches (Paris & Luxury)</h2>
            <p>Found <b>{len(top_jobs)}</b> matches based on your profile (SKEMA, Kering, Puig exp):</p>
            <table style="border-collapse: collapse; width: 100%; max-width: 800px;">
                <tr style="background-color: #f8f9fa; text-align: left;">
                    <th style="padding: 10px; border-bottom: 2px solid #ddd;">Score</th>
                    <th style="padding: 10px; border-bottom: 2px solid #ddd;">Title</th>
                    <th style="padding: 10px; border-bottom: 2px solid #ddd;">Company</th>
                    <th style="padding: 10px; border-bottom: 2px solid #ddd;">Why Match?</th>
                    <th style="padding: 10px; border-bottom: 2px solid #ddd;">Action</th>
                </tr>
        """

    for job in top_jobs:
        color = "#27ae60" if job['score'] >= 80 else "#d35400"
        html_body += f"""
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee; font-weight: bold; color: {color};">
                        {job['score']}
                    </td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{job['title']}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">{job['company']}</td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee; font-size: 14px; color: #555;">
                        {job['reason']}
                    </td>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">
                        <a href="{job['job_url']}" style="background-color: #007bff; color: white; padding: 5px 10px; text-decoration: none; border-radius: 4px; font-size: 12px;">Apply</a>
                    </td>
                </tr>
            """

    html_body += "</table></body></html>"

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print(f"📧  Email sent successfully to {receiver}!")
    except Exception as e:
        print(f"❌  Email sending failed: {e}")

def main():
    # 1. Scraping
    df = pd.DataFrame()
    for location in LOCATIONS:
        df = pd.concat([df,get_jobs_data(location)], ignore_index=True, sort=False)
    
    if df.empty:
        print("No jobs found via scraping.")
        return

    scored_jobs = []
    req_proxies = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None

    # 2. Evaluation Loop
    print(f"🧠  Analyzing {len(df)} jobs with AI...")

    for _, row in df.iterrows():
        title = row.get('title', 'Unknown')
        description = row.get('description')
        job_url = row.get('job_url')

        # 补救措施：如果没有JD，尝试手动抓取
        if not description or len(str(description)) < 50:
            if job_url:
                description = fetch_missing_description(job_url, proxies=req_proxies)

        evaluation = evaluate_job(title, description)

        print(f"   📝 [{evaluation['score']}] {title}: {evaluation['reason']}")

        # 阈值：只保留 50 分以上的职位
        if evaluation['score'] >= 50:
            scored_jobs.append({
                "title": title,
                "company": row.get('company'),
                "job_url": row.get('job_url'),
                "score": evaluation['score'],
                "reason": evaluation['reason']
            })

    # 3. Sorting & Sending
    scored_jobs.sort(key=lambda x: x['score'], reverse=True)
    top_jobs = scored_jobs[:15]
    
    send_email(top_jobs)

if __name__ == "__main__":
    main()
