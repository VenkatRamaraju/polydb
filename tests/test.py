import requests
import json

def insert_text(text):
    return requests.post("http://localhost:9000/insert", json={"test": text})

def find_similar(text):
    return requests.post("http://localhost:9000/find_similar", json={"test": text, "top_k": 5})

# Multilingual sample texts
MULTILINGUAL_TEXTS = {
    "English": {
        "original": "Machine learning enables computers to learn from data and improve over time",
        "query": "computers learning from data"
    },
    "Hebrew": {
        "original": "למידת מכונה מאפשרת למחשבים ללמוד מנתונים ולהשתפר עם הזמן",
        "query": "מחשבים לומדים מנתונים"
    },
    "Bengali": {
        "original": "মেশিন লার্নিং কম্পিউটারগুলিকে ডেটা থেকে শিখতে এবং সময়ের সাথে উন্নত করতে সক্ষম করে",
        "query": "কম্পিউটার ডেটা থেকে শেখা"
    },
    "Vietnamese": {
        "original": "Học máy cho phép máy tính học từ dữ liệu và cải thiện theo thời gian",
        "query": "máy tính học từ dữ liệu"
    },
    "Korean": {
        "original": "머신 러닝은 컴퓨터가 데이터에서 학습하고 시간이 지남에 따라 개선되도록 합니다",
        "query": "컴퓨터 데이터 학습"
    },
    "Arabic": {
        "original": "يتيح التعلم الآلي للحواسيب التعلم من البيانات والتحسن بمرور الوقت",
        "query": "الحواسيب تتعلم من البيانات"
    },
    "Russian": {
        "original": "Машинное обучение позволяет компьютерам учиться на данных и улучшаться со временем",
        "query": "компьютеры учатся на данных"
    },
    "Thai": {
        "original": "การเรียนรู้ของเครื่องช่วยให้คอมพิวเตอร์เรียนรู้จากข้อมูลและปรับปรุงเมื่อเวลาผ่านไป",
        "query": "คอมพิวเตอร์เรียนรู้จากข้อมูล"
    },
    "Chinese": {
        "original": "机器学习使计算机能够从数据中学习并随着时间推移而改进",
        "query": "计算机从数据中学习"
    },
    "Japanese": {
        "original": "機械学習により、コンピューターはデータから学習し、時間とともに改善することができます",
        "query": "コンピューターがデータから学習する"
    }
}

result = insert_text("machine learning is the future")
print(result.status_code)
print(result.json())

result = find_similar("deep learning changed the world")
print(result.status_code)
print(result.json())