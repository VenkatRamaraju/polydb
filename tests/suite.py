"""
Note: These tests are generated.
"""
#!/usr/bin/env python3
import unittest
import requests
import json
import time
import random
import string

# Suite of tests
class PolyDBAPITest(unittest.TestCase):
    """Test suite for PolyDB API"""
    
    BASE_URL = "http://localhost:9000"
    SAMPLE_TEXTS = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Vector databases are optimized for similarity search operations",
        "Python is a powerful programming language with simple syntax",
        "Natural language processing helps computers understand human language",
        "Embeddings represent text as high-dimensional vectors",
        "The Transformer architecture revolutionized NLP tasks",
        "Data structures are essential for efficient algorithms",
        "Distributed systems enable horizontal scaling of applications",
        "Cloud computing provides on-demand computing resources"
    ]
    
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
    
    def test_01_insert_single_text(self):
        """Test inserting a single text into the database"""
        text = "This is a test document for insertion"
        response = self._insert_text(text)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok", f"Failed to insert text: {data.get('error', 'Unknown error')}")
        
    def test_02_insert_and_find_similar(self):
        """Test inserting a text and then finding it with a similar query"""
        # Insert a specific text
        original_text = "Artificial intelligence is transforming the technology landscape"
        insert_response = self._insert_text(original_text)
        
        self.assertEqual(insert_response.status_code, 200)
        insert_data = insert_response.json()
        self.assertEqual(insert_data["status"], "ok", f"Failed to insert text: {insert_data.get('error', 'Unknown error')}")
        
        # Search with a similar query
        query_text = "AI is changing technology"
        search_response = self._find_similar(query_text)
        
        self.assertEqual(search_response.status_code, 200)
        search_data = search_response.json()
        self.assertEqual(search_data["status"], "ok", f"Failed to search: {search_data.get('error', 'Unknown error')}")
        
        # Verify that our original text is in the results
        self.assertIn(original_text, search_data["similar_texts"], 
                     "Original text not found in search results")
    
    def test_03_batch_insert_and_search(self):
        # Insert all sample texts
        for text in self.SAMPLE_TEXTS:
            response = self._insert_text(text)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok", f"Failed to insert text: {data.get('error', 'Unknown error')}")
        
        # Search with various queries
        search_queries = [
            ("machine learning", "Machine learning is a subset of artificial intelligence"),
            ("vector similarity", "Vector databases are optimized for similarity search operations"),
            ("python programming", "Python is a powerful programming language with simple syntax")
        ]
        
        for query, expected_result in search_queries:
            response = self._find_similar(query)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok", f"Failed to search: {data.get('error', 'Unknown error')}")
            self.assertIn(expected_result, data["similar_texts"], 
                         f"Expected text not found in search results for query: {query}")
    
    def test_04_top_k_parameter(self):
        # Search with different top_k values
        for top_k in [1, 3, 5]:
            response = self._find_similar("artificial intelligence", top_k)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok", f"Failed to search: {data.get('error', 'Unknown error')}")
            self.assertLessEqual(len(data["similar_texts"]), top_k, 
                               f"Got more results than requested with top_k={top_k}")
    
    def test_05_random_text_insertion(self):
        # Generate and insert 5 random texts
        random_texts = []
        for _ in range(5):
            text = self._generate_random_text(20)
            random_texts.append(text)
            response = self._insert_text(text)
            self.assertEqual(response.status_code, 200)
        
        # Try to retrieve each random text
        for original_text in random_texts:
            # Use the first few words as the query
            query = " ".join(original_text.split()[:3])
            response = self._find_similar(query, 10)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok", f"Failed to search: {data.get('error', 'Unknown error')}")
            
            # Check if the original text is in the results
            found = False
            for result in data["similar_texts"]:
                if original_text in result:
                    found = True
                    break
            
            self.assertTrue(found, f"Could not retrieve random text with query: {query}")
    
    def test_06_multilingual_support(self):
        print("\nTesting multilingual support for the following languages:")
        for lang in self.MULTILINGUAL_TEXTS.keys():
            print(f"- {lang}")
        
        # Insert all multilingual texts
        for lang, text_data in self.MULTILINGUAL_TEXTS.items():
            print(f"\nProcessing {lang} text...")
            original_text = text_data["original"]
            response = self._insert_text(original_text)
            
            self.assertEqual(response.status_code, 200, f"Failed to insert {lang} text")
            data = response.json()
            self.assertEqual(data["status"], "ok", 
                           f"Failed to insert {lang} text: {data.get('error', 'Unknown error')}")
        
        # Allow time for processing - multilingual texts may need more processing time
        print("Waiting for processing to complete...")
        
        # Search for each language with its corresponding query
        for lang, text_data in self.MULTILINGUAL_TEXTS.items():
            print(f"\nSearching for {lang} text with query: {text_data['query']}")
            original_text = text_data["original"]
            query_text = text_data["query"]
            
            response = self._find_similar(query_text, 10)
            self.assertEqual(response.status_code, 200, f"Failed search for {lang}")
            data = response.json()
            self.assertEqual(data["status"], "ok", 
                           f"Failed to search {lang}: {data.get('error', 'Unknown error')}")
            
            # Check if the original text is in the results
            self.assertIn(original_text, data["similar_texts"], 
                         f"{lang} text not found in search results")
            print(f"✓ {lang} text successfully retrieved")
    
    def test_07_cross_language_search(self):
        """Test searching across different languages"""
        # Define cross-language search pairs (query language -> target language)
        cross_language_pairs = [
            ("English", "Arabic"),
            ("Japanese", "Chinese"),
            ("Russian", "English"),
            ("Korean", "Vietnamese")
        ]
        
        print("\nTesting cross-language search capabilities...")
        for query_lang, target_lang in cross_language_pairs:
            print(f"\nSearching with {query_lang} query for {target_lang} content...")
            
            # Use the query from one language to search for content in another language
            query_text = self.MULTILINGUAL_TEXTS[query_lang]["query"]
            target_text = self.MULTILINGUAL_TEXTS[target_lang]["original"]
            
            # Perform the search with a higher top_k to increase chances of finding cross-language matches
            response = self._find_similar(query_text, 20)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Note: This test is informational rather than a strict pass/fail
            # It's testing the model's cross-language capabilities
            if target_text in data["similar_texts"]:
                print(f"✓ Successfully found {target_lang} content with {query_lang} query")
            else:
                print(f"ℹ {target_lang} content not found with {query_lang} query - this may be normal depending on the embedding model's cross-lingual capabilities")
    
    def _insert_text(self, text):
        """Helper method to insert text into the database"""
        endpoint = f"{self.BASE_URL}/insert"
        payload = {"text": text}
        return requests.post(endpoint, json=payload)
    
    def _find_similar(self, text, top_k=5):
        """Helper method to find similar texts"""
        endpoint = f"{self.BASE_URL}/find_similar"
        payload = {"text": text, "top_k": top_k}
        return requests.post(endpoint, json=payload)
    
    def _generate_random_text(self, word_count):
        """Generate random text with the specified number of words"""
        words = []
        for _ in range(word_count):
            # Generate a random word of length 3-10
            word_length = random.randint(3, 10)
            word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
            words.append(word)
        return " ".join(words)

if __name__ == "__main__":
    # Wait a moment for the server to be fully started
    print("Running tests...")
    
    # Run the tests
    unittest.main()