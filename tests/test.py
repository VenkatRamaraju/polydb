
#!/usr/bin/env python3
import unittest
import requests
import json
import time
import random
import string

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
    
    def test_01_insert_single_text(self):
        """Test inserting a single text into the database"""
        text = "This is a test document for insertion"
        response = self._insert_text(text)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok", f"Failed to insert text: {data.get('error', 'Unknown error')}")
        
        # Allow some time for processing
        time.sleep(1)
    
    def test_02_insert_and_find_similar(self):
        """Test inserting a text and then finding it with a similar query"""
        # Insert a specific text
        original_text = "Artificial intelligence is transforming the technology landscape"
        insert_response = self._insert_text(original_text)
        
        self.assertEqual(insert_response.status_code, 200)
        insert_data = insert_response.json()
        self.assertEqual(insert_data["status"], "ok", f"Failed to insert text: {insert_data.get('error', 'Unknown error')}")
        
        # Allow some time for processing
        time.sleep(1)
        
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
        """Test inserting multiple texts and searching among them"""
        # Insert all sample texts
        for text in self.SAMPLE_TEXTS:
            response = self._insert_text(text)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok", f"Failed to insert text: {data.get('error', 'Unknown error')}")
        
        # Allow time for processing
        time.sleep(2)
        
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
        """Test the top_k parameter for limiting search results"""
        # Search with different top_k values
        for top_k in [1, 3, 5]:
            response = self._find_similar("artificial intelligence", top_k)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "ok", f"Failed to search: {data.get('error', 'Unknown error')}")
            self.assertLessEqual(len(data["similar_texts"]), top_k, 
                               f"Got more results than requested with top_k={top_k}")
    
    def test_05_random_text_insertion(self):
        """Test inserting random texts and verifying they can be retrieved"""
        # Generate and insert 5 random texts
        random_texts = []
        for _ in range(5):
            text = self._generate_random_text(20)
            random_texts.append(text)
            response = self._insert_text(text)
            self.assertEqual(response.status_code, 200)
        
        # Allow time for processing
        time.sleep(2)
        
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
    print("Waiting 3 seconds for the server to be fully initialized...")
    time.sleep(3)
    
    # Run the tests
    unittest.main()