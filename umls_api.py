import requests
from requests.auth import HTTPBasicAuth
import json

class UMLSInterface:
    def __init__(self, api_key):
        self.api_key = api_key
        self.tgt = None
        self.service_url = "https://uts-ws.nlm.nih.gov/rest"
        
    def get_ticket_granting_ticket(self):
        auth_endpoint = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
        response = requests.post(auth_endpoint, data={'apikey': self.api_key})
        self.tgt = response.text
        return self.tgt
    
    def get_service_ticket(self):
        if not self.tgt:
            self.get_ticket_granting_ticket()
        response = requests.post(f"{self.tgt}", data={'service': self.service_url})
        return response.text
    
    def search_xray_concepts(self, query="chest x-ray"):
        ticket = self.get_service_ticket()
        search_endpoint = f"{self.service_url}/search/current"
        
        params = {
            'string': query,
            'ticket': ticket,
            'sabs': 'RADLEX,SNOMED CT',  # Focusing on radiology terminology
            'returnIdType': 'concept'
        }
        
        response = requests.get(search_endpoint, params=params)
        return response.json()
    
    def get_concept_info(self, cui):
        ticket = self.get_service_ticket()
        concept_endpoint = f"{self.service_url}/content/current/CUI/{cui}"
        
        params = {
            'ticket': ticket
        }
        
        response = requests.get(concept_endpoint, params=params)
        return response.json()

def main():
    # Example usage
    api_key = "your-api-key-here"
    umls = UMLSInterface(api_key)
    
    # Search for chest x-ray concepts
    results = umls.search_xray_concepts()
    
    # Process and display results
    for result in results.get('result', {}).get('results', []):
        print(f"Name: {result.get('name')}")
        print(f"URI: {result.get('uri')}")
        print(f"CUI: {result.get('ui')}")
        print("---")

if __name__ == "__main__":
    main()
