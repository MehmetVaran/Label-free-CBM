import requests
import json

UMLS_API_KEY = open(".umls_api_key", "r").read()  # Replace with your UMLS API key
UMLS_AUTH_ENDPOINT = 'https://utslogin.nlm.nih.gov/cas/v1/api-key'
UMLS_API_BASE = 'https://uts-ws.nlm.nih.gov/rest'

def get_umls_tgt(api_key):
    """Obtain a Ticket Granting Ticket (TGT) for UMLS authentication."""
    params = {'apikey': api_key}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(UMLS_AUTH_ENDPOINT, data=params, headers=headers)
    if response.status_code == 201:
        tgt = response.headers['location']
        return tgt
    else:
        raise Exception(f"Failed to get TGT: {response.text}")

def get_umls_service_ticket(tgt):
    """Obtain a Service Ticket (ST) using the TGT."""
    service = 'http://umlsks.nlm.nih.gov'
    params = {'service': service}
    response = requests.post(tgt, data=params)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to get Service Ticket: {response.text}")

def search_umls_concepts(query, api_key=UMLS_API_KEY):
    """Search for UMLS concepts using the /search endpoint."""
    tgt = get_umls_tgt(api_key)
    st = get_umls_service_ticket(tgt)
    search_url = f"{UMLS_API_BASE}/search/current"
    params = {
        'string': query,
        'ticket': st,
        'pageSize': 5
    }
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        results = response.json()
        return results.get('result', {}).get('results', [])
    else:
        raise Exception(f"UMLS search failed: {response.text}")

def load_diseases(filepath):
    """Load disease names from a file, excluding 'No Finding'."""
    with open(filepath, "r") as f:
        diseases = [line.strip() for line in f if line.strip() and line.strip().lower() != "no finding"]
    return diseases

def save_concepts_to_json(concepts_dict, out_path):
    """Save the concepts dictionary to a JSON file."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(concepts_dict, f, indent=2, ensure_ascii=False)

def generate_umls_concepts_for_diseases(diseases, api_key=UMLS_API_KEY):
    """Generate UMLS concepts for a list of diseases."""
    concepts_dict = {}
    for disease in diseases:
        try:
            concepts = search_umls_concepts(disease, api_key)
            concepts_dict[disease] = concepts
        except Exception as e:
            concepts_dict[disease] = {"error": str(e)}
    return concepts_dict

def get_related_concepts(uri, api_key=UMLS_API_KEY):
    """Fetch related concepts for a given UMLS concept URI."""
    tgt = get_umls_tgt(api_key)
    st = get_umls_service_ticket(tgt)
    relations_url = uri + "/relations"
    params = {'ticket': st}
    response = requests.get(relations_url, params=params)
    if response.status_code == 200:
        results = response.json()
        result = results.get('result', None)
        if isinstance(result, dict):
            return result.get('relations', [])
        elif isinstance(result, list):
            return result
        else:
            return []
    else:
        return []

def generate_related_concepts_for_diseases(concepts_json_path, out_path, api_key=UMLS_API_KEY):
    """For each disease, fetch related concepts for each of its UMLS CUIs."""
    with open(concepts_json_path, "r", encoding="utf-8") as f:
        disease_concepts = json.load(f)
    disease_related = {}
    for disease, concepts in disease_concepts.items():
        related = []
        if isinstance(concepts, list):
            for concept in concepts:
                uri = concept.get("uri")
                if uri:
                    rels = get_related_concepts(uri, api_key)
                    # Optionally filter or simplify the related concepts
                    for rel in rels:
                        related_concept = {
                            "relationLabel": rel.get("relationLabel"),
                            "relatedIdName": rel.get("relatedIdName"),
                            "relatedId": rel.get("relatedId"),
                            "relatedIdVersion": rel.get("relatedIdVersion"),
                            "relatedIdSource": rel.get("rootSource")
                        }
                        related.append(related_concept)
        disease_related[disease] = related
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(disease_related, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    diseases = load_diseases("data/chestxray_classes.txt")
    concepts_dict = generate_umls_concepts_for_diseases(diseases)
    save_concepts_to_json(concepts_dict, "data/chestxray_umls_concepts.json")
    print(f"Saved UMLS concepts for {len(diseases)} diseases to data/chestxray_umls_concepts.json")
    # After saving concepts, generate related concepts for each disease
    generate_related_concepts_for_diseases(
        "data/chestxray_umls_concepts.json",
        "data/chestxray_umls_related_concepts.json"
    )
    print("Saved related UMLS concepts to data/chestxray_umls_related_concepts.json")
