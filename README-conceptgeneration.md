# Concept Generation System: Technical Report

## Executive Summary

This document describes a multi-agent concept generation system designed to automatically discover and extract biomedical concepts from the Unified Medical Language System (UMLS) for use in Concept Bottleneck Models (CBMs). The system leverages LangChain's Plan-and-Execute framework to orchestrate intelligent agents that search, retrieve, and expand medical concepts related to specific diseases.

**Key Features:**
- Automated concept discovery from UMLS Metathesaurus
- Multi-agent orchestration using LangChain
- Intelligent concept expansion with related terminology
- Robust error handling and graceful degradation
- Batch processing capabilities for multiple diseases

---

## 1. Introduction

### 1.1 Background

Concept Bottleneck Models (CBMs) require high-quality concept sets to function effectively. Manually curating these concepts is time-consuming and may miss important relationships. This system automates the process by:

1. **Searching** UMLS for disease-related concepts
2. **Retrieving** detailed metadata (definitions, semantic types, sources)
3. **Expanding** concepts with clinically relevant related terms

### 1.2 System Overview

The concept generation system is built on three core technologies:

- **UMLS API**: Provides access to the Unified Medical Language System, a comprehensive biomedical terminology database
- **LangChain**: Enables multi-agent orchestration with planning and execution capabilities
- **OpenAI GPT Models**: Powers the intelligent agents that make decisions about concept selection and expansion

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input (Disease Name)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          LangChainConceptGenerator (Orchestrator)            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │         Plan-and-Execute Controller                    │  │
│  │  ┌──────────────┐          ┌──────────────┐           │  │
│  │  │   Planner    │─────────▶│   Executor   │           │  │
│  │  │  (GPT-4o)    │          │  (GPT-4o)    │           │  │
│  │  └──────────────┘          └──────┬───────┘           │  │
│  └──────────────────────────────────┼────────────────────┘  │
│                                     │                         │
│  ┌──────────────────────────────────┼────────────────────┐   │
│  │         LangChain Tools          │                    │   │
│  │  ┌──────────────┐  ┌──────────┐ │  ┌──────────────┐  │   │
│  │  │ SearchUMLS   │  │ FetchCUI │ │  │ FetchRelated │  │   │
│  │  │    Tool      │  │   Tool   │ │  │    Tool      │  │   │
│  │  └──────┬───────┘  └────┬─────┘ │  └──────┬───────┘  │   │
│  └─────────┼───────────────┼───────┼─────────┼─────────┘   │
│            │               │       │         │              │
└────────────┼───────────────┼───────┼─────────┼──────────────┘
             │               │       │         │
             ▼               ▼       ▼         ▼
┌─────────────────────────────────────────────────────────────┐
│                    UMLSClient (API Wrapper)                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Direct API Key Authentication (since May 2022)        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              UMLS REST API (NLM)                            │
│  • /search/current                                          │
│  • /content/current/CUI/{CUI}                              │
│  • /content/current/CUI/{CUI}/relations                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 UMLSClient
A lightweight HTTP client that handles:
- Direct API key authentication (modern method, no tickets)
- Request/response management
- Error handling and retries
- Three main operations:
  - `search_concepts()`: Search UMLS by term
  - `fetch_concept_details()`: Get detailed CUI information
  - `fetch_related_concepts()`: Retrieve related concepts

#### 2.2.2 LangChain Tools
Three specialized tools that wrap UMLS API calls:

1. **SearchUMLSTool**
   - Purpose: Search for concepts matching a disease name
   - Input: Query string, page size, search type
   - Output: JSON array of search results

2. **FetchConceptDetailsTool**
   - Purpose: Retrieve detailed metadata for a specific CUI
   - Input: CUI (Concept Unique Identifier)
   - Output: JSON object with name, definitions, semantic types, sources

3. **FetchRelatedConceptsTool**
   - Purpose: Expand a concept with related terms
   - Input: CUI, optional relation type filters
   - Output: JSON array of related concepts with relationship information

#### 2.2.3 Plan-and-Execute Controller
- **Planner**: Analyzes the task and creates a step-by-step plan
- **Executor**: Executes the plan by calling appropriate tools
- **Controller**: Coordinates between planner and executor, handles iterations

---

## 3. Workflow Description

### 3.1 Single Disease Processing

When processing a single disease, the system follows this workflow:

```
1. User provides disease name (e.g., "Pneumonia")
   │
   ▼
2. Planner analyzes the task and creates a plan:
   - Search UMLS for "Pneumonia"
   - Select top N most relevant CUIs
   - Fetch detailed information for each CUI
   - Expand each CUI with related concepts
   │
   ▼
3. Executor follows the plan:
   a. Calls SearchUMLSTool("Pneumonia")
      → Returns list of matching CUIs with scores
   │
   b. For each selected CUI:
      - Calls FetchConceptDetailsTool(CUI)
        → Gets name, definition, semantic types, sources
      - Calls FetchRelatedConceptsTool(CUI)
        → Gets related concepts (symptoms, treatments, etc.)
   │
   ▼
4. Executor aggregates results into structured JSON
   │
   ▼
5. System extracts and validates JSON output
   │
   ▼
6. Returns structured concept data
```

### 3.2 Agent Instructions

The system provides detailed instructions to the agents:

1. **Search Phase**: Find relevant CUIs for the disease
   - Check for empty results before accessing indices
   - Try alternative search terms if initial search fails

2. **Retrieval Phase**: Gather rich metadata
   - Fetch preferred names, semantic types, definitions
   - Collect source vocabulary information

3. **Expansion Phase**: Find related concepts
   - Identify symptoms, treatments, anatomical sites
   - Respect relation type constraints if provided

4. **Output Phase**: Structure the data
   - Follow strict JSON schema
   - Handle empty arrays gracefully

### 3.3 Error Handling

The system implements multiple layers of error handling:

1. **Index Error Prevention**: Instructions explicitly warn agents to check list lengths
2. **Try-Catch Blocks**: Wraps agent execution to catch IndexError and KeyError
3. **Graceful Degradation**: Returns empty structures with error messages instead of crashing
4. **Logging**: Comprehensive logging for debugging and monitoring

---

## 4. UMLS API Integration

### 4.1 Authentication

The system uses **direct API key authentication**, introduced by UMLS in May 2022:

- **Method**: API key passed as query parameter (`apiKey`)
- **Advantages**: Simpler, faster, no ticket expiration issues
- **Implementation**: `UMLSClient._authenticated_get()` and `_authenticated_post()`

### 4.2 API Endpoints Used

| Endpoint | Purpose | Parameters |
|----------|---------|------------|
| `/search/current` | Search for concepts | `string`, `pageSize`, `searchType` |
| `/content/current/CUI/{CUI}` | Get concept details | `returnIdType`, `format` |
| `/content/current/CUI/{CUI}/relations` | Get related concepts | `includedRelation` (optional) |

### 4.3 API Response Structure

**Search Response:**
```json
{
  "result": {
    "results": [
      {
        "ui": "C0032285",
        "name": "Pneumonia",
        "score": 100,
        "uri": "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0032285"
      }
    ]
  }
}
```

**Concept Details Response:**
```json
{
  "result": {
    "ui": "C0032285",
    "name": "Pneumonia",
    "definition": "Inflammation of the lung...",
    "semanticTypes": [
      {"name": "Disease or Syndrome", "uri": "..."}
    ],
    "atoms": [...]
  }
}
```

**Relations Response:**
```json
{
  "result": [
    {
      "relatedIdName": "Cough",
      "relationLabel": "may_cause",
      "relatedId": "C0010200"
    }
  ]
}
```

---

## 5. Multi-Agent System Details

### 5.1 LangChain Plan-and-Execute Framework

The system uses LangChain's experimental `PlanAndExecute` framework:

**Why Plan-and-Execute?**
- Complex tasks require multi-step reasoning
- Agents need to adapt based on intermediate results
- Better than single-shot execution for exploratory tasks

**How it Works:**
1. **Planning Phase**: LLM analyzes the task and creates a plan
2. **Execution Phase**: LLM executes each step, using tools as needed
3. **Iteration**: Can refine plan based on execution results

### 5.2 Agent Configuration

- **Model**: GPT-4o-mini (default, configurable)
- **Temperature**: 0.0 (deterministic for reproducibility)
- **Tools**: Three UMLS-specific tools
- **Verbose Mode**: Optional detailed logging

### 5.3 Tool Descriptions

Each tool has a detailed description that guides the agent:

- **SearchUMLSTool**: Warns about empty results, emphasizes index checking
- **FetchConceptDetailsTool**: Explains when to use (after searching)
- **FetchRelatedConceptsTool**: Describes expansion use case, warns about empty arrays

---

## 6. Output Format

### 6.1 JSON Schema

The system returns structured JSON following this schema:

```json
{
  "disease": "Pneumonia",
  "candidate_concepts": [
    {
      "cui": "C0032285",
      "name": "Pneumonia",
      "score": 100.0,
      "semantic_types": ["Disease or Syndrome"],
      "definition": "Inflammation of the lung...",
      "sources": ["SNOMEDCT_US", "ICD10CM"]
    }
  ],
  "related_concepts": {
    "C0032285": [
      {
        "related_cui": "C0010200",
        "name": "Cough",
        "relation_label": "may_cause",
        "relation": "RO",
        "additional_information": null
      }
    ]
  }
}
```

### 6.2 Error Output Format

When errors occur, the system returns:

```json
{
  "disease": "Pneumonia",
  "candidate_concepts": [],
  "related_concepts": {},
  "error": "Index error during execution: list index out of range"
}
```

---

## 7. Usage Examples

### 7.1 Command-Line Interface

**Single Disease:**
```bash
python umls_api.py --disease "Pneumonia" --verbose
```

**Multiple Diseases from File:**
```bash
python umls_api.py --disease-file data/diseases.txt --max-concepts 10 --out results.json
```

**With Relation Type Filters:**
```bash
python umls_api.py --disease "Pneumonia" --relation-types RO RB
```

### 7.2 Python API

**Basic Usage:**
```python
from umls_api import LangChainConceptGenerator

generator = LangChainConceptGenerator(verbose=True)
result = generator.generate("Pneumonia", max_concepts=5)
print(result)
```

**Batch Processing:**
```python
diseases = ["Pneumonia", "Diabetes", "Hypertension"]
results = generator.generate_batch(
    diseases,
    max_concepts=10,
    relation_types=["RO", "RB"]
)
```

**Custom LLM:**
```python
from langchain_openai import ChatOpenAI

custom_llm = ChatOpenAI(model="gpt-4", temperature=0.1)
generator = LangChainConceptGenerator(llm=custom_llm)
```

---

## 8. Configuration

### 8.1 API Keys

The system reads API keys from files:

- **UMLS API Key**: `.umls_api_key` (in project root)
- **OpenAI API Key**: `.openai_api_key` (in project root)

**Fallback**: Environment variables (`OPENAI_API_KEY`)

### 8.2 Default Parameters

- **LLM Model**: `gpt-4o-mini`
- **Temperature**: `0.0` (deterministic)
- **Max Concepts**: `5` per disease
- **Request Timeout**: `30` seconds
- **Page Size**: `10` search results

### 8.3 Customization Options

- Custom LLM models
- Adjustable temperature for creativity
- Configurable max concepts per disease
- Optional relation type filtering
- Verbose logging mode

---

## 9. Technical Implementation Details

### 9.1 Dependencies

**Core Libraries:**
- `langchain >= 0.2.0`: Multi-agent orchestration
- `langchain-openai >= 0.1.0`: OpenAI integration
- `langchain-experimental >= 0.0.50`: Plan-and-Execute framework
- `requests >= 2.28.0`: HTTP client for UMLS API
- `pydantic`: Data validation and type checking

### 9.2 Code Structure

```
umls_api.py
├── UMLSClient (lines 48-175)
│   ├── Authentication
│   ├── API Methods
│   └── Error Handling
│
├── LangChain Tools (lines 178-265)
│   ├── SearchUMLSTool
│   ├── FetchConceptDetailsTool
│   └── FetchRelatedConceptsTool
│
├── Multi-Agent Orchestrator (lines 312-469)
│   ├── LangChainConceptGenerator
│   ├── generate() - Single disease
│   └── generate_batch() - Multiple diseases
│
└── CLI Interface (lines 515-586)
    ├── Argument Parsing
    └── Main Entry Point
```

### 9.3 Key Design Decisions

1. **Direct API Key Auth**: Chose modern authentication over deprecated tickets
2. **Plan-and-Execute**: Selected for complex multi-step tasks
3. **Error Handling**: Comprehensive to ensure robustness
4. **Tool Wrapping**: UMLS API calls wrapped as LangChain tools for agent use
5. **JSON Extraction**: Robust parsing handles various output formats

---

## 10. Performance Considerations

### 10.1 API Rate Limits

- UMLS API: No explicit rate limits documented, but reasonable usage expected
- OpenAI API: Subject to rate limits based on tier
- **Recommendation**: Add delays for batch processing if needed

### 10.2 Caching

- **Not Currently Implemented**: Each request hits the API
- **Future Enhancement**: Cache search results and concept details

### 10.3 Parallelization

- **Current**: Sequential processing for batch operations
- **Future Enhancement**: Parallel agent execution for multiple diseases

---

## 11. Limitations and Future Improvements

### 11.1 Current Limitations

1. **No Caching**: Repeated searches hit the API every time
2. **Sequential Processing**: Batch operations process one disease at a time
3. **Error Recovery**: Limited retry logic for transient failures
4. **Cost**: OpenAI API calls incur costs (mitigated by using GPT-4o-mini)

### 11.2 Proposed Enhancements

1. **Result Caching**: Cache UMLS search results and concept details
2. **Parallel Execution**: Process multiple diseases concurrently
3. **Retry Logic**: Automatic retries for transient API failures
4. **Concept Filtering**: Post-processing to filter low-quality concepts
5. **Validation**: Validate generated concepts against existing concept sets
6. **Metrics**: Track concept quality, coverage, and relevance

---

## 12. Comparison with Previous Approach

### 12.1 Old System (Direct REST Calls)

**Characteristics:**
- Simple function calls to UMLS API
- Manual orchestration of search → fetch → expand
- No intelligent concept selection
- Limited error handling

**Limitations:**
- Required manual intervention for concept selection
- No adaptive search strategies
- Difficult to handle edge cases

### 12.2 New System (Multi-Agent)

**Advantages:**
- Intelligent planning and execution
- Adaptive search strategies
- Better error handling
- More flexible and extensible

**Trade-offs:**
- More complex architecture
- Requires OpenAI API (costs)
- Slightly slower due to LLM calls

---

## 13. Conclusion

This multi-agent concept generation system represents a significant advancement in automated concept discovery for Concept Bottleneck Models. By leveraging LangChain's Plan-and-Execute framework and the comprehensive UMLS database, the system can intelligently discover, retrieve, and expand biomedical concepts with minimal human intervention.

The system's architecture is designed for:
- **Robustness**: Comprehensive error handling
- **Flexibility**: Configurable parameters and extensible design
- **Intelligence**: Adaptive planning and execution
- **Usability**: Simple CLI and Python API

Future work should focus on performance optimization (caching, parallelization) and quality improvements (validation, filtering, metrics).

---

## 14. References

1. **UMLS API Documentation**: https://documentation.uts.nlm.nih.gov/rest/home.html
2. **LangChain Documentation**: https://python.langchain.com/
3. **Plan-and-Execute Framework**: https://python.langchain.com/docs/use_cases/agent_simulations/plan_and_execute
4. **Concept Bottleneck Models**: Oikarinen et al., "Label-free Concept Bottleneck Models", ICLR 2023

---

## Appendix A: Example Output

See `data/umls_concepts.json` for example outputs generated by the system.

## Appendix B: Troubleshooting

**Common Issues:**

1. **API Key Errors**: Ensure `.umls_api_key` and `.openai_api_key` files exist
2. **Index Errors**: Usually indicates empty search results - system handles gracefully
3. **Import Errors**: Install dependencies with `pip install -r requirements.txt`
4. **Rate Limits**: Add delays between requests if hitting API limits

**Debug Mode:**
Use `--verbose` flag for detailed logging:
```bash
python umls_api.py --disease "Pneumonia" --verbose
```

