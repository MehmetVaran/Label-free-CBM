# pyright: reportMissingImports=false
"""LangChain-powered multi-agent system for UMLS concept discovery.

This module exposes a high-level orchestration layer that coordinates multiple
LangChain agents to search the Unified Medical Language System (UMLS) for a
given disease, collect relevant concepts, and expand them with related
terminology. It replaces the previous direct REST calls with a more flexible
agentic workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import requests
from pydantic import BaseModel, Field
from requests import Response, Session

try:
    from langchain.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langchain_experimental.plan_and_execute import (
        PlanAndExecute,
        load_agent_executor,
        load_chat_planner,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain dependencies are missing. Install the packages listed in "
        "`requirements.txt` (langchain, langchain-openai, langgraph) before using this module."
    ) from exc

LOGGER = logging.getLogger(__name__)

UMLS_API_BASE = "https://uts-ws.nlm.nih.gov/rest"
DEFAULT_KEY_PATH = Path(__file__).resolve().parent / ".umls_api_key"
DEFAULT_OPENAI_KEY_PATH = Path(__file__).resolve().parent / ".openai_api_key"
# Note: Since May 2022, UMLS API supports direct API key authentication.
# The old ticket-granting ticket system is deprecated.


class UMLSClient:
    """Lightweight client that manages UMLS authentication and requests.
    
    Uses direct API key authentication (supported since May 2022).
    See: https://documentation.uts.nlm.nih.gov/rest/home.html
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        session: Optional[Session] = None,
        api_base: str = UMLS_API_BASE,
        request_timeout: int = 30,
    ) -> None:
        key = api_key or self._read_key_from_disk(DEFAULT_KEY_PATH)
        if not key:
            raise ValueError(
                "A UMLS API key is required. Provide one explicitly or store it in "
                f"{DEFAULT_KEY_PATH}"
            )

        self.api_key = key.strip()
        self.session = session or Session()
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout

    @staticmethod
    def _read_key_from_disk(path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            LOGGER.debug("UMLS API key file not found at %s", path)
        return None

    def _authenticated_get(
        self, path: str, *, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an authenticated GET request using direct API key authentication."""
        query = dict(params or {})
        query["apiKey"] = self.api_key
        url = f"{self.api_base}/{path.lstrip('/')}"
        LOGGER.debug("GET %s (apiKey included)", url)
        response = self.session.get(url, params=query, timeout=self.request_timeout)
        self._raise_for_status(response, f"UMLS GET request to {url} failed")
        return response.json()

    def _authenticated_post(
        self,
        path: str,
        *,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated POST request using direct API key authentication."""
        query = dict(params or {})
        query["apiKey"] = self.api_key
        url = f"{self.api_base}/{path.lstrip('/')}"
        LOGGER.debug("POST %s (apiKey included)", url)
        response = self.session.post(
            url, params=query, data=data, timeout=self.request_timeout
        )
        self._raise_for_status(response, f"UMLS POST request to {url} failed")
        return response.json()

    @staticmethod
    def _raise_for_status(response: Response, message: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"{message}: {detail}") from exc

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def search_concepts(
        self,
        query: str,
        *,
        page_size: int = 10,
        search_type: str = "words",
        include_obsolete: bool = False,
        include_suppressible: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search for UMLS concepts matching the query string."""
        params = {
            "string": query,
            "pageSize": page_size,
            "searchType": search_type,
            "includeObsolete": str(include_obsolete).lower(),
            "includeSuppressible": str(include_suppressible).lower(),
        }
        payload = self._authenticated_get("search/current", params=params)
        results = payload.get("result", {}).get("results", [])
        LOGGER.debug("Search returned %d results for %s", len(results), query)
        return results

    def fetch_concept_details(
        self, cui: str, *, return_format: str = "json"
    ) -> Dict[str, Any]:
        """Fetch detailed information for a concept identified by its CUI."""
        path = f"content/current/CUI/{cui}"
        params = {"returnIdType": "sourceConcept", "format": return_format}
        payload = self._authenticated_get(path, params=params)
        return payload.get("result", payload)

    def fetch_related_concepts(
        self,
        cui: str,
        *,
        relation_types: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve concepts related to the given CUI."""
        params: Dict[str, Any] = {}
        if relation_types:
            params["includedRelation"] = ",".join(sorted(set(relation_types)))
        path = f"content/current/CUI/{cui}/relations"
        payload = self._authenticated_get(path, params=params)
        results = payload.get("result", [])
        LOGGER.debug(
            "Relations request returned %d items for CUI %s", len(results), cui
        )
        return results


# ------------------------------------------------------------------------- #
# LangChain tool definitions
# ------------------------------------------------------------------------- #


class SearchUMLSInput(BaseModel):
    query: str = Field(..., description="Disease or keyword to search in UMLS.")
    page_size: int = Field(
        10, ge=1, le=100, description="Maximum number of search results to return."
    )
    search_type: str = Field(
        "words",
        description="UMLS searchType parameter (e.g., 'exact', 'words', 'approximate').",
    )


class SearchUMLSTool(BaseTool):
    name: str = "search_umls_concepts"
    description: str = (
        "Use this tool to search the UMLS Metathesaurus for concepts related to a "
        "disease, symptom, or biomedical term. Returns JSON containing UMLS search "
        "results. IMPORTANT: The results array may be empty. Always check if results "
        "is empty before accessing elements by index (e.g., results[0])."
    )
    args_schema: Type[BaseModel] = SearchUMLSInput
    client: UMLSClient

    def _run(  # type: ignore[override]
        self, query: str, page_size: int = 10, search_type: str = "words"
    ) -> str:
        results = self.client.search_concepts(
            query, page_size=page_size, search_type=search_type
        )
        return json.dumps(results, ensure_ascii=False)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError("Async operation is not supported.")


class ConceptDetailInput(BaseModel):
    cui: str = Field(..., description="UMLS Concept Unique Identifier (CUI).")


class FetchConceptDetailsTool(BaseTool):
    name: str = "fetch_umls_concept_details"
    description: str = (
        "Retrieve detailed information for a specific UMLS concept by its CUI. Use "
        "this after searching to gather definitions, semantic types, and source data."
    )
    args_schema: Type[BaseModel] = ConceptDetailInput
    client: UMLSClient

    def _run(self, cui: str) -> str:  # type: ignore[override]
        details = self.client.fetch_concept_details(cui)
        return json.dumps(details, ensure_ascii=False)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError("Async operation is not supported.")


class RelatedConceptsInput(BaseModel):
    cui: str = Field(..., description="UMLS Concept Unique Identifier (CUI).")
    relation_types: Optional[List[str]] = Field(
        default=None,
        description="Optional list of relation types to include (e.g., 'RO', 'RB').",
    )


class FetchRelatedConceptsTool(BaseTool):
    name: str = "fetch_umls_related_concepts"
    description: str = (
        "Fetch concepts related to a specific UMLS CUI. Use this to expand a disease "
        "concept into associated findings, treatments, anatomical sites, etc. "
        "IMPORTANT: The results array may be empty. Always check if results is empty "
        "before accessing elements by index."
    )
    args_schema: Type[BaseModel] = RelatedConceptsInput
    client: UMLSClient

    def _run(  # type: ignore[override]
        self, cui: str, relation_types: Optional[List[str]] = None
    ) -> str:
        relations = self.client.fetch_related_concepts(cui, relation_types=relation_types)
        return json.dumps(relations, ensure_ascii=False)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError("Async operation is not supported.")


# ------------------------------------------------------------------------- #
# Multi-agent Orchestrator
# ------------------------------------------------------------------------- #


def _read_openai_key() -> Optional[str]:
    """Read OpenAI API key from .openai_api_key file or environment variable."""
    # Try reading from file first
    try:
        key = DEFAULT_OPENAI_KEY_PATH.read_text(encoding="utf-8").strip()
        if key:
            return key
    except FileNotFoundError:
        LOGGER.debug("OpenAI API key file not found at %s", DEFAULT_OPENAI_KEY_PATH)
    except Exception as e:
        LOGGER.warning("Error reading OpenAI API key file: %s", e)
    
    # Fall back to environment variable
    return os.environ.get("OPENAI_API_KEY")


def _default_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """Factory for a deterministic ChatOpenAI instance."""
    api_key = _read_openai_key()
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    """Try to recover the outermost JSON object from a free-form string."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


class LangChainConceptGenerator:
    """High-level interface for orchestrating UMLS concept discovery."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        llm: Optional[ChatOpenAI] = None,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.client = UMLSClient(api_key=api_key)
        self.llm = llm or _default_llm(model=llm_model, temperature=llm_temperature)
        self.verbose = verbose

        tools = [
            SearchUMLSTool(client=self.client),
            FetchConceptDetailsTool(client=self.client),
            FetchRelatedConceptsTool(client=self.client),
        ]

        planner = load_chat_planner(self.llm)
        executor = load_agent_executor(self.llm, tools, verbose=verbose)
        self.controller = PlanAndExecute(
            planner=planner,
            executor=executor,
            verbose=verbose,
        )

    def generate(
        self,
        disease_name: str,
        *,
        max_concepts: int = 5,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the multi-agent workflow for a single disease."""
        instructions = (
            "You are a team of biomedical knowledge mining agents. Work together to:\n"
            "1. Search UMLS for the provided disease to identify the most pertinent CUIs.\n"
            "   IMPORTANT: Always check if search results are empty before accessing list indices.\n"
            "   If search returns an empty list, try alternative search terms or return an empty candidate_concepts array.\n"
            f"2. Retrieve rich metadata (preferred name, semantic types, definitions) for up to {max_concepts} concepts.\n"
            "   IMPORTANT: Only access list elements if the list is not empty. Check len() before accessing [0], [1], etc.\n"
            "3. Expand each chosen concept by collecting clinically relevant related concepts.\n"
            "   IMPORTANT: If related concepts return an empty list, use an empty array in the output.\n"
            "4. Produce a JSON object with the following schema:\n"
            "{\n"
            '  "disease": string,\n'
            '  "candidate_concepts": [\n'
            "    {\n"
            '      "cui": string,\n'
            '      "name": string,\n'
            '      "score": float | null,\n'
            '      "semantic_types": [string],\n'
            '      "definition": string | null,\n'
            '      "sources": [string]\n'
            "    }\n"
            "  ],\n"
            '  "related_concepts": {\n'
            '    "CUI": [\n'
            "      {\n"
            '        "related_cui": string,\n'
            '        "name": string | null,\n'
            '        "relation_label": string | null,\n'
            '        "relation": string | null,\n'
            '        "additional_information": string | null\n'
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
            "If relation types are provided you must respect them.\n"
            "CRITICAL: Always check if lists/arrays are empty before accessing elements by index. "
            "Use len() to check list length first, or iterate safely."
        )

        prompt = (
            f"Disease: {disease_name}\n"
            f"Relation types constraint: {relation_types or 'None'}\n"
            f"{instructions}\n"
            "Return ONLY the JSON object — no extra narration."
        )

        if self.verbose:
            LOGGER.info("Starting concept generation workflow for '%s'", disease_name)

        try:
            raw_output = self.controller.invoke({"input": prompt})
        except (IndexError, KeyError) as exc:
            # Handle index errors that might occur during agent execution
            LOGGER.warning(
                "Agent execution encountered an index/key error for '%s': %s. "
                "This may indicate empty search results. Returning empty structure.",
                disease_name, exc
            )
            return {
                "disease": disease_name,
                "candidate_concepts": [],
                "related_concepts": {},
                "error": f"Index error during execution: {exc}"
            }
        except Exception as exc:
            # Catch any other exceptions during agent execution
            LOGGER.exception("Agent execution failed for '%s'", disease_name)
            return {
                "disease": disease_name,
                "candidate_concepts": [],
                "related_concepts": {},
                "error": str(exc)
            }

        if isinstance(raw_output, dict):
            # Depending on LangChain version this may already be a dict.
            candidate = raw_output.get("output", raw_output)
        else:
            candidate = raw_output

        parsed = _extract_json_blob(candidate if isinstance(candidate, str) else json.dumps(candidate))

        if not parsed:
            LOGGER.warning("Unable to parse agent output into JSON for '%s'", disease_name)
            return {
                "disease": disease_name,
                "candidate_concepts": [],
                "related_concepts": {},
                "error": "Unable to parse agent output into JSON"
            }

        parsed.setdefault("disease", disease_name)
        parsed["candidate_concepts"] = parsed.get("candidate_concepts", [])[:max_concepts]
        # Ensure related_concepts is a dict if not present
        parsed.setdefault("related_concepts", {})
        return parsed

    def generate_batch(
        self,
        diseases: Iterable[str],
        *,
        max_concepts: int = 5,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run the workflow for many diseases, collecting results."""
        results: Dict[str, Dict[str, Any]] = {}
        for disease in diseases:
            disease = disease.strip()
            if not disease:
                continue
            try:
                results[disease] = self.generate(
                    disease,
                    max_concepts=max_concepts,
                    relation_types=relation_types,
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.exception("Failed to generate concepts for %s", disease)
                results[disease] = {"error": str(exc)}
        return results


# ------------------------------------------------------------------------- #
# Legacy helper functions preserved for compatibility
# ------------------------------------------------------------------------- #


def load_diseases(filepath: str) -> List[str]:
    """Load disease names from a file, excluding the 'No Finding' token."""
    with open(filepath, "r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and line.strip().lower() != "no finding"
        ]


def save_concepts_to_json(concepts_dict: Dict[str, Any], out_path: str) -> None:
    """Persist the concept dictionary to disk."""
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(concepts_dict, handle, indent=2, ensure_ascii=False)


def generate_umls_concepts_for_diseases(
    diseases: Iterable[str],
    *,
    api_key: Optional[str] = None,
    max_concepts: int = 5,
    relation_types: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Convenience wrapper to run the LangChain multi-agent system."""
    generator = LangChainConceptGenerator(api_key=api_key, verbose=verbose)
    return generator.generate_batch(
        diseases,
        max_concepts=max_concepts,
        relation_types=relation_types,
    )


# ------------------------------------------------------------------------- #
# CLI entry point
# ------------------------------------------------------------------------- #


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UMLS concepts via a LangChain multi-agent workflow."
    )
    parser.add_argument(
        "--disease",
        type=str,
        help="Single disease name to process.",
    )
    parser.add_argument(
        "--disease-file",
        type=str,
        help="Path to a newline-delimited list of diseases.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/umls_concepts.json",
        help="Destination JSON file for generated concepts.",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=5,
        help="Maximum number of candidate concepts per disease.",
    )
    parser.add_argument(
        "--relation-types",
        type=str,
        nargs="*",
        help="Optional list of relation types to include (space separated).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging agent behaviour.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    if not args.disease and not args.disease_file:
        raise ValueError("Provide either --disease or --disease-file.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    diseases: List[str] = []
    if args.disease:
        diseases.append(args.disease)
    if args.disease_file:
        diseases.extend(load_diseases(args.disease_file))

    generator = LangChainConceptGenerator(verbose=args.verbose)
    results = generator.generate_batch(
        diseases,
        max_concepts=args.max_concepts,
        relation_types=args.relation_types,
    )
    save_concepts_to_json(results, args.out)
    LOGGER.info(
        "Saved UMLS concept generations for %d diseases to %s",
        len(results),
        args.out,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
