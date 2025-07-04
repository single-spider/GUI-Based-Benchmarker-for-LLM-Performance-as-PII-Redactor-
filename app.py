import streamlit as st
import json
import time
import requests
import google.generativeai as genai
from openai import OpenAI
import os
import logging
import re
from collections import defaultdict, deque
from typing import List, Dict, Any
import uuid

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================================================================
# Rate Limiter to prevent exceeding API quotas.
# ==================================================================
class RateLimiter:
    """A simple rate limiter to manage API call frequency."""
    def __init__(self, max_requests: int, per_seconds: int):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.timestamps = deque()

    def wait(self):
        """Blocks until a new request can be made, if necessary."""
        while True:
            current_time = time.monotonic()
            # Remove timestamps older than the time window
            while self.timestamps and self.timestamps[0] <= current_time - self.per_seconds:
                self.timestamps.popleft()

            if len(self.timestamps) < self.max_requests:
                break

            # Calculate wait time until the oldest request expires
            time_to_wait = self.timestamps[0] + self.per_seconds - current_time
            if time_to_wait > 0:
                log_msg = f"⏳ Rate limit reached. Waiting for {time_to_wait:.2f}s..."
                logger.info(log_msg)
                if 'logs' in st.session_state:
                    st.session_state.logs.append(log_msg)
                time.sleep(time_to_wait)
        
        self.timestamps.append(time.monotonic())


# ==================================================================
# API Client Classes
# ==================================================================
class GoogleGenAIClient:
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("Google API keys list cannot be empty.")
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.configure_client()

    def configure_client(self):
        try:
            genai.configure(api_key=self.api_keys[self.current_key_idx])
            logger.info(f"Google GenAI client configured with key index {self.current_key_idx}")
            return True
        except Exception as e:
            logger.error(f"Google API key error (key index {self.current_key_idx}): {str(e)}")
            return False

    def rotate_key(self):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        logger.info(f"Rotated Google API key to index {self.current_key_idx}")
        return self.configure_client()

    def get_models(self) -> List[str]:
        for _ in range(len(self.api_keys)):
            try:
                return [m.name for m in genai.list_models()
                        if 'generateContent' in m.supported_generation_methods]
            except Exception as e:
                logger.error(f"Google model fetch error (key index {self.current_key_idx}): {str(e)}")
                if "quota" in str(e).lower() or "api key" in str(e).lower():
                    self.rotate_key()
                    continue
                else: 
                    break
        logger.error("Failed to fetch Google models with all available keys.")
        return []

    def generate(self, model: str, prompt: str, system_instruction: str = "") -> str:
        initial_key_idx = self.current_key_idx
        for i in range(len(self.api_keys)):
            try:
                if "gemma" in model.lower() and system_instruction:
                    full_prompt = f"SYSTEM INSTRUCTIONS (YOU MUST FOLLOW):\n{system_instruction}\n\nUSER INPUT:\n{prompt}\n\nYOU MUST RESPOND WITH ONLY THE REQUESTED FORMAT, NO ADDITIONAL TEXT."
                    model_instance = genai.GenerativeModel(model)
                    response = model_instance.generate_content(full_prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                else:
                    model_instance = genai.GenerativeModel(model, system_instruction=system_instruction)
                    response = model_instance.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.0))
                return response.text
            except Exception as e:
                logger.error(f"Google generation error with model {model} (key index {self.current_key_idx}): {str(e)}")
                err_str = str(e).lower()
                if "quota" in err_str or "api key" in err_str or "resource has been exhausted" in err_str:
                    logger.info("Rotating key due to quota/key error.")
                    self.rotate_key()
                    if self.current_key_idx == initial_key_idx and i > 0:
                        break
                else:
                    break
        return "ERROR_GENERATION"


class OpenRouterClient:
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("OpenRouter API keys list cannot be empty.")
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.client = None
        self.configure_client()

    def configure_client(self):
        try:
            self.client = OpenAI(api_key=self.api_keys[self.current_key_idx], base_url="https://openrouter.ai/api/v1")
            logger.info(f"OpenRouter client configured with key index {self.current_key_idx}")
            return True
        except Exception as e:
            logger.error(f"OpenRouter API key error (key index {self.current_key_idx}): {str(e)}")
            return False

    def rotate_key(self):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        logger.info(f"Rotated OpenRouter API key to index {self.current_key_idx}")
        return self.configure_client()

    def get_models(self) -> List[str]:
        for _ in range(len(self.api_keys)):
            try:
                headers = {"Authorization": f"Bearer {self.api_keys[self.current_key_idx]}"}
                response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
                response.raise_for_status()
                return [model['id'] for model in response.json()['data']]
            except requests.exceptions.HTTPError as e:
                logger.error(f"OpenRouter model fetch error (key index {self.current_key_idx}): {str(e)}")
                if e.response.status_code in [401, 403, 429]:
                     self.rotate_key()
                     continue
                else: break
            except Exception as e:
                logger.error(f"Unexpected OpenRouter model fetch error: {str(e)}")
                break
        logger.error("Failed to fetch OpenRouter models with all available keys.")
        return []

    def generate(self, model: str, prompt: str, system_instruction: str = "", response_is_json: bool = True) -> str:
        if not self.client: return "ERROR_CLIENT_NOT_CONFIGURED"
        initial_key_idx = self.current_key_idx
        for i in range(len(self.api_keys)):
            try:
                messages = []
                if system_instruction: messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})

                completion_kwargs = {"model": model, "messages": messages, "temperature": 0.0}
                if response_is_json: completion_kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**completion_kwargs)
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenRouter generation error with model {model} (key index {self.current_key_idx}): {str(e)}")
                err_str = str(e).lower()
                if "quota" in err_str or "invalid api key" in err_str or "rate limit" in err_str or \
                   (hasattr(e, 'status_code') and e.status_code in [401, 403, 429]):
                    self.rotate_key()
                    if self.current_key_idx == initial_key_idx and i > 0: break
                else: break
        return "ERROR_GENERATION"

# ==================================================================
# Evaluation Pipeline Core 
# ==================================================================
class EvaluationPipeline:
    LEVEL1_SYSTEM_PROMPT = """You are an expert PII detection system for Indic languages. 
Your task is to analyze the provided text and tokens, then output EXACTLY in this format:

{
  "labels": [
    ["token1", "CATEGORY"],
    ["token2", "CATEGORY"],
    ...
  ]
}

Allowed categories (USE THESE EXACTLY):
- PERSON_NAME (Full or partial names of people, aliases, maiden names. E.g., "John Smith", "Rajesh Kumar", "Mary")
- PHONE_NUMBER (All phone/mobile numbers, typically including country codes or area codes. E.g., "+91-9876543210", "(212) 555-1234")
- AADHAAR_NUMBER (12-digit unique identification numbers issued in India. E.g., "1234 5678 9012")
- VEHICLE_NUMBER (License plate numbers, vehicle registration numbers, or Vehicle Identification Numbers (VIN). E.g., "MH01AB1234", "VIN: ABCDE1234FGHIJ5KL")
- ADDRESS (Physical locations, street names, building numbers, city, state, postal codes, geographical details. E.g., "123, Liberty Ave, New York, NY, 10001", "45, Saaket Colony, Delhi")
- DATE_OF_BIRTH (Specific date a person was born. E.g., "15/08/1985", "January 1, 1990")
- PLACE_OF_BIRTH (City, state, or country where a person was born. E.g., "Nagpur, Maharashtra", "New York")
- EMAIL_ADDRESS (Personal or work email addresses. E.g., "john.smith85@example.com", "info@company.co.in")
- PAN_NUMBER (Indian Permanent Account Number, a 10-character alphanumeric ID issued by the Income Tax Department. E.g., "ABCDE1234F", "PSRCY1234D")
- USERNAME (Login names, screen names, nicknames, or handles. E.g., "johndoe", "user123")
- SOCIAL_MEDIA_HANDLE (Publicly available usernames or handles on social media platforms. E.g., "@john_doe", "fb.com/profile.id")
- PASSPORT_NUMBER (Unique identification number assigned to a person's passport. E.g., "A12345678", "P1234567")
- VOTER_ID_NUMBER (Indian Electoral Photo Identity Card (EPIC) number, typically a 10-digit alphanumeric code. E.g., "NUO1234561", "YCV0164822")
- BANK_ACCOUNT_NUMBER (Bank account numbers. E.g., "9876543210", "123456789")
- ORDER_ID (A unique alphanumeric identifier for a transaction or order. E.g., "ORD-2023-56789", "PO#98765")
- BLOOD_TYPE (A person's blood group. E.g., "A+", "O-negative")
- MEDICAL_CONDITION (Specific health diagnoses, diseases, syndromes, or disorders. E.g., "diabetes", "chronic fatigue syndrome", "hypertension")
- HEALTH_INSURANCE_ID (Health insurance policy numbers. E.g., "ABC123456", "HI7890123")
- GENETIC_INFO (DNA data, genetic test results, biometric data like fingerprints or retinal scans used for identification. E.g., "BRCA1 gene mutation detected", "fingerprint scan data", "A-T-G-C sequence")
- IP_ADDRESS (Internet Protocol address assigned to a device. E.g., "192.168.1.1", "203.0.113.45")
- DEVICE_ID (Unique identifier for a computing device. E.g., "ABC123456", "IMEI: 987654321098765")
- MOTHERS_MAIDEN_NAME (The surname of a person's mother before marriage. E.g., "Sharma", "Johnson")
- RELIGION (A person's religious or philosophical beliefs. E.g., "Hindu", "Christian", "Islam")
- MARITAL_STATUS (A person's marital state. E.g., "Married", "Single", "Divorced")
- DRIVERS_LICENSE_NUMBER (Unique identification number on a driver's license. E.g., "DL1234567890", "MH0120230000123")
- EMPLOYMENT_INFO (Details about a person's employment, including company, position, salary, employment history, disciplinary actions, performance reviews. E.g., "Software Engineer at Google", "Former HR Manager", "annual salary of 15 lakhs")
- EDUCATION_INFO (Details about a person's educational background, including degrees, institutions, transcripts, school ID numbers. E.g., "B.Tech from IIT Delhi", "PhD in Physics", "Student ID: 12345")
- GSTIN (Indian Goods and Services Tax Identification Number, a 15-digit alphanumeric code. E.g., "27ABCDE1234F1Z5", "07AAAAA0000A1Z5")
- CREDIT_CARD_NUMBER (Credit or debit card numbers. E.g., "1234-5678-9012-3456", "4111222233334444")
- O (Not PII)

RULES:
1. Output MUST be valid JSON with ONLY the "labels" field.
2. MUST include EVERY token from the input exactly once.
3. Only use the specified categories (no variations).
4. Be conservative - when uncertain, use "O".
5. NEVER add explanations, notes, or other fields.
6. Preserve the EXACT token spelling/case from input.
7. DO NOT OUTPUT JSON IN CODE BLOCK OR ANY WRAPPERS, GIVE RAW OUTPUT ONLY."""

    ### MODIFIED ###
    SYNTHETIC_DATA_PROMPT = """You are an expert data generator for testing PII detection systems.
Your task is to create `{num_samples}` realistic text samples in `{language}`. Each sample MUST include fictional examples of the PII types listed below.

PII CATEGORIES TO USE:
{pii_definitions}

You MUST produce a single, valid JSON object as your output. The JSON object must have a single top-level key: "samples".
The value of "samples" MUST be a JSON list where EACH element is a distinct sample object with three keys: "text", "tokens", and "labels".
- "text": A single string containing the complete, natural-sounding sentence you generated for that sample.
- "tokens": A list of strings, representing the "text" tokenized by words and punctuation.
- "labels": A list of lists, where each inner list is `[token, LABEL]`. The LABEL must be one of the requested PII types or "O" for non-PII tokens.

RULES:
1. The output MUST be a single, raw JSON object. NO markdown wrappers (```json), explanations, or other text.
2. The final output must be `{{"samples": [...]}}`.
3. For each sample object inside the list, the `tokens` and `labels` lists MUST have the same length.
4. The labels MUST EXACTLY match the PII category names provided in the request.
5. Use "O" for any token that is not part of the requested PII types.
6. The generated data MUST be fictional and not contain any real personal information.
7. Create varied and realistic sentences for each of the `{num_samples}` samples.

Example Output (for a request of 2 samples in Hindi with PERSON_NAME, PHONE_NUMBER):
{{
  "samples": [
    {{
      "text": "मेरा नाम प्रिया है और मेरा फोन नंबर +91-9876543210 है।",
      "tokens": ["मेरा", "नाम", "प्रिया", "है", "और", "मेरा", "फोन", "नंबर", "+91-9876543210", "है", "।"],
      "labels": [["मेरा", "O"], ["नाम", "O"], ["प्रिया", "PERSON_NAME"], ["है", "O"], ["और", "O"], ["मेरा", "O"], ["फोन", "O"], ["नंबर", "O"], ["+91-9876543210", "PHONE_NUMBER"], ["है", "O"], ["।", "O"]]
    }},
    {{
      "text": "अगर आपको मदद चाहिए, तो आप राहुल को +91-8765432109 पर कॉल कर सकते हैं।",
      "tokens": ["अगर", "आपको", "मदद", "चाहिए", ",", "तो", "आप", "राहुल", "को", "+91-8765432109", "पर", "कॉल", "कर", "सकते", "हैं", "।"],
      "labels": [["अगर", "O"], ["आपको", "O"], ["मदद", "O"], ["चाहिए", "O"], [",", "O"], ["तो", "O"], ["आप", "O"], ["राहुल", "PERSON_NAME"], ["को", "O"], ["+91-8765432109", "PHONE_NUMBER"], ["पर", "O"], ["कॉल", "O"], ["कर", "O"], ["सकते", "O"], ["हैं", "O"], ["।", "O"]]
    }}
  ]
}}
"""

    LEVEL2_SYSTEM_PROMPT = """You are a strict PII redaction system. 
REDACT ALL PII using THESE EXACT replacements:
1. PERSON_NAME → [REDACTED_NAME]
2. PHONE_NUMBER → [REDACTED_PHONE]
3. AADHAAR_NUMBER → [REDACTED_AADHAAR]
4. VEHICLE_NUMBER → [REDACTED_VEHICLE]
5. ADDRESS → [REDACTED_ADDRESS]
6. DATE_OF_BIRTH → [REDACTED_DATE_OF_BIRTH]
7. PLACE_OF_BIRTH → [REDACTED_PLACE_OF_BIRTH]
8. EMAIL_ADDRESS → [REDACTED_EMAIL]
9. PAN_NUMBER → [REDACTED_PAN]
10. USERNAME → [REDACTED_USERNAME]
11. SOCIAL_MEDIA_HANDLE → [REDACTED_SOCIAL_MEDIA]
12. PASSPORT_NUMBER → [REDACTED_PASSPORT]
13. VOTER_ID_NUMBER → [REDACTED_VOTER_ID]
14. BANK_ACCOUNT_NUMBER → [REDACTED_BANK_ACCOUNT]
15. ORDER_ID → [REDACTED_ORDER_ID]
16. BLOOD_TYPE → [REDACTED_BLOOD_TYPE]
17. MEDICAL_CONDITION → [REDACTED_MEDICAL_CONDITION]
18. HEALTH_INSURANCE_ID → [REDACTED_HEALTH_INSURANCE_ID]
19. GENETIC_INFO → [REDACTED_GENETIC_INFO]
20. IP_ADDRESS → [REDACTED_IP_ADDRESS]
21. DEVICE_ID → [REDACTED_DEVICE_ID]
22. MOTHERS_MAIDEN_NAME → [REDACTED_MOTHERS_MAIDEN_NAME]
23. RELIGION → [REDACTED_RELIGION]
24. MARITAL_STATUS → [REDACTED_MARITAL_STATUS]
25. DRIVERS_LICENSE_NUMBER → [REDACTED_DRIVERS_LICENSE]
26. EMPLOYMENT_INFO → [REDACTED_EMPLOYMENT_INFO]
27. EDUCATION_INFO → [REDACTED_EDUCATION_INFO]
28. GSTIN → [REDACTED_GSTIN]
29. CREDIT_CARD_NUMBER → [REDACTED_CREDIT_CARD]

RULES:
1. Output ONLY the redacted text (no JSON, no explanations)
2. Preserve ALL original formatting, punctuation, spacing
3. Be CONSERVATIVE - when in doubt, redact
4. NEVER add any text beyond the redacted content"""

    LEVEL2_TESTER_PROMPT = """Evaluate if:
1. ALL PII was redacted (compare with original)
2. Used CORRECT redaction tags exactly (as per the defined list: e.g. PERSON_NAME -> [REDACTED_NAME])
3. Preserved ALL non-PII content exactly

Original: {original}
Redacted: {redacted}

Output ONLY one word: "Pass" if ALL criteria are met perfectly, or "Fail" if ANY criteria fail."""

    def __init__(self):
        self.google_client = None
        self.openrouter_client = None
        self.rate_limiter = RateLimiter(max_requests=25, per_seconds=60)
        self.pii_definitions = self._get_pii_definitions_from_prompt()
        self.pii_categories = sorted(list(self.pii_definitions.keys()))

    def _get_pii_definitions_from_prompt(self) -> Dict[str, str]:
        """Extracts PII categories and their descriptions from the Level 1 system prompt."""
        matches = re.findall(r"-\s([A-Z_]+)\s\((.*?)\)", self.LEVEL1_SYSTEM_PROMPT)
        return {name: desc for name, desc in matches}
    
    def initialize_clients(self, google_keys, openrouter_keys):
        if google_keys: self.google_client = GoogleGenAIClient(google_keys)
        if openrouter_keys: self.openrouter_client = OpenRouterClient(openrouter_keys)

    def parse_response(self, response: str, expected_tokens: List[str]) -> Dict:
        if response in ["ERROR_GENERATION", "ERROR_CLIENT_NOT_CONFIGURED"]:
            return {"labels": [[token, "O"] for token in expected_tokens]}
        try:
            result = json.loads(response)
            if "labels" in result and isinstance(result["labels"], list): return result
        except json.JSONDecodeError: pass
        
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL) or re.search(r'(\{.*?\})', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if "labels" in result and isinstance(result["labels"], list): return result
            except json.JSONDecodeError: pass
        
        logger.error(f"Could not parse response into valid label format. Defaulting all to 'O'. Response: {response[:200]}")
        return {"labels": [[token, "O"] for token in expected_tokens]}

    def run_level1(self, model: str, data: Dict) -> Dict:
        prompt_data = {"id": data["id"], "language": data["language"], "text": data["text"], "tokens": data["tokens"]}
        prompt_json = json.dumps(prompt_data, ensure_ascii=False)
        self.rate_limiter.wait()
        
        if model.startswith("models/"):
            response_text = self.google_client.generate(model, prompt_json, self.LEVEL1_SYSTEM_PROMPT) if self.google_client else "ERROR_CLIENT_NOT_CONFIGURED"
        else:
            response_text = self.openrouter_client.generate(model, prompt_json, self.LEVEL1_SYSTEM_PROMPT, response_is_json=True) if self.openrouter_client else "ERROR_CLIENT_NOT_CONFIGURED"

        parsed = self.parse_response(response_text, data["tokens"])
        
        # Robustly create the final label list, handling missing tokens in the response
        llm_labels_map = defaultdict(deque)
        for token_text, tag in parsed.get("labels", []):
            llm_labels_map[token_text].append(tag)
        
        final_labels = []
        for token_in_data in data["tokens"]:
            if llm_labels_map[token_in_data]:
                final_labels.append(llm_labels_map[token_in_data].popleft())
            else:
                final_labels.append("O")
        
        if len(final_labels) != len(data["tokens"]):
            return {"labels": ["O"] * len(data["tokens"])}
        return {"labels": final_labels}

    ### MODIFIED ###
    def generate_synthetic_data(self, generator_model: str, language: str, pii_types: List[str], custom_pii_defs: Dict[str, str], num_samples: int) -> Dict[str, Any]:
        """Generates a batch of synthetic gold-standard data using a single LLM call."""
        all_pii_defs = self.pii_definitions.copy()
        all_pii_defs.update(custom_pii_defs)
        
        pii_definitions_str = "\n".join(f"- {name} ({all_pii_defs[name]})" for name in pii_types)
        
        system_prompt = self.SYNTHETIC_DATA_PROMPT.format(
            num_samples=num_samples,
            language=language,
            pii_definitions=pii_definitions_str
        )
        user_prompt = f"Please generate {num_samples} samples now."
        
        self.rate_limiter.wait()
        
        response_text = ""
        if generator_model.startswith("models/"):
            if not self.google_client: return {}
            response_text = self.google_client.generate(generator_model, user_prompt, system_prompt)
        else:
            if not self.openrouter_client: return {}
            response_text = self.openrouter_client.generate(generator_model, user_prompt, system_prompt, response_is_json=True)

        if response_text in ["ERROR_GENERATION", "ERROR_CLIENT_NOT_CONFIGURED"]:
            st.session_state.logs.append(f"❌ Generation failed entirely. Check API keys and model status.")
            return {}

        try:
            data = json.loads(response_text)
            if "samples" in data and isinstance(data["samples"], list):
                synthetic_data = {}
                for sample in data["samples"]:
                    if all(k in sample for k in ["text", "tokens", "labels"]) and len(sample["tokens"]) == len(sample["labels"]):
                        sample_id = f"synth_{language.lower()}_{uuid.uuid4().hex[:8]}"
                        sample['id'] = sample_id
                        sample['language'] = language
                        synthetic_data[sample_id] = sample
                    else:
                        st.session_state.logs.append(f"⚠️ A generated sample has invalid structure and was discarded.")
                return synthetic_data
            else:
                st.session_state.logs.append(f"⚠️ LLM output did not contain a 'samples' list. Response: {response_text[:200]}")
                return {}
        except json.JSONDecodeError:
            st.session_state.logs.append(f"⚠️ Failed to parse JSON from LLM response. Response: {response_text[:200]}")
            return {}

    def run_level2(self, model: str, text: str) -> str:
        self.rate_limiter.wait()
        if model.startswith("models/"):
            return self.google_client.generate(model, text, self.LEVEL2_SYSTEM_PROMPT) if self.google_client else "ERROR_CLIENT_NOT_INITIALIZED"
        else:
            return self.openrouter_client.generate(model, text, self.LEVEL2_SYSTEM_PROMPT, response_is_json=False) if self.openrouter_client else "ERROR_CLIENT_NOT_INITIALIZED"

    def evaluate_level2(self, tester_model: str, original: str, redacted: str) -> str:
        prompt = self.LEVEL2_TESTER_PROMPT.format(original=original, redacted=redacted)
        self.rate_limiter.wait()
        
        if tester_model.startswith("models/"):
            result = self.google_client.generate(tester_model, prompt) if self.google_client else "Fail"
        else:
            result = self.openrouter_client.generate(tester_model, prompt, response_is_json=False) if self.openrouter_client else "Fail"
        return "Pass" if "pass" in result.lower() else "Fail"

    ### MODIFIED ###
    def calculate_metrics(self, gold_data: Dict[str, Any], predictions: List[Dict]) -> Dict:
        """Calculates precision, recall, and F1 scores, now with the bug fixed."""
        metrics = defaultdict(lambda: defaultdict(float))
        metrics["overall"] = {"correct": 0, "total": 0}
        
        all_pii_categories = set(self.pii_categories)

        for pred_item in predictions:
            sample_id = pred_item["id"]
            if sample_id not in gold_data: continue

            # BUG FIX: Extract only the label string from the gold data.
            # Gold labels are in the format [['token', 'LABEL']], so we take the second element.
            gold_labels = [label[1] for label in gold_data[sample_id].get("labels", [])]
            predicted_labels = pred_item["predicted"]
            
            if len(gold_labels) != len(predicted_labels): continue

            for g_label, p_label in zip(gold_labels, predicted_labels):
                metrics["overall"]["total"] += 1
                if g_label == p_label: metrics["overall"]["correct"] += 1
                
                # Add any unforeseen labels from gold or predictions to the set for comprehensive metrics
                if g_label != "O" and g_label not in all_pii_categories: all_pii_categories.add(g_label)
                if p_label != "O" and p_label not in all_pii_categories: all_pii_categories.add(p_label)

        # Initialize counts for all known categories
        for cat in all_pii_categories:
            metrics["true_positives"][cat] = 0
            metrics["false_positives"][cat] = 0
            metrics["false_negatives"][cat] = 0
            metrics["support"][cat] = 0

        # Second pass to calculate TP, FP, FN
        for pred_item in predictions:
            sample_id = pred_item["id"]
            if sample_id not in gold_data: continue
            
            gold_labels = [label[1] for label in gold_data[sample_id].get("labels", [])]
            predicted_labels = pred_item["predicted"]
            if len(gold_labels) != len(predicted_labels): continue
            
            for g_label, p_label in zip(gold_labels, predicted_labels):
                if g_label != "O":
                    metrics["support"][g_label] += 1
                    if p_label == g_label:
                        metrics["true_positives"][g_label] += 1
                    else:
                        metrics["false_negatives"][g_label] += 1
                if p_label != "O" and g_label != p_label:
                    metrics["false_positives"][p_label] += 1
        
        for cat in all_pii_categories:
            if cat == "O": continue
            tp, fp, fn = metrics["true_positives"][cat], metrics["false_positives"][cat], metrics["false_negatives"][cat]
            metrics["precision"][cat] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            metrics["recall"][cat] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics["f1"][cat] = 2 * (metrics["precision"][cat] * metrics["recall"][cat]) / (metrics["precision"][cat] + metrics["recall"][cat]) if (metrics["precision"][cat] + metrics["recall"][cat]) > 0 else 0.0

        total_tp = sum(metrics["true_positives"].values())
        total_fp = sum(metrics["false_positives"].values())
        total_fn = sum(metrics["false_negatives"].values())

        metrics["overall"]["accuracy"] = metrics["overall"]["correct"] / metrics["overall"]["total"] if metrics["overall"]["total"] > 0 else 0.0
        metrics["overall"]["micro_precision"] = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        metrics["overall"]["micro_recall"] = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        metrics["overall"]["micro_f1"] = 2 * (metrics["overall"]["micro_precision"] * metrics["overall"]["micro_recall"]) / (metrics["overall"]["micro_precision"] + metrics["overall"]["micro_recall"]) if (metrics["overall"]["micro_precision"] + metrics["overall"]["micro_recall"]) > 0 else 0.0
        
        return json.loads(json.dumps(metrics, default=dict)) # Convert defaultdicts for clean output

# ==================================================================
# Streamlit UI Implementation
# ==================================================================
def main():
    st.set_page_config(page_title="Indic PII Evaluation Pipeline", layout="wide")
    st.title("Multi-Level LLM Evaluation Pipeline for Indic PII")

    if 'pipeline' not in st.session_state: st.session_state.pipeline = EvaluationPipeline()
    if 'logs' not in st.session_state: st.session_state.logs = []
    if 'results' not in st.session_state: st.session_state.results = defaultdict(dict)
    if 'file_metrics' not in st.session_state: st.session_state.file_metrics = defaultdict(dict)
    if 'cumulative_metrics' not in st.session_state: st.session_state.cumulative_metrics = {}
    if 'active_models' not in st.session_state: st.session_state.active_models = {"Level1": [], "Level2": []}
    if 'gold_data_cache' not in st.session_state: st.session_state.gold_data_cache = {}
    if 'stop_requested' not in st.session_state: st.session_state.stop_requested = False

    with st.expander("API Configuration", expanded=not (st.session_state.pipeline.google_client or st.session_state.pipeline.openrouter_client)):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Google Gen AI")
            google_keys_str = st.text_area("Google API Keys (one per line)", height=100, key="google_keys_input")
        with col2:
            st.subheader("OpenRouter")
            openrouter_keys_str = st.text_area("OpenRouter API Keys (one per line)", height=100, key="openrouter_keys_input")
        if st.button("Initialize Clients"):
            google_keys = [k.strip() for k in google_keys_str.split('\n') if k.strip()]
            openrouter_keys = [k.strip() for k in openrouter_keys_str.split('\n') if k.strip()]
            if google_keys or openrouter_keys:
                st.session_state.pipeline.initialize_clients(google_keys, openrouter_keys)
                st.success("Clients initialized.")
                st.rerun()
            else: st.warning("Please provide at least one API key.")

    all_available_models = []
    with st.expander("Model Configuration"):
        google_model_list, openrouter_model_list = [], []
        if st.session_state.pipeline.google_client:
            with st.spinner("Fetching Google Models..."): google_model_list = st.session_state.pipeline.google_client.get_models()
        if st.session_state.pipeline.openrouter_client:
            with st.spinner("Fetching OpenRouter Models..."): openrouter_model_list = st.session_state.pipeline.openrouter_client.get_models()
        all_available_models = ["(Select Model)"] + google_model_list + openrouter_model_list
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Level 1: PII Detection")
            st.session_state.active_models["Level1"] = st.multiselect("Models (L1)", google_model_list + openrouter_model_list, key="l1_select")
        with col2:
            st.subheader("Level 2: Redaction")
            st.session_state.active_models["Level2"] = st.multiselect("Models (L2)", google_model_list + openrouter_model_list, key="l2_select")
        with col3:
            st.subheader("Level 2 Evaluation")
            tester_model = st.selectbox("Tester Model", all_available_models, key="tester_model_select")

    with st.expander("Evaluation Data Management", expanded=True):
        data_tabs = st.tabs(["Generate Synthetic Data", "Upload Gold Data", "Manage Data Cache"])
        
        ### MODIFIED ###
        with data_tabs[0]: # Generate Synthetic Data
            st.info("This module uses an LLM to generate a batch of test data in a single, efficient API call.")
            gen_col1, gen_col2 = st.columns(2)
            with gen_col1:
                generator_model = st.selectbox("Generator Model", all_available_models, key="generator_model_select")
                gen_lang = st.text_input("Language to Generate", "Hindi")
                gen_num_samples = st.number_input("Number of Samples to Generate", 1, 100, 10)
            with gen_col2:
                 pii_options = st.session_state.pipeline.pii_categories
                 gen_pii_types = st.multiselect(
                     "Standard PII Types to Include",
                     options=pii_options,
                     default=["PERSON_NAME", "PHONE_NUMBER", "ADDRESS"]
                 )
            
            st.subheader("Define Custom PII Types (Optional)")
            custom_pii_input = st.text_area("Add new types, one per line, in `NAME: Description` format.", 
                                            placeholder="STUDENT_ID: A unique ID assigned to a student by an institution.\nFAVORITE_COLOR: A person's preferred color.", height=100)

            if st.button("✨ Generate Synthetic Data", use_container_width=True):
                if generator_model == "(Select Model)" or not gen_lang or not gen_pii_types:
                    st.warning("Please select a generator model, specify a language, and choose at least one PII type.")
                else:
                    custom_pii_defs = {}
                    custom_pii_types_list = []
                    for line in custom_pii_input.split('\n'):
                        if ':' in line:
                            name, desc = line.split(':', 1)
                            name = name.strip().upper().replace(" ", "_")
                            if name:
                                custom_pii_defs[name] = desc.strip()
                                custom_pii_types_list.append(name)
                    
                    all_pii_to_gen = gen_pii_types + custom_pii_types_list
                    
                    with st.spinner(f"Generating {gen_num_samples} samples for {gen_lang}..."):
                        st.session_state.logs.append(f"🔥 Starting synthetic data generation for {gen_lang} using {generator_model}.")
                        generated_data = st.session_state.pipeline.generate_synthetic_data(
                            generator_model, gen_lang, all_pii_to_gen, custom_pii_defs, gen_num_samples
                        )
                    if generated_data:
                        file_name = f"Synthetic_{gen_lang}_{len(generated_data)}samples.json"
                        st.session_state.gold_data_cache[file_name] = generated_data
                        st.success(f"Generated {len(generated_data)} samples and added as '{file_name}' to the data cache.")
                        st.rerun()
                    else:
                        st.error("Data generation failed. Check the logs for more details.")

        with data_tabs[1]: # Upload Gold Data
            uploaded_files = st.file_uploader("Upload Gold Standard Data (JSON files)", type=["json"], accept_multiple_files=True)
            if st.button("Load Uploaded Files"):
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        try:
                            st.session_state.gold_data_cache[uploaded_file.name] = json.load(uploaded_file)
                        except Exception as e:
                            st.error(f"Error loading {uploaded_file.name}: {e}")
                    st.success(f"Loaded {len(uploaded_files)} file(s) into cache.")
                    st.rerun()

        ### MODIFIED ###
        with data_tabs[2]: # Manage Data Cache
            st.subheader("Current Data Cache")
            if not st.session_state.gold_data_cache:
                st.write("No data loaded or generated yet.")
            else:
                for name, data in st.session_state.gold_data_cache.items():
                    cols = st.columns([3, 1, 1])
                    cols[0].write(f"📄 **{name}** ({len(data)} samples)")
                    if name.startswith("Synthetic"):
                        cols[1].download_button(
                            "Download", 
                            data=json.dumps(data, indent=2, ensure_ascii=False),
                            file_name=name,
                            mime="application/json",
                            key=f"dl_{name}"
                        )
            
            sample_size = st.number_input("Sample Size for Evaluation (per file)", min_value=1, value=10, max_value=1000)

    # ... The rest of the pipeline controls, execution, and results display logic remains largely the same ...
    # It will now correctly use the bug-fixed `calculate_metrics` function.

    st.subheader("Pipeline Controls")
    control_cols = st.columns([1, 1, 1, 2])
    run_level1_button = control_cols[0].button("▶️ Run Level 1", use_container_width=True)
    run_level2_button = control_cols[1].button("▶️ Run Level 2", use_container_width=True)
    
    def request_stop():
        st.session_state.stop_requested = True
        st.warning("Stop requested. The current process will halt after this sample.")
    control_cols[2].button("⏹️ Request Stop", on_click=request_stop, use_container_width=True)

    if control_cols[3].button("🗑️ Clear All Results & Logs", type="secondary", use_container_width=True):
        keys_to_clear = ['logs', 'results', 'file_metrics', 'cumulative_metrics', 'gold_data_cache', 'stop_requested']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("All results, logs, and cached data have been cleared.")
        st.rerun()

    st.subheader("Execution Logs")
    log_display_area = st.empty()
    def update_logs_display():
        log_text = "\n".join(st.session_state.get('logs', [])[-20:])
        log_display_area.text_area("Logs", value=log_text, height=200, label_visibility="collapsed", disabled=True)
    update_logs_display()

    if run_level1_button:
        if not st.session_state.active_models["Level1"] or not st.session_state.gold_data_cache:
            st.warning("Please select models for Level 1 and ensure data is loaded or generated.")
        else:
            # Main processing loop... (this part is unchanged and should work with the fixed metrics function)
            st.session_state.stop_requested = False
            all_gold_data_combined = {k: v for d in st.session_state.gold_data_cache.values() for k, v in d.items()}

            for model_name in st.session_state.active_models["Level1"]:
                if st.session_state.get('stop_requested', False): break
                st.session_state.logs.append(f"🚀 Starting Level 1 for {model_name}...")
                with st.spinner(f"Running Level 1: {model_name}..."):
                    for file_name, gold_data in st.session_state.gold_data_cache.items():
                        if st.session_state.get('stop_requested', False): break
                        st.session_state.logs.append(f"Processing {file_name} for {model_name}...")
                        
                        test_samples = list(gold_data.values())[:sample_size]
                        file_predictions = []
                        for i, sample in enumerate(test_samples):
                            if st.session_state.get('stop_requested', False): break
                            st.session_state.logs.append(f"Model: {model_name}, File: {file_name}, Sample: {i+1}/{len(test_samples)}")
                            update_logs_display()
                            
                            result = st.session_state.pipeline.run_level1(model_name, sample)
                            file_predictions.append({"id": sample["id"], "predicted": result["labels"]})
                        
                        if file_predictions:
                            st.session_state.results[model_name][file_name] = file_predictions
                            metrics = st.session_state.pipeline.calculate_metrics(gold_data, file_predictions)
                            st.session_state.file_metrics[model_name][file_name] = metrics
                            st.session_state.logs.append(f"✅ Progress saved for {model_name} on {file_name}. Accuracy: {metrics['overall']['accuracy']:.2%}")

                    if model_name in st.session_state.results:
                        all_model_preds = [item for file_res in st.session_state.results[model_name].values() for item in file_res]
                        if all_model_preds:
                             st.session_state.cumulative_metrics[model_name] = st.session_state.pipeline.calculate_metrics(all_gold_data_combined, all_model_preds)
                             st.session_state.logs.append(f"📊 Cumulative metrics updated for {model_name}.")
            st.rerun()

    if run_level2_button:
        if not st.session_state.active_models["Level2"] or not st.session_state.gold_data_cache or tester_model == "(Select Model)":
            st.warning("Please select models for Level 2, a tester model, and ensure data is loaded or generated.")
        else:
            # Main processing loop... (unchanged)
            st.session_state.stop_requested = False
            for model_name in st.session_state.active_models["Level2"]:
                if st.session_state.get('stop_requested', False): break
                st.session_state.logs.append(f"🚀 Starting Level 2 for {model_name}...")
                with st.spinner(f"Running Level 2: {model_name}..."):
                    for file_name, gold_data in st.session_state.gold_data_cache.items():
                        if st.session_state.get('stop_requested', False): break
                        st.session_state.logs.append(f"Processing {file_name} for {model_name} (Level 2)...")
                        
                        test_samples = list(gold_data.values())[:sample_size]
                        file_l2_results = []
                        for i, sample in enumerate(test_samples):
                            if st.session_state.get('stop_requested', False): break
                            st.session_state.logs.append(f"Model L2: {model_name}, File: {file_name}, Sample: {i+1}/{len(test_samples)}")
                            update_logs_display()

                            redacted = st.session_state.pipeline.run_level2(model_name, sample["text"])
                            eval_result = st.session_state.pipeline.evaluate_level2(tester_model, sample["text"], redacted)
                            file_l2_results.append({"id": sample["id"], "original": sample["text"], "redacted": redacted, "evaluation": eval_result})
                        
                        if file_l2_results:
                            l2_file_key = f"{file_name}_L2"
                            st.session_state.results[model_name][l2_file_key] = file_l2_results
                            pass_count = sum(1 for r in file_l2_results if r["evaluation"] == "Pass")
                            pass_rate = pass_count / len(file_l2_results) if file_l2_results else 0.0
                            st.session_state.file_metrics[model_name][f"{file_name}_L2_PassRate"] = {"pass_rate": pass_rate}
                            st.session_state.logs.append(f"✅ L2 Progress saved. Pass Rate: {pass_rate:.2%}")
                
                    l2_cum_key = f"{model_name}_L2_PassRate"
                    all_l2_res = [item for key, res_list in st.session_state.results.get(model_name, {}).items() if key.endswith("_L2") for item in res_list]
                    if all_l2_res:
                        total_passed = sum(1 for r in all_l2_res if r["evaluation"] == "Pass")
                        st.session_state.cumulative_metrics[l2_cum_key] = {"pass_rate": total_passed / len(all_l2_res)}
            st.rerun()
    
    # Results display... (unchanged, will work correctly)
    if st.session_state.results or st.session_state.file_metrics or st.session_state.cumulative_metrics:
        st.subheader("📊 Evaluation Results & Metrics")
        models_with_data = set(st.session_state.results.keys()) | set(st.session_state.file_metrics.keys()) | set(st.session_state.cumulative_metrics.keys())
        displayable_model_names = sorted(list(set([m.split("_L2_PassRate")[0] for m in models_with_data])))
        
        if displayable_model_names:
            selected_display_model = st.selectbox("Select Model to View Detailed Metrics", options=displayable_model_names)
            if selected_display_model:
                tab_titles = ["Cumulative Metrics", "Per-File Metrics"]
                if selected_display_model in st.session_state.results: tab_titles.append("Raw Results")
                tabs = st.tabs(tab_titles)

                with tabs[0]: # Cumulative
                    st.write(f"#### Cumulative Metrics for: `{selected_display_model}` (All Files Combined)")
                    if selected_display_model in st.session_state.cumulative_metrics:
                        metrics = st.session_state.cumulative_metrics[selected_display_model]
                        st.markdown("##### Level 1 Cumulative")
                        cols = st.columns(4)
                        cols[0].metric("Accuracy", f"{metrics['overall']['accuracy']:.2%}")
                        cols[1].metric("Micro Precision", f"{metrics['overall']['micro_precision']:.2%}")
                        cols[2].metric("Micro Recall", f"{metrics['overall']['micro_recall']:.2%}")
                        cols[3].metric("Micro F1", f"{metrics['overall']['micro_f1']:.2%}")
                        cat_df_data = [{"Category": cat, "Precision": f"{metrics['precision'].get(cat, 0):.2%}", "Recall": f"{metrics['recall'].get(cat, 0):.2%}", "F1": f"{metrics['f1'].get(cat, 0):.2%}", "Support": metrics["support"][cat]} for cat in sorted(metrics.get("support", {}).keys()) if metrics["support"][cat] > 0]
                        if cat_df_data: st.dataframe(cat_df_data, use_container_width=True)
                    l2_cum_key = selected_display_model + "_L2_PassRate"
                    if l2_cum_key in st.session_state.cumulative_metrics:
                        st.markdown("##### Level 2 Cumulative")
                        st.metric(label="Overall Redaction Pass Rate", value=f"{st.session_state.cumulative_metrics[l2_cum_key]['pass_rate']:.2%}")

                with tabs[1]: # Per-File
                    st.write(f"#### Per-File Metrics for: `{selected_display_model}`")
                    # L1
                    for file_name, metrics in st.session_state.file_metrics.get(selected_display_model, {}).items():
                        if "_L2_PassRate" in file_name: continue
                        with st.expander(f"📄 File: `{file_name}` (Level 1)"):
                            cols = st.columns(4)
                            cols[0].metric("Accuracy", f"{metrics['overall']['accuracy']:.2%}")
                            cols[1].metric("Micro Precision", f"{metrics['overall']['micro_precision']:.2%}")
                            cols[2].metric("Micro Recall", f"{metrics['overall']['micro_recall']:.2%}")
                            cols[3].metric("Micro F1", f"{metrics['overall']['micro_f1']:.2%}")
                            cat_df_data = [{"Category": cat, "Precision": f"{metrics['precision'].get(cat, 0):.2%}", "Recall": f"{metrics['recall'].get(cat, 0):.2%}", "F1": f"{metrics['f1'].get(cat, 0):.2%}", "Support": metrics["support"][cat]} for cat in sorted(metrics.get("support", {}).keys()) if metrics["support"][cat] > 0]
                            if cat_df_data: st.dataframe(cat_df_data, use_container_width=True)
                    # L2
                    for file_key, metric_val in st.session_state.file_metrics.get(selected_display_model, {}).items():
                         if "_L2_PassRate" in file_key and "pass_rate" in metric_val:
                            with st.expander(f"📄 File: `{file_key.replace('_L2_PassRate', '')}` (Level 2)"):
                                st.metric(label="Redaction Pass Rate", value=f"{metric_val['pass_rate']:.2%}")

                if "Raw Results" in tab_titles:
                    with tabs[2]:
                        st.write(f"#### Raw Prediction Results for: `{selected_display_model}`")
                        for file_key, file_data in st.session_state.results.get(selected_display_model, {}).items():
                            with st.expander(f"📄 Output: `{file_key}`"):
                                st.json(file_data)
        
        st.subheader("📤 Export Metrics")
        if displayable_model_names:
            export_model_select = st.selectbox("Select Model to Export Metrics For", options=displayable_model_names, key="export_model_selector")
            export_data = {
                "model_name": export_model_select,
                "per_file_metrics": st.session_state.file_metrics.get(export_model_select, {}),
                "cumulative_metrics": { "Level1": st.session_state.cumulative_metrics.get(export_model_select), "Level2_PassRate": st.session_state.cumulative_metrics.get(f"{export_model_select}_L2_PassRate")}
            }
            st.download_button(label=f"Download Metrics for {export_model_select}", data=json.dumps(export_data, indent=2), file_name=f"metrics_{export_model_select.replace('/', '_')}.json", mime="application/json", use_container_width=True)

if __name__ == "__main__":
    main()
