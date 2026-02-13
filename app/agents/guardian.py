"""
Guardian Agent - Safety and Policy Compliance Gate

Reviews outputs for safety, accuracy, and policy compliance before
delivering to users. Acts as the final quality gate in the pipeline.
"""

import re
from urllib.parse import urlparse

from app.agents.base_agent import create_agent
from app.model_factory import ModelFactory
from google.adk.agents import LlmAgent


# PII patterns (reduced set)
_PII_PATTERNS: list[tuple[str, str, str]] = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email", "Email address"),
    (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "ssn", "Possible SSN"),
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone", "Phone number"),
]

# Sensitive topic keywords (reduced)
_SENSITIVE_KEYWORDS: dict[str, list[str]] = {
    "medical": ["diagnosis", "treatment", "medication", "dosage", "symptoms"],
    "financial": ["invest", "stock", "guaranteed return", "financial advice"],
    "legal": ["legal advice", "sue", "lawsuit", "liable"],
}

# Source tiers (reduced)
_TRUSTED_DOMAINS: set[str] = {
    "arxiv.org", "nature.com", "science.org", "ieee.org", "nih.gov",
    "cdc.gov", "who.int", "github.com", "wikipedia.org", "reuters.com",
}

# Disclaimers
_DISCLAIMERS: dict[str, str] = {
    "medical": "\n\n> **Medical Disclaimer:** For educational purposes only. Not a substitute for professional medical advice.",
    "financial": "\n\n> **Financial Disclaimer:** Not financial advice. Consult a qualified advisor.",
    "legal": "\n\n> **Legal Disclaimer:** For general reference only. Consult a qualified attorney.",
    "general": "\n\n---\n*Information provided for reference. Verify from primary sources.*",
}


# Tool 1: check_content_safety
def check_content_safety(content: str) -> dict:
    """Scan content for PII and sensitive topics."""
    issues = []
    pii_found = []
    
    # Check PII
    for pattern, pii_type, desc in _PII_PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            pii_found.extend([f"{pii_type}:{m[:20]}..." for m in matches[:3]])
            issues.append(f"{desc} detected ({len(matches)} found)")
    
    # Check sensitive topics
    content_lower = content.lower()
    topic_flags = []
    for topic, keywords in _SENSITIVE_KEYWORDS.items():
        if any(kw in content_lower for kw in keywords):
            topic_flags.append(topic)
    
    # Check for misinformation signals
    misinfo_signals = []
    vague_phrases = ["studies show", "research says", "experts agree", "it is known"]
    for phrase in vague_phrases:
        if phrase in content_lower:
            misinfo_signals.append(f"Vague citation: '{phrase}'")
    
    return {
        "pii_detected": len(pii_found) > 0,
        "pii_summary": pii_found[:5],
        "sensitive_topics": topic_flags,
        "misinformation_signals": misinfo_signals[:3],
        "needs_review": len(pii_found) > 0 or len(topic_flags) > 0 or len(misinfo_signals) > 3,
        "risk_level": "high" if len(pii_found) > 0 else ("medium" if topic_flags else "low"),
    }


# Tool 2: validate_sources
def validate_sources(urls: list[str]) -> dict:
    """Validate source credibility."""
    if not urls:
        return {"trusted_count": 0, "untrusted_count": 0, "overall_quality": "none"}
    
    trusted = 0
    untrusted = []
    
    for url in urls:
        domain = urlparse(url).netloc.lower()
        # Remove www. prefix for matching
        if domain.startswith("www."):
            domain = domain[4:]
        
        is_trusted = any(trusted_domain in domain for trusted_domain in _TRUSTED_DOMAINS)
        if is_trusted:
            trusted += 1
        else:
            untrusted.append(domain)
    
    trusted_pct = (trusted / len(urls)) * 100 if urls else 0
    
    return {
        "total_urls": len(urls),
        "trusted_count": trusted,
        "untrusted_count": len(untrusted),
        "trusted_percentage": round(trusted_pct, 1),
        "overall_quality": "strong" if trusted_pct >= 60 else ("moderate" if trusted_pct >= 30 else "weak"),
    }


# Tool 3: format_safe_response
def format_safe_response(content: str, disclaimer_type: str = "auto", redact_pii: bool = True) -> dict:
    """Format content with disclaimers and optional PII redaction."""
    formatted = content
    pii_redacted = 0
    
    # Redact PII
    if redact_pii:
        for pattern, pii_type, _ in _PII_PATTERNS:
            matches = re.findall(pattern, formatted)
            if matches:
                pii_redacted += len(matches)
                formatted = re.sub(pattern, f"[REDACTED]", formatted)
    
    # Add disclaimer
    content_lower = content.lower()
    if disclaimer_type == "auto":
        if any(w in content_lower for w in _SENSITIVE_KEYWORDS["medical"]):
            disclaimer_type = "medical"
        elif any(w in content_lower for w in _SENSITIVE_KEYWORDS["financial"]):
            disclaimer_type = "financial"
        elif any(w in content_lower for w in _SENSITIVE_KEYWORDS["legal"]):
            disclaimer_type = "legal"
        elif len(content) > 200:
            disclaimer_type = "general"
        else:
            disclaimer_type = "none"
    
    disclaimer_added = "none"
    if disclaimer_type != "none" and disclaimer_type in _DISCLAIMERS:
        formatted += _DISCLAIMERS[disclaimer_type]
        disclaimer_added = disclaimer_type
    
    return {
        "formatted_content": formatted,
        "disclaimer_added": disclaimer_added,
        "pii_redacted_count": pii_redacted,
    }


# Guardian Agent Definition
GUARDIAN_INSTRUCTION = """You are the Guardian Agent - the final safety gate.

Every response passes through you before reaching the user.

Review Process:
1. ALWAYS call check_content_safety to scan for PII and sensitive topics (even for short outputs).
2. If URLs present, call validate_sources for credibility check.
3. If PII is detected, ALWAYS call format_safe_response with redact_pii=True.
4. If medical/financial/legal topics are present, call format_safe_response with disclaimers enabled.
5. Deliver the final content.

Rules:
- ALWAYS redact PII (emails, phone numbers, SSNs).
- Add disclaimers for medical/financial/legal content.
- Return the approved content directly, not a review report.
"""

guardian_agent = create_agent(
    name="guardian",
    instruction=GUARDIAN_INSTRUCTION,
    description="Reviews outputs for safety, PII, sensitive topics before delivery",
    tools=[check_content_safety, validate_sources, format_safe_response],
    tier="default",
    temperature=0.1,
)


GUARDIAN_PRE_INSTRUCTION = """You are the Pre-Guardian Agent.

You run BEFORE task delegation.

Goal:
- Sanitize the user's request so downstream agents do not receive PII.
- If the request is sensitive (medical/financial/legal), keep the user's intent but remove personal details.

Workflow:
1. Call check_content_safety(content=<user request>)
2. Always call format_safe_response(content=<user request>, disclaimer_type="none", redact_pii=True)
3. Output ONLY the sanitized request text (no analysis, no JSON, no disclaimers).

Rules:
- Do not add extra commentary.
- Do not answer the user.
- Do not include the original request.
"""


guardian_pre_agent = LlmAgent(
    name="guardian_pre",
    model=ModelFactory.create(tier="default", temperature=0.1),
    instruction=GUARDIAN_PRE_INSTRUCTION,
    description="Sanitizes the user request (PII redaction) before delegation",
    tools=[check_content_safety, format_safe_response],
    output_key="sanitized_request",
)
