import json
import unittest
from unittest.mock import patch

from gtm_agent import data_service
from gtm_agent import gtm_agent


SENSITIVE_FIELDS = {
    "billing_qualification",
    "tax_id",
    "date_of_birth",
    "card_on_file",
    "credit_check_ref",
}


def flatten_keys(value):
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(flatten_keys(child))
        return keys
    if isinstance(value, list):
        keys = set()
        for child in value:
            keys.update(flatten_keys(child))
        return keys
    return set()


class FakeScore:
    def model_dump(self):
        return {"score": 80}


class FakeScoringLlm:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeScore()


class ProspectPrivacyTests(unittest.TestCase):
    prospect_id = "LEAD-12853"

    def setUp(self):
        data_service._PROFILES.clear()

    def test_get_prospect_returns_only_contact_fields(self):
        result = gtm_agent.get_prospect.invoke({"prospect_id": self.prospect_id})

        self.assertEqual(set(result["prospect"]), {"prospect_id", "name", "email"})
        self.assertTrue(SENSITIVE_FIELDS.isdisjoint(flatten_keys(result)))

    def test_build_profile_and_cache_exclude_sensitive_fields(self):
        result = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        cached = data_service._PROFILES[self.prospect_id]

        self.assertTrue(SENSITIVE_FIELDS.isdisjoint(flatten_keys(result)))
        self.assertTrue(SENSITIVE_FIELDS.isdisjoint(flatten_keys(cached)))

    def test_build_profile_sanitizes_existing_cached_profile(self):
        data_service._PROFILES[self.prospect_id] = data_service.get_prospect_record(self.prospect_id)

        result = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        cached = data_service._PROFILES[self.prospect_id]

        self.assertTrue(SENSITIVE_FIELDS.isdisjoint(flatten_keys(result)))
        self.assertTrue(SENSITIVE_FIELDS.isdisjoint(flatten_keys(cached)))

    def test_score_prompt_uses_only_allowlisted_profile_fields(self):
        fake_llm = FakeScoringLlm()
        profile = {
            "prospect_id": self.prospect_id,
            "name": "Omar Okafor",
            "annual_revenue": 70000000,
            "tech_stack": ["AWS"],
            "account_details": [{"industry": "Software"}],
            "engagement_history": [],
            "billing_qualification": {
                "tax_id": "secret",
                "date_of_birth": "secret",
                "card_on_file": "secret",
                "credit_check_ref": "secret",
            },
            "unexpected": "secret",
        }
        offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 1,
            "description": "A platform",
        }

        with patch.object(gtm_agent, "_scoring_llm", fake_llm):
            gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        prompt = fake_llm.messages[1]["content"]
        self.assertNotIn("billing_qualification", prompt)
        for field in SENSITIVE_FIELDS - {"billing_qualification"}:
            self.assertNotIn(field, prompt)
        self.assertNotIn("unexpected", prompt)
        payload = json.loads(prompt.split("Prospect profile:\n", 1)[1])
        self.assertEqual(
            set(payload),
            {
                "prospect_id",
                "name",
                "annual_revenue",
                "tech_stack",
                "account_details",
                "engagement_history",
            },
        )


if __name__ == "__main__":
    unittest.main()
