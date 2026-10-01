import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTest(unittest.TestCase):
    runtime = SimpleNamespace(config={"metadata": {"user_id": "rep-001"}})

    def test_disqualified_prospect_is_blocked(self):
        result = send_prospect_email.func(
            {
                "prospect_id": "LEAD-50001",
                "email": "priya.nair@brightwaveapps.com",
            },
            "Checking in",
            "Hello",
            self.runtime,
        )

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is marked disqualified; email not sent."},
        )

    def test_eligible_prospect_is_sent(self):
        result = send_prospect_email.func(
            {
                "prospect_id": "LEAD-12853",
                "email": "omar.okafor@northstaranalytics.com",
            },
            "Checking in",
            "Hello",
            self.runtime,
        )

        self.assertEqual(result["status"], "sent")

    def test_prompt_requires_reporting_blocked_result(self):
        self.assertIn("Never email a prospect marked disqualified", SYSTEM_PROMPT)
        self.assertIn("status is not sent", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
