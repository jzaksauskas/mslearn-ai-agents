import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel

# Add references
from azure.identity import DefaultAzureCredential
from agent_framework import workflow
from agent_framework.foundry import FoundryAgent

load_dotenv()
PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
CONFIDENCE_THRESHOLD = 0.6

# The sample tickets that used to live in the "Set variable" workflow node
SUPPORT_TICKETS = [
    "The API returns a 403 error when creating invoices, but our API key hasn't changed.",
    "Is there a way to export all invoices as a CSV?",
    "I was charged twice for the same invoice last Friday and my customer is also seeing two receipts. Can someone fix this?",
]


# Matches the JSON schema response format configured on Triage-Agent in Foundry
class TriageResult(BaseModel):
    customer_issue: str
    category: str
    confidence: float


def print_ticket_result(
    ticket_number: int, triage: TriageResult, response_text: str
) -> None:
    print("\n" + "=" * 80)
    print(
        f"Ticket {ticket_number}: {triage.category} ({triage.confidence:.0%} confidence)"
    )
    print("-" * 80)
    print(f"Issue: {triage.customer_issue}")
    print("\nResponse:")
    print(response_text.strip())


async def main() -> None:
    with DefaultAzureCredential() as credential:
        # Connect to the agents already created in the Foundry portal
        triage_agent = FoundryAgent(
            project_endpoint=PROJECT_ENDPOINT,
            agent_name="Triage-Agent",
            credential=credential,
        )
        resolution_agent = FoundryAgent(
            project_endpoint=PROJECT_ENDPOINT,
            agent_name="Resolution-Agent",
            credential=credential,
        )

        # The For-each / If-Else / Invoke-agent nodes from the deprecated
        # Foundry Workflow builder are now expressed as plain Python control
        # flow, closing over the two agents connected above.
        @workflow
        async def triage_support_tickets(tickets: list[str]) -> int:
            for ticket_number, ticket in enumerate(tickets, start=1):
                triage_response = await triage_agent.run(
                    ticket, options={"response_format": TriageResult}
                )
                triage: TriageResult = triage_response.value

                if triage.confidence <= CONFIDENCE_THRESHOLD:
                    print_ticket_result(
                        ticket_number,
                        triage,
                        f'The support ticket classification has low confidence. Requesting more details about the issue: "{ticket}"',
                    )
                    continue

                if triage.category == "Billing":
                    print_ticket_result(
                        ticket_number,
                        triage,
                        "Escalate billing issue to human support team.",
                    )
                    continue

                resolution_response = await resolution_agent.run(triage_response.text)
                print_ticket_result(ticket_number, triage, resolution_response.text)

            return len(tickets)

        workflow_instance = triage_support_tickets.build()
        result = await workflow_instance.run(SUPPORT_TICKETS)

        print("=" * 80)
        print(f"\nProcessed {result.get_outputs()[0]} ticket(s)")


if __name__ == "__main__":
    asyncio.run(main())
