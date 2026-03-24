"""Agent integration test for the ICU Coordinator environment.

Runs a full agent loop using the OpenAI responses API, logging every
tool call and result to a .jsonl trajectory file for inspection.

Usage:
    python test_agent.py
"""

import asyncio
import json
import os
from datetime import datetime

from openai import AsyncOpenAI
from icu_coordinator import (
    ICUCoordinatorEnvironment,
    ViewDashboardParams,
    AdmitPatientParams,
    TransferPatientParams,
    DischargePatientParams,
    SetDiversionParams,
    RequestStaffParams,
    CancelElectiveParams,
    AdvanceTimeParams,
)


def get_secrets():
    """Load secrets from .env file in parent directory."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    secrets = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    secrets[key.strip().lower()] = val.strip()
    return secrets


TOOLS = [
    {
        "type": "function",
        "name": "view_dashboard",
        "description": (
            "View the complete hospital dashboard showing unit occupancy, staffing "
            "levels, patient census, pending surgeries, and key performance metrics."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "admit_patient",
        "description": (
            "Admit a patient from the ED to an inpatient unit. The patient must be "
            "in the ED and require admission. Valid units: icu, stepdown, medsurg_a, medsurg_b."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The ID of the patient to admit (e.g. 'P0042')",
                },
                "unit": {
                    "type": "string",
                    "enum": ["icu", "stepdown", "medsurg_a", "medsurg_b"],
                    "description": "Target inpatient unit",
                },
            },
            "required": ["patient_id", "unit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "transfer_patient",
        "description": (
            "Transfer a patient between inpatient units. Use to step down from ICU "
            "or escalate to ICU. Valid units: icu, stepdown, medsurg_a, medsurg_b."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The ID of the patient to transfer",
                },
                "to_unit": {
                    "type": "string",
                    "enum": ["icu", "stepdown", "medsurg_a", "medsurg_b"],
                    "description": "Destination unit",
                },
            },
            "required": ["patient_id", "to_unit"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "discharge_patient",
        "description": (
            "Discharge a patient who is ready for discharge. The bed will need "
            "45-60 minutes of cleaning before becoming available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The ID of the patient to discharge",
                },
            },
            "required": ["patient_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_diversion",
        "description": (
            "Toggle ambulance diversion. When active, ambulance arrivals decrease "
            "by ~60% but incurs a 0.05/hr scoring penalty."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "active": {
                    "type": "boolean",
                    "description": "True to activate diversion, False to deactivate",
                },
            },
            "required": ["active"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "request_staff",
        "description": (
            "Request additional nursing staff. Agency nurses arrive after 4-hour delay "
            "at 2x cost. Valid units: ed, icu, stepdown, medsurg_a, medsurg_b, pacu."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "unit": {
                    "type": "string",
                    "enum": ["ed", "icu", "stepdown", "medsurg_a", "medsurg_b", "pacu"],
                    "description": "Unit to assign staff to",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of nurses to request (1-10)",
                },
                "staff_type": {
                    "type": "string",
                    "enum": ["agency"],
                    "description": "Type of staff to request",
                    "default": "agency",
                },
            },
            "required": ["unit", "count"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "cancel_elective",
        "description": (
            "Cancel a scheduled elective surgery. The surgery must not have already "
            "started. Each cancellation incurs a small scoring penalty."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "surgery_id": {
                    "type": "string",
                    "description": "The ID of the surgery to cancel (e.g. 'S0003')",
                },
            },
            "required": ["surgery_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "advance_time",
        "description": (
            "Advance the simulation by 1-4 hours. Processes arrivals, surgeries, "
            "deterioration, mortality, and shift changes. Returns events and scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": "Number of hours to advance (1-4)",
                    "default": 1,
                },
            },
            "additionalProperties": False,
        },
    },
]


PARAM_MAP = {
    "view_dashboard": ViewDashboardParams,
    "admit_patient": AdmitPatientParams,
    "transfer_patient": TransferPatientParams,
    "discharge_patient": DischargePatientParams,
    "set_diversion": SetDiversionParams,
    "request_staff": RequestStaffParams,
    "cancel_elective": CancelElectiveParams,
    "advance_time": AdvanceTimeParams,
}


async def dispatch_tool(env, tool_name, args):
    """Dispatch a tool call to the environment."""
    handler = getattr(env, tool_name, None)
    if handler is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    params_cls = PARAM_MAP[tool_name]
    params = params_cls(**args)
    return await handler(params)


async def run_agent_test(max_turns=300):
    secrets = get_secrets()
    oai_key = secrets.get("openai_api_key")
    if not oai_key:
        print("ERROR: No openai_api_key found in .env file")
        print("Expected .env in parent directory with: OPENAI_API_KEY=sk-...")
        return

    oai_client = AsyncOpenAI(api_key=oai_key)

    tasks = ICUCoordinatorEnvironment.list_tasks(split="train")
    task = tasks[0]  # normal_weekday_seed0

    print(f"=== ICU Coordinator Agent Test ===")
    print(f"Task: {task['id']}")
    print(f"Scenario: {task['scenario']}")
    print()

    env = ICUCoordinatorEnvironment(task_spec=task, secrets=secrets)
    await env.setup()
    prompt = await env.get_prompt()

    input_list = [{"role": "user", "content": prompt[0].text}]
    finished = False
    turn = 0
    trajectory = []

    while not finished and turn < max_turns:
        turn += 1
        try:
            response = await oai_client.responses.create(
                model="gpt-4.1-mini",
                tools=TOOLS,
                input=input_list,
            )
        except Exception as e:
            print(f"  API error at turn {turn}: {e}")
            break

        input_list += response.output

        for item in response.output:
            if item.type == "function_call":
                args = json.loads(str(item.arguments))

                try:
                    result = await dispatch_tool(env, item.name, args)
                except Exception as e:
                    error_msg = f"Tool error: {e}"
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": error_msg,
                    })
                    trajectory.append({
                        "turn": turn,
                        "tool": item.name,
                        "args": args,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    print(f"  Turn {turn}: {item.name}({json.dumps(args)[:80]}) ERROR: {e}")
                    continue

                finished = result.finished
                reward = result.reward

                output_text = result.blocks[0].text if result.blocks else ""

                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": output_text,
                })

                # Log trajectory entry
                entry = {
                    "turn": turn,
                    "tool": item.name,
                    "args": args,
                    "reward": reward,
                    "finished": finished,
                    "output_preview": output_text[:200],
                    "timestamp": datetime.utcnow().isoformat(),
                }
                if finished and result.metadata:
                    entry["final_metadata"] = {
                        k: v for k, v in result.metadata.items()
                        if k != "hourly_scores" and k != "deaths"
                    }
                trajectory.append(entry)

                # Print progress
                if item.name == "advance_time":
                    hour_info = result.metadata.get("current_hour", "?")
                    remaining = result.metadata.get("hours_remaining", "?")
                    print(f"  Turn {turn}: advance_time -> hour {hour_info}, "
                          f"remaining: {remaining}, reward: {reward:.4f}")
                elif item.name == "view_dashboard":
                    print(f"  Turn {turn}: view_dashboard")
                else:
                    success = result.metadata.get("success", "")
                    short_args = json.dumps(args)[:60]
                    print(f"  Turn {turn}: {item.name}({short_args}) "
                          f"success={success} reward={reward:.4f}")

                if finished:
                    print()
                    print("=" * 50)
                    print("SIMULATION COMPLETE")
                    print("=" * 50)
                    print(f"Final Reward: {reward:.4f}")
                    if result.metadata:
                        print(f"Total Deaths: {result.metadata.get('total_deaths', '?')}")
                        print(f"Total Patients: {result.metadata.get('total_patients', '?')}")
                        print(f"Discharged: {result.metadata.get('discharged', '?')}")
                        print(f"Surgeries Cancelled: {result.metadata.get('surgeries_cancelled', '?')}")
                    break

    if not finished:
        print(f"\nHit max turns ({max_turns}) without finishing")

    # Write trajectory to JSONL
    trajectory_file = f"trajectory_{task['id']}.jsonl"
    with open(trajectory_file, "w") as f:
        for entry in trajectory:
            f.write(json.dumps(entry) + "\n")
    print(f"\nTrajectory written to {trajectory_file} ({len(trajectory)} entries)")

    await env.teardown()


if __name__ == "__main__":
    asyncio.run(run_agent_test())
