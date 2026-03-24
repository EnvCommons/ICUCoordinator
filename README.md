# ICU Coordinator

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/icu-coordinator)

## Description

ICU Coordinator is a hospital bed management simulation where agents act as the bed coordinator at a 150-bed community hospital. Agents must allocate beds, manage staffing, handle patient admissions and discharges, schedule operating rooms, and make ambulance diversion decisions under stochastic patient arrivals. The simulation parameters are grounded in medical literature including SCCM ICU admission guidelines, AHRQ nurse staffing research, and MIMIC-III clinical data patterns.

The environment models patient flow from ED triage through admission, treatment, and discharge, with stochastic deterioration, mortality from ICU delays and understaffing, bed turnover times, 12-hour nursing shifts, and surgical patient pipelines. Agents face constrained planning problems with ethical tradeoffs between competing demands for scarce resources.

Note: this is a synthetic environment which is primarily AI-generated; please test thoroughly before use in any RL pipeline.

## Capabilities

- Multi-step hospital resource allocation under uncertainty
- Managing competing priorities (ICU placement vs. bed availability vs. staffing)
- Reasoning about stochastic arrival patterns and patient deterioration
- Planning ahead for shift changes, surgical schedules, and capacity constraints
- Balancing short-term actions (admissions) with long-term consequences (staffing requests, diversion)

## Compute Requirements

This environment is computationally lightweight. The simulation runs entirely in-memory with no sandbox, GPU, or external API requirements. A standard CPU instance with minimal RAM is sufficient.

## License

[ORLv1](https://openreward.ai/orlv1.md).

## Tasks

There are 20 training tasks across 4 scenarios (5 seeds each):

- **Normal Weekday** (48h): Typical Tuesday-Wednesday operations at 70% occupancy.
- **Winter Surge** (72h): Flu/respiratory surge with +30% arrivals and higher acuity.
- **Mass Casualty** (48h): Multi-vehicle accident at hour 6 brings 15 trauma patients.
- **Staffing Crisis** (48h): 30% nurse shortage due to call-outs and vacancies.

And 10 test tasks across 2 scenarios (5 seeds each):

- **Pandemic Wave** (72h): +50% arrivals, 90% initial occupancy, 15% staff shortage.
- **Holiday Weekend** (72h): Skeleton staffing with +20% trauma cases.

Each task simulates a continuous period of hospital operations. The agent makes decisions each hour using 8 tools, then advances time. Simulations last 48-72 hours.

## Reward Structure

This is a dense, verifiable reward environment with no LLM grader. Each hour, the agent receives a score from 0 to 1:

$$\text{score}_t = \max(0, \; 1.0 - \text{penalties})$$

Penalties include:
- **0.50** per patient death
- **0.01** per critical patient (ESI-1/2) waiting for ICU placement
- **0.02** per understaffed unit (nurse:patient ratio exceeds safe threshold)
- **0.05** per hour of ambulance diversion
- **0.001** per patient boarding in the ED awaiting an inpatient bed

The final reward is the mean of all hourly scores, with a small additional penalty per cancelled elective surgery.

## Data

No external data files are required. The simulation is fully procedural, generating patient arrivals via Poisson processes and lengths of stay via log-normal distributions, all parameterized from published medical literature. Scenario seeds ensure reproducibility.

## Tools

Agents have access to 8 tools:

| Tool | Description |
|------|-------------|
| `view_dashboard` | View full hospital status: occupancy, staffing, patient census, OR schedule |
| `admit_patient` | Admit an ED patient to ICU, step-down, or med-surg unit |
| `transfer_patient` | Transfer patient between inpatient units (e.g., ICU step-down) |
| `discharge_patient` | Discharge a patient who is ready, freeing the bed after cleaning |
| `set_diversion` | Toggle ambulance diversion on/off |
| `request_staff` | Request agency nurses (4-hour arrival delay, 2x cost) |
| `cancel_elective` | Cancel a scheduled elective surgery |
| `advance_time` | Advance simulation 1-4 hours, processing all events |

## Time Horizon

ICU Coordinator is a multi-turn environment. Each simulation lasts 48-72 hours with decisions made each hour. A typical task involves viewing the dashboard, making management decisions, and advancing time in a loop.

## Environment Difficulty

The environment presents significant challenges due to stochastic arrivals, capacity constraints, and cascading consequences of poor decisions. Deaths are rare but devastating to the score. Effective management requires proactive discharge planning, ICU prioritization, and anticipatory staffing requests.

## Other Environment Requirements

No external API keys or secrets are required. The environment is fully self-contained.

## Safety

This environment simulates hospital resource allocation decisions that involve patient mortality risk. While the simulation is entirely artificial, the ethical dimensions are: agents must make triage-like decisions about which patients receive scarce ICU beds, when to cancel elective surgeries, and how to balance staff wellbeing against patient safety. This environment is for research purposes only.

## Citations

```bibtex
@dataset{GRICUCoordinator,
  author    = {General Reasoning Inc. Team},
  title     = {ICU Coordinator},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/icu-coordinator}
}
```
