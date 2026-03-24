"""ICU Coordinator OpenReward environment.

A hyper-realistic hospital bed/ICU coordination simulation where agents manage
bed allocation, staffing, patient admissions/transfers/discharges, OR scheduling,
and ambulance diversion under stochastic patient arrivals. Parameters are grounded
in medical literature (SCCM guidelines, MIMIC-III studies, AHRQ staffing research).
"""

from typing import List
import numpy as np
from pydantic import BaseModel, field_validator

from openreward.environments import (
    Environment, JSONObject, ToolOutput, tool, TextBlock,
)
from simulation import (
    HospitalSimulation, UnitName, PatientStatus,
    SCENARIO_CONFIGS,
)


# ---------------------------------------------------------------------------
# Pydantic models for task spec and tool parameters
# ---------------------------------------------------------------------------

class TaskSpec(BaseModel):
    id: str
    scenario: str
    seed: int


class ViewDashboardParams(BaseModel, extra="forbid"):
    """No parameters needed."""
    pass


class AdmitPatientParams(BaseModel, extra="forbid"):
    patient_id: str
    unit: str

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        valid = {"icu", "stepdown", "medsurg_a", "medsurg_b"}
        if v not in valid:
            raise ValueError(f"Unit must be one of {sorted(valid)}, got '{v}'")
        return v


class TransferPatientParams(BaseModel, extra="forbid"):
    patient_id: str
    to_unit: str

    @field_validator("to_unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        valid = {"icu", "stepdown", "medsurg_a", "medsurg_b"}
        if v not in valid:
            raise ValueError(f"to_unit must be one of {sorted(valid)}, got '{v}'")
        return v


class DischargePatientParams(BaseModel, extra="forbid"):
    patient_id: str


class SetDiversionParams(BaseModel, extra="forbid"):
    active: bool


class RequestStaffParams(BaseModel, extra="forbid"):
    unit: str
    count: int
    staff_type: str = "agency"

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        valid = {"ed", "icu", "stepdown", "medsurg_a", "medsurg_b", "pacu"}
        if v not in valid:
            raise ValueError(f"Unit must be one of {sorted(valid)}, got '{v}'")
        return v

    @field_validator("count")
    @classmethod
    def validate_count(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError("Count must be between 1 and 10")
        return v

    @field_validator("staff_type")
    @classmethod
    def validate_staff_type(cls, v: str) -> str:
        if v not in ("regular", "agency"):
            raise ValueError("staff_type must be 'regular' or 'agency'")
        return v


class CancelElectiveParams(BaseModel, extra="forbid"):
    surgery_id: str


class AdvanceTimeParams(BaseModel, extra="forbid"):
    hours: int = 1

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, v: int) -> int:
        if v < 1 or v > 4:
            raise ValueError("Can only advance 1-4 hours at a time")
        return v


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ICUCoordinatorEnvironment(Environment):
    """Hospital bed / ICU coordinator environment.

    Agents manage a 150-bed community hospital, making hourly decisions about
    bed allocation, staffing, patient flow, and resource management under
    stochastic demand.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.sim: HospitalSimulation | None = None
        self.finished = False

    async def setup(self):
        self.sim = HospitalSimulation(
            scenario=self.config.scenario,
            seed=self.config.seed,
        )

    async def teardown(self):
        self.sim = None

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "train":
            tasks = []
            for scenario in ["normal_weekday", "winter_surge",
                             "mass_casualty", "staffing_crisis"]:
                for seed in range(5):
                    tasks.append({
                        "id": f"{scenario}_seed{seed}",
                        "scenario": scenario,
                        "seed": seed,
                    })
            return tasks
        elif split == "test":
            tasks = []
            for scenario in ["pandemic_wave", "holiday_weekend"]:
                for seed in range(5):
                    tasks.append({
                        "id": f"{scenario}_seed{seed}",
                        "scenario": scenario,
                        "seed": seed,
                    })
            return tasks
        return []

    async def get_prompt(self) -> List[TextBlock]:
        assert self.sim is not None, "Must call setup() before get_prompt()"
        sc = self.sim.config
        dashboard = self.sim.format_dashboard()

        prompt = f"""You are the Bed Coordinator at Mercy General Hospital, a 150-bed community hospital.
Your role is to manage bed allocation, patient admissions, transfers, discharges, staffing,
operating room scheduling, and ambulance diversion decisions over a {sc.duration_hours}-hour period.

SCENARIO: {sc.name} - {sc.description}

HOSPITAL LAYOUT:
  Emergency Dept (ED):  24 beds  | Safe ratio: 1 nurse per 4 patients
  ICU (combined):       18 beds  | Safe ratio: 1 nurse per 2 patients
  Step-down/IMC:        20 beds  | Safe ratio: 1 nurse per 3 patients
  Med-Surg A:           32 beds  | Safe ratio: 1 nurse per 5 patients
  Med-Surg B:           32 beds  | Safe ratio: 1 nurse per 5 patients
  OR:                    6 rooms
  PACU:                  8 beds  | Safe ratio: 1 nurse per 2 patients

PATIENT FLOW:
  - ED arrivals follow a Poisson process (~4-5/hr) varying by time of day.
  - ~22% of ED patients require inpatient admission.
  - ESI-1/2 patients are critical and often need ICU beds urgently.
  - Surgical patients go: OR -> PACU (1-2h) -> inpatient unit.
  - Patients become "ready for discharge" when their length of stay completes.

KEY RISKS:
  - ESI-1/2 patients waiting for ICU beds face +0.5%/hr mortality risk.
  - ED boarding beyond 4 hours increases mortality.
  - Understaffed units (ratio above safe threshold) increase mortality up to 3x.
  - Deaths are the worst outcome (-0.5 per death in hourly score).

AVAILABLE TOOLS:
  1. view_dashboard() - See full hospital status, occupancy, staffing, patients.
  2. admit_patient(patient_id, unit) - Admit an ED patient to "icu", "stepdown", "medsurg_a", or "medsurg_b".
  3. transfer_patient(patient_id, to_unit) - Transfer between inpatient units.
  4. discharge_patient(patient_id) - Discharge patients who are ready.
  5. set_diversion(active) - Toggle ambulance diversion (reduces arrivals 60%, incurs score penalty).
  6. request_staff(unit, count, staff_type) - Request "agency" nurses (4h delay, 2x cost) for a unit.
  7. cancel_elective(surgery_id) - Cancel an elective surgery (frees OR, but penalizes score).
  8. advance_time(hours) - Advance 1-4 hours. Processes all events and returns results.

SCORING:
  Each hour you receive a score from 0 to 1:
    1.0 - penalties where penalties include:
    - 0.50 per death
    - 0.001 per patient currently boarding in ED
    - 0.01 per critical patient (ESI-1/2) waiting for ICU
    - 0.02 per understaffed unit
    - 0.05 while diversion is active
  Final reward = average of all hourly scores (minus small penalty per cancelled surgery).

STRATEGY TIPS:
  - Prioritize ICU placement for ESI-1/2 patients to prevent deaths.
  - Discharge ready patients promptly to free beds (beds need 45-60 min cleaning after discharge).
  - Transfer patients to lower-acuity units when they improve (e.g., ICU -> stepdown).
  - Request agency nurses early (4-hour delay) if you anticipate staffing shortfalls.
  - Use diversion sparingly - it reduces arrivals but costs 0.05/hr in score.
  - Plan ahead for shift changes (every 12 hours) and surgical schedules.
  - View the dashboard frequently to stay informed.

WORKFLOW: View dashboard -> Make decisions (admit/transfer/discharge/staff/diversion) -> Advance time -> Repeat

CURRENT HOSPITAL STATUS:
{dashboard}

Begin managing the hospital. Your first action should be to review the dashboard and make any immediate decisions before advancing time."""

        return [TextBlock(text=prompt)]

    # ----- Tools -----

    @tool
    async def view_dashboard(self, params: ViewDashboardParams) -> ToolOutput:
        """View the complete hospital dashboard showing unit occupancy, staffing
        levels, patient census, pending surgeries, and key performance metrics."""
        assert self.sim is not None
        dashboard = self.sim.format_dashboard()
        metadata = self.sim.get_dashboard_metadata()
        return ToolOutput(
            blocks=[TextBlock(text=dashboard)],
            metadata=metadata,
            reward=0.0,
            finished=False,
        )

    @tool
    async def admit_patient(self, params: AdmitPatientParams) -> ToolOutput:
        """Admit a patient from the ED to an inpatient unit. The patient must be
        in the ED and require admission. Target unit must have available beds.
        Valid units: icu, stepdown, medsurg_a, medsurg_b."""
        assert self.sim is not None
        target = UnitName(params.unit)
        result_msg = self.sim.admit_patient(params.patient_id, target)
        is_error = result_msg.startswith("Error")
        return ToolOutput(
            blocks=[TextBlock(text=result_msg)],
            metadata={
                "patient_id": params.patient_id,
                "unit": params.unit,
                "success": not is_error,
            },
            reward=0.0,
            finished=False,
        )

    @tool
    async def transfer_patient(self, params: TransferPatientParams) -> ToolOutput:
        """Transfer a patient between inpatient units. Use this to step down
        patients from ICU to lower-acuity units, or escalate deteriorating
        patients to ICU. Valid units: icu, stepdown, medsurg_a, medsurg_b."""
        assert self.sim is not None
        target = UnitName(params.to_unit)
        result_msg = self.sim.transfer_patient(params.patient_id, target)
        is_error = result_msg.startswith("Error")
        return ToolOutput(
            blocks=[TextBlock(text=result_msg)],
            metadata={
                "patient_id": params.patient_id,
                "to_unit": params.to_unit,
                "success": not is_error,
            },
            reward=0.0,
            finished=False,
        )

    @tool
    async def discharge_patient(self, params: DischargePatientParams) -> ToolOutput:
        """Discharge a patient who is ready for discharge. The bed will need
        45-60 minutes of cleaning before it becomes available again."""
        assert self.sim is not None
        result_msg = self.sim.discharge_patient(params.patient_id)
        is_error = result_msg.startswith("Error")
        return ToolOutput(
            blocks=[TextBlock(text=result_msg)],
            metadata={
                "patient_id": params.patient_id,
                "success": not is_error,
            },
            reward=0.0,
            finished=False,
        )

    @tool
    async def set_diversion(self, params: SetDiversionParams) -> ToolOutput:
        """Toggle ambulance diversion status. When active, ambulance arrivals
        decrease by ~60% (walk-ins only), but incurs a 0.05/hr scoring penalty.
        Use sparingly for capacity relief."""
        assert self.sim is not None
        self.sim.state.diversion_active = params.active
        status = "ACTIVATED" if params.active else "DEACTIVATED"
        msg = f"Ambulance diversion {status}."
        if params.active:
            msg += " Ambulance arrivals reduced ~60%. Scoring penalty: -0.05/hr."
        else:
            msg += " Normal ambulance arrivals resumed."
        return ToolOutput(
            blocks=[TextBlock(text=msg)],
            metadata={"diversion_active": params.active},
            reward=0.0,
            finished=False,
        )

    @tool
    async def request_staff(self, params: RequestStaffParams) -> ToolOutput:
        """Request additional nursing staff for a unit. Agency nurses arrive
        after a 4-hour delay at 2x cost. Valid units: ed, icu, stepdown,
        medsurg_a, medsurg_b, pacu."""
        assert self.sim is not None
        if params.staff_type == "agency":
            result_msg = self.sim.request_agency_staff(params.unit, params.count)
        else:
            result_msg = (f"Float pool nurses not available at this time. "
                          f"Use staff_type='agency' to request agency nurses.")
        return ToolOutput(
            blocks=[TextBlock(text=result_msg)],
            metadata={
                "unit": params.unit,
                "count": params.count,
                "staff_type": params.staff_type,
            },
            reward=0.0,
            finished=False,
        )

    @tool
    async def cancel_elective(self, params: CancelElectiveParams) -> ToolOutput:
        """Cancel a scheduled elective surgery. The surgery must not have
        already started. Each cancellation incurs a small scoring penalty."""
        assert self.sim is not None
        result_msg = self.sim.cancel_surgery(params.surgery_id)
        is_error = result_msg.startswith("Error")
        return ToolOutput(
            blocks=[TextBlock(text=result_msg)],
            metadata={
                "surgery_id": params.surgery_id,
                "success": not is_error,
            },
            reward=0.0,
            finished=False,
        )

    @tool
    async def advance_time(self, params: AdvanceTimeParams) -> ToolOutput:
        """Advance the simulation by 1-4 hours. Processes all events including
        patient arrivals, surgeries, deterioration, mortality, and shift changes.
        Returns a summary of events and the hourly score(s). When the simulation
        period ends, returns finished=True with the final cumulative reward."""
        assert self.sim is not None

        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation already completed.")],
                metadata={"error": "already_finished"},
                reward=0.0,
                finished=True,
            )

        all_events: list[str] = []
        hourly_scores: list[float] = []

        for _ in range(params.hours):
            if self.sim.state.current_hour >= self.sim.state.simulation_duration:
                break
            result = self.sim.advance_one_hour()
            all_events.extend(result["events"])
            hourly_scores.append(result["score"])

        is_final = self.sim.state.current_hour >= self.sim.state.simulation_duration
        hour = self.sim.state.current_hour
        duration = self.sim.state.simulation_duration

        # Format summary
        lines: list[str] = []
        lines.append(f"=== Hour {int(hour)}/{int(duration)} ===")
        lines.append(f"Events this period: {len(all_events)}")
        for event in all_events[-15:]:
            lines.append(f"  - {event}")
        if len(all_events) > 15:
            lines.append(f"  ... and {len(all_events) - 15} more events")

        scores_str = ", ".join(f"{s:.3f}" for s in hourly_scores)
        lines.append(f"Hourly scores: [{scores_str}]")

        if is_final:
            self.finished = True
            final_reward = self.sim.get_final_reward()
            total_deaths = len(self.sim.state.deaths)
            total_patients = len(self.sim.state.patients)
            discharged = sum(1 for p in self.sim.state.patients.values()
                             if p.status == PatientStatus.DISCHARGED)

            lines.append("")
            lines.append("=" * 50)
            lines.append("SIMULATION COMPLETE")
            lines.append("=" * 50)
            lines.append(f"Final Reward: {final_reward:.4f}")
            lines.append(f"Total Deaths: {total_deaths}")
            lines.append(f"Total Patients Seen: {total_patients}")
            lines.append(f"Patients Discharged: {discharged}")
            lines.append(f"Surgeries Cancelled: {self.sim.state.surgeries_cancelled}")
            lines.append(f"Mean Hourly Score: {np.mean(self.sim.state.hourly_scores):.4f}")

            return ToolOutput(
                blocks=[TextBlock(text="\n".join(lines))],
                metadata={
                    "final_reward": final_reward,
                    "total_deaths": total_deaths,
                    "total_patients": total_patients,
                    "discharged": discharged,
                    "surgeries_cancelled": self.sim.state.surgeries_cancelled,
                    "hourly_scores": [round(s, 4) for s in self.sim.state.hourly_scores],
                    "deaths": self.sim.state.deaths,
                },
                reward=final_reward,
                finished=True,
            )
        else:
            step_reward = float(np.mean(hourly_scores)) if hourly_scores else 0.0
            return ToolOutput(
                blocks=[TextBlock(text="\n".join(lines))],
                metadata={
                    "current_hour": hour,
                    "hours_remaining": duration - hour,
                    "hourly_scores": hourly_scores,
                    "events_count": len(all_events),
                },
                reward=step_reward,
                finished=False,
            )
