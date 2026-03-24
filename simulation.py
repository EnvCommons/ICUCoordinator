"""Hospital simulation engine for the ICU Coordinator environment.

Models a medium community hospital (~150 inpatient beds) with realistic
patient flow, staffing, mortality, and capacity dynamics. All parameters
are grounded in medical literature (SCCM guidelines, MIMIC-III studies,
AHRQ staffing research).
"""

from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class UnitName(str, Enum):
    ED = "ed"
    ICU = "icu"
    STEPDOWN = "stepdown"
    MEDSURG_A = "medsurg_a"
    MEDSURG_B = "medsurg_b"
    OR = "or"
    PACU = "pacu"


class ESI(IntEnum):
    """Emergency Severity Index (1 = most critical, 5 = least)."""
    ESI1 = 1
    ESI2 = 2
    ESI3 = 3
    ESI4 = 4
    ESI5 = 5


class PatientStatus(str, Enum):
    WAITING_IN_ED = "waiting_in_ed"
    BOARDING_IN_ED = "boarding_in_ed"
    ADMITTED = "admitted"
    IN_SURGERY = "in_surgery"
    IN_PACU = "in_pacu"
    READY_FOR_DISCHARGE = "ready_for_discharge"
    DISCHARGED = "discharged"
    DECEASED = "deceased"


class StaffType(str, Enum):
    REGULAR = "regular"
    AGENCY = "agency"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Patient:
    id: str
    esi: int  # 1-5
    arrival_hour: float
    status: PatientStatus
    location: UnitName
    needs_admission: bool
    target_unit: Optional[UnitName]
    los_remaining_hours: float  # hours left in current unit before discharge-ready
    ed_treatment_hours_remaining: float = 0.0  # hours of ED treatment before admission-eligible
    ed_boarding_hours: float = 0.0
    icu_delay_hours: float = 0.0
    is_surgical: bool = False
    surgery_id: Optional[str] = None
    pacu_hours_remaining: float = 0.0
    admitted_hour: Optional[float] = None


@dataclass
class Surgery:
    id: str
    scheduled_hour: float
    duration_hours: float
    patient_id: str
    or_room: int
    is_elective: bool = True
    is_cancelled: bool = False
    post_op_destination: UnitName = UnitName.MEDSURG_A
    started: bool = False
    completed: bool = False


@dataclass
class Nurse:
    id: str
    unit: UnitName
    staff_type: StaffType
    shift_start: float
    shift_end: float


@dataclass
class UnitState:
    unit: UnitName
    total_beds: int
    patients: list[str] = field(default_factory=list)
    nurses: list[str] = field(default_factory=list)
    safe_ratio: float = 5.0  # max patients per nurse before "understaffed"
    beds_being_cleaned: int = 0
    cleaning_finish_times: list[float] = field(default_factory=list)

    @property
    def occupied_beds(self) -> int:
        return len(self.patients)

    @property
    def available_beds(self) -> int:
        return self.total_beds - self.occupied_beds - self.beds_being_cleaned

    @property
    def nurse_patient_ratio(self) -> float:
        """Patients per nurse (lower is better). Returns inf if no nurses."""
        n = len(self.nurses)
        if n == 0:
            return float("inf") if self.occupied_beds > 0 else 0.0
        return self.occupied_beds / n

    @property
    def is_understaffed(self) -> bool:
        return self.nurse_patient_ratio > self.safe_ratio


@dataclass
class HospitalState:
    current_hour: float = 0.0
    simulation_duration: float = 48.0

    units: dict[str, UnitState] = field(default_factory=dict)
    patients: dict[str, Patient] = field(default_factory=dict)
    surgeries: dict[str, Surgery] = field(default_factory=dict)
    nurses: dict[str, Nurse] = field(default_factory=dict)

    diversion_active: bool = False
    next_patient_id: int = 0
    next_nurse_id: int = 0
    next_surgery_id: int = 0

    hourly_scores: list[float] = field(default_factory=list)
    deaths: list[dict] = field(default_factory=list)
    events_log: list[str] = field(default_factory=list)

    pending_agency: list[dict] = field(default_factory=list)

    # Track cancellations for reward
    surgeries_cancelled: int = 0


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    name: str
    duration_hours: int
    initial_occupancy: float
    arrival_rate_multiplier: float
    staffing_multiplier: float
    trauma_multiplier: float
    mass_casualty_hour: Optional[int]
    mass_casualty_count: int = 0
    acuity_distribution: list[float] = field(
        # Source: ESI Implementation Handbook (Gilboy et al., 2020) expected
        # ranges: ESI-1 1-3%, ESI-2 20-30%, ESI-3 30-40%, ESI-4+5 20-35%.
        # Observed distribution: Mistry et al., Adv Emerg Nurs J, 2022
        # (n=955K). Adjusted for community-hospital acuity mix.
        default_factory=lambda: [0.01, 0.15, 0.45, 0.28, 0.11]
    )
    description: str = ""


SCENARIO_CONFIGS: dict[str, ScenarioConfig] = {
    "normal_weekday": ScenarioConfig(
        name="normal_weekday",
        duration_hours=48,
        initial_occupancy=0.70,
        arrival_rate_multiplier=1.0,
        staffing_multiplier=1.0,
        trauma_multiplier=1.0,
        mass_casualty_hour=None,
        description="Typical Tuesday-Wednesday operations at Asimov General Hospital.",
    ),
    "winter_surge": ScenarioConfig(
        name="winter_surge",
        duration_hours=72,
        initial_occupancy=0.85,
        arrival_rate_multiplier=1.3,
        staffing_multiplier=1.0,
        trauma_multiplier=1.0,
        mass_casualty_hour=None,
        acuity_distribution=[0.02, 0.18, 0.48, 0.23, 0.09],
        description="Winter flu/respiratory surge. +30% arrivals, higher acuity mix.",
    ),
    "mass_casualty": ScenarioConfig(
        name="mass_casualty",
        duration_hours=48,
        initial_occupancy=0.70,
        arrival_rate_multiplier=1.0,
        staffing_multiplier=1.0,
        trauma_multiplier=1.0,
        mass_casualty_hour=6,
        mass_casualty_count=15,
        description="Multi-vehicle accident at hour 6 brings 15 trauma patients.",
    ),
    "staffing_crisis": ScenarioConfig(
        name="staffing_crisis",
        duration_hours=48,
        initial_occupancy=0.70,
        arrival_rate_multiplier=1.0,
        staffing_multiplier=0.70,
        trauma_multiplier=1.0,
        mass_casualty_hour=None,
        description="30% nurse shortage due to call-outs and vacancies.",
    ),
    "pandemic_wave": ScenarioConfig(
        name="pandemic_wave",
        duration_hours=72,
        initial_occupancy=0.90,
        arrival_rate_multiplier=1.5,
        staffing_multiplier=0.85,
        trauma_multiplier=1.0,
        mass_casualty_hour=None,
        acuity_distribution=[0.03, 0.20, 0.48, 0.21, 0.08],
        description="Pandemic surge: +50% arrivals, 90% initial occupancy, 15% staff shortage.",
    ),
    "holiday_weekend": ScenarioConfig(
        name="holiday_weekend",
        duration_hours=72,
        initial_occupancy=0.75,
        arrival_rate_multiplier=1.0,
        staffing_multiplier=0.75,
        trauma_multiplier=1.2,
        mass_casualty_hour=None,
        description="Holiday weekend: skeleton staffing, +20% trauma cases.",
    ),
}

# ---------------------------------------------------------------------------
# Hospital unit definitions
# ---------------------------------------------------------------------------

# Source: SCCM Critical Care Statistics 2023. ICU beds are ~12-17% of
# total beds nationally. 18/150=12%, conservative for a community hospital.
# Safe ratios: SCCM recommends 1:2 for ICU; AHRQ Nurse Staffing Evidence
# Report (2007) documents 1:4-6 for med-surg; 1:3 for step-down.
UNIT_DEFS = {
    UnitName.ED:        {"beds": 24, "safe_ratio": 4.0},
    UnitName.ICU:       {"beds": 18, "safe_ratio": 2.0},
    UnitName.STEPDOWN:  {"beds": 20, "safe_ratio": 3.0},
    UnitName.MEDSURG_A: {"beds": 32, "safe_ratio": 5.0},
    UnitName.MEDSURG_B: {"beds": 32, "safe_ratio": 5.0},
    UnitName.OR:        {"beds": 6,  "safe_ratio": 99.0},
    UnitName.PACU:      {"beds": 8,  "safe_ratio": 2.0},
}

# Base nurses per shift at full staffing (calibrated to safe ratios at ~80% occupancy)
BASE_NURSES = {
    UnitName.ED:        6,
    UnitName.ICU:       9,
    UnitName.STEPDOWN:  7,
    UnitName.MEDSURG_A: 7,
    UnitName.MEDSURG_B: 7,
    UnitName.PACU:      4,
}

# Length-of-stay parameters (log-normal: median in hours, sigma).
# Source: MIMIC-III median ICU LOS = 2.1 days (IQR 1.2-4.6), per
# Johnson et al., Scientific Data, 2016. 60h ≈ 2.5 days. Log-normal
# distribution validated by Verburg et al., PLOS ONE, 2014.
LOS_PARAMS = {
    UnitName.ICU:       {"median": 60.0,  "sigma": 0.6},
    UnitName.STEPDOWN:  {"median": 48.0,  "sigma": 0.5},
    UnitName.MEDSURG_A: {"median": 72.0,  "sigma": 0.5},
    UnitName.MEDSURG_B: {"median": 72.0,  "sigma": 0.5},
}

# ED treatment time before patient is ready for admission decision (hours)
ED_TREATMENT_TIME = {
    1: 0.5,   # ESI-1: stabilize quickly, needs immediate placement
    2: 1.0,
    3: 2.0,
    4: 3.0,
    5: 3.5,
}

# ED-only discharge LOS for non-admitted patients (hours)
ED_DISCHARGE_TIME = {
    3: 4.0,
    4: 3.0,
    5: 2.0,
}

# Time-of-day arrival modulation (hour -> multiplier on base lambda).
# Source: Well-established ED diurnal arrival pattern; see Hertzum,
# The Ergonomics Open J, 2016 and Rostami-Tabar et al., Health Systems, 2023.
# Peak 10am-2pm (~1.3x), trough 3am-6am (~0.55x).
TIME_OF_DAY_MOD = {
    0: 0.70, 1: 0.60, 2: 0.55, 3: 0.55, 4: 0.55, 5: 0.60,
    6: 0.75, 7: 0.90, 8: 1.10, 9: 1.25, 10: 1.35, 11: 1.35,
    12: 1.30, 13: 1.30, 14: 1.20, 15: 1.15, 16: 1.10, 17: 1.05,
    18: 1.00, 19: 0.95, 20: 0.90, 21: 0.85, 22: 0.80, 23: 0.75,
}

# Base ED arrival rate (patients per hour)
BASE_ARRIVAL_RATE = 4.5

# Admission probability by ESI
# Source: Tanabe et al., "Reliability and validity of scores on The
# Emergency Severity Index version 3," Acad Emerg Med, 2004.
# ESI-1 published at 80% admission; we use 90% for a community hospital
# that typically admits sicker patients. Overall admission rate ~25%.
ADMISSION_PROB = {1: 0.90, 2: 0.75, 3: 0.45, 4: 0.08, 5: 0.03}

# Target unit distribution for admitted patients by ESI
# Returns (unit, probability) pairs
TARGET_UNIT_DIST: dict[int, list[tuple[UnitName, float]]] = {
    1: [(UnitName.ICU, 0.85), (UnitName.STEPDOWN, 0.15)],
    2: [(UnitName.ICU, 0.35), (UnitName.STEPDOWN, 0.40), (UnitName.MEDSURG_A, 0.25)],
    3: [(UnitName.STEPDOWN, 0.20), (UnitName.MEDSURG_A, 0.40), (UnitName.MEDSURG_B, 0.40)],
    4: [(UnitName.MEDSURG_A, 0.50), (UnitName.MEDSURG_B, 0.50)],
    5: [(UnitName.MEDSURG_A, 0.50), (UnitName.MEDSURG_B, 0.50)],
}

# Base hourly mortality by ESI (when properly placed).
# Calibrated simulation parameters. No single published source provides
# per-hour mortality by ESI; rates are calibrated to produce realistic
# overall inpatient mortality (~2% for ESI-1 admits, <0.1% for ESI-4/5).
BASE_MORTALITY = {1: 0.003, 2: 0.001, 3: 0.0002, 4: 0.00005, 5: 0.00001}

# Surgery post-op destination distribution
# Source: ACS NSQIP data; post-op ICU admission common for elective
# non-cardiac surgery (Bruceta et al., Acta Anaesthesiol Scand, 2020).
# We use 10% ICU, 20% step-down for community hospital elective mix.
SURGERY_DEST_DIST = [
    (UnitName.ICU, 0.10),
    (UnitName.STEPDOWN, 0.20),
    (UnitName.MEDSURG_A, 0.35),
    (UnitName.MEDSURG_B, 0.35),
]


# ---------------------------------------------------------------------------
# Hospital Simulation
# ---------------------------------------------------------------------------

class HospitalSimulation:
    """Discrete-event hospital simulation with hourly time steps."""

    def __init__(self, scenario: str, seed: int):
        if scenario not in SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario: {scenario}. Valid: {list(SCENARIO_CONFIGS.keys())}")
        self.rng = np.random.default_rng(seed)
        self.config = SCENARIO_CONFIGS[scenario]
        self.state = HospitalState(simulation_duration=float(self.config.duration_hours))

        self._init_units()
        self._init_staffing()
        self._init_existing_patients()
        self._schedule_surgeries()

    # ----- Initialization -----

    def _init_units(self):
        for unit_name, defs in UNIT_DEFS.items():
            self.state.units[unit_name.value] = UnitState(
                unit=unit_name,
                total_beds=defs["beds"],
                safe_ratio=defs["safe_ratio"],
            )

    def _init_staffing(self):
        """Create initial nurses for the first shift (hour 0-12)."""
        for unit_name, base_count in BASE_NURSES.items():
            count = max(1, int(base_count * self.config.staffing_multiplier))
            for _ in range(count):
                nurse = self._create_nurse(unit_name, 0.0, 12.0, StaffType.REGULAR)
                self.state.nurses[nurse.id] = nurse
                self.state.units[unit_name.value].nurses.append(nurse.id)

    def _create_nurse(self, unit: UnitName, shift_start: float, shift_end: float,
                      staff_type: StaffType) -> Nurse:
        nid = f"N{self.state.next_nurse_id:04d}"
        self.state.next_nurse_id += 1
        return Nurse(
            id=nid, unit=unit, staff_type=staff_type,
            shift_start=shift_start, shift_end=shift_end,
        )

    def _init_existing_patients(self):
        """Pre-populate beds to initial_occupancy."""
        inpatient_units = [UnitName.ICU, UnitName.STEPDOWN, UnitName.MEDSURG_A, UnitName.MEDSURG_B]
        for unit_name in inpatient_units:
            unit_state = self.state.units[unit_name.value]
            target_occupied = int(unit_state.total_beds * self.config.initial_occupancy)
            for _ in range(target_occupied):
                pid = f"P{self.state.next_patient_id:04d}"
                self.state.next_patient_id += 1

                # Sample remaining LOS (patient is partway through stay)
                params = LOS_PARAMS[unit_name]
                full_los = float(self.rng.lognormal(np.log(params["median"]), params["sigma"]))
                # Patient has been here for some fraction of their stay
                fraction_completed = float(self.rng.uniform(0.1, 0.9))
                remaining = max(1.0, full_los * (1.0 - fraction_completed))

                esi = self._esi_for_unit(unit_name)
                patient = Patient(
                    id=pid,
                    esi=esi,
                    arrival_hour=-remaining,  # Arrived before simulation start
                    status=PatientStatus.ADMITTED,
                    location=unit_name,
                    needs_admission=True,
                    target_unit=unit_name,
                    los_remaining_hours=remaining,
                    admitted_hour=-remaining,
                )
                self.state.patients[pid] = patient
                unit_state.patients.append(pid)

    def _esi_for_unit(self, unit: UnitName) -> int:
        """Return a plausible ESI for a patient already in a unit."""
        if unit == UnitName.ICU:
            return int(self.rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2]))
        elif unit == UnitName.STEPDOWN:
            return int(self.rng.choice([2, 3, 4], p=[0.2, 0.6, 0.2]))
        else:
            return int(self.rng.choice([3, 4, 5], p=[0.4, 0.4, 0.2]))

    def _schedule_surgeries(self):
        """Schedule elective surgeries for each day of the simulation."""
        num_days = int(np.ceil(self.config.duration_hours / 24))
        for day in range(num_days):
            num_surgeries = int(self.rng.integers(4, 7))  # 4-6 per day
            for i in range(num_surgeries):
                sid = f"S{self.state.next_surgery_id:04d}"
                self.state.next_surgery_id += 1

                # Schedule between 8am and 3pm (hours 8-15)
                sched_hour = day * 24 + float(self.rng.uniform(8.0, 15.0))
                if sched_hour >= self.config.duration_hours:
                    continue

                duration = max(0.5, float(self.rng.lognormal(np.log(2.0), 0.5)))
                duration = min(duration, 6.0)

                # Create a surgical patient
                pid = f"P{self.state.next_patient_id:04d}"
                self.state.next_patient_id += 1

                # Pick post-op destination
                dest_units = [d[0] for d in SURGERY_DEST_DIST]
                dest_probs = [d[1] for d in SURGERY_DEST_DIST]
                post_op_dest = UnitName(self.rng.choice([u.value for u in dest_units], p=dest_probs))

                patient = Patient(
                    id=pid,
                    esi=3,  # Surgical patients are typically ESI-3
                    arrival_hour=sched_hour - 1.0,  # Arrive 1h before surgery
                    status=PatientStatus.WAITING_IN_ED,  # Will transition when surgery starts
                    location=UnitName.ED,
                    needs_admission=True,
                    target_unit=post_op_dest,
                    los_remaining_hours=0.0,
                    is_surgical=True,
                    surgery_id=sid,
                )
                # Don't add surgical patients to state yet; they arrive at scheduled time
                surgery = Surgery(
                    id=sid,
                    scheduled_hour=sched_hour,
                    duration_hours=duration,
                    patient_id=pid,
                    or_room=i % 6,
                    post_op_destination=post_op_dest,
                )
                self.state.surgeries[sid] = surgery

    # ----- Core Simulation Loop -----

    def advance_one_hour(self) -> dict:
        """Advance simulation by one hour. Returns hourly result dict."""
        hour = self.state.current_hour
        events: list[str] = []

        # 1. Generate new ED arrivals
        events += self._generate_arrivals()

        # 2. Mass casualty event
        if (self.config.mass_casualty_hour is not None
                and int(hour) == self.config.mass_casualty_hour):
            events += self._generate_mass_casualty()

        # 3. Admit surgical patients arriving this hour
        events += self._admit_surgical_patients()

        # 4. Process surgeries (start / complete)
        events += self._process_surgeries()

        # 5. Process PACU patients
        events += self._process_pacu()

        # 6. Update bed cleaning
        self._update_bed_cleaning()

        # 7. Update ED treatment timers and boarding status
        events += self._update_ed_patients()

        # 8. Update inpatient LOS and discharge readiness
        events += self._update_patient_los()

        # 9. Patient deterioration
        events += self._process_deterioration()

        # 10. Mortality from boarding / ICU delay
        events += self._process_boarding_mortality()

        # 11. Understaffing mortality
        events += self._process_understaffing_mortality()

        # 12. Agency nurse arrivals
        events += self._process_agency_arrivals()

        # 13. Shift changes (every 12 hours)
        if int(hour) > 0 and int(hour) % 12 == 0:
            # Only trigger once per 12-hour boundary
            frac = hour - int(hour)
            if frac < 0.01:
                events += self._process_shift_change()

        # 14. Calculate hourly score
        score = self._calculate_hourly_score()
        self.state.hourly_scores.append(score)

        # 15. Advance clock
        self.state.current_hour += 1.0
        self.state.events_log.extend(events)

        return {
            "hour": hour,
            "events": events,
            "score": score,
            "is_final": self.state.current_hour >= self.state.simulation_duration,
        }

    # ----- Arrival Generation -----

    def _generate_arrivals(self) -> list[str]:
        hour_of_day = int(self.state.current_hour % 24)
        day_of_week = int(self.state.current_hour / 24) % 7
        weekend_mod = 1.05 if day_of_week >= 5 else 1.0

        effective_lambda = (
            BASE_ARRIVAL_RATE
            * TIME_OF_DAY_MOD[hour_of_day]
            * weekend_mod
            * self.config.arrival_rate_multiplier
        )
        if self.state.diversion_active:
            effective_lambda *= 0.40

        num_arrivals = int(self.rng.poisson(effective_lambda))
        events: list[str] = []

        for _ in range(num_arrivals):
            patient = self._create_ed_patient()
            self.state.patients[patient.id] = patient
            ed = self.state.units[UnitName.ED.value]
            if ed.occupied_beds < ed.total_beds:
                ed.patients.append(patient.id)
            else:
                # ED is full - patient still arrives but waits (hallway bed)
                ed.patients.append(patient.id)
            events.append(f"Patient {patient.id} arrived in ED (ESI-{patient.esi})")

        return events

    def _create_ed_patient(self) -> Patient:
        pid = f"P{self.state.next_patient_id:04d}"
        self.state.next_patient_id += 1

        # Sample ESI from scenario distribution
        esi = int(self.rng.choice([1, 2, 3, 4, 5], p=self.config.acuity_distribution))

        # Determine admission need
        needs_admission = bool(self.rng.random() < ADMISSION_PROB[esi])

        # Determine target unit
        target_unit = None
        if needs_admission:
            units = [d[0] for d in TARGET_UNIT_DIST[esi]]
            probs = [d[1] for d in TARGET_UNIT_DIST[esi]]
            target_unit = UnitName(self.rng.choice([u.value for u in units], p=probs))

        # ED treatment time
        ed_time = ED_TREATMENT_TIME.get(esi, 2.0) + float(self.rng.exponential(0.5))

        # LOS for non-admitted patients (they'll be discharged from ED)
        if not needs_admission:
            los = ED_DISCHARGE_TIME.get(esi, 3.0) + float(self.rng.exponential(0.5))
        else:
            los = 0.0  # Will be set when admitted to unit

        return Patient(
            id=pid,
            esi=esi,
            arrival_hour=self.state.current_hour,
            status=PatientStatus.WAITING_IN_ED,
            location=UnitName.ED,
            needs_admission=needs_admission,
            target_unit=target_unit,
            los_remaining_hours=los,
            ed_treatment_hours_remaining=ed_time,
        )

    def _generate_mass_casualty(self) -> list[str]:
        events: list[str] = []
        count = self.config.mass_casualty_count
        events.append(f"*** MASS CASUALTY INCIDENT: {count} trauma patients arriving ***")

        for _ in range(count):
            pid = f"P{self.state.next_patient_id:04d}"
            self.state.next_patient_id += 1
            esi = int(self.rng.choice([1, 2, 3], p=[0.20, 0.50, 0.30]))

            target_unit = UnitName.ICU if esi <= 2 else UnitName.STEPDOWN

            patient = Patient(
                id=pid,
                esi=esi,
                arrival_hour=self.state.current_hour,
                status=PatientStatus.WAITING_IN_ED,
                location=UnitName.ED,
                needs_admission=True,
                target_unit=target_unit,
                los_remaining_hours=0.0,
                ed_treatment_hours_remaining=0.5,  # Fast triage for MCI
            )
            self.state.patients[pid] = patient
            self.state.units[UnitName.ED.value].patients.append(pid)
            events.append(f"MCI Patient {pid} arrived (ESI-{esi})")

        return events

    # ----- Surgical Patient Flow -----

    def _admit_surgical_patients(self) -> list[str]:
        """Admit surgical patients who are scheduled to arrive this hour."""
        events: list[str] = []
        for sid, surgery in self.state.surgeries.items():
            if surgery.is_cancelled or surgery.started:
                continue
            # Patients arrive 1 hour before scheduled surgery
            arrival_hour = surgery.scheduled_hour - 1.0
            if int(arrival_hour) == int(self.state.current_hour):
                pid = surgery.patient_id
                if pid not in self.state.patients:
                    patient = Patient(
                        id=pid,
                        esi=3,
                        arrival_hour=self.state.current_hour,
                        status=PatientStatus.WAITING_IN_ED,
                        location=UnitName.ED,
                        needs_admission=True,
                        target_unit=surgery.post_op_destination,
                        los_remaining_hours=0.0,
                        is_surgical=True,
                        surgery_id=sid,
                        ed_treatment_hours_remaining=0.5,
                    )
                    self.state.patients[pid] = patient
                    self.state.units[UnitName.ED.value].patients.append(pid)
                    events.append(f"Surgical patient {pid} arrived for surgery {sid}")
        return events

    def _process_surgeries(self) -> list[str]:
        events: list[str] = []
        for sid, surgery in self.state.surgeries.items():
            if surgery.is_cancelled or surgery.completed:
                continue

            # Start surgery if scheduled
            if (not surgery.started
                    and self.state.current_hour >= surgery.scheduled_hour):
                pid = surgery.patient_id
                patient = self.state.patients.get(pid)
                if patient is None:
                    continue

                or_unit = self.state.units[UnitName.OR.value]
                if or_unit.occupied_beds < or_unit.total_beds:
                    surgery.started = True
                    or_unit.patients.append(pid)

                    # Remove from ED
                    ed = self.state.units[UnitName.ED.value]
                    if pid in ed.patients:
                        ed.patients.remove(pid)

                    patient.status = PatientStatus.IN_SURGERY
                    patient.location = UnitName.OR
                    events.append(f"Surgery {sid} started (patient {pid}, OR room {surgery.or_room})")

            # Complete surgery if duration elapsed
            if (surgery.started and not surgery.completed
                    and self.state.current_hour >= surgery.scheduled_hour + surgery.duration_hours):
                surgery.completed = True
                pid = surgery.patient_id
                patient = self.state.patients.get(pid)
                if patient is None:
                    continue

                or_unit = self.state.units[UnitName.OR.value]
                if pid in or_unit.patients:
                    or_unit.patients.remove(pid)

                # Move to PACU
                pacu = self.state.units[UnitName.PACU.value]
                patient.status = PatientStatus.IN_PACU
                patient.location = UnitName.PACU
                patient.pacu_hours_remaining = 1.0 + float(self.rng.random())  # 1-2h
                patient.target_unit = surgery.post_op_destination
                pacu.patients.append(pid)
                events.append(f"Surgery {sid} completed, patient {pid} -> PACU")

        return events

    def _process_pacu(self) -> list[str]:
        """Move PACU patients to boarding status when recovery is complete."""
        events: list[str] = []
        pacu = self.state.units[UnitName.PACU.value]

        for pid in list(pacu.patients):
            patient = self.state.patients[pid]
            if patient.status != PatientStatus.IN_PACU:
                continue
            patient.pacu_hours_remaining -= 1.0
            if patient.pacu_hours_remaining <= 0:
                # PACU complete - patient needs to be placed in target unit
                pacu.patients.remove(pid)
                patient.status = PatientStatus.BOARDING_IN_ED
                patient.location = UnitName.ED
                patient.needs_admission = True
                # Set LOS for destination unit
                if patient.target_unit and patient.target_unit in LOS_PARAMS:
                    params = LOS_PARAMS[patient.target_unit]
                    patient.los_remaining_hours = float(
                        self.rng.lognormal(np.log(params["median"]), params["sigma"])
                    )
                else:
                    patient.los_remaining_hours = 48.0
                ed = self.state.units[UnitName.ED.value]
                ed.patients.append(pid)
                events.append(f"Patient {pid} PACU complete, boarding for {patient.target_unit.value if patient.target_unit else 'unit'}")

        return events

    # ----- Bed Cleaning -----

    def _start_bed_cleaning(self, unit_name: UnitName):
        # Source: AHE (Association for the Healthcare Environment)
        # terminal cleaning standard: 40-45 min (we add overhead for 45-60 min total).
        unit_state = self.state.units[unit_name.value]
        cleaning_duration = 0.75 + float(self.rng.random()) * 0.25  # 45-60 min
        finish_time = self.state.current_hour + cleaning_duration
        unit_state.beds_being_cleaned += 1
        unit_state.cleaning_finish_times.append(finish_time)

    def _update_bed_cleaning(self):
        for unit_state in self.state.units.values():
            completed = [t for t in unit_state.cleaning_finish_times
                         if t <= self.state.current_hour]
            unit_state.beds_being_cleaned -= len(completed)
            unit_state.cleaning_finish_times = [
                t for t in unit_state.cleaning_finish_times
                if t > self.state.current_hour
            ]

    # ----- ED Patient Updates -----

    def _update_ed_patients(self) -> list[str]:
        """Update ED treatment timers; transition to boarding when treatment complete."""
        events: list[str] = []
        for pid in list(self.state.units[UnitName.ED.value].patients):
            patient = self.state.patients.get(pid)
            if patient is None:
                continue

            if patient.status == PatientStatus.WAITING_IN_ED:
                patient.ed_treatment_hours_remaining -= 1.0
                if patient.ed_treatment_hours_remaining <= 0:
                    if patient.needs_admission:
                        patient.status = PatientStatus.BOARDING_IN_ED
                        events.append(f"Patient {pid} (ESI-{patient.esi}) now boarding in ED, needs {patient.target_unit.value if patient.target_unit else 'unit'}")
                    else:
                        # Non-admitted patient ready to go
                        patient.los_remaining_hours -= 1.0
                        if patient.los_remaining_hours <= 0:
                            self._discharge_ed_patient(pid)
                            events.append(f"Patient {pid} treated and discharged from ED")

            elif patient.status == PatientStatus.BOARDING_IN_ED:
                if not patient.needs_admission:
                    # Still in ED treatment, count down
                    patient.los_remaining_hours -= 1.0
                    if patient.los_remaining_hours <= 0:
                        self._discharge_ed_patient(pid)
                        events.append(f"Patient {pid} treated and discharged from ED")

        return events

    def _discharge_ed_patient(self, pid: str):
        patient = self.state.patients[pid]
        patient.status = PatientStatus.DISCHARGED
        patient.location = UnitName.ED
        ed = self.state.units[UnitName.ED.value]
        if pid in ed.patients:
            ed.patients.remove(pid)

    # ----- Inpatient LOS Updates -----

    def _update_patient_los(self) -> list[str]:
        events: list[str] = []
        inpatient_units = [UnitName.ICU, UnitName.STEPDOWN, UnitName.MEDSURG_A, UnitName.MEDSURG_B]
        for unit_name in inpatient_units:
            unit_state = self.state.units[unit_name.value]
            for pid in list(unit_state.patients):
                patient = self.state.patients.get(pid)
                if patient is None or patient.status != PatientStatus.ADMITTED:
                    continue
                patient.los_remaining_hours -= 1.0
                if patient.los_remaining_hours <= 0:
                    patient.status = PatientStatus.READY_FOR_DISCHARGE
                    events.append(f"Patient {pid} in {unit_name.value} is ready for discharge")
        return events

    # ----- Deterioration -----

    def _process_deterioration(self) -> list[str]:
        events: list[str] = []
        for pid, patient in list(self.state.patients.items()):
            if patient.status in (PatientStatus.DISCHARGED, PatientStatus.DECEASED,
                                  PatientStatus.IN_SURGERY, PatientStatus.IN_PACU):
                continue

            base_rate = 0.0
            if patient.status == PatientStatus.BOARDING_IN_ED:
                base_rate = 0.02 if patient.esi <= 2 else 0.005
            elif patient.status == PatientStatus.ADMITTED:
                base_rate = 0.002
            elif patient.status == PatientStatus.WAITING_IN_ED:
                base_rate = 0.001

            # Understaffing multiplier
            if patient.location.value in self.state.units:
                unit_state = self.state.units[patient.location.value]
                if unit_state.is_understaffed and len(unit_state.nurses) > 0:
                    over = unit_state.nurse_patient_ratio / unit_state.safe_ratio
                    # Source: Aiken et al., JAMA, 2002: each additional patient per
                    # nurse increases mortality odds by 7% (OR 1.07). Peutere et al.,
                    # Int J Nursing Studies, 2024: 1.05x mortality per 20% increase.
                    # We use 2.0x cap as upper bound for simulation purposes.
                    base_rate *= min(2.0, max(1.0, over))

            if base_rate > 0 and self.rng.random() < base_rate:
                if patient.esi > 1:
                    old_esi = patient.esi
                    patient.esi -= 1
                    # Re-evaluate target unit
                    if patient.esi <= 2:
                        patient.target_unit = UnitName.ICU
                    elif patient.esi == 3 and patient.target_unit in (UnitName.MEDSURG_A, UnitName.MEDSURG_B):
                        patient.target_unit = UnitName.STEPDOWN
                    events.append(f"Patient {pid} deteriorated: ESI-{old_esi} -> ESI-{patient.esi}")

        return events

    # ----- Mortality -----

    def _process_boarding_mortality(self) -> list[str]:
        events: list[str] = []
        for pid, patient in list(self.state.patients.items()):
            if patient.status in (PatientStatus.DISCHARGED, PatientStatus.DECEASED,
                                  PatientStatus.IN_SURGERY):
                continue

            mortality_prob = 0.0

            # ICU delay mortality
            if (patient.esi <= 2
                    and patient.target_unit == UnitName.ICU
                    and patient.location != UnitName.ICU):
                patient.icu_delay_hours += 1.0
                mortality_prob += 0.005

            # ED boarding mortality beyond 4h
            if patient.status == PatientStatus.BOARDING_IN_ED:
                patient.ed_boarding_hours += 1.0
                if patient.ed_boarding_hours > 4.0:
                    extra = patient.ed_boarding_hours - 4.0
                    mortality_prob += 0.002 * extra

            # Base mortality
            mortality_prob += BASE_MORTALITY.get(patient.esi, 0.0)

            # Wrong-unit penalty
            if patient.status == PatientStatus.ADMITTED:
                if patient.target_unit == UnitName.ICU and patient.location != UnitName.ICU:
                    mortality_prob *= 1.5
                elif (patient.target_unit == UnitName.STEPDOWN
                      and patient.location in (UnitName.MEDSURG_A, UnitName.MEDSURG_B)):
                    mortality_prob *= 1.2

            if mortality_prob > 0 and self.rng.random() < mortality_prob:
                self._kill_patient(pid, "clinical_deterioration")
                events.append(
                    f"DEATH: Patient {pid} (ESI-{patient.esi}, "
                    f"boarding={patient.ed_boarding_hours:.1f}h, "
                    f"icu_delay={patient.icu_delay_hours:.1f}h)"
                )

        return events

    def _process_understaffing_mortality(self) -> list[str]:
        events: list[str] = []
        for unit_name_str, unit_state in self.state.units.items():
            if unit_state.unit in (UnitName.OR, UnitName.ED):
                continue
            if unit_state.is_understaffed and unit_state.occupied_beds > 0 and len(unit_state.nurses) > 0:
                over = unit_state.nurse_patient_ratio / unit_state.safe_ratio
                extra_mort = 0.001 * (over - 1.0)
                for pid in list(unit_state.patients):
                    patient = self.state.patients.get(pid)
                    if patient and patient.status == PatientStatus.ADMITTED:
                        if self.rng.random() < extra_mort:
                            self._kill_patient(pid, "understaffing")
                            events.append(f"DEATH: Patient {pid} in {unit_name_str} (understaffing, ratio={over:.1f}x)")
        return events

    def _kill_patient(self, pid: str, cause: str):
        patient = self.state.patients[pid]
        patient.status = PatientStatus.DECEASED

        # Remove from unit
        for unit_state in self.state.units.values():
            if pid in unit_state.patients:
                unit_state.patients.remove(pid)
                self._start_bed_cleaning(unit_state.unit)
                break

        self.state.deaths.append({
            "patient_id": pid,
            "esi": patient.esi,
            "cause": cause,
            "hour": self.state.current_hour,
            "ed_boarding_hours": patient.ed_boarding_hours,
            "icu_delay_hours": patient.icu_delay_hours,
        })

    # ----- Staffing -----

    def _process_agency_arrivals(self) -> list[str]:
        events: list[str] = []
        remaining: list[dict] = []
        for req in self.state.pending_agency:
            if self.state.current_hour >= req["available_at"]:
                unit_name = UnitName(req["unit"])
                for _ in range(req["count"]):
                    # Agency nurse works until next shift boundary
                    shift_end = (int(self.state.current_hour / 12) + 1) * 12.0
                    nurse = self._create_nurse(unit_name, self.state.current_hour, shift_end, StaffType.AGENCY)
                    self.state.nurses[nurse.id] = nurse
                    self.state.units[unit_name.value].nurses.append(nurse.id)
                events.append(f"{req['count']} agency nurse(s) arrived for {req['unit']}")
            else:
                remaining.append(req)
        self.state.pending_agency = remaining
        return events

    def _process_shift_change(self) -> list[str]:
        """Replace nurses whose shift has ended."""
        events: list[str] = []
        current = self.state.current_hour
        shift_end_new = current + 12.0

        for unit_name, base_count in BASE_NURSES.items():
            unit_state = self.state.units[unit_name.value]

            # Remove nurses whose shift ended
            expired = [nid for nid in unit_state.nurses
                       if self.state.nurses[nid].shift_end <= current]
            for nid in expired:
                unit_state.nurses.remove(nid)
                del self.state.nurses[nid]

            # Add new shift nurses
            count = max(1, int(base_count * self.config.staffing_multiplier))
            for _ in range(count):
                nurse = self._create_nurse(unit_name, current, shift_end_new, StaffType.REGULAR)
                self.state.nurses[nurse.id] = nurse
                unit_state.nurses.append(nurse.id)

        events.append(f"Shift change at hour {int(current)}")
        return events

    # ----- Reward Calculation -----

    def _calculate_hourly_score(self) -> float:
        penalty = 0.0

        # 1. Deaths this hour
        deaths_this_hour = [d for d in self.state.deaths
                            if int(d["hour"]) == int(self.state.current_hour)]
        penalty += 0.5 * len(deaths_this_hour)

        # 2. ED boarding (per patient currently boarding)
        boarding = sum(1 for p in self.state.patients.values()
                       if p.status == PatientStatus.BOARDING_IN_ED)
        penalty += 0.001 * boarding

        # 3. ICU delay (critical patients not in ICU)
        icu_delayed = sum(1 for p in self.state.patients.values()
                          if (p.esi <= 2
                              and p.target_unit == UnitName.ICU
                              and p.location != UnitName.ICU
                              and p.status not in (PatientStatus.DISCHARGED,
                                                   PatientStatus.DECEASED,
                                                   PatientStatus.IN_SURGERY)))
        penalty += 0.01 * icu_delayed

        # 4. Understaffing
        understaffed = sum(1 for u in self.state.units.values()
                           if u.is_understaffed
                           and u.unit not in (UnitName.OR,)
                           and u.occupied_beds > 0)
        penalty += 0.02 * understaffed

        # 5. Diversion
        if self.state.diversion_active:
            penalty += 0.05

        return max(0.0, 1.0 - penalty)

    def get_final_reward(self) -> float:
        if not self.state.hourly_scores:
            return 0.0
        base = float(np.mean(self.state.hourly_scores))
        # Subtract cancellation penalty
        cancel_penalty = self.state.surgeries_cancelled * 0.01
        return max(0.0, base - cancel_penalty)

    # ----- Action Methods (called by tools) -----

    def admit_patient(self, patient_id: str, target_unit: UnitName) -> str:
        """Admit a patient from ED to an inpatient unit. Returns status message."""
        patient = self.state.patients.get(patient_id)
        if patient is None:
            return f"Error: Patient {patient_id} not found."
        if patient.status not in (PatientStatus.WAITING_IN_ED, PatientStatus.BOARDING_IN_ED):
            return f"Error: Patient {patient_id} is not in ED (status: {patient.status.value})."
        if not patient.needs_admission:
            return f"Error: Patient {patient_id} does not require inpatient admission."

        unit_state = self.state.units[target_unit.value]
        if unit_state.available_beds <= 0:
            return (f"Error: No available beds in {target_unit.value} "
                    f"({unit_state.occupied_beds}/{unit_state.total_beds} occupied, "
                    f"{unit_state.beds_being_cleaned} cleaning).")

        # Execute admission
        ed = self.state.units[UnitName.ED.value]
        if patient_id in ed.patients:
            ed.patients.remove(patient_id)

        unit_state.patients.append(patient_id)
        patient.status = PatientStatus.ADMITTED
        patient.location = target_unit
        patient.admitted_hour = self.state.current_hour

        # Set LOS for destination
        if target_unit in LOS_PARAMS:
            params = LOS_PARAMS[target_unit]
            patient.los_remaining_hours = float(
                self.rng.lognormal(np.log(params["median"]), params["sigma"])
            )
        else:
            patient.los_remaining_hours = 48.0

        return (f"Patient {patient_id} (ESI-{patient.esi}) admitted to {target_unit.value}. "
                f"Beds remaining: {unit_state.available_beds}. "
                f"ED boarding time was {patient.ed_boarding_hours:.1f}h.")

    def transfer_patient(self, patient_id: str, to_unit: UnitName) -> str:
        """Transfer a patient between inpatient units."""
        patient = self.state.patients.get(patient_id)
        if patient is None:
            return f"Error: Patient {patient_id} not found."
        if patient.status not in (PatientStatus.ADMITTED, PatientStatus.READY_FOR_DISCHARGE):
            return f"Error: Patient {patient_id} cannot be transferred (status: {patient.status.value})."

        target_state = self.state.units[to_unit.value]
        if target_state.available_beds <= 0:
            return (f"Error: No available beds in {to_unit.value} "
                    f"({target_state.occupied_beds}/{target_state.total_beds} occupied).")

        # Remove from source
        source_unit = patient.location
        source_state = self.state.units[source_unit.value]
        if patient_id in source_state.patients:
            source_state.patients.remove(patient_id)
            self._start_bed_cleaning(source_unit)

        # Add to target
        target_state.patients.append(patient_id)
        patient.location = to_unit
        patient.status = PatientStatus.ADMITTED

        # Reset LOS for new unit
        if to_unit in LOS_PARAMS:
            params = LOS_PARAMS[to_unit]
            patient.los_remaining_hours = float(
                self.rng.lognormal(np.log(params["median"]), params["sigma"])
            )

        return (f"Patient {patient_id} transferred from {source_unit.value} to {to_unit.value}. "
                f"Beds remaining in {to_unit.value}: {target_state.available_beds}.")

    def discharge_patient(self, patient_id: str) -> str:
        """Discharge a patient who is ready."""
        patient = self.state.patients.get(patient_id)
        if patient is None:
            return f"Error: Patient {patient_id} not found."
        if patient.status != PatientStatus.READY_FOR_DISCHARGE:
            return f"Error: Patient {patient_id} is not ready for discharge (status: {patient.status.value})."

        # Remove from unit
        unit_state = self.state.units[patient.location.value]
        if patient_id in unit_state.patients:
            unit_state.patients.remove(patient_id)
            self._start_bed_cleaning(patient.location)

        patient.status = PatientStatus.DISCHARGED
        return (f"Patient {patient_id} discharged from {patient.location.value}. "
                f"Bed will be available after cleaning (~45-60 min).")

    def cancel_surgery(self, surgery_id: str) -> str:
        """Cancel an elective surgery."""
        surgery = self.state.surgeries.get(surgery_id)
        if surgery is None:
            return f"Error: Surgery {surgery_id} not found."
        if surgery.is_cancelled:
            return f"Error: Surgery {surgery_id} is already cancelled."
        if surgery.started:
            return f"Error: Surgery {surgery_id} has already started and cannot be cancelled."

        surgery.is_cancelled = True
        self.state.surgeries_cancelled += 1

        # Remove surgical patient if they haven't arrived yet
        pid = surgery.patient_id
        patient = self.state.patients.get(pid)
        if patient:
            if patient.status == PatientStatus.WAITING_IN_ED:
                ed = self.state.units[UnitName.ED.value]
                if pid in ed.patients:
                    ed.patients.remove(pid)
                patient.status = PatientStatus.DISCHARGED

        return f"Surgery {surgery_id} cancelled. Patient impact: elective case postponed."

    def request_agency_staff(self, unit: str, count: int) -> str:
        """Request agency nurses for a unit."""
        available_at = self.state.current_hour + 4.0
        self.state.pending_agency.append({
            "unit": unit,
            "count": count,
            "available_at": available_at,
        })
        return (f"Requested {count} agency nurse(s) for {unit}. "
                f"Estimated arrival: hour {available_at:.0f} (4-hour delay). "
                f"Cost: 2x standard rate.")

    # ----- Dashboard Formatting -----

    def format_dashboard(self) -> str:
        """Format full hospital dashboard as text."""
        lines: list[str] = []
        hour = self.state.current_hour
        remaining = self.state.simulation_duration - hour

        lines.append(f"{'='*60}")
        lines.append(f"MERCY GENERAL HOSPITAL - HOUR {int(hour)}/{int(self.state.simulation_duration)}")
        lines.append(f"Time: Day {int(hour//24)+1}, {int(hour%24):02d}:00 | Hours remaining: {int(remaining)}")
        lines.append(f"Diversion: {'ACTIVE' if self.state.diversion_active else 'Off'}")
        if self.state.hourly_scores:
            avg = np.mean(self.state.hourly_scores)
            lines.append(f"Running score: {avg:.4f}")
        lines.append(f"{'='*60}")

        # Unit status table
        lines.append("")
        lines.append(f"{'Unit':<12} {'Beds':>5} {'Occ':>4} {'Clean':>5} {'Avail':>5} {'Nurses':>6} {'Ratio':>6} {'Status':>8}")
        lines.append(f"{'-'*12} {'-'*5} {'-'*4} {'-'*5} {'-'*5} {'-'*6} {'-'*6} {'-'*8}")

        for unit_name in [UnitName.ED, UnitName.ICU, UnitName.STEPDOWN,
                          UnitName.MEDSURG_A, UnitName.MEDSURG_B, UnitName.PACU]:
            us = self.state.units[unit_name.value]
            ratio = f"{us.nurse_patient_ratio:.1f}" if us.occupied_beds > 0 else "-"
            status = "UNSAFE" if us.is_understaffed else "OK"
            lines.append(
                f"{unit_name.value:<12} {us.total_beds:>5} {us.occupied_beds:>4} "
                f"{us.beds_being_cleaned:>5} {us.available_beds:>5} "
                f"{len(us.nurses):>6} {ratio:>6} {status:>8}"
            )

        # OR status
        or_unit = self.state.units[UnitName.OR.value]
        lines.append(f"{'or':<12} {or_unit.total_beds:>5} {or_unit.occupied_beds:>4} "
                      f"{'':>5} {or_unit.total_beds - or_unit.occupied_beds:>5} "
                      f"{'':>6} {'':>6} {'':>8}")

        # ED Patients needing admission
        boarding = [p for p in self.state.patients.values()
                    if p.status == PatientStatus.BOARDING_IN_ED]
        if boarding:
            lines.append("")
            lines.append(f"PATIENTS BOARDING IN ED ({len(boarding)}):")
            boarding.sort(key=lambda p: (-p.esi, -p.ed_boarding_hours), reverse=True)
            boarding.sort(key=lambda p: p.esi)
            for p in boarding[:20]:
                target = p.target_unit.value if p.target_unit else "?"
                lines.append(
                    f"  {p.id} | ESI-{p.esi} | Boarding: {p.ed_boarding_hours:.1f}h | "
                    f"Needs: {target} | ICU delay: {p.icu_delay_hours:.1f}h"
                )
            if len(boarding) > 20:
                lines.append(f"  ... and {len(boarding)-20} more")

        # Patients waiting in ED (not yet boarding)
        waiting = [p for p in self.state.patients.values()
                   if p.status == PatientStatus.WAITING_IN_ED and p.needs_admission]
        if waiting:
            lines.append("")
            lines.append(f"PATIENTS WAITING IN ED - PENDING WORKUP ({len(waiting)}):")
            for p in waiting[:10]:
                target = p.target_unit.value if p.target_unit else "?"
                lines.append(
                    f"  {p.id} | ESI-{p.esi} | ED time remaining: {p.ed_treatment_hours_remaining:.1f}h | "
                    f"Will need: {target}"
                )

        # Patients ready for discharge
        ready = [p for p in self.state.patients.values()
                 if p.status == PatientStatus.READY_FOR_DISCHARGE]
        if ready:
            lines.append("")
            lines.append(f"PATIENTS READY FOR DISCHARGE ({len(ready)}):")
            for p in ready[:15]:
                lines.append(f"  {p.id} | Unit: {p.location.value} | ESI-{p.esi}")
            if len(ready) > 15:
                lines.append(f"  ... and {len(ready)-15} more")

        # Upcoming surgeries
        upcoming = [(sid, s) for sid, s in self.state.surgeries.items()
                    if not s.is_cancelled and not s.completed
                    and s.scheduled_hour >= hour
                    and s.scheduled_hour < hour + 12]
        in_progress = [(sid, s) for sid, s in self.state.surgeries.items()
                       if s.started and not s.completed]
        if upcoming or in_progress:
            lines.append("")
            lines.append("OR SCHEDULE:")
            for sid, s in in_progress:
                remaining_h = max(0, s.scheduled_hour + s.duration_hours - hour)
                lines.append(f"  {sid} IN PROGRESS | Patient {s.patient_id} | "
                             f"~{remaining_h:.1f}h remaining | Post-op: {s.post_op_destination.value}")
            for sid, s in upcoming[:8]:
                if not s.started:
                    lines.append(f"  {sid} | Scheduled hour {int(s.scheduled_hour)} | "
                                 f"Duration: {s.duration_hours:.1f}h | Post-op: {s.post_op_destination.value}")

        # Pending agency requests
        if self.state.pending_agency:
            lines.append("")
            lines.append("PENDING AGENCY STAFF:")
            for req in self.state.pending_agency:
                lines.append(f"  {req['count']} nurse(s) for {req['unit']} | "
                             f"Arrives hour {int(req['available_at'])}")

        # Deaths this simulation
        total_deaths = len(self.state.deaths)
        if total_deaths > 0:
            lines.append("")
            lines.append(f"TOTAL DEATHS THIS SIMULATION: {total_deaths}")
            recent_deaths = [d for d in self.state.deaths if d["hour"] >= hour - 4]
            for d in recent_deaths[-5:]:
                lines.append(f"  Hour {int(d['hour'])}: Patient {d['patient_id']} "
                             f"(ESI-{d['esi']}, cause: {d['cause']})")

        # Recent events
        recent = self.state.events_log[-10:]
        if recent:
            lines.append("")
            lines.append("RECENT EVENTS:")
            for e in recent:
                lines.append(f"  {e}")

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def get_dashboard_metadata(self) -> dict:
        """Return structured dashboard data."""
        unit_data = {}
        for name, us in self.state.units.items():
            unit_data[name] = {
                "total_beds": us.total_beds,
                "occupied": us.occupied_beds,
                "available": us.available_beds,
                "cleaning": us.beds_being_cleaned,
                "nurses": len(us.nurses),
                "ratio": round(us.nurse_patient_ratio, 2),
                "understaffed": us.is_understaffed,
            }

        return {
            "hour": self.state.current_hour,
            "hours_remaining": self.state.simulation_duration - self.state.current_hour,
            "diversion": self.state.diversion_active,
            "units": unit_data,
            "boarding_count": sum(1 for p in self.state.patients.values()
                                  if p.status == PatientStatus.BOARDING_IN_ED),
            "deaths_total": len(self.state.deaths),
            "running_score": float(np.mean(self.state.hourly_scores)) if self.state.hourly_scores else 1.0,
        }
