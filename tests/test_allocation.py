import copy

import pandas as pd
import pytest

from src.allocation import (
    BELOW_MINIMUM,
    CANDIDATE_COLUMNS,
    CONSTRAINT,
    FIT_WEIGHTS,
    NO_SKILL,
    PARTIAL,
    QUALIFICATION_COLUMNS,
    QUALIFIED,
    STRONG,
    classify_proficiency,
    evaluate_candidates,
    fit_label,
    get_qualified_resources,
    get_resource_qualifications,
)
from src.capacity import build_capacity_table
from src.data_loader import load_data


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def capacity(data):
    return build_capacity_table(data)


def cand(df, resource):
    return df[df["ResourceID"] == resource].iloc[0]


# =====================================================================
# Hand-made dataset: one requirement, six resources, every answer known in advance
#
# Requirement Q1: role "Dev", skill Python, minimum proficiency 4, 10h/week, weeks 1-2
#
#   A  Dev      python 5  no load            -> 50 + 30 + 20  = 100          Strong
#   B  Dev      python 2  no load            -> 25 + 30 + 20  =  75 (Strong) capped to Partial
#   C  Dev      python 4  35h wk1, 40h wk2   -> 50 +  0 +  0  =  50          Partial, fully allocated
#   D  Dev      python 2  40h both weeks     -> 25 +  0 +  0  =  25          Constraint, no availability
#   E  Dev      python 4  38h both weeks     -> 50 +  6 +  1  =  57          Partial, insufficient hours
#   F  Analyst  python 4  no load            -> 50 + 30 + 20  = 100          Strong, role mismatch (soft)
#   G  Dev      (no python skill)            -> not a candidate
# =====================================================================
def tiny():
    resources = [(r, f"Person {r}", "Analyst" if r == "F" else "Dev", 40) for r in "ABCDEFG"]
    skills = [("A", "Python", 5), ("B", "Python", 2), ("C", "Python", 4), ("D", "Python", 2),
              ("E", "Python", 4), ("F", "Python", 4), ("G", "SQL", 5)]
    allocs = [("C", 1, 35), ("C", 2, 40), ("D", 1, 40), ("D", 2, 40), ("E", 1, 38), ("E", 2, 38)]
    return {
        "Resources": pd.DataFrame(resources, columns=["ResourceID", "ResourceName", "Role", "WeeklyCapacityHrs"]),
        "Resource_Skills": pd.DataFrame(skills, columns=["ResourceID", "Skill", "Proficiency"]),
        "Project_Requirements": pd.DataFrame(
            [("Q1", "P1", "Dev", "Python", 4, 10, 1, 2)],
            columns=["RequirementID", "ProjectID", "Role", "Skill", "MinProficiency", "HoursPerWeek", "StartWeek", "EndWeek"],
        ),
        "Allocations": pd.DataFrame(
            [(r, "Q1", "P1", w, h) for r, w, h in allocs],
            columns=["ResourceID", "RequirementID", "ProjectID", "Week", "PlannedHrs"],
        ),
        "Calendar": pd.DataFrame({"Week": [1, 2, 3]}),
    }


@pytest.fixture
def small():
    return evaluate_candidates(tiny(), "Q1")


# ---------- Step 9: qualification ----------
@pytest.mark.parametrize(
    "proficiency, minimum, expected",
    [(5, 4, QUALIFIED), (4, 4, QUALIFIED), (3, 4, BELOW_MINIMUM), (1, 4, BELOW_MINIMUM),
     (0, 4, NO_SKILL), (None, 4, NO_SKILL), (float("nan"), 4, NO_SKILL)],
)
def test_classify_proficiency(proficiency, minimum, expected):
    assert classify_proficiency(proficiency, minimum) == expected


def test_spec_example_qualification_on_real_data(data):
    # Q020 needs Cloud Infrastructure >= 4: R15 has 5 (qualified), R08 has 3 (not qualified)
    q = get_resource_qualifications(data, "Q020").set_index("ResourceID")
    assert q.loc["R15", "Qualification"] == QUALIFIED
    assert q.loc["R08", "Qualification"] == BELOW_MINIMUM
    assert q.loc["R21", "Qualification"] == NO_SKILL  # a UX designer has no cloud skill
    assert list(get_resource_qualifications(data, "Q020").columns) == QUALIFICATION_COLUMNS


def test_every_resource_gets_a_qualification_row(data):
    assert len(get_resource_qualifications(data, "Q020")) == 28


def test_only_one_person_is_qualified_for_q020(data):
    assert get_qualified_resources(data, "Q020") == ["R15"]


def test_qualification_depends_on_the_requirements_minimum_not_just_the_skill(data):
    # Q043 needs Data Visualization >= 5; R12 has 4 so is below, R11 has 5
    assert get_qualified_resources(data, "Q043") == ["R11"]


# ---------- Step 10: candidate dimensions ----------
def test_candidate_columns_and_sorting(small):
    assert list(small.columns) == CANDIDATE_COLUMNS
    assert list(small["FitScore"]) == sorted(small["FitScore"], reverse=True)


def test_resource_with_no_skill_is_not_a_candidate(small):
    assert "G" not in set(small["ResourceID"])
    assert set(small["ResourceID"]) == set("ABCDEF")


def test_dimensions_are_shown_separately(small):
    e = cand(small, "E")
    assert e["SkillMatch"] == pytest.approx(1.0)
    assert e["TightestFreeHrs"] == 2
    assert e["Availability"] == pytest.approx(0.2)
    assert e["PeakUtilization"] == pytest.approx(0.95)
    assert e["LoadHeadroom"] == pytest.approx(0.05)


def test_skill_match_is_capped_at_one(small):
    assert cand(small, "A")["SkillMatch"] == 1.0  # proficiency 5 vs minimum 4


def test_availability_uses_the_tightest_week_not_the_average(small):
    c = cand(small, "C")  # free 5h in week 1, 0h in week 2
    assert c["TightestFreeHrs"] == 0
    assert c["Availability"] == 0


# ---------- Step 11: fit score ----------
def test_fit_scores_match_hand_calculation(small):
    expected = {"A": 100, "B": 75, "C": 50, "D": 25, "E": 57, "F": 100}
    for resource, score in expected.items():
        assert cand(small, resource)["FitScore"] == pytest.approx(score), resource


def test_fit_labels(small):
    expected = {"A": STRONG, "B": PARTIAL, "C": PARTIAL, "D": CONSTRAINT, "E": PARTIAL, "F": STRONG}
    for resource, label in expected.items():
        assert cand(small, resource)["FitLabel"] == label, resource


def test_below_minimum_proficiency_is_capped_at_partial_even_when_score_is_strong(small):
    b = cand(small, "B")
    assert b["FitScore"] == pytest.approx(75)       # score alone would be Strong
    assert b["FitLabel"] == PARTIAL                  # but the cap applies
    assert "Below required proficiency" in b["Constraints"]


@pytest.mark.parametrize("score, label", [(100, STRONG), (75, STRONG), (74.99, PARTIAL),
                                          (50, PARTIAL), (49.99, CONSTRAINT), (0, CONSTRAINT)])
def test_label_thresholds(score, label):
    assert fit_label(score) == label


def test_cap_only_applies_to_below_minimum():
    assert fit_label(90, QUALIFIED) == STRONG
    assert fit_label(90, BELOW_MINIMUM) == PARTIAL
    assert fit_label(30, BELOW_MINIMUM) == CONSTRAINT


def test_weights_are_the_documented_assumptions():
    assert FIT_WEIGHTS == {"skill": 50, "availability": 30, "headroom": 20}


def test_weights_are_configurable():
    availability_only = {"skill": 0, "availability": 100, "headroom": 0}
    result = evaluate_candidates(tiny(), "Q1", weights=availability_only)
    assert cand(result, "A")["FitScore"] == pytest.approx(100)
    assert cand(result, "E")["FitScore"] == pytest.approx(20)  # 0.2 availability


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError):
        evaluate_candidates(tiny(), "Q1", weights={"skill": 1})
    with pytest.raises(ValueError):
        evaluate_candidates(tiny(), "Q1", weights={"skill": -1, "availability": 1, "headroom": 1})


# ---------- Step 12: constraint explanations ----------
def test_fully_allocated_flag(small):
    assert cand(small, "C")["Constraints"] == "Already fully allocated"


def test_no_availability_flag(small):
    d = cand(small, "D")
    assert "No availability" in d["Constraints"]
    assert "Below required proficiency" in d["Constraints"]


def test_insufficient_hours_flag(small):
    assert cand(small, "E")["Constraints"] == "Insufficient hours"


def test_unconstrained_candidate_has_no_flags(small):
    assert cand(small, "A")["Constraints"] == ""


def test_role_mismatch_is_a_soft_flag_and_does_not_remove_the_candidate(small):
    f = cand(small, "F")
    assert f["Constraints"] == "Role mismatch"
    assert f["FitLabel"] == STRONG


# ---------- real data, hand-checked from the raw rows ----------
def test_q020_open_cloud_requirement(data, capacity):
    result = evaluate_candidates(data, "Q020", capacity=capacity)
    assert set(result["ResourceID"]) == {"R01", "R08", "R15", "R16", "R17"}

    r15 = cand(result, "R15")  # only qualified person, but nearly full
    assert r15["TightestFreeHrs"] == 2
    assert r15["Availability"] == pytest.approx(0.2)
    assert r15["FitScore"] == pytest.approx(57.0)
    assert r15["FitLabel"] == PARTIAL
    assert r15["Constraints"] == "Insufficient hours"

    r16 = cand(result, "R16")  # plenty of room but one level below the minimum
    assert r16["FitScore"] == pytest.approx(78.5)
    assert r16["FitLabel"] == PARTIAL  # capped, not Strong
    assert r16["Constraints"] == "Below required proficiency"


def test_skill_fit_is_not_the_same_as_suitability_on_q043(data, capacity):
    result = evaluate_candidates(data, "Q043", capacity=capacity)
    r11, r12 = cand(result, "R11"), cand(result, "R12")
    assert r11["SkillMatch"] == 1.0 and r12["SkillMatch"] == pytest.approx(0.8)
    assert r11["TightestFreeHrs"] == 4 and r12["TightestFreeHrs"] == 24
    assert r11["FitScore"] == pytest.approx(58.667, abs=1e-3)
    assert r12["FitScore"] == pytest.approx(82.0)
    assert r12["FitScore"] > r11["FitScore"]  # the less-skilled but free person ranks higher
    assert r11["Constraints"] == "Insufficient hours"


def test_capacity_argument_gives_the_same_answer_as_computing_it(data, capacity):
    pd.testing.assert_frame_equal(
        evaluate_candidates(data, "Q020"), evaluate_candidates(data, "Q020", capacity=capacity)
    )


# ---------- edge cases ----------
def test_unknown_requirement_raises_a_clear_error(data):
    with pytest.raises(ValueError, match="Unknown requirement"):
        evaluate_candidates(data, "Q999")
    with pytest.raises(ValueError, match="Unknown requirement"):
        get_resource_qualifications(data, "Q999")


def test_requirement_nobody_has_the_skill_for_returns_no_candidates():
    d = tiny()
    d["Project_Requirements"].loc[0, "Skill"] = "Welding"
    result = evaluate_candidates(d, "Q1")
    assert result.empty
    assert list(result.columns) == CANDIDATE_COLUMNS


def test_zero_required_hours_means_full_availability():
    d = tiny()
    d["Project_Requirements"].loc[0, "HoursPerWeek"] = 0
    assert cand(evaluate_candidates(d, "Q1"), "D")["Availability"] == 1.0


def test_zero_capacity_resource_counts_as_fully_loaded():
    d = tiny()
    d["Resources"].loc[d["Resources"]["ResourceID"] == "A", "WeeklyCapacityHrs"] = 0
    a = cand(evaluate_candidates(d, "Q1"), "A")
    assert a["PeakUtilization"] == 1.0
    assert a["LoadHeadroom"] == 0
    assert a["Availability"] == 0


def test_over_allocated_resource_gets_zero_availability_not_negative():
    d = tiny()
    d["Allocations"] = pd.concat(
        [d["Allocations"], pd.DataFrame([("A", 1, 50), ("A", 2, 50)], columns=["ResourceID", "Week", "PlannedHrs"])],
        ignore_index=True,
    )
    a = cand(evaluate_candidates(d, "Q1"), "A")
    assert a["TightestFreeHrs"] == -10
    assert a["Availability"] == 0
    assert a["LoadHeadroom"] == 0
    assert a["FitScore"] == pytest.approx(50)  # skill only


# ---------- purity ----------
def test_inputs_are_not_modified(data):
    before = copy.deepcopy(data)
    evaluate_candidates(data, "Q020")
    get_resource_qualifications(data, "Q020")
    get_qualified_resources(data, "Q020")
    for name in data:
        pd.testing.assert_frame_equal(data[name], before[name])


# =====================================================================
# Step 13: conflict detection
# =====================================================================
from src.allocation import (  # noqa: E402
    BROADER_POOL,
    CAPACITY_SHORTAGE,
    COMPETING_REQUIREMENTS,
    CONFLICT_COLUMNS,
    LIMITED_BACKUP,
    NO_QUALIFIED,
    OVER_ALLOCATION,
    SCARCITY_COLUMNS,
    SINGLE_POINT,
    SPOF_COLUMNS,
    STAFFING_GAP,
    analyze_scarce_skills,
    detect_conflicts,
    find_single_points_of_failure,
    format_weeks,
    scarcity_category,
)


@pytest.fixture(scope="module")
def conflicts(data):
    return detect_conflicts(data)


def of_type(conflicts, kind):
    return conflicts[conflicts["ConflictType"] == kind]


def conflict_row(conflicts, kind, requirement):
    return of_type(conflicts, kind).query("RequirementID == @requirement").iloc[0]


@pytest.mark.parametrize(
    "weeks, text",
    [([4, 5, 6, 7, 8], "4-8"), ([4, 5, 6, 8], "4-6, 8"), ([3], "3"), ([], ""), ([9, 7, 8, 1], "1, 7-9"), ([2, 2, 3], "2-3")],
)
def test_format_weeks(weeks, text):
    assert format_weeks(weeks) == text


def test_conflict_columns(conflicts):
    assert list(conflicts.columns) == CONFLICT_COLUMNS


def test_baseline_has_no_over_allocation(conflicts):
    assert of_type(conflicts, OVER_ALLOCATION).empty  # the tension comes from unmet demand, not overbooking


def test_staffing_gaps_are_the_three_open_requirements(conflicts):
    gaps = of_type(conflicts, STAFFING_GAP).set_index("RequirementID")
    assert set(gaps.index) == {"Q020", "Q043", "Q045"}
    assert gaps.loc["Q020", "Hours"] == 70 and gaps.loc["Q020", "Weeks"] == "4-10"
    assert gaps.loc["Q043", "Hours"] == 162 and gaps.loc["Q043", "Weeks"] == "3-11"
    assert gaps.loc["Q045", "Hours"] == 56 and gaps.loc["Q045", "ProjectID"] == "P08"


def test_q020_capacity_shortage_matches_hand_calculation(conflicts):
    # R15 is the only qualified person and has 2h free in weeks 4-8: (10 - 2) x 5 weeks = 40h
    r = conflict_row(conflicts, CAPACITY_SHORTAGE, "Q020")
    assert r["Weeks"] == "4-8"
    assert r["Hours"] == 40
    assert "R15" in r["Detail"]


def test_q043_capacity_shortage_matches_hand_calculation(conflicts):
    # R11 is the only qualified person: 4h free in weeks 4-8 (shortfall 14) and 14h free in weeks 9-11 (shortfall 4)
    r = conflict_row(conflicts, CAPACITY_SHORTAGE, "Q043")
    assert r["Weeks"] == "4-11"
    assert r["Hours"] == 14 * 5 + 4 * 3
    assert "R11" in r["Detail"]


def test_requirement_with_a_large_qualified_pool_has_no_capacity_shortage(conflicts):
    # Q045 is unstaffed too, but many people could take it, so it is a gap and not a shortage
    assert conflict_row(conflicts, STAFFING_GAP, "Q045")["Hours"] == 56
    assert of_type(conflicts, CAPACITY_SHORTAGE).query("RequirementID == 'Q045'").empty


def test_competing_requirements_for_q020(conflicts):
    r = conflict_row(conflicts, COMPETING_REQUIREMENTS, "Q020")
    assert r["Hours"] == 60 + 70 + 60
    assert "Q001 (P01) 60h" in r["Detail"]
    assert "Q011 (P02) 70h" in r["Detail"]
    assert "Q035 (P06) 60h" in r["Detail"]


def test_competing_requirements_for_q043(conflicts):
    r = conflict_row(conflicts, COMPETING_REQUIREMENTS, "Q043")
    assert r["Hours"] == 128 + 50 + 80
    assert "Q024 (P05) 128h" in r["Detail"]


def test_total_conflict_rows(conflicts):
    assert conflicts["ConflictType"].value_counts().to_dict() == {
        STAFFING_GAP: 3, CAPACITY_SHORTAGE: 2, COMPETING_REQUIREMENTS: 2,
    }


def test_over_allocation_is_detected_on_hand_made_data():
    d = tiny()
    d["Allocations"] = pd.concat(
        [d["Allocations"], pd.DataFrame([("A", "Q1", "P1", 1, 30), ("A", "Q1", "P1", 1, 20), ("A", "Q1", "P1", 2, 45)],
                                         columns=["ResourceID", "RequirementID", "ProjectID", "Week", "PlannedHrs"])],
        ignore_index=True,
    )
    d["Project_Requirements"].loc[0, "HoursPerWeek"] = 0  # keep demand out of the picture
    over = of_type(detect_conflicts(d), OVER_ALLOCATION)
    a = over[over["ResourceID"] == "A"].iloc[0]
    assert a["Weeks"] == "1-2"
    assert a["Hours"] == 10 + 5            # 50 vs 40, 45 vs 40
    assert "up to 10h" in a["Detail"]
    assert set(over["ResourceID"]) == {"A"}  # D is exactly full (40 of 40), which is not over-allocation


def test_requirement_nobody_qualified_for_is_a_shortage_with_no_one_listed():
    d = tiny()
    d["Project_Requirements"].loc[0, "MinProficiency"] = 5
    d["Resource_Skills"].loc[d["Resource_Skills"]["ResourceID"] == "A", "Proficiency"] = 4
    d["Allocations"] = d["Allocations"].iloc[0:0]  # nobody is allocated, so the 10h/week is wholly unmet
    shortage = of_type(detect_conflicts(d), CAPACITY_SHORTAGE)
    assert len(shortage) == 1
    assert "nobody" in shortage.iloc[0]["Detail"]
    assert shortage.iloc[0]["Hours"] == 20


def test_conflict_detection_does_not_modify_input(data):
    before = copy.deepcopy(data)
    detect_conflicts(data)
    analyze_scarce_skills(data)
    for name in data:
        pd.testing.assert_frame_equal(data[name], before[name])


# =====================================================================
# Step 14: scarce skills / single points of failure
# =====================================================================
@pytest.fixture(scope="module")
def scarcity(data):
    return analyze_scarce_skills(data)


def test_scarcity_columns_and_row_count(scarcity):
    assert list(scarcity.columns) == SCARCITY_COLUMNS
    assert len(scarcity) == 50  # every V2 project is Active or Planned


@pytest.mark.parametrize("count, category",
                         [(0, NO_QUALIFIED), (1, SINGLE_POINT), (2, LIMITED_BACKUP), (3, BROADER_POOL), (10, BROADER_POOL)])
def test_scarcity_category_boundaries(count, category):
    assert scarcity_category(count) == category


def test_category_counts_on_the_v2_data(scarcity):
    assert scarcity["Category"].value_counts().to_dict() == {
        BROADER_POOL: 34, LIMITED_BACKUP: 12, SINGLE_POINT: 4,
    }


def test_the_four_single_point_requirements(scarcity):
    sole = scarcity[scarcity["Category"] == SINGLE_POINT].set_index("RequirementID")["QualifiedResources"].to_dict()
    assert sole == {"Q001": "R15", "Q020": "R15", "Q012": "R06", "Q043": "R11"}


def test_data_engineering_has_limited_backup(scarcity):
    de = scarcity[scarcity["Skill"] == "Data Engineering"]
    assert set(de["QualifiedResources"]) == {"R08, R09"}
    assert set(de["Category"]) == {LIMITED_BACKUP}


def test_single_points_of_failure_by_resource(data, scarcity):
    spof = find_single_points_of_failure(data, scarcity=scarcity).set_index("ResourceID")
    assert list(find_single_points_of_failure(data).columns) == SPOF_COLUMNS
    assert set(spof.index) == {"R15", "R06", "R11"}
    assert spof.loc["R15", "RequirementCount"] == 2
    assert spof.loc["R15", "Requirements"] == "Q001, Q020"
    assert spof.loc["R15", "Projects"] == "P01, P04"


def test_completed_projects_are_excluded_from_scarcity(data):
    d = copy.deepcopy(data)
    d["Projects"].loc[d["Projects"]["ProjectID"] == "P04", "Status"] = "Completed"
    result = analyze_scarce_skills(d)
    assert "P04" not in set(result["ProjectID"])
    assert len(result) == 50 - 3  # P04 has three requirements


def test_scarcity_on_hand_made_data():
    d = tiny()
    d["Projects"] = pd.DataFrame({"ProjectID": ["P1"], "Status": ["Active"]})
    row = analyze_scarce_skills(d).iloc[0]
    assert row["QualifiedCount"] == 4                    # python >= 4: A (5), C (4), E (4), F (4); B and D (2) do not qualify
    assert row["QualifiedResources"] == "A, C, E, F"


# =====================================================================
# Step 17: the four planted situations, DISCOVERED by the real engine
#
# Each test first discovers the situation with generic rules (no IDs named in the discovery),
# then compares it with the dataset design. IDs appear only in the "design match" asserts.
# The thresholds are simulation assumptions, not industry standards.
# =====================================================================
from src.capacity import build_requirement_demand  # noqa: E402
from src.dependencies import exposed_project_ids, projects_supported_by_resource  # noqa: E402
from src.validator import validate_data  # noqa: E402

HIGH_UTILIZATION = 0.90       # "heavily allocated": peak weekly utilization at or above this
ROOM_UTILIZATION = 0.70       # "has room": peak weekly utilization below this
UNDERSTAFFED_COVERAGE = 0.60  # "understaffed": allocated hours cover less than this share of required hours


def overlap_weeks(a, b):
    return list(range(int(max(a.StartWeek, b.StartWeek)), int(min(a.EndWeek, b.EndWeek)) + 1))


def discover_critical_concentration(data, scarcity, capacity, conflicts):
    """A heavily loaded resource that is the ONLY qualified person for two requirements of different
    projects with overlapping windows, where at least one of those requirements has a capacity shortage."""
    req = data["Project_Requirements"].set_index("RequirementID")
    shortage_reqs = set(of_type(conflicts, CAPACITY_SHORTAGE)["RequirementID"])
    found = []
    for sole in find_single_points_of_failure(data, scarcity).itertuples():
        ids = sole.Requirements.split(", ")
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ra, rb = req.loc[a], req.loc[b]
                weeks = overlap_weeks(ra, rb)
                if ra.ProjectID == rb.ProjectID or not weeks:
                    continue
                peak = capacity[(capacity["ResourceID"] == sole.ResourceID) & capacity["Week"].isin(weeks)]["Utilization"].max()
                if peak >= HIGH_UTILIZATION and {a, b} & shortage_reqs:
                    found.append({"resource": sole.ResourceID, "requirements": (a, b), "weeks": weeks})
    return found


def test_planted_1_critical_skill_concentration(data, scarcity, capacity, conflicts):
    found = discover_critical_concentration(data, scarcity, capacity, conflicts)
    assert len(found) == 1, "expected exactly one critically concentrated resource"
    # design match
    assert found[0]["resource"] == "R15"
    assert found[0]["requirements"] == ("Q001", "Q020")
    assert found[0]["weeks"] == [4, 5, 6, 7, 8, 9, 10]
    # emergent numbers: 2h free against a 10h need
    week4 = capacity[(capacity["ResourceID"] == "R15") & (capacity["Week"] == 4)].iloc[0]
    assert week4["AvailableHrs"] == 2 and week4["Utilization"] == pytest.approx(0.95)
    assert conflict_row(conflicts, CAPACITY_SHORTAGE, "Q020")["Hours"] == 40


def test_planted_2_dependency_exposure(data, scarcity, capacity, conflicts):
    found = discover_critical_concentration(data, scarcity, capacity, conflicts)[0]
    shortage_projects = set(of_type(conflicts, CAPACITY_SHORTAGE)["ProjectID"])
    affected = set(projects_supported_by_resource(data, found["resource"], weeks=found["weeks"])) | shortage_projects
    exposed = exposed_project_ids(data, affected)
    assert exposed, "losing or overloading this resource must expose at least one downstream project"
    # design match: P01 blocks P05, P02 blocks P09, and the starved P04 blocks P03
    assert {"P05", "P03"} <= set(exposed)
    assert exposed == ["P03", "P05", "P09"]


def test_planted_3_limited_backup_with_one_heavily_loaded_person(data, scarcity, capacity):
    peak = capacity.groupby("ResourceID")["Utilization"].max()
    found = []
    for row in scarcity[scarcity["QualifiedCount"] == 2].itertuples():
        pool = sorted(row.QualifiedResources.split(", "), key=lambda r: peak[r])
        low, high = pool
        if peak[high] >= HIGH_UTILIZATION and peak[low] < ROOM_UTILIZATION:
            found.append((row.Skill, high, low))
    assert found, "no two-person skill with one heavily loaded person was discovered"
    # design match: Data Engineering (R08 heavily loaded, R09 has room), reported for both P05 and P06
    assert set(found) == {("Data Engineering", "R08", "R09")}
    assert peak["R08"] == pytest.approx(0.925) and peak["R09"] == pytest.approx(0.60)


def test_planted_4_existing_staffing_gap(data):
    demand = build_requirement_demand(data)
    by_project = demand.groupby("ProjectID")[["RequiredHrs", "AllocatedHrs"]].sum()
    coverage = by_project["AllocatedHrs"] / by_project["RequiredHrs"]
    understaffed = set(coverage[coverage < UNDERSTAFFED_COVERAGE].index)
    open_requirements = demand[demand["AllocatedHrs"] == 0]
    assert understaffed and not open_requirements.empty
    # design match: P08 is the most under-staffed; P04 (61%) also starts with a wholly open requirement
    assert understaffed == {"P08"}
    assert coverage.idxmin() == "P08"
    assert set(open_requirements["RequirementID"]) == {"Q020", "Q043", "Q045"}
    assert set(open_requirements["ProjectID"]) == {"P04", "P08"}


def test_planted_5_misleading_candidate_skill_fit_is_not_suitability(data, capacity):
    """Open requirements where the best-ranked candidate is NOT the most skilled one."""
    demand = build_requirement_demand(data)
    found = {}
    for rid in demand.loc[demand["AllocatedHrs"] == 0, "RequirementID"]:
        ranked = evaluate_candidates(data, rid, capacity=capacity)
        if ranked.empty:
            continue
        top = ranked.iloc[0]
        most_skilled = ranked["SkillMatch"].max()
        if top["SkillMatch"] < most_skilled:
            found[rid] = top["ResourceID"]
    assert {"Q020", "Q043"} <= set(found)          # design match
    assert found["Q020"] == "R16" and found["Q043"] == "R12"


# =====================================================================
# Step 18: edge cases on the real data
# (resource-unavailable, deadline-change and effort-increase edge cases are tested with the
#  scenario engine in tests/test_scenarios.py, once Steps 20-22 exist)
# =====================================================================
def with_changes(data, changes):
    """Deep copy of the data with cell edits given as {(table, key_column, key, column): new_value}."""
    d = copy.deepcopy(data)
    for (table, key_col, key, col), new in changes.items():
        d[table].loc[d[table][key_col] == key, col] = new
    return d


def test_edge_zero_available_hours_in_part_of_the_window(data):
    # capacity 38h: R15 is exactly full in weeks 4-8, but has room in weeks 9-10
    d = with_changes(data, {("Resources", "ResourceID", "R15", "WeeklyCapacityHrs"): 38})
    r15 = cand(evaluate_candidates(d, "Q020"), "R15")
    assert r15["TightestFreeHrs"] == 0
    assert r15["Availability"] == 0
    assert r15["Constraints"] == "Already fully allocated"


def test_edge_no_qualified_candidate(data):
    # raise the minimum to 5 and drop the only level-5 cloud engineer to 4: nobody qualifies any more
    d = with_changes(data, {("Project_Requirements", "RequirementID", "Q020", "MinProficiency"): 5})
    cloud_r15 = (d["Resource_Skills"]["ResourceID"] == "R15") & (d["Resource_Skills"]["Skill"] == "Cloud Infrastructure")
    d["Resource_Skills"].loc[cloud_r15, "Proficiency"] = 4
    assert get_qualified_resources(d, "Q020") == []
    assert analyze_scarce_skills(d).set_index("RequirementID").loc["Q020", "Category"] == NO_QUALIFIED
    shortage = conflict_row(detect_conflicts(d), CAPACITY_SHORTAGE, "Q020")
    assert "nobody" in shortage["Detail"]
    assert shortage["Hours"] == 70  # the whole requirement is unmeetable by anyone qualified
    ranked = evaluate_candidates(d, "Q020")
    assert not ranked.empty and (ranked["FitLabel"] != STRONG).all()  # below-minimum people are still shown, capped


def test_edge_resource_fully_allocated_has_zero_headroom(data):
    d = with_changes(data, {("Resources", "ResourceID", "R15", "WeeklyCapacityHrs"): 38})
    week4 = build_capacity_table(d).query("ResourceID == 'R15' and Week == 4").iloc[0]
    assert week4["AvailableHrs"] == 0
    assert week4["Utilization"] == 1
    assert week4["LoadHeadroom"] == 0


def test_edge_requirement_with_no_allocation_flows_through_every_engine(data):
    demand = build_requirement_demand(data).set_index("RequirementID").loc["Q020"]
    assert demand["AllocatedHrs"] == 0 and demand["Coverage"] == 0
    assert "Q020" in set(of_type(detect_conflicts(data), STAFFING_GAP)["RequirementID"])
    assert analyze_scarce_skills(data).set_index("RequirementID").loc["Q020", "Category"] == SINGLE_POINT
    assert not evaluate_candidates(data, "Q020").empty


def test_edge_over_booking_is_detected_but_is_not_a_validation_error(data):
    d = copy.deepcopy(data)
    row = d["Allocations"].index[(d["Allocations"]["ResourceID"] == "R15") & (d["Allocations"]["Week"] == 4)][0]
    d["Allocations"].loc[row, "PlannedHrs"] = d["Allocations"].loc[row, "PlannedHrs"] + 10  # 48h planned vs 40h capacity
    assert validate_data(d) == []  # the validator accepts it; discovering it is the engine's job
    over = of_type(detect_conflicts(d), OVER_ALLOCATION)
    assert list(over["ResourceID"]) == ["R15"]
    assert over.iloc[0]["Weeks"] == "4" and over.iloc[0]["Hours"] == 8
    assert build_capacity_table(d).query("ResourceID == 'R15' and Week == 4").iloc[0]["AvailableHrs"] == -8


def test_edge_every_engine_runs_end_to_end_on_the_v2_data(data):
    assert validate_data(data) == []
    cap = build_capacity_table(data)
    assert len(cap) == 28 * 12
    assert len(build_requirement_demand(data)) == 50
    assert len(detect_conflicts(data, capacity=cap)) == 7
    assert len(analyze_scarce_skills(data)) == 50
    assert exposed_project_ids(data, ["P01", "P04"]) == ["P03", "P05"]


def test_edge_engines_are_deterministic(data):
    first = detect_conflicts(data), analyze_scarce_skills(data), evaluate_candidates(data, "Q020")
    second = detect_conflicts(data), analyze_scarce_skills(data), evaluate_candidates(data, "Q020")
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


# ---------- Unavailability flag (scenario shock) ----------
def test_unavailable_in_part_of_the_window_is_flagged_instead_of_the_hours_flags(data):
    from src.scenarios import apply_resource_unavailable
    d = apply_resource_unavailable(data, "R15", [5])  # week 5 is inside Q020's window (weeks 4-8)
    r15 = cand(evaluate_candidates(d, "Q020"), "R15")
    assert r15["TightestFreeHrs"] == 0
    assert r15["Availability"] == 0
    assert r15["Constraints"] == "Unavailable in part of the window"


def test_unavailability_outside_the_window_adds_no_flag(data):
    d = copy.deepcopy(data)
    d["Unavailability"] = pd.DataFrame({"ResourceID": ["R15"], "Week": [11]})  # Q020 ends in week 8
    base = cand(evaluate_candidates(data, "Q020"), "R15")
    r15 = cand(evaluate_candidates(d, "Q020"), "R15")
    assert r15["Constraints"] == base["Constraints"]
    assert r15["FitScore"] == pytest.approx(base["FitScore"])
