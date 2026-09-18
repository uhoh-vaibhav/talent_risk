"""
2x2 Talent Matrix & Actionable HR Recommendation Engine
Combines Attrition Risk P(attrition) and Promotion Readiness P(promotion) into 4 distinct quadrants.
"""

from typing import Dict, Any


class TalentMatrixEngine:
    """Derives 2x2 Talent Matrix classification and actionable HR strategies."""

    QUADRANT_RETAIN_URGENTLY = "Urgent Retention & Key Talent"
    QUADRANT_INVEST_FAST_TRACK = "Invest & Fast-Track"
    QUADRANT_MONITOR_ENGAGE = "Monitor & Engage"
    QUADRANT_CORE_PERFORMER = "Core Performer / Low Priority"

    @classmethod
    def evaluate_quadrant(
        cls,
        p_attrition: float,
        p_promotion: float,
        attrition_threshold: float = 0.50,
        promotion_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """Maps attrition risk and promotion readiness scores into the 2x2 matrix."""

        is_high_attrition = p_attrition > attrition_threshold
        is_high_promotion = p_promotion > promotion_threshold

        if is_high_attrition and is_high_promotion:
            quadrant = cls.QUADRANT_RETAIN_URGENTLY
            action_code = "URGENT_RETENTION"
            hr_action = "Immediate intervention: Conduct stay interview, review compensation, fast-track promotion timeline, and assign executive mentor."
            priority_level = "CRITICAL"
        elif not is_high_attrition and is_high_promotion:
            quadrant = cls.QUADRANT_INVEST_FAST_TRACK
            action_code = "DEVELOPMENT_INVESTMENT"
            hr_action = "High potential investment: Enroll in leadership development, assign strategic projects, and prepare for formal promotion."
            priority_level = "HIGH"
        elif is_high_attrition and not is_high_promotion:
            quadrant = cls.QUADRANT_MONITOR_ENGAGE
            action_code = "FLIGHT_RISK_MONITORING"
            hr_action = "Flight risk mitigation: Investigate workload, satisfaction drivers, and manager relationship to improve engagement."
            priority_level = "MEDIUM"
        else:
            quadrant = cls.QUADRANT_CORE_PERFORMER
            action_code = "STABLE_PERFORMANCE"
            hr_action = "Maintain steady engagement: Provide ongoing skill training and standard performance incentives."
            priority_level = "LOW"

        return {
            "quadrant": quadrant,
            "action_code": action_code,
            "priority_level": priority_level,
            "hr_action_recommendation": hr_action,
            "attrition_risk_score": round(float(p_attrition), 4),
            "promotion_readiness_score": round(float(p_promotion), 4)
        }
