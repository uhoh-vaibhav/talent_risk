"""
2x2 Talent Matrix & Actionable HR Recommendation Engine
Combines Attrition Risk P(attrition) and Promotion Readiness P(promotion) into 4 distinct quadrants.
Includes cautious, decision-support advisory language rather than prescriptive mandates.
"""

from typing import Dict, Any, Optional


class TalentMatrixEngine:
    """Derives 2x2 Talent Matrix classification and advisory HR recommendations."""

    QUADRANT_RETAIN_URGENTLY = "Urgent Retention & Key Talent"
    QUADRANT_INVEST_FAST_TRACK = "Invest & Fast-Track"
    QUADRANT_MONITOR_ENGAGE = "Monitor & Engage"
    QUADRANT_CORE_PERFORMER = "Core Performer / Low Priority"

    DISCLAIMER = (
        "Advisory Notice: These indicators represent probabilistic machine learning estimates "
        "and should be used as decision support alongside qualitative managerial assessment."
    )

    @classmethod
    def evaluate_quadrant(
        cls,
        p_attrition: float,
        p_promotion: float,
        attrition_threshold: float = 0.50,
        promotion_threshold: float = 0.50
    ) -> Dict[str, Any]:
        """Maps attrition risk and promotion readiness scores into the 2x2 matrix with cautious recommendations."""

        is_high_attrition = p_attrition >= attrition_threshold
        is_high_promotion = p_promotion >= promotion_threshold

        if is_high_attrition and is_high_promotion:
            quadrant = cls.QUADRANT_RETAIN_URGENTLY
            action_code = "URGENT_RETENTION"
            hr_action = (
                "Suggested Retention Strategy: Model indicates elevated flight risk alongside strong promotion signals. "
                "Consider scheduling a stay interview, exploring internal growth avenues, and reviewing compensation alignment."
            )
            priority_level = "CRITICAL"
        elif not is_high_attrition and is_high_promotion:
            quadrant = cls.QUADRANT_INVEST_FAST_TRACK
            action_code = "DEVELOPMENT_INVESTMENT"
            hr_action = (
                "Suggested Growth Strategy: Model reflects high readiness for advancement with stable retention indicators. "
                "Consider leadership training enrollment, cross-functional project assignments, and discussing a promotion roadmap."
            )
            priority_level = "HIGH"
        elif is_high_attrition and not is_high_promotion:
            quadrant = cls.QUADRANT_MONITOR_ENGAGE
            action_code = "FLIGHT_RISK_MONITORING"
            hr_action = (
                "Suggested Engagement Strategy: Model indicates potential attrition risk without clear promotion indicators. "
                "Consider an informal 1-on-1 check-in to understand work-life balance, project satisfaction, and team dynamics."
            )
            priority_level = "MEDIUM"
        else:
            quadrant = cls.QUADRANT_CORE_PERFORMER
            action_code = "STABLE_PERFORMANCE"
            hr_action = (
                "Suggested Sustained Engagement: Steady performer indicators with low immediate flight risk. "
                "Continue standard performance incentives, career mentorship, and regular quarterly reviews."
            )
            priority_level = "LOW"

        return {
            "quadrant": quadrant,
            "action_code": action_code,
            "priority_level": priority_level,
            "hr_action_recommendation": hr_action,
            "attrition_risk_score": round(float(p_attrition), 4),
            "promotion_readiness_score": round(float(p_promotion), 4),
            "attrition_threshold": round(float(attrition_threshold), 4),
            "promotion_threshold": round(float(promotion_threshold), 4),
            "disclaimer": cls.DISCLAIMER
        }
