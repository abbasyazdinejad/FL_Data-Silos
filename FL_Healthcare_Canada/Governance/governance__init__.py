"""
Demo of AI Governance module for federated learning in healthcare

Demo Implemention of comprehensive governance mechanisms:
- Consent management (SMART-on-FHIR compliant)
- Audit logging (hash-chained for immutability)
- Fairness auditing (demographic parity, bias detection)
- Compliance checking (PIPEDA, PHIPA, OCAP®)

These components ensure ethical, legal, and accountable FL deployment.
"""

from .consent_manager import (
    ConsentManager
)

from .audit_logger import (
    AuditLogger
)

from .fairness_auditor import (
    FairnessAuditor
)

from .compliance_checker import (
    ComplianceChecker
)

__all__ = [
    'ConsentManager',
    'AuditLogger',
    'FairnessAuditor',
    'ComplianceChecker'
]

__version__ = '1.0.0'
__author__ = 'Abbas Yazdinejad and Jude Kong'

# Canadian healthcare regulations reference
REGULATIONS = {
    'PIPEDA': {
        'name': 'Personal Information Protection and Electronic Documents Act',
        'scope': 'Federal privacy law for private sector',
        'requirements': [
            'Consent for data collection and use',
            'Purpose limitation',
            'Data minimization',
            'Right to access and correction',
            'Security safeguards'
        ]
    },
    'PHIPA': {
        'name': 'Personal Health Information Protection Act',
        'scope': 'Ontario provincial health privacy law',
        'requirements': [
            'Express consent for health information',
            'Circle of care provisions',
            'Data localization (Ontario)',
            'Breach notification',
            'Privacy impact assessments'
        ]
    },
    'OCAP': {
        'name': 'Ownership, Control, Access, and Possession',
        'scope': 'Indigenous data sovereignty principles',
        'requirements': [
            'Community ownership of data',
            'Community control over data management',
            'Community access to data',
            'Community possession of data'
        ]
    }
}

# Governance metrics from paper (Table 5)
PAPER_METRICS = {
    'consent': {
        'latency_ms': 5.6,
        'checks_per_round': 5,  # Check per client
        'grant_rate': 1.0  # 100% in simulations
    },
    'audit': {
        'latency_ms': 2.0,
        'events_per_round': 1,  # Log aggregation event
        'chain_integrity': True
    },
    'fairness': {
        'bias_flags_%': 2.3,  # From paper
        'threshold': 0.10,  # 10% disparity threshold
        'checks_per_epoch': 1
    },
    'compliance': {
        'privacy_budget_max': 2.0,
        'privacy_budget_target': 1.0,
        'violations_target': 0
    }
}


def create_governance_pipeline(
    enable_consent: bool = True,
    enable_audit: bool = True,
    enable_fairness: bool = True,
    enable_compliance: bool = True,
    config: dict = None
) -> dict:
    """
    Create complete governance pipeline for FL
    
    Args:
        enable_consent: Enable consent management
        enable_audit: Enable audit logging
        enable_fairness: Enable fairness auditing
        enable_compliance: Enable compliance checking
        config: Optional configuration dict
        
    Returns:
        Dict containing all governance components
    """
    if config is None:
        config = PAPER_METRICS.copy()
    
    pipeline = {}
    
    if enable_consent:
        pipeline['consent_manager'] = ConsentManager(
            latency_ms=config.get('consent', {}).get('latency_ms', 5.6)
        )
    
    if enable_audit:
        pipeline['audit_logger'] = AuditLogger(
            latency_ms=config.get('audit', {}).get('latency_ms', 2.0)
        )
    
    if enable_fairness:
        pipeline['fairness_auditor'] = FairnessAuditor(
            threshold=config.get('fairness', {}).get('threshold', 0.10)
        )
    
    if enable_compliance:
        pipeline['compliance_checker'] = ComplianceChecker()
    
    return pipeline


def get_governance_overhead(
    consent_latency_ms: float = 5.6,
    audit_latency_ms: float = 2.0,
    num_clients: int = 3,
    training_time_ms: float = 1000.0
) -> dict:
    """
    Estimate governance overhead
    
    From paper: Governance overhead is ~1.7% of total training time
    
    Args:
        consent_latency_ms: Consent check latency per client
        audit_latency_ms: Audit logging latency per event
        num_clients: Number of federated clients
        training_time_ms: Training time per round (milliseconds)
        
    Returns:
        Dict with overhead analysis
    """
    consent_overhead = consent_latency_ms * num_clients
    audit_overhead = audit_latency_ms  # One audit event per round
    
    total_governance_overhead = consent_overhead + audit_overhead
    total_time = training_time_ms + total_governance_overhead
    
    overhead_percentage = (total_governance_overhead / total_time) * 100
    
    return {
        'consent_overhead_ms': consent_overhead,
        'audit_overhead_ms': audit_overhead,
        'total_governance_overhead_ms': total_governance_overhead,
        'training_time_ms': training_time_ms,
        'total_time_ms': total_time,
        'overhead_percentage': overhead_percentage
    }


def validate_governance_compliance(governance_stats: dict) -> dict:
    """
    Validate that governance meets paper standards
    
    Args:
        governance_stats: Statistics from governance components
        
    Returns:
        Validation results
    """
    validation = {
        'consent': False,
        'audit': False,
        'fairness': False,
        'privacy': False,
        'overall': False
    }
    
    issues = []
    
    # Check consent
    if 'consent' in governance_stats:
        consent = governance_stats['consent']
        if consent.get('grant_rate', 0) >= 0.95:  # 95% consent rate minimum
            validation['consent'] = True
        else:
            issues.append("Consent rate below 95%")
    
    # Check audit
    if 'audit' in governance_stats:
        audit = governance_stats['audit']
        if audit.get('chain_valid', False):
            validation['audit'] = True
        else:
            issues.append("Audit chain integrity compromised")
    
    # Check fairness
    if 'fairness' in governance_stats:
        fairness = governance_stats['fairness']
        if fairness.get('violation_rate_%', 100) < 5.0:  # <5% violations
            validation['fairness'] = True
        else:
            issues.append(f"Fairness violations too high: {fairness.get('violation_rate_%')}%")
    
    # Check privacy
    if 'privacy' in governance_stats:
        privacy = governance_stats['privacy']
        if privacy.get('epsilon_spent', 10) <= 2.0:  # Within budget
            validation['privacy'] = True
        else:
            issues.append(f"Privacy budget exceeded: ε={privacy.get('epsilon_spent')}")
    
    # Overall compliance
    validation['overall'] = all(validation.values())
    validation['issues'] = issues
    
    return validation


# Example usage
def example_usage():
    """
    Example of using the governance module
    """
    print("Creating governance pipeline...")
    pipeline = create_governance_pipeline()
    
    print("\nGovernance components:")
    for name, component in pipeline.items():
        print(f"  - {name}: {type(component).__name__}")
    
    print("\nEstimating governance overhead...")
    overhead = get_governance_overhead(num_clients=3)
    print(f"  Overhead: {overhead['overhead_percentage']:.2f}%")
    print(f"  (Paper reports ~1.7%)")
    
    print("\nCanadian regulations:")
    for reg_name, reg_info in REGULATIONS.items():
        print(f"  {reg_name}: {reg_info['name']}")


if __name__ == "__main__":
    example_usage()
