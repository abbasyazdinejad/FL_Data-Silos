"""
Example of Compliance Checking for Canadian Healthcare Regulations

Checks compliance with:
- PIPEDA (Personal Information Protection and Electronic Documents Act)
- PHIPA (Personal Health Information Protection Act - Ontario)
- OCAP® (Indigenous data sovereignty principles)
- Differential Privacy budgets
"""

from typing import Dict, List, Optional
import numpy as np
from datetime import datetime


class ComplianceChecker:
    """
    Check compliance with privacy and regulatory requirements
    
    Canadian healthcare regulations:
    - PIPEDA: Federal privacy law
    - PHIPA: Provincial health privacy (Ontario, others)
    - OCAP®: Indigenous data sovereignty (Ownership, Control, Access, Possession)
    
    Privacy compliance:
    - Differential privacy budget limits (ε ≤ 2.0)
    - Data localization requirements
    - Minimum model performance standards
    """
    
    def __init__(self):
        """Initialize compliance checker"""
        self.checks_performed = 0
        self.violations_found = 0
        self.compliance_history = []
        
        # Define compliance rules
        self.rules = self._define_compliance_rules()
    
    def _define_compliance_rules(self) -> Dict:
        """
        Define compliance rules based on Canadian regulations
        
        Returns:
            Dict of compliance rules
        """
        return {
            'privacy_budget': {
                'max_epsilon': 2.0,      # Maximum allowed (regulatory limit)
                'target_epsilon': 1.0,   # Target from paper
                'max_delta': 1e-5        # Maximum delta
            },
            'data_retention': {
                'max_days': 365,         # Maximum retention period
                'min_days': 30           # Minimum for audit trail
            },
            'consent': {
                'required': True,
                'min_grant_rate': 0.95   # 95% consent minimum
            },
            'audit': {
                'required': True,
                'chain_validation': True
            },
            'data_localization': {
                'required': True,         # Data must stay in jurisdiction
                'cross_border_transfer': False
            },
            'model_performance': {
                'min_accuracy': 0.70,    # 70% minimum acceptable
                'max_bias_rate': 5.0     # <5% bias flags acceptable
            },
            'fairness': {
                'max_disparity': 0.15,   # 15% disparity maximum
                'demographic_parity': True
            }
        }
    
    def check_privacy_compliance(
        self,
        epsilon_spent: float,
        delta: float = 1e-5,
        mechanism: str = 'DP-SGD'
    ) -> Dict:
        """
        Check differential privacy compliance
        
        Args:
            epsilon_spent: Privacy budget spent
            delta: Delta parameter
            mechanism: DP mechanism used
            
        Returns:
            Compliance status dict
        """
        self.checks_performed += 1
        
        max_epsilon = self.rules['privacy_budget']['max_epsilon']
        target_epsilon = self.rules['privacy_budget']['target_epsilon']
        max_delta = self.rules['privacy_budget']['max_delta']
        
        # Check epsilon compliance
        epsilon_compliant = epsilon_spent <= max_epsilon
        meets_target = epsilon_spent <= target_epsilon
        
        # Check delta compliance
        delta_compliant = delta <= max_delta
        
        # Overall compliance
        compliant = epsilon_compliant and delta_compliant
        
        if not compliant:
            self.violations_found += 1
            violation_reason = []
            if not epsilon_compliant:
                violation_reason.append(f"ε={epsilon_spent:.2f} exceeds max={max_epsilon}")
            if not delta_compliant:
                violation_reason.append(f"δ={delta} exceeds max={max_delta}")
        else:
            violation_reason = []
        
        result = {
            'check_type': 'privacy_budget',
            'compliant': compliant,
            'meets_target': meets_target,
            'epsilon_spent': epsilon_spent,
            'max_epsilon': max_epsilon,
            'target_epsilon': target_epsilon,
            'delta': delta,
            'max_delta': max_delta,
            'mechanism': mechanism,
            'violations': violation_reason
        }
        
        self.compliance_history.append(result)
        
        return result
    
    def check_consent_compliance(
        self,
        consent_rate: float,
        num_checks: int,
        num_granted: int
    ) -> Dict:
        """
        Check consent compliance (PIPEDA/PHIPA requirement)
        
        Args:
            consent_rate: Rate of consents granted (0.0 to 1.0)
            num_checks: Number of consent checks
            num_granted: Number of consents granted
            
        Returns:
            Compliance status
        """
        self.checks_performed += 1
        
        min_rate = self.rules['consent']['min_grant_rate']
        consent_required = self.rules['consent']['required']
        
        if consent_required:
            compliant = consent_rate >= min_rate
        else:
            compliant = True
        
        if not compliant:
            self.violations_found += 1
        
        result = {
            'check_type': 'consent',
            'compliant': compliant,
            'consent_rate': consent_rate,
            'min_rate': min_rate,
            'num_checks': num_checks,
            'num_granted': num_granted,
            'required': consent_required
        }
        
        self.compliance_history.append(result)
        
        return result
    
    def check_audit_compliance(
        self,
        audit_trail_exists: bool,
        chain_valid: bool,
        num_events: int
    ) -> Dict:
        """
        Check audit trail compliance
        
        Args:
            audit_trail_exists: Whether audit trail exists
            chain_valid: Whether hash chain is valid
            num_events: Number of audited events
            
        Returns:
            Compliance status
        """
        self.checks_performed += 1
        
        audit_required = self.rules['audit']['required']
        chain_validation_required = self.rules['audit']['chain_validation']
        
        if audit_required:
            if chain_validation_required:
                compliant = audit_trail_exists and chain_valid
            else:
                compliant = audit_trail_exists
        else:
            compliant = True
        
        if not compliant:
            self.violations_found += 1
        
        result = {
            'check_type': 'audit',
            'compliant': compliant,
            'audit_exists': audit_trail_exists,
            'chain_valid': chain_valid,
            'num_events': num_events,
            'required': audit_required
        }
        
        self.compliance_history.append(result)
        
        return result
    
    def check_data_localization(
        self,
        data_location: str,
        expected_location: str,
        cross_border: bool = False
    ) -> Dict:
        """
        Check data localization compliance (PHIPA requirement)
        
        Args:
            data_location: Actual data location
            expected_location: Expected location (e.g., 'ontario', 'canada')
            cross_border: Whether data crossed borders
            
        Returns:
            Compliance status
        """
        self.checks_performed += 1
        
        localization_required = self.rules['data_localization']['required']
        cross_border_allowed = self.rules['data_localization']['cross_border_transfer']
        
        if localization_required:
            location_match = (data_location == expected_location)
            cross_border_compliant = not cross_border or cross_border_allowed
            compliant = location_match and cross_border_compliant
        else:
            compliant = True
        
        if not compliant:
            self.violations_found += 1
        
        result = {
            'check_type': 'data_localization',
            'compliant': compliant,
            'data_location': data_location,
            'expected_location': expected_location,
            'cross_border': cross_border,
            'cross_border_allowed': cross_border_allowed
        }
        
        self.compliance_history.append(result)
        
        return result
    
    def check_model_performance(
        self,
        accuracy: float,
        bias_rate: float,
        task_type: str = 'classification'
    ) -> Dict:
        """
        Check if model meets minimum performance standards
        
        Args:
            accuracy: Model accuracy (0.0 to 1.0)
            bias_rate: Bias violation rate percentage (0-100)
            task_type: Type of ML task
            
        Returns:
            Compliance status
        """
        self.checks_performed += 1
        
        min_acc = self.rules['model_performance']['min_accuracy']
        max_bias = self.rules['model_performance']['max_bias_rate']
        
        accuracy_compliant = accuracy >= min_acc
        bias_compliant = bias_rate <= max_bias
        
        compliant = accuracy_compliant and bias_compliant
        
        if not compliant:
            self.violations_found += 1
        
        result = {
            'check_type': 'model_performance',
            'compliant': compliant,
            'accuracy': accuracy,
            'min_accuracy': min_acc,
            'accuracy_compliant': accuracy_compliant,
            'bias_rate_%': bias_rate,
            'max_bias_%': max_bias,
            'bias_compliant': bias_compliant,
            'task_type': task_type
        }
        
        self.compliance_history.append(result)
        
        return result
    
    def check_fairness_compliance(
        self,
        max_disparity: float,
        fairness_type: str = 'demographic_parity'
    ) -> Dict:
        """
        Check fairness compliance
        
        Args:
            max_disparity: Maximum disparity across groups
            fairness_type: Type of fairness metric
            
        Returns:
            Compliance status
        """
        self.checks_performed += 1
        
        max_allowed_disparity = self.rules['fairness']['max_disparity']
        
        compliant = max_disparity <= max_allowed_disparity
        
        if not compliant:
            self.violations_found += 1
        
        result = {
            'check_type': 'fairness',
            'compliant': compliant,
            'max_disparity': max_disparity,
            'max_allowed_disparity': max_allowed_disparity,
            'fairness_type': fairness_type
        }
        
        self.compliance_history.append(result)
        
        return result
    
    def get_statistics(self) -> Dict:
        """
        Get compliance checking statistics
        
        Returns:
            Dict of statistics
        """
        compliance_rate = 1.0 - (self.violations_found / max(1, self.checks_performed))
        
        # Count violations by type
        violations_by_type = {}
        for check in self.compliance_history:
            if not check['compliant']:
                check_type = check['check_type']
                violations_by_type[check_type] = violations_by_type.get(check_type, 0) + 1
        
        return {
            'checks_performed': self.checks_performed,
            'violations_found': self.violations_found,
            'compliance_rate': compliance_rate,
            'violations_by_type': violations_by_type,
            'rules': self.rules
        }
    
    def generate_compliance_report(self) -> str:
        """
        Generate human-readable compliance report
        
        Returns:
            Formatted report string
        """
        stats = self.get_statistics()
        
        report = []
        report.append("="*70)
        report.append("COMPLIANCE REPORT - CANADIAN HEALTHCARE FL")
        report.append("="*70)
        report.append(f"Checks Performed: {stats['checks_performed']}")
        report.append(f"Violations Found: {stats['violations_found']}")
        report.append(f"Compliance Rate: {stats['compliance_rate']:.1%}")
        report.append("")
        
        if stats['violations_found'] > 0:
            report.append("Violations by Type:")
            for vtype, count in stats['violations_by_type'].items():
                report.append(f"  - {vtype}: {count}")
            report.append("")
        
        report.append("Regulatory Standards:")
        report.append("  - PIPEDA: Personal Information Protection (Federal)")
        report.append("  - PHIPA: Health Privacy Act (Provincial)")
        report.append("  - OCAP®: Indigenous Data Sovereignty")
        report.append("")
        
        if stats['violations_found'] == 0:
            report.append("✓ FULLY COMPLIANT with all regulations")
        else:
            report.append("✗ COMPLIANCE ISSUES DETECTED - Review required")
        
        report.append("="*70)
        
        return "\n".join(report)
    
    def reset_statistics(self):
        """Reset compliance statistics"""
        self.checks_performed = 0
        self.violations_found = 0
        self.compliance_history = []


if __name__ == "__main__":
    print("="*70)
    print("Testing Compliance Checker")
    print("="*70)
    
    # Create compliance checker
    compliance = ComplianceChecker()
    
    # Test privacy compliance
    print("\n1. Privacy Budget Compliance")
    print("-"*70)
    result = compliance.check_privacy_compliance(epsilon_spent=1.0, delta=1e-5)
    print(f"Compliant: {result['compliant']}")
    print(f"Meets target: {result['meets_target']}")
    
    # Test consent compliance
    print("\n2. Consent Compliance")
    print("-"*70)
    result = compliance.check_consent_compliance(
        consent_rate=1.0, num_checks=100, num_granted=100
    )
    print(f"Compliant: {result['compliant']}")
    print(f"Consent rate: {result['consent_rate']:.1%}")
    
    # Test audit compliance
    print("\n3. Audit Compliance")
    print("-"*70)
    result = compliance.check_audit_compliance(
        audit_trail_exists=True, chain_valid=True, num_events=50
    )
    print(f"Compliant: {result['compliant']}")
    print(f"Chain valid: {result['chain_valid']}")
    
    # Test model performance
    print("\n4. Model Performance Compliance")
    print("-"*70)
    result = compliance.check_model_performance(
        accuracy=0.92, bias_rate=2.3
    )
    print(f"Compliant: {result['compliant']}")
    print(f"Accuracy: {result['accuracy']:.1%}")
    print(f"Bias rate: {result['bias_rate_%']:.1f}%")
    
    # Get statistics
    print("\n5. Statistics")
    print("-"*70)
    stats = compliance.get_statistics()
    print(f"Checks: {stats['checks_performed']}")
    print(f"Violations: {stats['violations_found']}")
    print(f"Compliance rate: {stats['compliance_rate']:.1%}")
    
    # Generate report
    print("\n6. Compliance Report")
    print("-"*70)
    print(compliance.generate_compliance_report())
    
    print("\n" + "="*70)
    print("Compliance Checker tests passed!")
    print("="*70)
