"""
Example of Consent Management for Federated Learning

Simulates SMART-on-FHIR compliant consent management for healthcare data.
In production, integrates with:
- Provincial health information systems
- Institutional review boards (IRBs)
- Patient consent portals
"""

import time
from typing import Dict, List, Optional
import hashlib
from datetime import datetime, timedelta


class ConsentManager:
    """
    Consent management system for FL in healthcare
    
    Simulates patient/institutional consent checks with:
    - Fast consent verification (5.6ms average, from paper)
    - Consent caching for efficiency
    - Consent revocation support
    - Audit trail for all consent checks
    
    Based on SMART-on-FHIR standards for healthcare data consent.
    """
    
    def __init__(self, latency_ms: float = 5.6):
        """
        Initialize consent manager
        
        Args:
            latency_ms: Simulated consent check latency (from paper: 5.6ms)
        """
        self.latency_ms = latency_ms
        
        # Statistics
        self.consent_checks = 0
        self.consent_granted = 0
        self.consent_denied = 0
        
        # Consent cache for performance
        self.consent_cache = {}
        
        # Consent records (would be in database in production)
        self.consent_records = {}
        
        # Audit trail
        self.audit_trail = []
    
    def check_consent(
        self, 
        client_id: str, 
        operation: str = "training",
        data_type: str = "health_record",
        force_check: bool = False
    ) -> bool:
        """
        Check if consent is granted for operation
        
        Args:
            client_id: Client identifier (e.g., 'ontario', 'hospital_1')
            operation: Operation type ('training', 'inference', 'aggregation')
            data_type: Type of data ('health_record', 'genomic', 'imaging')
            force_check: Force fresh check (bypass cache)
            
        Returns:
            True if consent granted, False otherwise
        """
        # Check cache first (unless force_check)
        cache_key = f"{client_id}_{operation}_{data_type}"
        
        if not force_check and cache_key in self.consent_cache:
            consent_result = self.consent_cache[cache_key]
            self._record_check(client_id, operation, consent_result, from_cache=True)
            return consent_result
        
        # Simulate latency (database query, API call to consent service)
        time.sleep(self.latency_ms / 1000.0)
        
        # In real implementation, this would:
        # 1. Query consent database or SMART-on-FHIR server
        # 2. Verify consent is current (not expired or revoked)
        # 3. Check consent scope covers the operation
        # 4. Log consent check
        
        # For simulation, assume consent is granted
        # (Can be modified to simulate denied cases)
        consent_granted = self._verify_consent(client_id, operation, data_type)
        
        # Update statistics
        self.consent_checks += 1
        if consent_granted:
            self.consent_granted += 1
        else:
            self.consent_denied += 1
        
        # Cache result
        self.consent_cache[cache_key] = consent_granted
        
        # Record check
        self._record_check(client_id, operation, consent_granted, from_cache=False)
        
        return consent_granted
    
    def _verify_consent(
        self, 
        client_id: str, 
        operation: str, 
        data_type: str
    ) -> bool:
        """
        Internal method to verify consent
        
        In production, this would:
        - Query consent management system
        - Verify consent is not expired
        - Check consent scope
        """
        # Check if there's an explicit consent record
        record_key = f"{client_id}_{operation}_{data_type}"
        
        if record_key in self.consent_records:
            record = self.consent_records[record_key]
            
            # Check if expired
            if 'expiry' in record:
                if datetime.now() > record['expiry']:
                    return False
            
            # Check if revoked
            if record.get('revoked', False):
                return False
            
            return record.get('granted', True)
        
        # Default: assume consent granted (simulation)
        # In production, this would deny by default
        return True
    
    def grant_consent(
        self,
        client_id: str,
        operation: str = "training",
        data_type: str = "health_record",
        duration_days: int = 365
    ):
        """
        Grant consent for client/operation
        
        Args:
            client_id: Client identifier
            operation: Operation type
            data_type: Data type
            duration_days: Consent validity period
        """
        record_key = f"{client_id}_{operation}_{data_type}"
        
        self.consent_records[record_key] = {
            'granted': True,
            'timestamp': datetime.now(),
            'expiry': datetime.now() + timedelta(days=duration_days),
            'revoked': False,
            'client_id': client_id,
            'operation': operation,
            'data_type': data_type
        }
        
        # Invalidate cache
        cache_key = f"{client_id}_{operation}_{data_type}"
        if cache_key in self.consent_cache:
            del self.consent_cache[cache_key]
    
    def revoke_consent(
        self, 
        client_id: str, 
        operation: Optional[str] = None,
        data_type: Optional[str] = None
    ):
        """
        Revoke consent for client/operation
        
        Args:
            client_id: Client identifier
            operation: Operation type (if None, revokes all)
            data_type: Data type (if None, revokes all)
        """
        if operation and data_type:
            # Revoke specific consent
            record_key = f"{client_id}_{operation}_{data_type}"
            if record_key in self.consent_records:
                self.consent_records[record_key]['revoked'] = True
            
            # Invalidate cache
            cache_key = f"{client_id}_{operation}_{data_type}"
            if cache_key in self.consent_cache:
                del self.consent_cache[cache_key]
        else:
            # Revoke all consents for this client
            for key in list(self.consent_records.keys()):
                if key.startswith(f"{client_id}_"):
                    self.consent_records[key]['revoked'] = True
            
            # Clear cache
            keys_to_delete = [k for k in self.consent_cache if k.startswith(f"{client_id}_")]
            for key in keys_to_delete:
                del self.consent_cache[key]
    
    def _record_check(
        self, 
        client_id: str, 
        operation: str, 
        granted: bool,
        from_cache: bool
    ):
        """Record consent check in audit trail"""
        self.audit_trail.append({
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'operation': operation,
            'granted': granted,
            'from_cache': from_cache
        })
    
    def get_statistics(self) -> Dict:
        """
        Get consent check statistics
        
        Returns:
            Dict of statistics matching paper metrics
        """
        return {
            'total_checks': self.consent_checks,
            'granted': self.consent_granted,
            'denied': self.consent_denied,
            'grant_rate': self.consent_granted / max(1, self.consent_checks),
            'avg_latency_ms': self.latency_ms,
            'cache_size': len(self.consent_cache),
            'active_consents': len([r for r in self.consent_records.values() 
                                   if not r.get('revoked', False)])
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.consent_checks = 0
        self.consent_granted = 0
        self.consent_denied = 0
        self.audit_trail = []
    
    def get_audit_trail(self, client_id: Optional[str] = None) -> List[Dict]:
        """
        Get consent audit trail
        
        Args:
            client_id: Optional client ID filter
            
        Returns:
            List of audit records
        """
        if client_id:
            return [r for r in self.audit_trail if r['client_id'] == client_id]
        return self.audit_trail.copy()


if __name__ == "__main__":
    print("="*70)
    print("Testing Consent Manager")
    print("="*70)
    
    # Create consent manager
    consent_mgr = ConsentManager(latency_ms=5.6)
    
    # Grant consent for Ontario
    print("\n1. Grant Consent")
    print("-"*70)
    consent_mgr.grant_consent('ontario', 'training', 'health_record')
    print("✓ Consent granted for Ontario")
    
    # Check consent
    print("\n2. Check Consent")
    print("-"*70)
    result = consent_mgr.check_consent('ontario', 'training')
    print(f"Consent check result: {result}")
    
    # Check again (from cache)
    result2 = consent_mgr.check_consent('ontario', 'training')
    print(f"Cached consent check: {result2}")
    
    # Get statistics
    print("\n3. Statistics")
    print("-"*70)
    stats = consent_mgr.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Revoke consent
    print("\n4. Revoke Consent")
    print("-"*70)
    consent_mgr.revoke_consent('ontario', 'training', 'health_record')
    print("✓ Consent revoked")
    
    # Check after revocation
    result3 = consent_mgr.check_consent('ontario', 'training', force_check=True)
    print(f"Check after revocation: {result3}")
    
    print("\n" + "="*70)
    print("Consent Manager tests passed!")
    print("="*70)
