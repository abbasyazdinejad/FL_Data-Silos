"""
Example of Hash-chained Audit Logging for Federated Learning

Implements blockchain-style immutable audit trails:
- Each record hashes the previous record
- Tampering detection through chain verification
- Cryptographic integrity (SHA-256)
- Fast logging (2.0ms average, from paper)
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime


class AuditLogger:
    """
    Hash-chained audit logger for immutable audit trails
    
    Each audit record contains:
    - Timestamp
    - Event details (type, client, round, data)
    - Hash of previous record (chain linkage)
    - Hash of current record (integrity)
    
    This creates an immutable chain similar to blockchain,
    making it cryptographically hard to tamper with logs.
    """
    
    def __init__(self, latency_ms: float = 2.0):
        """
        Initialize audit logger
        
        Args:
            latency_ms: Simulated audit logging latency (from paper: 2.0ms)
        """
        self.latency_ms = latency_ms
        self.audit_chain = []
        self.previous_hash = "0" * 64  # Genesis hash (all zeros)
    
    def log_event(
        self,
        event_type: str,
        client_id: str,
        round_num: int,
        details: Dict[str, Any]
    ) -> str:
        """
        Log an audit event with hash chaining
        
        Args:
            event_type: Type of event ('training', 'aggregation', 'evaluation', etc.)
            client_id: Client identifier
            round_num: Federated round number
            details: Additional event details (loss, accuracy, etc.)
            
        Returns:
            Hash of the logged event (for verification)
        """
        # Simulate latency (disk I/O, database write)
        time.sleep(self.latency_ms / 1000.0)
        
        # Create audit record
        timestamp = datetime.now().isoformat()
        record = {
            'timestamp': timestamp,
            'event_type': event_type,
            'client_id': client_id,
            'round_num': round_num,
            'details': details,
            'previous_hash': self.previous_hash
        }
        
        # Compute hash of current record
        record_str = json.dumps(record, sort_keys=True)
        current_hash = hashlib.sha256(record_str.encode()).hexdigest()
        record['hash'] = current_hash
        
        # Add to chain
        self.audit_chain.append(record)
        self.previous_hash = current_hash
        
        return current_hash
    
    def log_training_event(
        self,
        client_id: str,
        round_num: int,
        loss: float,
        num_samples: int
    ) -> str:
        """
        Convenience method for logging training events
        
        Args:
            client_id: Client identifier
            round_num: Federated round
            loss: Training loss
            num_samples: Number of samples
            
        Returns:
            Event hash
        """
        return self.log_event(
            event_type='training',
            client_id=client_id,
            round_num=round_num,
            details={
                'loss': loss,
                'num_samples': num_samples
            }
        )
    
    def log_aggregation_event(
        self,
        round_num: int,
        num_clients: int,
        aggregation_method: str
    ) -> str:
        """
        Convenience method for logging aggregation events
        
        Args:
            round_num: Federated round
            num_clients: Number of participating clients
            aggregation_method: Aggregation method used
            
        Returns:
            Event hash
        """
        return self.log_event(
            event_type='aggregation',
            client_id='server',
            round_num=round_num,
            details={
                'num_clients': num_clients,
                'aggregation_method': aggregation_method
            }
        )
    
    def verify_chain(self) -> bool:
        """
        Verify integrity of the entire audit chain
        
        This checks:
        1. Each record's previous_hash matches the actual previous hash
        2. Each record's hash is correctly computed
        
        Returns:
            True if chain is valid, False if tampering detected
        """
        if not self.audit_chain:
            return True  # Empty chain is valid
        
        previous_hash = "0" * 64  # Genesis hash
        
        for i, record in enumerate(self.audit_chain):
            # Verify previous hash matches
            if record['previous_hash'] != previous_hash:
                print(f"Chain broken at record {i}: previous_hash mismatch")
                return False
            
            # Recompute hash
            record_copy = record.copy()
            stored_hash = record_copy.pop('hash')
            record_str = json.dumps(record_copy, sort_keys=True)
            computed_hash = hashlib.sha256(record_str.encode()).hexdigest()
            
            # Verify hash matches
            if computed_hash != stored_hash:
                print(f"Chain broken at record {i}: hash mismatch")
                return False
            
            previous_hash = stored_hash
        
        return True
    
    def get_audit_trail(
        self, 
        client_id: Optional[str] = None,
        event_type: Optional[str] = None,
        round_num: Optional[int] = None
    ) -> List[Dict]:
        """
        Get audit trail with optional filters
        
        Args:
            client_id: Filter by client ID (optional)
            event_type: Filter by event type (optional)
            round_num: Filter by round number (optional)
            
        Returns:
            List of filtered audit records
        """
        trail = self.audit_chain.copy()
        
        if client_id:
            trail = [r for r in trail if r['client_id'] == client_id]
        
        if event_type:
            trail = [r for r in trail if r['event_type'] == event_type]
        
        if round_num is not None:
            trail = [r for r in trail if r['round_num'] == round_num]
        
        return trail
    
    def get_statistics(self) -> Dict:
        """
        Get audit logging statistics
        
        Returns:
            Dict of statistics matching paper metrics
        """
        event_types = self._count_event_types()
        
        return {
            'total_events': len(self.audit_chain),
            'chain_valid': self.verify_chain(),
            'event_types': event_types,
            'avg_latency_ms': self.latency_ms,
            'chain_length': len(self.audit_chain),
            'genesis_hash': "0" * 64,
            'latest_hash': self.previous_hash
        }
    
    def _count_event_types(self) -> Dict[str, int]:
        """Count events by type"""
        counts = {}
        for record in self.audit_chain:
            event_type = record['event_type']
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts
    
    def export_audit_log(self, filepath: str):
        """
        Export audit log to JSON file
        
        Args:
            filepath: Path to output file
        """
        with open(filepath, 'w') as f:
            json.dump({
                'audit_chain': self.audit_chain,
                'chain_valid': self.verify_chain(),
                'export_time': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Audit log exported to {filepath}")
    
    def import_audit_log(self, filepath: str):
        """
        Import audit log from JSON file
        
        Args:
            filepath: Path to input file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.audit_chain = data['audit_chain']
        if self.audit_chain:
            self.previous_hash = self.audit_chain[-1]['hash']
        
        # Verify imported chain
        if not self.verify_chain():
            raise ValueError("Imported audit chain failed integrity check!")
        
        print(f"Audit log imported from {filepath}")
        print(f"  Events: {len(self.audit_chain)}")
        print(f"  Chain valid: {self.verify_chain()}")


if __name__ == "__main__":
    print("="*70)
    print("Testing Audit Logger")
    print("="*70)
    
    # Create audit logger
    audit_log = AuditLogger(latency_ms=2.0)
    
    # Log some events
    print("\n1. Log Training Events")
    print("-"*70)
    hash1 = audit_log.log_training_event('ontario', 1, 0.5, 1000)
    print(f"Event 1 hash: {hash1[:16]}...")
    
    hash2 = audit_log.log_training_event('alberta', 1, 0.6, 500)
    print(f"Event 2 hash: {hash2[:16]}...")
    
    # Log aggregation
    print("\n2. Log Aggregation Event")
    print("-"*70)
    hash3 = audit_log.log_aggregation_event(1, 3, 'weighted_avg')
    print(f"Aggregation hash: {hash3[:16]}...")
    
    # Verify chain
    print("\n3. Verify Chain Integrity")
    print("-"*70)
    is_valid = audit_log.verify_chain()
    print(f"Chain valid: {is_valid}")
    
    # Get statistics
    print("\n4. Statistics")
    print("-"*70)
    stats = audit_log.get_statistics()
    for key, value in stats.items():
        if key != 'genesis_hash' and key != 'latest_hash':
            print(f"  {key}: {value}")
    
    # Test tampering detection
    print("\n5. Tampering Detection")
    print("-"*70)
    print("Original chain valid:", audit_log.verify_chain())
    
    # Tamper with a record
    if len(audit_log.audit_chain) > 0:
        audit_log.audit_chain[0]['details']['loss'] = 999.0
        print("After tampering:", audit_log.verify_chain())
    
    print("\n" + "="*70)
    print("Audit Logger tests completed!")
    print("="*70)
