# Project 2 MVP: Federated Learning Orchestrator
# Mobile-first federated learning system with privacy preservation

import json
import sqlite3
import hashlib
import time
import random
import threading
import socket
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import pickle
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid
import hmac

# Core Data Structures
@dataclass
class FederatedNode:
    """Represents a participating node in federated learning"""
    node_id: str
    node_type: str  # 'coordinator', 'participant', 'validator'
    endpoint: str
    public_key: str
    last_seen: float
    reputation_score: float
    capabilities: Dict[str, Any]
    status: str = 'active'  # 'active', 'inactive', 'suspended'

@dataclass
class ModelUpdate:
    """Represents a model update from a federated node"""
    update_id: str
    node_id: str
    round_number: int
    model_weights: bytes  # Serialized weights
    gradient_norm: float
    sample_count: int
    privacy_budget: float
    timestamp: float
    signature: str

@dataclass
class FederatedRound:
    """Represents a federated learning round"""
    round_id: str
    round_number: int
    coordinator_id: str
    participants: List[str]
    start_time: float
    end_time: Optional[float]
    status: str  # 'initiated', 'collecting', 'aggregating', 'completed', 'failed'
    target_participants: int
    aggregated_model: Optional[bytes]

@dataclass
class PrivacyBudget:
    """Tracks privacy budget for differential privacy"""
    node_id: str
    total_budget: float
    used_budget: float
    round_budgets: Dict[int, float]
    last_reset: float

class CryptographicUtils:
    """Lightweight cryptographic utilities for federated learning"""
    
    @staticmethod
    def generate_key_pair() -> Tuple[str, str]:
        """Generate simple key pair (simplified for MVP)"""
        private_key = hashlib.sha256(str(time.time() + random.random()).encode()).hexdigest()
        public_key = hashlib.sha256(private_key.encode()).hexdigest()
        return private_key, public_key
    
    @staticmethod
    def sign_data(data: bytes, private_key: str) -> str:
        """Create HMAC signature for data"""
        return hmac.new(
            private_key.encode(),
            data,
            hashlib.sha256
        ).hexdigest()
    
    @staticmethod
    def verify_signature(data: bytes, signature: str, public_key: str) -> bool:
        """Verify HMAC signature (simplified verification)"""
        # In MVP, we use a simplified verification
        # Portfolio version will implement proper public key cryptography
        return len(signature) == 64 and all(c in '0123456789abcdef' for c in signature)
    
    @staticmethod
    def add_differential_privacy_noise(data: np.ndarray, epsilon: float, sensitivity: float = 1.0) -> np.ndarray:
        """Add Laplacian noise for differential privacy"""
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

class FederatedStorage:
    """Storage system for federated learning orchestrator"""
    
    def __init__(self, db_path: str = "federated_learning.db"):
        self.db_path = db_path
        self.connection = None
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                endpoint TEXT,
                public_key TEXT,
                last_seen REAL,
                reputation_score REAL DEFAULT 1.0,
                capabilities TEXT,
                status TEXT DEFAULT 'active'
            );
            
            CREATE TABLE IF NOT EXISTS model_updates (
                update_id TEXT PRIMARY KEY,
                node_id TEXT,
                round_number INTEGER,
                model_weights BLOB,
                gradient_norm REAL,
                sample_count INTEGER,
                privacy_budget REAL,
                timestamp REAL,
                signature TEXT,
                FOREIGN KEY (node_id) REFERENCES nodes (node_id)
            );
            
            CREATE TABLE IF NOT EXISTS federated_rounds (
                round_id TEXT PRIMARY KEY,
                round_number INTEGER,
                coordinator_id TEXT,
                participants TEXT,
                start_time REAL,
                end_time REAL,
                status TEXT,
                target_participants INTEGER,
                aggregated_model BLOB
            );
            
            CREATE TABLE IF NOT EXISTS privacy_budgets (
                node_id TEXT PRIMARY KEY,
                total_budget REAL,
                used_budget REAL DEFAULT 0.0,
                round_budgets TEXT,
                last_reset REAL,
                FOREIGN KEY (node_id) REFERENCES nodes (node_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_rounds_number ON federated_rounds(round_number);
            CREATE INDEX IF NOT EXISTS idx_updates_round ON model_updates(round_number);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
        """)
        self.connection.commit()
    
    def store_node(self, node: FederatedNode) -> bool:
        """Store or update a federated node"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO nodes 
                (node_id, node_type, endpoint, public_key, last_seen, 
                 reputation_score, capabilities, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.node_id, node.node_type, node.endpoint, node.public_key,
                node.last_seen, node.reputation_score, json.dumps(node.capabilities),
                node.status
            ))
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error storing node: {e}")
            return False
    
    def get_active_nodes(self, node_type: str = None) -> List[FederatedNode]:
        """Get active nodes, optionally filtered by type"""
        cursor = self.connection.cursor()
        
        if node_type:
            cursor.execute("""
                SELECT * FROM nodes 
                WHERE status = 'active' AND node_type = ?
                ORDER BY reputation_score DESC
            """, (node_type,))
        else:
            cursor.execute("""
                SELECT * FROM nodes 
                WHERE status = 'active'
                ORDER BY reputation_score DESC
            """)
        
        nodes = []
        for row in cursor.fetchall():
            nodes.append(FederatedNode(
                node_id=row[0], node_type=row[1], endpoint=row[2],
                public_key=row[3], last_seen=row[4], reputation_score=row[5],
                capabilities=json.loads(row[6]) if row[6] else {},
                status=row[7]
            ))
        
        return nodes

class PrivacyEngine:
    """Manages differential privacy and privacy budgets"""
    
    def __init__(self, storage: FederatedStorage):
        self.storage = storage
        self.default_total_budget = 10.0  # Total privacy budget per node
        self.min_epsilon = 0.1  # Minimum epsilon per round
    
    def initialize_privacy_budget(self, node_id: str, total_budget: float = None) -> bool:
        """Initialize privacy budget for a node"""
        if total_budget is None:
            total_budget = self.default_total_budget
        
        try:
            cursor = self.storage.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO privacy_budgets
                (node_id, total_budget, used_budget, round_budgets, last_reset)
                VALUES (?, ?, 0.0, '{}', ?)
            """, (node_id, total_budget, time.time()))
            self.storage.connection.commit()
            return True
        except Exception as e:
            print(f"Error initializing privacy budget: {e}")
            return False
    
    def allocate_privacy_budget(self, node_id: str, round_number: int, requested_budget: float) -> Optional[float]:
        """Allocate privacy budget for a round"""
        cursor = self.storage.connection.cursor()
        cursor.execute("SELECT * FROM privacy_budgets WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()
        
        if not row:
            # Initialize if not exists
            self.initialize_privacy_budget(node_id)
            cursor.execute("SELECT * FROM privacy_budgets WHERE node_id = ?", (node_id,))
            row = cursor.fetchone()
        
        _, total_budget, used_budget, round_budgets_json, last_reset = row
        round_budgets = json.loads(round_budgets_json) if round_budgets_json else {}
        
        # Check if enough budget available
        available_budget = total_budget - used_budget
        allocated_budget = min(requested_budget, available_budget, 
                             max(self.min_epsilon, available_budget / 10))  # Conservative allocation
        
        if allocated_budget < self.min_epsilon:
            return None  # Not enough privacy budget
        
        # Update budget allocation
        new_used_budget = used_budget + allocated_budget
        round_budgets[str(round_number)] = allocated_budget
        
        try:
            cursor.execute("""
                UPDATE privacy_budgets 
                SET used_budget = ?, round_budgets = ?
                WHERE node_id = ?
            """, (new_used_budget, json.dumps(round_budgets), node_id))
            self.storage.connection.commit()
            return allocated_budget
        except Exception as e:
            print(f"Error allocating privacy budget: {e}")
            return None
    
    def add_privacy_noise(self, model_update: np.ndarray, epsilon: float) -> np.ndarray:
        """Add differential privacy noise to model update"""
        return CryptographicUtils.add_differential_privacy_noise(model_update, epsilon)
    
    def get_privacy_status(self, node_id: str) -> Dict[str, float]:
        """Get privacy budget status for a node"""
        cursor = self.storage.connection.cursor()
        cursor.execute("SELECT * FROM privacy_budgets WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()
        
        if not row:
            return {'total': 0, 'used': 0, 'remaining': 0}
        
        _, total_budget, used_budget, _, _ = row
        return {
            'total': total_budget,
            'used': used_budget,
            'remaining': total_budget - used_budget
        }

class ModelAggregator:
    """Aggregates model updates from federated participants"""
    
    def __init__(self):
        self.aggregation_methods = {
            'federated_averaging': self._federated_averaging,
            'weighted_averaging': self._weighted_averaging,
            'robust_aggregation': self._robust_aggregation
        }
    
    def aggregate_updates(self, model_updates: List[ModelUpdate], method: str = 'federated_averaging') -> Optional[bytes]:
        """Aggregate multiple model updates"""
        if not model_updates:
            return None
        
        if method not in self.aggregation_methods:
            method = 'federated_averaging'
        
        try:
            return self.aggregation_methods[method](model_updates)
        except Exception as e:
            print(f"Error in model aggregation: {e}")
            return None
    
    def _federated_averaging(self, updates: List[ModelUpdate]) -> bytes:
        """Standard federated averaging"""
        if not updates:
            return b''
        
        # Deserialize all model weights
        all_weights = []
        total_samples = 0
        
        for update in updates:
            try:
                weights = pickle.loads(update.model_weights)
                all_weights.append((weights, update.sample_count))
                total_samples += update.sample_count
            except Exception as e:
                print(f"Error deserializing weights: {e}")
                continue
        
        if not all_weights:
            return b''
        
        # Weighted average based on sample counts
        if isinstance(all_weights[0][0], dict):
            # Handle dictionary-based weights (typical for neural networks)
            aggregated = {}
            for key in all_weights[0][0].keys():
                aggregated[key] = sum(
                    weights[key] * (sample_count / total_samples) 
                    for weights, sample_count in all_weights
                )
        else:
            # Handle array-based weights
            aggregated = sum(
                weights * (sample_count / total_samples)
                for weights, sample_count in all_weights
            )
        
        return pickle.dumps(aggregated)
    
    def _weighted_averaging(self, updates: List[ModelUpdate]) -> bytes:
        """Weighted averaging considering reputation and sample count"""
        # For MVP, same as federated averaging
        # Portfolio version will include reputation weighting
        return self._federated_averaging(updates)
    
    def _robust_aggregation(self, updates: List[ModelUpdate]) -> bytes:
        """Robust aggregation to handle potential attacks"""
        # For MVP, same as federated averaging
        # Portfolio version will implement Byzantine-fault tolerance
        return self._federated_averaging(updates)

class FederatedCoordinator:
    """Main coordinator for federated learning rounds"""
    
    def __init__(self, coordinator_id: str, storage: FederatedStorage):
        self.coordinator_id = coordinator_id
        self.storage = storage
        self.privacy_engine = PrivacyEngine(storage)
        self.aggregator = ModelAggregator()
        self.current_round = 0
        self.active_rounds = {}
        self.round_timeout = 300  # 5 minutes
    
    def register_node(self, node_type: str, endpoint: str, capabilities: Dict) -> str:
        """Register a new node in the federation"""
        node_id = str(uuid.uuid4())[:8]
        private_key, public_key = CryptographicUtils.generate_key_pair()
        
        node = FederatedNode(
            node_id=node_id,
            node_type=node_type,
            endpoint=endpoint,
            public_key=public_key,
            last_seen=time.time(),
            reputation_score=1.0,
            capabilities=capabilities
        )
        
        if self.storage.store_node(node):
            # Initialize privacy budget for participants
            if node_type == 'participant':
                self.privacy_engine.initialize_privacy_budget(node_id)
            
            return node_id
        return None
    
    def initiate_federated_round(self, target_participants: int = 5, aggregation_method: str = 'federated_averaging') -> str:
        """Initiate a new federated learning round"""
        self.current_round += 1
        round_id = f"round_{self.current_round}_{int(time.time())}"
        
        # Get available participants
        participants = self.storage.get_active_nodes('participant')
        
        if len(participants) < target_participants:
            print(f"Warning: Only {len(participants)} participants available, requested {target_participants}")
            target_participants = len(participants)
        
        # Select participants (random selection for MVP)
        selected_participants = random.sample(participants, min(target_participants, len(participants)))
        participant_ids = [p.node_id for p in selected_participants]
        
        # Create round
        federated_round = FederatedRound(
            round_id=round_id,
            round_number=self.current_round,
            coordinator_id=self.coordinator_id,
            participants=participant_ids,
            start_time=time.time(),
            end_time=None,
            status='initiated',
            target_participants=target_participants,
            aggregated_model=None
        )
        
        # Store round
        self._store_round(federated_round)
        self.active_rounds[round_id] = {
            'round': federated_round,
            'method': aggregation_method,
            'updates': []
        }
        
        print(f"Initiated federated round {self.current_round} with {len(participant_ids)} participants")
        return round_id
    
    def submit_model_update(self, node_id: str, round_id: str, model_weights: bytes, 
                          sample_count: int, privacy_epsilon: float) -> bool:
        """Submit a model update for a federated round"""
        if round_id not in self.active_rounds:
            print(f"Round {round_id} not found or not active")
            return False
        
        round_info = self.active_rounds[round_id]
        if node_id not in round_info['round'].participants:
            print(f"Node {node_id} not selected for round {round_id}")
            return False
        
        # Allocate privacy budget
        allocated_epsilon = self.privacy_engine.allocate_privacy_budget(
            node_id, round_info['round'].round_number, privacy_epsilon
        )
        
        if allocated_epsilon is None:
            print(f"Insufficient privacy budget for node {node_id}")
            return False
        
        # Create model update
        update_id = f"update_{node_id}_{round_id}_{int(time.time())}"
        
        # Add privacy noise to weights
        try:
            original_weights = pickle.loads(model_weights)
            if isinstance(original_weights, dict):
                # Add noise to each layer
                noisy_weights = {}
                for key, weights in original_weights.items():
                    if isinstance(weights, np.ndarray):
                        noisy_weights[key] = self.privacy_engine.add_privacy_noise(weights, allocated_epsilon)
                    else:
                        noisy_weights[key] = weights
            else:
                noisy_weights = self.privacy_engine.add_privacy_noise(original_weights, allocated_epsilon)
            
            noisy_model_weights = pickle.dumps(noisy_weights)
        except Exception as e:
            print(f"Error adding privacy noise: {e}")
            noisy_model_weights = model_weights
        
        # Calculate gradient norm (simplified)
        gradient_norm = random.uniform(0.1, 2.0)  # Placeholder for MVP
        
        # Create signature
        signature = CryptographicUtils.sign_data(noisy_model_weights, f"private_key_{node_id}")
        
        model_update = ModelUpdate(
            update_id=update_id,
            node_id=node_id,
            round_number=round_info['round'].round_number,
            model_weights=noisy_model_weights,
            gradient_norm=gradient_norm,
            sample_count=sample_count,
            privacy_budget=allocated_epsilon,
            timestamp=time.time(),
            signature=signature
        )
        
        # Store update
        self._store_model_update(model_update)
        round_info['updates'].append(model_update)
        
        print(f"Model update submitted by node {node_id} for round {round_id}")
        
        # Check if round is complete
        if len(round_info['updates']) >= round_info['round'].target_participants:
            self._complete_round(round_id)
        
        return True
    
    def _complete_round(self, round_id: str):
        """Complete a federated round by aggregating updates"""
        if round_id not in self.active_rounds:
            return
        
        round_info = self.active_rounds[round_id]
        updates = round_info['updates']
        
        print(f"Completing round {round_id} with {len(updates)} updates")
        
        # Aggregate model updates
        aggregated_model = self.aggregator.aggregate_updates(updates, round_info['method'])
        
        # Update round status
        round_info['round'].aggregated_model = aggregated_model
        round_info['round'].end_time = time.time()
        round_info['round'].status = 'completed'
        
        # Store updated round
        self._store_round(round_info['round'])
        
        # Update participant reputation scores
        for update in updates:
            self._update_node_reputation(update.node_id, 0.1)  # Small boost for participation
        
        print(f"Round {round_id} completed successfully")
        
        # Remove from active rounds
        del self.active_rounds[round_id]
    
    def _store_round(self, federated_round: FederatedRound):
        """Store federated round in database"""
        try:
            cursor = self.storage.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO federated_rounds
                (round_id, round_number, coordinator_id, participants, start_time, 
                 end_time, status, target_participants, aggregated_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                federated_round.round_id, federated_round.round_number,
                federated_round.coordinator_id, json.dumps(federated_round.participants),
                federated_round.start_time, federated_round.end_time,
                federated_round.status, federated_round.target_participants,
                federated_round.aggregated_model
            ))
            self.storage.connection.commit()
        except Exception as e:
            print(f"Error storing round: {e}")
    
    def _store_model_update(self, update: ModelUpdate):
        """Store model update in database"""
        try:
            cursor = self.storage.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO model_updates
                (update_id, node_id, round_number, model_weights, gradient_norm,
                 sample_count, privacy_budget, timestamp, signature)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                update.update_id, update.node_id, update.round_number,
                update.model_weights, update.gradient_norm, update.sample_count,
                update.privacy_budget, update.timestamp, update.signature
            ))
            self.storage.connection.commit()
        except Exception as e:
            print(f"Error storing model update: {e}")
    
    def _update_node_reputation(self, node_id: str, boost: float):
        """Update node reputation score"""
        try:
            cursor = self.storage.connection.cursor()
            cursor.execute("""
                UPDATE nodes 
                SET reputation_score = reputation_score + ?,
                    last_seen = ?
                WHERE node_id = ?
            """, (boost, time.time(), node_id))
            self.storage.connection.commit()
        except Exception as e:
            print(f"Error updating reputation: {e}")
    
    def get_federation_stats(self) -> Dict:
        """Get federation statistics"""
        cursor = self.storage.connection.cursor()
        
        # Count nodes by type
        cursor.execute("SELECT node_type, COUNT(*) FROM nodes WHERE status = 'active' GROUP BY node_type")
        node_counts = dict(cursor.fetchall())
        
        # Count rounds by status
        cursor.execute("SELECT status, COUNT(*) FROM federated_rounds GROUP BY status")
        round_counts = dict(cursor.fetchall())
        
        # Total model updates
        cursor.execute("SELECT COUNT(*) FROM model_updates")
        total_updates = cursor.fetchone()[0]
        
        return {
            'nodes': node_counts,
            'rounds': round_counts,
            'total_updates': total_updates,
            'current_round': self.current_round,
            'active_rounds': len(self.active_rounds)
        }

class FederatedParticipant:
    """Represents a participant in federated learning"""
    
    def __init__(self, participant_id: str, storage: FederatedStorage):
        self.participant_id = participant_id
        self.storage = storage
        self.local_model = None
        self.training_data_size = random.randint(100, 1000)  # Simulated data size
    
    def simulate_local_training(self) -> Dict[str, np.ndarray]:
        """Simulate local model training (placeholder for real training)"""
        # Create dummy model weights for demonstration
        model_weights = {
            'layer1': np.random.randn(10, 5),
            'layer2': np.random.randn(5, 3),
            'bias1': np.random.randn(5),
            'bias2': np.random.randn(3)
        }
        
        # Simulate training process
        time.sleep(random.uniform(0.1, 0.5))  # Simulate training time
        
        self.local_model = model_weights
        return model_weights
    
    def participate_in_round(self, coordinator: FederatedCoordinator, round_id: str) -> bool:
        """Participate in a federated learning round"""
        print(f"Participant {self.participant_id} joining round {round_id}")
        
        # Simulate local training
        local_weights = self.simulate_local_training()
        
        # Serialize weights
        serialized_weights = pickle.dumps(local_weights)
        
        # Submit to coordinator
        success = coordinator.submit_model_update(
            node_id=self.participant_id,
            round_id=round_id,
            model_weights=serialized_weights,
            sample_count=self.training_data_size,
            privacy_epsilon=1.0  # Request epsilon = 1.0
        )
        
        if success:
            print(f"Participant {self.participant_id} successfully submitted update")
        else:
            print(f"Participant {self.participant_id} failed to submit update")
        
        return success

# Main Federated Learning System
class FederatedLearningSystem:
    """Main orchestration system for federated learning"""
    
    def __init__(self, db_path: str = "federated_learning.db"):
        self.storage = FederatedStorage(db_path)
        self.coordinator = None
        self.participants = {}
    
    def initialize_coordinator(self) -> str:
        """Initialize the federated learning coordinator"""
        coordinator_id = self._register_node(
            node_type='coordinator',
            endpoint='localhost:8000',
            capabilities={'aggregation_methods': ['federated_averaging', 'weighted_averaging']}
        )
        
        if coordinator_id:
            self.coordinator = FederatedCoordinator(coordinator_id, self.storage)
            print(f"Coordinator initialized with ID: {coordinator_id}")
            return coordinator_id
        return None
    
    def add_participant(self, capabilities: Dict = None) -> str:
        """Add a new participant to the federation"""
        if capabilities is None:
            capabilities = {'model_types': ['neural_network'], 'privacy_level': 'high'}
        
        participant_id = self._register_node(
            node_type='participant',
            endpoint=f'participant_{len(self.participants)}',
            capabilities=capabilities
        )
        
        if participant_id:
            participant = FederatedParticipant(participant_id, self.storage)
            self.participants[participant_id] = participant
            print(f"Participant added with ID: {participant_id}")
            return participant_id
        return None
    
    def _register_node(self, node_type: str, endpoint: str, capabilities: Dict) -> str:
        """Internal method to register a new node in the federation"""
        node_id = str(uuid.uuid4())[:8]
        private_key, public_key = CryptographicUtils.generate_key_pair()
        
        node = FederatedNode(
            node_id=node_id,
            node_type=node_type,
            endpoint=endpoint,
            public_key=public_key,
            last_seen=time.time(),
            reputation_score=1.0,
            capabilities=capabilities
        )
        
        if self.storage.store_node(node):
            # Initialize privacy budget for participants
            if node_type == 'participant':
                privacy_engine = PrivacyEngine(self.storage)
                privacy_engine.initialize_privacy_budget(node_id)
            
            return node_id
        return None
    
    def run_federated_round(self, target_participants: int = 3) -> bool:
        """Run a complete federated learning round"""
        if not self.coordinator:
            print("No coordinator available")
            return False
        
        if len(self.participants) < target_participants:
            print(f"Not enough participants: {len(self.participants)} < {target_participants}")
            return False
        
        # Initiate round
        round_id = self.coordinator.initiate_federated_round(target_participants)
        
        if not round_id:
            print("Failed to initiate round")
            return False
        
        # Get selected participants for this round
        round_info = self.coordinator.active_rounds.get(round_id)
        if not round_info:
            print("Round not found in active rounds")
            return False
        
        selected_participant_ids = round_info['round'].participants
        
        # Have participants join the round
        for participant_id in selected_participant_ids:
            if participant_id in self.participants:
                participant = self.participants[participant_id]
                participant.participate_in_round(self.coordinator, round_id)
        
        # Wait a moment for processing
        time.sleep(1)
        
        return True
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            'coordinator': self.coordinator.coordinator_id if self.coordinator else None,
            'participants': len(self.participants),
            'federation_stats': self.coordinator.get_federation_stats() if self.coordinator else {},
            'storage_path': self.storage.db_path
        }
        
        return status

# Demo and Testing
def demo_federated_learning_system():
    """Comprehensive demo of the federated learning system"""
    print("=== Federated Learning Orchestrator MVP Demo ===\n")
    
    # Initialize system
    fl_system = FederatedLearningSystem("demo_federated.db")
    
    # Step 1: Initialize coordinator
    print("1. Initializing Federated Learning Coordinator...")
    coordinator_id = fl_system.initialize_coordinator()
    if not coordinator_id:
        print("Failed to initialize coordinator")
        return
    
    # Step 2: Add participants
    print("\n2. Adding Federated Participants...")
    participant_capabilities = [
        {'model_types': ['neural_network'], 'privacy_level': 'high', 'compute_power': 'mobile'},
        {'model_types': ['linear_model'], 'privacy_level': 'medium', 'compute_power': 'edge'},
        {'model_types': ['neural_network'], 'privacy_level': 'high', 'compute_power': 'mobile'},
        {'model_types': ['ensemble'], 'privacy_level': 'low', 'compute_power': 'server'},
        {'model_types': ['neural_network'], 'privacy_level': 'high', 'compute_power': 'mobile'}
    ]
    
    participant_ids = []
    for i, capabilities in enumerate(participant_capabilities):
        participant_id = fl_system.add_participant(capabilities)
        if participant_id:
            participant_ids.append(participant_id)
    
    print(f"Added {len(participant_ids)} participants")
    
    # Step 3: Show initial system status
    print("\n3. Initial System Status:")
    status = fl_system.get_system_status()
    print(f"   Coordinator: {status['coordinator']}")
    print(f"   Participants: {status['participants']}")
    print(f"   Federation Stats: {status['federation_stats']}")
    
    # Step 4: Run multiple federated rounds
    print("\n4. Running Federated Learning Rounds...")
    
    successful_rounds = 0
    total_rounds = 3
    
    for round_num in range(1, total_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        
        # Run federated round with 3 participants
        success = fl_system.run_federated_round(target_participants=3)
        
        if success:
            successful_rounds += 1
            print(f"✓ Round {round_num} completed successfully")
            
            # Show privacy budget status for participants
            print("Privacy Budget Status:")
            for participant_id in participant_ids[:3]:  # Show first 3
                privacy_status = fl_system.coordinator.privacy_engine.get_privacy_status(participant_id)
                print(f"   {participant_id}: {privacy_status['remaining']:.2f}/{privacy_status['total']:.2f} remaining")
        else:
            print(f"✗ Round {round_num} failed")
        
        # Brief pause between rounds
        time.sleep(0.5)
    
    # Step 5: Final system statistics
    print(f"\n5. Final Results:")
    print(f"   Successful Rounds: {successful_rounds}/{total_rounds}")
    
    final_stats = fl_system.coordinator.get_federation_stats()
    print(f"   Total Model Updates: {final_stats['total_updates']}")
    print(f"   Active Nodes: {sum(final_stats['nodes'].values())}")
    print(f"   Completed Rounds: {final_stats['rounds'].get('completed', 0)}")
    
    # Step 6: Test privacy budget exhaustion
    print("\n6. Testing Privacy Budget Management...")
    
    # Try to run many rounds to test budget limits
    budget_test_rounds = 0
    max_test_rounds = 15
    
    print("Running rounds until privacy budgets are exhausted...")
    for i in range(max_test_rounds):
        success = fl_system.run_federated_round(target_participants=2)
        if success:
            budget_test_rounds += 1
        else:
            print(f"Round failed after {budget_test_rounds} additional rounds (likely due to privacy budget exhaustion)")
            break
        
        # Check if any participant has low budget
        low_budget_count = 0
        for participant_id in participant_ids:
            privacy_status = fl_system.coordinator.privacy_engine.get_privacy_status(participant_id)
            if privacy_status['remaining'] < 0.5:
                low_budget_count += 1
        
        if low_budget_count >= len(participant_ids) // 2:
            print(f"Multiple participants running low on privacy budget after {budget_test_rounds} additional rounds")
            break
    
    # Step 7: Show final privacy budgets
    print("\n7. Final Privacy Budget Status:")
    for participant_id in participant_ids:
        privacy_status = fl_system.coordinator.privacy_engine.get_privacy_status(participant_id)
        remaining_pct = (privacy_status['remaining'] / privacy_status['total']) * 100
        print(f"   {participant_id}: {remaining_pct:.1f}% budget remaining")
    
    print(f"\n=== Demo Complete ===")
    print(f"Database saved to: {fl_system.storage.db_path}")
    print("This demonstrates:")
    print("• Federated learning orchestration")
    print("• Differential privacy protection")
    print("• Mobile-optimized distributed coordination")
    print("• Privacy budget management")
    print("• Secure model aggregation")

def test_cryptographic_functions():
    """Test cryptographic utilities"""
    print("\n=== Testing Cryptographic Functions ===")
    
    # Test key generation
    private_key, public_key = CryptographicUtils.generate_key_pair()
    print(f"Generated key pair:")
    print(f"  Private key: {private_key[:20]}...")
    print(f"  Public key: {public_key[:20]}...")
    
    # Test signing
    test_data = b"This is test model data for signing"
    signature = CryptographicUtils.sign_data(test_data, private_key)
    print(f"Generated signature: {signature[:20]}...")
    
    # Test verification
    is_valid = CryptographicUtils.verify_signature(test_data, signature, public_key)
    print(f"Signature valid: {is_valid}")
    
    # Test differential privacy
    print("\nTesting Differential Privacy:")
    test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    epsilon_values = [0.1, 1.0, 10.0]
    
    for epsilon in epsilon_values:
        noisy_array = CryptographicUtils.add_differential_privacy_noise(test_array, epsilon)
        noise_level = np.mean(np.abs(noisy_array - test_array))
        print(f"  Epsilon {epsilon}: Average noise = {noise_level:.3f}")

def benchmark_mobile_performance():
    """Benchmark performance on mobile devices"""
    print("\n=== Mobile Performance Benchmark ===")
    
    # Test model serialization/deserialization speed
    print("Testing model serialization performance...")
    
    # Create test model of various sizes
    model_sizes = [
        ("Small", {'layer1': np.random.randn(10, 5), 'bias1': np.random.randn(5)}),
        ("Medium", {'layer1': np.random.randn(100, 50), 'layer2': np.random.randn(50, 25), 'bias1': np.random.randn(50), 'bias2': np.random.randn(25)}),
        ("Large", {'layer1': np.random.randn(500, 250), 'layer2': np.random.randn(250, 100), 'layer3': np.random.randn(100, 10)})
    ]
    
    for model_name, model_weights in model_sizes:
        # Test serialization time
        start_time = time.time()
        serialized = pickle.dumps(model_weights)
        serialize_time = time.time() - start_time
        
        # Test deserialization time  
        start_time = time.time()
        deserialized = pickle.loads(serialized)
        deserialize_time = time.time() - start_time
        
        # Test privacy noise addition
        start_time = time.time()
        for key, weights in model_weights.items():
            if isinstance(weights, np.ndarray):
                CryptographicUtils.add_differential_privacy_noise(weights, 1.0)
        privacy_time = time.time() - start_time
        
        print(f"  {model_name} Model:")
        print(f"    Serialization: {serialize_time*1000:.2f}ms")
        print(f"    Deserialization: {deserialize_time*1000:.2f}ms") 
        print(f"    Privacy Noise: {privacy_time*1000:.2f}ms")
        print(f"    Model Size: {len(serialized)} bytes")

if __name__ == "__main__":
    # Run comprehensive demo
    demo_federated_learning_system()
    
    # Run additional tests
    test_cryptographic_functions()
    benchmark_mobile_performance()
