"""Pydantic models for coordinator REST API.

These models match the style defined in COMMUNICATION_OUTLINE.md for
request/response shapes used in the control plane communication.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import time


class WorkerStatus(str, Enum):
    """Worker status values."""
    PENDING = "pending"
    ONLINE = "online"
    OFFLINE = "offline"
    STANDBY = "standby"


# =============================================================================
# Worker Registration
# =============================================================================

class WorkerRegistration(BaseModel):
    """Request body for POST /workers/register."""
    worker_id: str
    nickname: Optional[str] = None
    hardware_type: str = "apple_silicon"
    device_name: Optional[str] = None
    vram_gb: Optional[float] = None
    requested_layers: Optional[int] = None
    # ShardCompute-specific fields (for tensor parallelism)
    rank: Optional[int] = None
    host: Optional[str] = None
    port: Optional[int] = None
    collective_port: Optional[int] = None
    device_info: Optional[Dict[str, Any]] = None


class LayerAssignment(BaseModel):
    """Response from POST /workers/register."""
    worker_id: str
    layers_start: int
    layers_end: int
    model_name: str
    initial_peers: List[str]  # multiaddr format: /ip4/.../tcp/.../p2p/...
    is_idle: bool = False
    status: WorkerStatus = WorkerStatus.ONLINE
    # ShardCompute-specific fields
    rank: Optional[int] = None
    world_size: Optional[int] = None


# =============================================================================
# Worker Heartbeat
# =============================================================================

class WorkerHeartbeat(BaseModel):
    """Request body for POST /workers/heartbeat."""
    worker_id: str
    tokens_served_since_last: int = 0
    status: str = "online"
    # ShardCompute-specific fields
    rank: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None


class HeartbeatResponse(BaseModel):
    """Response from POST /workers/heartbeat."""
    status: str = "ok"
    worker_status: WorkerStatus = WorkerStatus.ONLINE
    layers_start: Optional[int] = None
    layers_end: Optional[int] = None


# =============================================================================
# Network Information
# =============================================================================

class NetworkInfo(BaseModel):
    """Response from GET /network/info."""
    model_name: str
    total_layers: int
    bootstrap_peers: List[str]  # multiaddr format
    online_workers: int
    coverage_percent: float
    ready_for_inference: bool
    quantization_mode: Optional[str] = None


class NetworkHealth(BaseModel):
    """Response from GET /network/health."""
    healthy: bool
    healthy_count: int
    total_count: int
    layer_coverage: Dict[int, bool] = Field(default_factory=dict)


class NetworkStats(BaseModel):
    """Response from GET /network/stats."""
    uptime_seconds: float
    total_requests: int
    total_tokens_generated: int
    avg_latency_ms: float
    p99_latency_ms: float
    avg_throughput_tokens_per_sec: float
    coverage_percent: float


# =============================================================================
# User Management
# =============================================================================

class UserRegistration(BaseModel):
    """Request body for POST /users/register."""
    user_id: str
    display_name: Optional[str] = None


class UserInfo(BaseModel):
    """Response from GET /users/{id}."""
    user_id: str
    display_name: Optional[str] = None
    total_prompts: int = 0
    total_tokens: int = 0
    created_at: float = Field(default_factory=time.time)


# =============================================================================
# Usage Tracking
# =============================================================================

class UsageLog(BaseModel):
    """Request body for POST /usage/log."""
    user_id: str
    prompt_tokens: int
    completion_tokens: int
    worker_ids: List[str] = Field(default_factory=list)


class UsageEntry(BaseModel):
    """Entry in usage log (returned by GET /usage/recent)."""
    user_id: str
    prompt_tokens: int
    completion_tokens: int
    worker_ids: List[str]
    timestamp: float = Field(default_factory=time.time)


# =============================================================================
# Worker List
# =============================================================================

class WorkerListEntry(BaseModel):
    """Entry in worker list (returned by GET /workers)."""
    worker_id: str
    rank: int
    host: str
    port: int
    collective_port: int
    status: WorkerStatus
    last_heartbeat: float
    device_info: Dict[str, Any] = Field(default_factory=dict)


class WorkerListResponse(BaseModel):
    """Response from GET /workers."""
    workers: List[WorkerListEntry]
    cluster_ready: bool
