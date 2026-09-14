"""
Node Identity
Per-machine identity, config versioning, and per-camera tracking sessions.

Deployment puts the same image on every edge box; only the mounted config
directory and the environment differ. This module owns the facts that are
true of the machine rather than of any camera:

  NODE_ID    stable name for this box, used to attribute every stored row
  NODE_ROLE  default role its cameras inherit ("seating" or "flow")

It also owns two counters that stored rows depend on:

  config_version  bumped whenever zone geometry, capacity or a camera's model
                  changes. Persisted, so "before/after the fix" stays
                  answerable across a restart.
  session_id      per camera, scoping tracker IDs. Tracker state resets on a
                  restart and on a model switch; without a session scope a
                  query spanning either silently merges two different people.
"""

import json
import logging
import os
import socket
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

ROLE_SEATING = "seating"
ROLE_FLOW = "flow"
VALID_ROLES = (ROLE_SEATING, ROLE_FLOW)


class NodeIdentity:
    """Identity and versioning for one edge node."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.state_file = self.config_dir / "node_state.json"
        self.lock = threading.Lock()

        self.node_id: str = os.getenv("NODE_ID") or socket.gethostname()

        role = (os.getenv("NODE_ROLE") or ROLE_SEATING).strip().lower()
        if role not in VALID_ROLES:
            logger.warning(
                f"NODE_ROLE '{role}' is not one of {VALID_ROLES}; defaulting to {ROLE_SEATING}"
            )
            role = ROLE_SEATING
        self.node_role: str = role

        # Bumped on config change, persisted across restarts.
        self.config_version: int = 1

        # Per-camera tracker session. Regenerated on restart and model switch.
        self._sessions: Dict[str, str] = {}

        self._load()
        logger.info(
            f"Node identity: id={self.node_id} role={self.node_role} "
            f"config_version={self.config_version}"
        )

    # ---------- persistence ----------

    def _load(self):
        """Restore config_version so history stays comparable across restarts."""
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            self.config_version = int(data.get("config_version", 1))
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning(f"Could not read {self.state_file} ({e}); starting at version 1")

    def _save(self):
        """Caller holds self.lock."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(
                    {
                        "node_id": self.node_id,
                        "config_version": self.config_version,
                        "updated_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except OSError as e:
            # A failed write costs comparability, not correctness - keep running.
            logger.error(f"Could not persist node state to {self.state_file}: {e}")

    # ---------- config version ----------

    def bump_config_version(self, reason: str = "") -> int:
        """
        Record that something affecting stored numbers has changed.

        Call on zone geometry edits, capacity changes and model switches - any
        change that makes rows before and after incomparable.
        """
        with self.lock:
            self.config_version += 1
            self._save()
            version = self.config_version
        logger.info(f"config_version -> {version}" + (f" ({reason})" if reason else ""))
        return version

    # ---------- tracker sessions ----------

    def get_session_id(self, camera_id: str) -> str:
        """Current tracker session for a camera, created on first use."""
        with self.lock:
            session = self._sessions.get(camera_id)
            if session is None:
                session = uuid.uuid4().hex[:12]
                self._sessions[camera_id] = session
        return session

    def new_session_id(self, camera_id: str, reason: str = "") -> str:
        """
        Start a new tracker session for a camera.

        Call whenever tracker state is discarded - a model switch, a stream
        reconnect - so track IDs either side are never treated as the same
        person.
        """
        session = uuid.uuid4().hex[:12]
        with self.lock:
            self._sessions[camera_id] = session
        logger.info(
            f"New tracking session for {camera_id}: {session}"
            + (f" ({reason})" if reason else "")
        )
        return session

    # ---------- roles ----------

    def resolve_role(self, camera: Optional[Dict]) -> str:
        """
        Role for a camera: its own if set and valid, otherwise the node default.

        Cameras configured before roles existed carry none, and inherit the
        node's - which keeps existing deployments behaving as they did.
        """
        if camera:
            role = camera.get("role")
            if isinstance(role, str):
                role = role.strip().lower()
                if role in VALID_ROLES:
                    return role
                if role:
                    logger.warning(
                        f"Camera {camera.get('id')} has unknown role '{role}'; "
                        f"falling back to node role {self.node_role}"
                    )
        return self.node_role

    def to_dict(self) -> Dict:
        """Identity as reported by the API and attached to stored rows."""
        return {
            "node_id": self.node_id,
            "node_role": self.node_role,
            "config_version": self.config_version,
        }
