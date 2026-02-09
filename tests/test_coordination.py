"""Tests for Splinter Coordination Layer."""

import pytest
import tempfile
import shutil
from pathlib import Path

from splinter.coordination.state import SharedState
from splinter.coordination.ownership import StateOwnershipManager, ProtectedState
from splinter.coordination.schema import SchemaValidator, HandoffManager, create_schema_from_example
from splinter.coordination.checkpoint import (
    CheckpointManager,
    InMemoryCheckpointStorage,
    FileCheckpointStorage,
)
from splinter.schemas import AgentStatus, ExecutionMetrics
from splinter.exceptions import StateOwnershipError, SchemaValidationError, CheckpointNotFoundError


class TestSharedState:
    """Tests for SharedState."""

    def test_basic_operations(self):
        """Test basic get/set/delete operations."""
        state = SharedState()

        state.set("key", "value")
        assert state.get("key") == "value"

        state.delete("key")
        assert state.get("key") is None

    def test_nested_paths(self):
        """Test dot notation for nested paths."""
        state = SharedState()

        state.set("research.results", ["item1", "item2"])
        state.set("research.metadata.source", "web")

        assert state.get("research.results") == ["item1", "item2"]
        assert state.get("research.metadata.source") == "web"

    def test_versioning(self):
        """Test that each change increments version."""
        state = SharedState()

        assert state.version == 0

        state.set("a", 1)
        assert state.version == 1

        state.set("b", 2)
        assert state.version == 2

        state.update({"c": 3, "d": 4})
        assert state.version == 3  # Single update = single version bump

    def test_snapshot_immutability(self):
        """Test that snapshots are immutable copies."""
        state = SharedState()
        state.set("key", {"nested": "value"})

        snapshot = state.snapshot()
        original_data = snapshot.data["key"]["nested"]

        # Modify state
        state.set("key.nested", "modified")

        # Snapshot should be unchanged
        assert snapshot.data["key"]["nested"] == original_data

    def test_history(self):
        """Test state history tracking."""
        state = SharedState()

        state.set("a", 1)
        state.set("b", 2)
        state.set("c", 3)

        history = state.get_history(limit=2)
        assert len(history) == 2
        assert history[0].version.version == 3  # Most recent first
        assert history[1].version.version == 2

    def test_restore(self):
        """Test restoring to previous version."""
        state = SharedState()

        state.set("value", 1)
        state.set("value", 2)
        state.set("value", 3)

        # Restore to version 1
        state.restore(1)

        assert state.get("value") == 1

    def test_merge(self):
        """Test deep merge of data."""
        state = SharedState()
        state.set("config", {"a": 1, "b": {"x": 10}})

        state.merge({"config": {"b": {"y": 20}, "c": 3}})

        assert state.get("config.a") == 1
        assert state.get("config.b.x") == 10
        assert state.get("config.b.y") == 20
        assert state.get("config.c") == 3


class TestStateOwnership:
    """Tests for StateOwnershipManager."""

    def test_owner_can_write(self):
        """Test that owner can write to their fields."""
        manager = StateOwnershipManager()
        manager.register("research.*", owner="researcher")

        # Should not raise
        manager.check_write("researcher", "research.results")
        manager.check_write("researcher", "research.metadata")

    def test_non_owner_blocked(self):
        """Test that non-owners are blocked."""
        manager = StateOwnershipManager()
        manager.register("research.*", owner="researcher")

        with pytest.raises(StateOwnershipError):
            manager.check_write("other_agent", "research.results")

    def test_protected_state(self):
        """Test ProtectedState wrapper."""
        state = SharedState()
        manager = StateOwnershipManager()
        manager.register("research.*", owner="researcher")

        protected = ProtectedState(state, manager)

        # Owner can write
        protected.set("research.data", "value", agent_id="researcher")
        assert protected.get("research.data") == "value"

        # Non-owner blocked
        with pytest.raises(StateOwnershipError):
            protected.set("research.data", "bad", agent_id="attacker")

    def test_get_owned_fields(self):
        """Test getting fields owned by agent."""
        manager = StateOwnershipManager()
        manager.register("research.*", owner="researcher")
        manager.register("summary.*", owner="summarizer")
        manager.register("plan.*", owner="researcher")

        owned = manager.get_owned_fields("researcher")
        assert "research.*" in owned
        assert "plan.*" in owned
        assert "summary.*" not in owned


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_valid_object(self):
        """Test validation of valid object."""
        validator = SchemaValidator()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name"],
        }

        errors = validator.validate({"name": "test", "count": 5}, schema)
        assert len(errors) == 0

    def test_missing_required(self):
        """Test detection of missing required field."""
        validator = SchemaValidator()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }

        errors = validator.validate({}, schema)
        assert len(errors) > 0
        assert any("required" in e for e in errors)

    def test_wrong_type(self):
        """Test detection of wrong type."""
        validator = SchemaValidator()

        schema = {"type": "string"}

        errors = validator.validate(123, schema)
        assert len(errors) > 0
        assert any("type" in e for e in errors)

    def test_array_validation(self):
        """Test array validation with items."""
        validator = SchemaValidator()

        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }

        # Valid
        errors = validator.validate(["a", "b"], schema)
        assert len(errors) == 0

        # Wrong item type
        errors = validator.validate(["a", 123], schema)
        assert len(errors) > 0

        # Too few items
        errors = validator.validate([], schema)
        assert len(errors) > 0


class TestHandoffManager:
    """Tests for HandoffManager."""

    def test_valid_handoff(self):
        """Test validation passes for valid output."""
        manager = HandoffManager(strict=True)

        manager.register_handoff(
            source="researcher",
            target="summarizer",
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array"},
                },
                "required": ["results"],
            },
        )

        # Should not raise
        manager.validate_output(
            source="researcher",
            target="summarizer",
            data={"results": ["finding1", "finding2"]},
        )

    def test_invalid_handoff_strict(self):
        """Test that invalid output raises in strict mode."""
        manager = HandoffManager(strict=True)

        manager.register_handoff(
            source="researcher",
            target="summarizer",
            output_schema={
                "type": "object",
                "required": ["results"],
            },
        )

        with pytest.raises(SchemaValidationError):
            manager.validate_output(
                source="researcher",
                target="summarizer",
                data={"wrong_field": "value"},
            )


class TestCreateSchemaFromExample:
    """Tests for create_schema_from_example utility."""

    def test_basic_types(self):
        """Test schema inference for basic types."""
        example = {
            "name": "test",
            "count": 42,
            "price": 19.99,
            "active": True,
        }

        schema = create_schema_from_example(example)

        assert schema["type"] == "object"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["price"]["type"] == "number"
        assert schema["properties"]["active"]["type"] == "boolean"

    def test_nested_objects(self):
        """Test schema inference for nested objects."""
        example = {
            "user": {
                "name": "test",
                "age": 25,
            }
        }

        schema = create_schema_from_example(example)

        assert schema["properties"]["user"]["type"] == "object"
        assert schema["properties"]["user"]["properties"]["name"]["type"] == "string"

    def test_arrays(self):
        """Test schema inference for arrays."""
        example = {
            "tags": ["a", "b", "c"],
            "scores": [1, 2, 3],
        }

        schema = create_schema_from_example(example)

        assert schema["properties"]["tags"]["type"] == "array"
        assert schema["properties"]["tags"]["items"]["type"] == "string"
        assert schema["properties"]["scores"]["items"]["type"] == "integer"


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_create_and_load(self):
        """Test creating and loading checkpoints."""
        manager = CheckpointManager()

        state = SharedState()
        state.set("key", "value")

        checkpoint = manager.create_checkpoint(
            workflow_id="test-wf",
            step=1,
            agent_id="agent_1",
            status=AgentStatus.COMPLETED,
            state=state,
            metrics=ExecutionMetrics(),
        )

        loaded = manager.get_checkpoint("test-wf", step=1)

        assert loaded.workflow_id == "test-wf"
        assert loaded.step == 1
        assert loaded.state_snapshot.data["key"] == "value"

    def test_get_latest(self):
        """Test getting latest checkpoint."""
        manager = CheckpointManager()
        state = SharedState()

        for i in range(3):
            state.set("step", i)
            manager.create_checkpoint(
                workflow_id="test-wf",
                step=i,
                agent_id="agent",
                status=AgentStatus.COMPLETED,
                state=state,
                metrics=ExecutionMetrics(),
            )

        latest = manager.get_latest_checkpoint("test-wf")
        assert latest.step == 2

    def test_checkpoint_not_found(self):
        """Test error when checkpoint not found."""
        manager = CheckpointManager()

        with pytest.raises(CheckpointNotFoundError):
            manager.get_checkpoint("nonexistent", step=1)

    def test_file_storage(self):
        """Test file-based checkpoint storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileCheckpointStorage(tmpdir)
            manager = CheckpointManager(storage=storage)

            state = SharedState()
            state.set("data", {"nested": "value"})

            manager.create_checkpoint(
                workflow_id="file-test",
                step=0,
                agent_id="agent",
                status=AgentStatus.COMPLETED,
                state=state,
                metrics=ExecutionMetrics(),
            )

            # Create new manager with same storage (simulates restart)
            manager2 = CheckpointManager(storage=FileCheckpointStorage(tmpdir))

            loaded = manager2.get_latest_checkpoint("file-test")
            assert loaded.state_snapshot.data["data"]["nested"] == "value"

    def test_get_resume_point(self):
        """Test getting resume point from checkpoint."""
        manager = CheckpointManager()
        state = SharedState()
        state.set("progress", "step1_done")

        manager.create_checkpoint(
            workflow_id="resume-test",
            step=1,
            agent_id="agent",
            status=AgentStatus.COMPLETED,
            state=state,
            metrics=ExecutionMetrics(total_cost=0.5, total_steps=5),
        )

        resume = manager.get_resume_point("resume-test")

        assert resume is not None
        resume_step, state_snapshot, metrics = resume
        assert resume_step == 2  # Should resume from next step
        assert state_snapshot.data["progress"] == "step1_done"
        assert metrics.total_cost == 0.5
