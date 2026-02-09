"""Tests for Splinter Cloud module."""

import asyncio
import pytest
import time

from splinter.cloud import (
    CloudClient,
    CloudConfig,
    Command,
    CommandHandler,
    CommandType,
    StateSync,
    SyncEvent,
)
from splinter.cloud.sync import SyncEventType


class TestCloudConfig:
    """Tests for CloudConfig."""

    def test_default_config(self):
        """Test default config values."""
        config = CloudConfig(api_key="sk-test-key")
        assert config.api_key == "sk-test-key"
        assert config.endpoint == "https://api.splinter.dev"
        assert config.sync_interval == 1.0
        assert config.auto_reconnect is True
        assert config.enable_telemetry is True

    def test_custom_config(self):
        """Test custom config values."""
        config = CloudConfig(
            api_key="sk-custom",
            endpoint="https://custom.api.com",
            sync_interval=0.5,
            auto_reconnect=False,
            project_id="proj-123",
        )
        assert config.api_key == "sk-custom"
        assert config.endpoint == "https://custom.api.com"
        assert config.sync_interval == 0.5
        assert config.auto_reconnect is False
        assert config.project_id == "proj-123"


class TestCloudClient:
    """Tests for CloudClient."""

    def test_create_client(self):
        """Test creating a cloud client."""
        client = CloudClient(api_key="sk-test-key")
        assert client.config.api_key == "sk-test-key"
        assert client.is_connected is False

    def test_client_with_options(self):
        """Test creating client with custom options."""
        client = CloudClient(
            api_key="sk-test-key",
            endpoint="https://custom.api.com",
            sync_interval=2.0,
        )
        assert client.config.endpoint == "https://custom.api.com"
        assert client.config.sync_interval == 2.0

    def test_register_object(self):
        """Test registering objects for sync."""
        client = CloudClient(api_key="sk-test-key")
        mock_gateway = {"test": "gateway"}
        client.register("gateway", mock_gateway)
        assert client._local_refs["gateway"] == mock_gateway

    def test_event_callbacks(self):
        """Test event callback registration."""
        client = CloudClient(api_key="sk-test-key")
        events_received = []

        def on_event(data):
            events_received.append(data)

        client.on("test_event", on_event)
        client._emit("test_event", {"key": "value"})

        assert len(events_received) == 1
        assert events_received[0] == {"key": "value"}

    def test_remove_callback(self):
        """Test removing event callbacks."""
        client = CloudClient(api_key="sk-test-key")
        events_received = []

        def on_event(data):
            events_received.append(data)

        client.on("test_event", on_event)
        client.off("test_event", on_event)
        client._emit("test_event", {"key": "value"})

        assert len(events_received) == 0


class TestSyncEvent:
    """Tests for SyncEvent."""

    def test_create_event(self):
        """Test creating a sync event."""
        event = SyncEvent(
            type=SyncEventType.AGENT_STARTED,
            agent_id="agent-1",
            payload={"provider": "openai", "model": "gpt-4o"},
        )
        assert event.type == SyncEventType.AGENT_STARTED
        assert event.agent_id == "agent-1"
        assert event.payload["provider"] == "openai"
        assert event.timestamp > 0

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = SyncEvent(
            type=SyncEventType.LOOP_DETECTED,
            agent_id="agent-1",
            payload={"pattern": "repeated"},
        )
        d = event.to_dict()
        assert d["type"] == "loop_detected"
        assert d["agent_id"] == "agent-1"
        assert d["payload"]["pattern"] == "repeated"


class TestStateSync:
    """Tests for StateSync."""

    def test_create_sync(self):
        """Test creating state sync."""
        config = CloudConfig(api_key="sk-test")
        sync = StateSync(config)
        assert sync.config == config

    def test_register_objects(self):
        """Test registering objects for sync."""
        config = CloudConfig(api_key="sk-test")
        sync = StateSync(config)

        mock_workflow = {"id": "wf-1"}
        sync.register("workflow", mock_workflow)
        assert sync._registered_objects["workflow"] == mock_workflow

    def test_queue_event(self):
        """Test queuing events."""
        config = CloudConfig(api_key="sk-test")
        sync = StateSync(config)

        event = SyncEvent(
            type=SyncEventType.AGENT_COMPLETED,
            agent_id="agent-1",
        )
        sync.queue_event(event)
        assert len(sync._event_queue) == 1

    def test_convenience_methods(self):
        """Test convenience methods for common events."""
        config = CloudConfig(api_key="sk-test")
        sync = StateSync(config)

        sync.agent_started("agent-1", "openai", "gpt-4o")
        sync.agent_completed("agent-1", {"result": "done"})
        sync.agent_failed("agent-1", "Error message")
        sync.loop_detected("agent-1", "repeated output")
        sync.deadlock_detected(["agent-1", "agent-2"], "mutual wait")
        sync.bottleneck_detected("agent-1", 30.0, "waiting for data")

        assert len(sync._event_queue) == 6


class TestCommand:
    """Tests for Command."""

    def test_create_command(self):
        """Test creating a command."""
        cmd = Command(
            id="cmd-1",
            type=CommandType.PAUSE_AGENT,
            payload={"agent_id": "agent-1"},
        )
        assert cmd.id == "cmd-1"
        assert cmd.type == CommandType.PAUSE_AGENT
        assert cmd.payload["agent_id"] == "agent-1"

    def test_command_from_dict(self):
        """Test creating command from dictionary."""
        data = {
            "id": "cmd-2",
            "type": "stop_agent",
            "payload": {"agent_id": "agent-2"},
        }
        cmd = Command.from_dict(data)
        assert cmd.id == "cmd-2"
        assert cmd.type == CommandType.STOP_AGENT
        assert cmd.payload["agent_id"] == "agent-2"

    def test_all_command_types(self):
        """Test all command types are accessible."""
        assert CommandType.PAUSE_AGENT.value == "pause_agent"
        assert CommandType.RESUME_AGENT.value == "resume_agent"
        assert CommandType.STOP_AGENT.value == "stop_agent"
        assert CommandType.GLOBAL_STOP.value == "global_stop"
        assert CommandType.UPDATE_RULES.value == "update_rules"
        assert CommandType.UPDATE_LIMITS.value == "update_limits"
        assert CommandType.ROLLBACK.value == "rollback"
        assert CommandType.BREAK_LOOP.value == "break_loop"


class TestCommandHandler:
    """Tests for CommandHandler."""

    @pytest.mark.asyncio
    async def test_create_handler(self):
        """Test creating command handler."""
        config = CloudConfig(api_key="sk-test")

        async def mock_handler(cmd):
            return {"success": True}

        handler = CommandHandler(config, mock_handler)
        assert handler.config == config

    @pytest.mark.asyncio
    async def test_get_result(self):
        """Test getting command result."""
        config = CloudConfig(api_key="sk-test")

        async def mock_handler(cmd):
            return {"success": True}

        handler = CommandHandler(config, mock_handler)
        # No result yet
        assert handler.get_result("cmd-1") is None


class TestSplinterCloudIntegration:
    """Tests for Splinter class cloud integration."""

    def test_splinter_without_cloud(self):
        """Test Splinter works without cloud API key."""
        from splinter import Splinter

        s = Splinter()
        assert s.cloud is None
        assert s.is_cloud_connected is False

    def test_splinter_with_cloud_key(self):
        """Test Splinter initializes cloud client with API key."""
        from splinter import Splinter

        s = Splinter(api_key="sk-test-cloud-key")
        assert s.cloud is not None
        assert s.cloud.config.api_key == "sk-test-cloud-key"
        # Not connected yet (async connect not called)
        assert s.is_cloud_connected is False

    @pytest.mark.asyncio
    async def test_connect_cloud(self):
        """Test connecting to cloud."""
        from splinter import Splinter

        s = Splinter()
        assert s.cloud is None

        # Connect with API key
        # Note: This will fail with "invalid API key" since we're not using a real one
        # but it tests the flow
        with pytest.raises(ValueError):
            await s.connect_cloud()  # No API key

    @pytest.mark.asyncio
    async def test_disconnect_cloud(self):
        """Test disconnecting from cloud."""
        from splinter import Splinter

        s = Splinter(api_key="sk-test-key")
        await s.disconnect_cloud()
        assert s.cloud is None


class TestCloudEventTypes:
    """Tests for sync event types."""

    def test_agent_events(self):
        """Test agent-related event types."""
        assert SyncEventType.AGENT_STARTED.value == "agent_started"
        assert SyncEventType.AGENT_COMPLETED.value == "agent_completed"
        assert SyncEventType.AGENT_FAILED.value == "agent_failed"
        assert SyncEventType.AGENT_PAUSED.value == "agent_paused"
        assert SyncEventType.AGENT_RESUMED.value == "agent_resumed"

    def test_control_events(self):
        """Test control-related event types."""
        assert SyncEventType.BUDGET_UPDATE.value == "budget_update"
        assert SyncEventType.RATE_LIMIT_HIT.value == "rate_limit_hit"
        assert SyncEventType.CIRCUIT_OPEN.value == "circuit_open"
        assert SyncEventType.LOOP_DETECTED.value == "loop_detected"
        assert SyncEventType.RULE_TRIGGERED.value == "rule_triggered"

    def test_coordination_events(self):
        """Test coordination-related event types."""
        assert SyncEventType.STATE_UPDATE.value == "state_update"
        assert SyncEventType.CHECKPOINT_CREATED.value == "checkpoint_created"
        assert SyncEventType.HANDOFF_STARTED.value == "handoff_started"
        assert SyncEventType.GOAL_PROGRESS.value == "goal_progress"
        assert SyncEventType.DEADLOCK_DETECTED.value == "deadlock_detected"
        assert SyncEventType.BOTTLENECK_DETECTED.value == "bottleneck_detected"
