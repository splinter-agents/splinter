"""Schema-enforced handoffs for Splinter.

This module provides strict validation of:
- Agent output structure
- Compatibility with downstream agent input

This eliminates defensive prompting and silent breakage.
Agents now depend on each other safely.
"""

import re
import threading
from typing import Any, Callable

from ..exceptions import SchemaValidationError
from ..schemas import HandoffSchema


class SchemaValidator:
    """Validates data against JSON Schema definitions.

    This is a lightweight JSON Schema validator that handles the
    most common validation cases without external dependencies.

    Example:
        validator = SchemaValidator()

        schema = {
            "type": "object",
            "properties": {
                "results": {"type": "array"},
                "summary": {"type": "string"}
            },
            "required": ["results"]
        }

        validator.validate(data, schema)  # Raises SchemaValidationError if invalid
    """

    def validate(self, data: Any, schema: dict[str, Any]) -> list[str]:
        """Validate data against a JSON Schema.

        Args:
            data: Data to validate.
            schema: JSON Schema definition.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []
        self._validate_value(data, schema, "", errors)
        return errors

    def _validate_value(
        self, value: Any, schema: dict[str, Any], path: str, errors: list[str]
    ) -> None:
        """Recursively validate a value against schema."""
        # Handle type validation
        if "type" in schema:
            expected_type = schema["type"]
            if not self._check_type(value, expected_type):
                errors.append(
                    f"{path or 'root'}: expected type '{expected_type}', "
                    f"got '{type(value).__name__}'"
                )
                return  # Skip further validation if type is wrong

        # Handle enum
        if "enum" in schema:
            if value not in schema["enum"]:
                errors.append(f"{path or 'root'}: value must be one of {schema['enum']}")

        # Handle const
        if "const" in schema:
            if value != schema["const"]:
                errors.append(f"{path or 'root'}: value must be {schema['const']}")

        # Handle object validation
        if schema.get("type") == "object" and isinstance(value, dict):
            self._validate_object(value, schema, path, errors)

        # Handle array validation
        if schema.get("type") == "array" and isinstance(value, list):
            self._validate_array(value, schema, path, errors)

        # Handle string validation
        if schema.get("type") == "string" and isinstance(value, str):
            self._validate_string(value, schema, path, errors)

        # Handle number validation
        if schema.get("type") in ("number", "integer") and isinstance(value, (int, float)):
            self._validate_number(value, schema, path, errors)

    def _check_type(self, value: Any, expected: str | list[str]) -> bool:
        """Check if value matches expected type(s)."""
        if isinstance(expected, list):
            return any(self._check_type(value, t) for t in expected)

        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_types = type_map.get(expected)
        if expected_types is None:
            return True  # Unknown type, assume valid

        return isinstance(value, expected_types)

    def _validate_object(
        self, obj: dict, schema: dict[str, Any], path: str, errors: list[str]
    ) -> None:
        """Validate object properties."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional = schema.get("additionalProperties", True)

        # Check required properties
        for prop in required:
            if prop not in obj:
                errors.append(f"{path}.{prop}" if path else prop + ": required property missing")

        # Validate each property
        for key, value in obj.items():
            prop_path = f"{path}.{key}" if path else key

            if key in properties:
                self._validate_value(value, properties[key], prop_path, errors)
            elif additional is False:
                errors.append(f"{prop_path}: additional property not allowed")
            elif isinstance(additional, dict):
                self._validate_value(value, additional, prop_path, errors)

    def _validate_array(
        self, arr: list, schema: dict[str, Any], path: str, errors: list[str]
    ) -> None:
        """Validate array items."""
        items_schema = schema.get("items")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        unique = schema.get("uniqueItems", False)

        if min_items is not None and len(arr) < min_items:
            errors.append(f"{path or 'root'}: array must have at least {min_items} items")

        if max_items is not None and len(arr) > max_items:
            errors.append(f"{path or 'root'}: array must have at most {max_items} items")

        if unique and len(arr) != len(set(map(str, arr))):
            errors.append(f"{path or 'root'}: array items must be unique")

        if items_schema:
            for i, item in enumerate(arr):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                self._validate_value(item, items_schema, item_path, errors)

    def _validate_string(
        self, s: str, schema: dict[str, Any], path: str, errors: list[str]
    ) -> None:
        """Validate string constraints."""
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")

        if min_length is not None and len(s) < min_length:
            errors.append(f"{path or 'root'}: string must be at least {min_length} characters")

        if max_length is not None and len(s) > max_length:
            errors.append(f"{path or 'root'}: string must be at most {max_length} characters")

        if pattern and not re.match(pattern, s):
            errors.append(f"{path or 'root'}: string must match pattern '{pattern}'")

    def _validate_number(
        self, n: int | float, schema: dict[str, Any], path: str, errors: list[str]
    ) -> None:
        """Validate number constraints."""
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_min = schema.get("exclusiveMinimum")
        exclusive_max = schema.get("exclusiveMaximum")

        if minimum is not None and n < minimum:
            errors.append(f"{path or 'root'}: number must be >= {minimum}")

        if maximum is not None and n > maximum:
            errors.append(f"{path or 'root'}: number must be <= {maximum}")

        if exclusive_min is not None and n <= exclusive_min:
            errors.append(f"{path or 'root'}: number must be > {exclusive_min}")

        if exclusive_max is not None and n >= exclusive_max:
            errors.append(f"{path or 'root'}: number must be < {exclusive_max}")


class HandoffManager:
    """Manages schema-enforced handoffs between agents.

    This class validates agent outputs against schemas before
    passing them to downstream agents.

    Example:
        manager = HandoffManager()

        # Define handoff schema
        manager.register_handoff(
            source="researcher",
            target="summarizer",
            output_schema={
                "type": "object",
                "properties": {
                    "results": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"}
                },
                "required": ["results"]
            }
        )

        # Validate before handoff
        manager.validate_output("researcher", "summarizer", data)  # Raises if invalid
    """

    def __init__(
        self,
        strict: bool = True,
        on_validation_error: Callable[[str, str, list[str]], None] | None = None,
    ):
        """Initialize the handoff manager.

        Args:
            strict: If True, reject on any validation failure.
            on_validation_error: Callback on validation errors (source, target, errors).
        """
        self._strict = strict
        self._on_validation_error = on_validation_error
        self._handoffs: dict[tuple[str, str], HandoffSchema] = {}
        self._output_schemas: dict[str, dict[str, Any]] = {}
        self._validator = SchemaValidator()
        self._lock = threading.RLock()

    def register_handoff(
        self,
        source: str,
        target: str,
        output_schema: dict[str, Any],
        required_fields: list[str] | None = None,
        strict: bool | None = None,
    ) -> None:
        """Register a handoff schema between two agents.

        Args:
            source: Source agent ID.
            target: Target agent ID.
            output_schema: JSON Schema for output validation.
            required_fields: Additional required fields.
            strict: Override default strict setting.
        """
        with self._lock:
            schema = HandoffSchema(
                source_agent_id=source,
                target_agent_id=target,
                output_schema=output_schema,
                required_fields=required_fields or [],
                strict=strict if strict is not None else self._strict,
            )
            self._handoffs[(source, target)] = schema

    def register_handoffs(self, handoffs: list[HandoffSchema]) -> None:
        """Register multiple handoff schemas.

        Args:
            handoffs: List of handoff schemas.
        """
        with self._lock:
            for h in handoffs:
                self._handoffs[(h.source_agent_id, h.target_agent_id)] = h

    def register_output_schema(self, agent_id: str, schema: dict[str, Any]) -> None:
        """Register an output schema for an agent.

        This schema will be used for all handoffs from this agent.

        Args:
            agent_id: The agent ID.
            schema: JSON Schema for output validation.
        """
        with self._lock:
            self._output_schemas[agent_id] = schema

    def validate_output(
        self,
        source: str,
        target: str | None,
        data: Any,
    ) -> bool:
        """Validate agent output for a handoff.

        Args:
            source: Source agent ID.
            target: Target agent ID (None to just validate against source schema).
            data: Output data to validate.

        Returns:
            True if valid.

        Raises:
            SchemaValidationError: If validation fails and strict mode is on.
        """
        with self._lock:
            errors: list[str] = []

            # Check specific handoff schema
            if target:
                key = (source, target)
                if key in self._handoffs:
                    handoff = self._handoffs[key]
                    errors.extend(self._validator.validate(data, handoff.output_schema))

                    # Check additional required fields
                    if isinstance(data, dict):
                        for field in handoff.required_fields:
                            if field not in data:
                                errors.append(f"Required field '{field}' missing")

            # Check agent's output schema
            if source in self._output_schemas:
                errors.extend(self._validator.validate(data, self._output_schemas[source]))

            if errors:
                if self._on_validation_error:
                    self._on_validation_error(source, target or "", errors)

                if self._strict:
                    raise SchemaValidationError(source, errors)

            return len(errors) == 0

    def get_expected_schema(self, source: str, target: str) -> dict[str, Any] | None:
        """Get the expected output schema for a handoff.

        Args:
            source: Source agent ID.
            target: Target agent ID.

        Returns:
            The schema or None if not defined.
        """
        with self._lock:
            key = (source, target)
            if key in self._handoffs:
                return self._handoffs[key].output_schema

            if source in self._output_schemas:
                return self._output_schemas[source]

            return None

    def get_downstream_agents(self, source: str) -> list[str]:
        """Get agents that receive handoffs from a source.

        Args:
            source: Source agent ID.

        Returns:
            List of target agent IDs.
        """
        with self._lock:
            return [target for (src, target) in self._handoffs.keys() if src == source]

    def get_upstream_agents(self, target: str) -> list[str]:
        """Get agents that send handoffs to a target.

        Args:
            target: Target agent ID.

        Returns:
            List of source agent IDs.
        """
        with self._lock:
            return [source for (source, tgt) in self._handoffs.keys() if tgt == target]

    def clear(self) -> None:
        """Clear all registered handoffs."""
        with self._lock:
            self._handoffs.clear()
            self._output_schemas.clear()


def create_schema_from_example(example: dict[str, Any], required: bool = True) -> dict[str, Any]:
    """Create a JSON Schema from an example object.

    This is a helper to quickly create schemas from example data.

    Args:
        example: Example data object.
        required: If True, mark all fields as required.

    Returns:
        JSON Schema dict.

    Example:
        schema = create_schema_from_example({
            "results": ["item1", "item2"],
            "count": 10,
            "metadata": {"source": "web"}
        })
    """

    def infer_type(value: Any) -> dict[str, Any]:
        if value is None:
            return {"type": "null"}
        elif isinstance(value, bool):
            return {"type": "boolean"}
        elif isinstance(value, int):
            return {"type": "integer"}
        elif isinstance(value, float):
            return {"type": "number"}
        elif isinstance(value, str):
            return {"type": "string"}
        elif isinstance(value, list):
            if len(value) > 0:
                return {"type": "array", "items": infer_type(value[0])}
            return {"type": "array"}
        elif isinstance(value, dict):
            properties = {k: infer_type(v) for k, v in value.items()}
            schema: dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                schema["required"] = list(value.keys())
            return schema
        else:
            return {}

    return infer_type(example)
