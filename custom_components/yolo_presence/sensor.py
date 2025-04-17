"""Sensor platform for YOLO Presence Detection."""

from __future__ import annotations

import logging
from typing import Any, Callable

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
import homeassistant.util.dt as dt_util

from .const import (
    DOMAIN,
    CONF_NAME,
    ATTR_HUMAN_COUNT,
    ATTR_PET_COUNT,
    ATTR_LAST_DETECTION,
    ATTR_MODEL_TYPE,
    ATTR_CONNECTION_STATUS,
)
from .api_client import YoloProcessingApiClient

_LOGGER = logging.getLogger(__name__)


class YoloPresenceSensorEntityDescription:
    """Describes a YOLO Presence sensor entity.

    This class is a wrapper around SensorEntityDescription that adds value_fn.
    """

    def __init__(
        self,
        *,
        key: str,
        name: str | None = None,
        device_class: SensorDeviceClass | None = None,
        state_class: SensorStateClass | None = None,
        icon: str | None = None,
        entity_category: EntityCategory | None = None,
        value_fn: Callable[[Any], Any] = lambda _: None,
    ) -> None:
        """Initialize the sensor description."""
        self.entity_description = SensorEntityDescription(
            key=key,
            name=name,
            device_class=device_class,
            state_class=state_class,
            icon=icon,
            entity_category=entity_category,
        )
        self.key = key
        self.name = name
        self.device_class = device_class
        self.state_class = state_class
        self.icon = icon
        self.entity_category = entity_category
        self.value_fn = value_fn


SENSOR_TYPES: tuple[YoloPresenceSensorEntityDescription, ...] = (
    YoloPresenceSensorEntityDescription(
        key=ATTR_HUMAN_COUNT,
        name="People Count",
        icon="mdi:account-group",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda client: client.people_count,
    ),
    YoloPresenceSensorEntityDescription(
        key=ATTR_PET_COUNT,
        name="Pet Count",
        icon="mdi:paw",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda client: client.pet_count,
    ),
    YoloPresenceSensorEntityDescription(
        key=ATTR_LAST_DETECTION,
        name="Last Detection",
        device_class=SensorDeviceClass.TIMESTAMP,
        entity_category=EntityCategory.DIAGNOSTIC,
        icon="mdi:clock-outline",
        value_fn=lambda client: (
            dt_util.utc_from_timestamp(client.last_update_time)
            if client.last_update_time
            else None
        ),
    ),
    YoloPresenceSensorEntityDescription(
        key=ATTR_MODEL_TYPE,
        name="Model Type",
        icon="mdi:chip",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda client: client.model_name,
    ),
    YoloPresenceSensorEntityDescription(
        key=ATTR_CONNECTION_STATUS,
        name="Connection Status",
        icon="mdi:connection",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda client: client.connection_status,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up YOLO Presence sensors based on a config entry."""
    client = hass.data[DOMAIN][entry.entry_id]

    # Add all sensor types
    async_add_entities(
        YoloPresenceSensor(client, entry, description) for description in SENSOR_TYPES
    )


class YoloPresenceSensor(SensorEntity):
    """Representation of a YOLO Presence sensor."""

    _attr_has_entity_name = True

    def __init__(
        self,
        client: YoloProcessingApiClient,
        entry: ConfigEntry,
        description: YoloPresenceSensorEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        self.entity_description = description.entity_description
        self._description = description
        self._client = client
        self._entry = entry

        # Set up entity attributes
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.data.get(CONF_NAME, "YOLO Presence"),
            manufacturer="Ultralytics",
            model="YOLO Presence (Remote Processing)",
            sw_version="1.0",
        )

        # Set initial state
        self._update_state()

    async def async_added_to_hass(self) -> None:
        """Register callbacks when entity is added."""
        # Register callback for state changes
        self.async_on_remove(
            self._client.register_update_callback(self._update_callback)
        )

    @callback
    def _update_callback(self) -> None:
        """Update the sensor state when client state changes."""
        self._update_state()
        self.async_write_ha_state()

    def _update_state(self) -> None:
        """Update the state from the client."""
        # Call the function from the description wrapper
        self._attr_native_value = self._description.value_fn(self._client)

    @property
    def available(self) -> bool:
        """Return if the sensor is available."""
        # For connection status, always available
        if self._description.key == ATTR_CONNECTION_STATUS:
            return True

        # For other sensors, available only when connected
        return self._client.available
