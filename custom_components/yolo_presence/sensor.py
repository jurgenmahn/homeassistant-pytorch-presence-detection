"""Sensor platform for YOLO Presence Detection."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional

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
from homeassistant.helpers.update_coordinator import CoordinatorEntity
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

_LOGGER = logging.getLogger(__name__)


@dataclass
class YoloPresenceSensorEntityDescription(SensorEntityDescription):
    """Describes a YOLO Presence sensor entity."""
    value_fn: Callable[[Any], Any] = None


SENSOR_TYPES: tuple[YoloPresenceSensorEntityDescription, ...] = (
    YoloPresenceSensorEntityDescription(
        key=ATTR_HUMAN_COUNT,
        name="People Count",
        icon="mdi:account-group",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda detector: detector.people_count,
    ),
    YoloPresenceSensorEntityDescription(
        key=ATTR_PET_COUNT,
        name="Pet Count",
        icon="mdi:paw",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda detector: detector.pet_count,
    ),
    YoloPresenceSensorEntityDescription(
        key=ATTR_LAST_DETECTION,
        name="Last Detection",
        device_class=SensorDeviceClass.TIMESTAMP,
        entity_category=EntityCategory.DIAGNOSTIC,
        icon="mdi:clock-outline",
        value_fn=lambda detector: detector.last_detection_time,
    ),
    YoloPresenceSensorEntityDescription(
        key=ATTR_MODEL_TYPE,
        name="Model Type",
        icon="mdi:chip",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda detector: detector.model_name,
    ),
    YoloPresenceSensorEntityDescription(
        key=ATTR_CONNECTION_STATUS,
        name="Connection Status",
        icon="mdi:connection",
        entity_category=EntityCategory.DIAGNOSTIC,
        value_fn=lambda detector: detector.connection_status,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up YOLO Presence sensors based on a config entry."""
    detector = hass.data[DOMAIN][entry.entry_id]
    
    # Add all sensor types
    async_add_entities(
        YoloPresenceSensor(detector, entry, description)
        for description in SENSOR_TYPES
    )


class YoloPresenceSensor(SensorEntity):
    """Representation of a YOLO Presence sensor."""

    _attr_has_entity_name = True
    entity_description: YoloPresenceSensorEntityDescription

    def __init__(
        self, 
        detector: Any, 
        entry: ConfigEntry,
        description: YoloPresenceSensorEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        self.entity_description = description
        self._detector = detector
        self._entry = entry
        
        # Set up entity attributes
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.data.get(CONF_NAME, "YOLO Presence"),
            manufacturer="Ultralytics",
            model=f"YOLO Presence ({detector.model_name})",
            sw_version="1.0",
        )
        
        # Set initial state
        self._update_state()
        
    async def async_added_to_hass(self) -> None:
        """Register callbacks when entity is added."""
        # Register callback for state changes
        self.async_on_remove(
            self._detector.register_update_callback(self._update_callback)
        )
        
    @callback
    def _update_callback(self) -> None:
        """Update the sensor state when detector state changes."""
        self._update_state()
        self.async_write_ha_state()
        
    def _update_state(self) -> None:
        """Update the state from the detector."""
        if self.entity_description.value_fn:
            self._attr_native_value = self.entity_description.value_fn(self._detector)
            
    @property
    def available(self) -> bool:
        """Return if the sensor is available."""
        # For connection status, always available
        if self.entity_description.key == ATTR_CONNECTION_STATUS:
            return True
            
        # For other sensors, available only when connected
        return self._detector.connection_status == "connected"