"""Binary sensor platform for YOLO Presence Detection."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    DOMAIN,
    CONF_NAME,
    ATTR_HUMANS_DETECTED,
    ATTR_PETS_DETECTED,
)
from .api_client import YoloProcessingApiClient

_LOGGER = logging.getLogger(__name__)


@dataclass
class YoloPresenceBinarySensorEntityDescription(BinarySensorEntityDescription):
    """Describes YOLO Presence binary sensor entity."""
    is_on_fn: Callable[[Any], bool] = None


BINARY_SENSOR_TYPES: tuple[YoloPresenceBinarySensorEntityDescription, ...] = (
    YoloPresenceBinarySensorEntityDescription(
        key=ATTR_HUMANS_DETECTED,
        name="Person Detected",
        device_class=BinarySensorDeviceClass.PRESENCE,
        icon="mdi:account",
        is_on_fn=lambda client: client.people_detected,
    ),
    YoloPresenceBinarySensorEntityDescription(
        key=ATTR_PETS_DETECTED,
        name="Pet Detected",
        device_class=BinarySensorDeviceClass.PRESENCE,
        icon="mdi:paw",
        is_on_fn=lambda client: client.pets_detected,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up YOLO Presence binary sensors based on a config entry."""
    client = hass.data[DOMAIN][entry.entry_id]
    
    async_add_entities(
        YoloPresenceBinarySensor(client, entry, description)
        for description in BINARY_SENSOR_TYPES
    )


class YoloPresenceBinarySensor(BinarySensorEntity):
    """YOLO Presence binary sensor."""

    _attr_has_entity_name = True
    entity_description: YoloPresenceBinarySensorEntityDescription

    def __init__(
        self, 
        client: YoloProcessingApiClient,
        entry: ConfigEntry,
        description: YoloPresenceBinarySensorEntityDescription,
    ) -> None:
        """Initialize the binary sensor."""
        self.entity_description = description
        self._client = client
        self._entry = entry
        
        # Entity attributes
        self._attr_unique_id = f"{entry.entry_id}_{description.key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.data.get(CONF_NAME, "YOLO Presence"),
            manufacturer="Ultralytics",
            model=f"YOLO Presence (Remote Processing)",
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
        """Update the binary sensor state when client state changes."""
        self._update_state()
        self.async_write_ha_state()
    
    def _update_state(self) -> None:
        """Update the state from the client."""
        if self.entity_description.is_on_fn:
            self._attr_is_on = self.entity_description.is_on_fn(self._client)

    @property
    def available(self) -> bool:
        """Return if the binary sensor is available."""
        return self._client.available