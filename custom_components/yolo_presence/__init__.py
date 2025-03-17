"""YOLO Presence Detection integration for Home Assistant."""
import asyncio
import logging
import os

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_interval
import datetime as dt

from .const import DOMAIN, SCAN_INTERVAL, DATA_YOLO_PRESENCE, ATTR_DEVICE_ID
from .presence_detector import YoloPresenceDetector

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.BINARY_SENSOR, Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up YOLO Presence from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    # Create detector instance
    detector = YoloPresenceDetector(
        hass=hass,
        config_entry=entry,
    )

    # Store detector instance
    hass.data[DOMAIN][entry.entry_id] = detector
    
    # Start detector
    await detector.async_initialize()
    
    # Load integration platforms (sensors)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    # Register reload handler
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Unload sensors
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok and entry.entry_id in hass.data[DOMAIN]:
        # Stop the detector
        detector = hass.data[DOMAIN][entry.entry_id]
        await detector.async_shutdown()
        
        # Remove the entry
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)