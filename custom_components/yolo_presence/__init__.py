"""YOLO Presence Detection integration for Home Assistant."""
import asyncio
import logging
import os

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_interval
import datetime as dt

from .const import (
    DOMAIN, 
    SCAN_INTERVAL, 
    DATA_YOLO_PRESENCE,
    ATTR_DEVICE_ID,
    CONF_PROCESSING_SERVER,
    CONF_PROCESSING_SERVER_PORT,
    CONF_USE_TCP_CONNECTION,
    DEFAULT_PROCESSING_SERVER_PORT,
    DEFAULT_USE_TCP_CONNECTION,
)
from .api_client import YoloProcessingApiClient

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.BINARY_SENSOR, Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up YOLO Presence from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    # Get configuration
    config = {**entry.data, **entry.options}
    
    # Create API client instance
    server_url = config.get(CONF_PROCESSING_SERVER)
    server_port = config.get(CONF_PROCESSING_SERVER_PORT, DEFAULT_PROCESSING_SERVER_PORT)
    use_tcp = config.get(CONF_USE_TCP_CONNECTION, DEFAULT_USE_TCP_CONNECTION)
    
    client = YoloProcessingApiClient(
        hass=hass,
        server_url=server_url,
        detector_id=entry.entry_id,
        update_interval=1,
        server_port=server_port,
        use_tcp=use_tcp,
    )

    # Store client instance
    hass.data[DOMAIN][entry.entry_id] = client
    
    # Initialize the client and connect to the processing server
    if not await client.async_initialize(config):
        _LOGGER.warning(f"Failed to initialize YOLO processing client for {entry.title}. Check the processing server status.")
        return False
    
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
        # Stop the API client
        client = hass.data[DOMAIN][entry.entry_id]
        await client.async_shutdown()
        
        # Remove the entry
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)