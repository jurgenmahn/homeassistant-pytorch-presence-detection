"""YOLO Presence Detection integration for Home Assistant."""
import asyncio
import logging
import os
from datetime import timedelta

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
DETECTOR_CHECK_INTERVAL = timedelta(minutes=1)


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
    
    # Set up periodic detector check
    async def check_detector_status(now: dt.datetime = None) -> None:
        """Check if detector is running on server and recreate if needed."""
        _LOGGER.debug(f"Checking detector status for {entry.title}")
        
        if not client.is_connected:
            _LOGGER.debug(f"Client not connected, attempting to connect for {entry.title}")
            await client.async_connect()
            
        if client.is_connected:
            # Request current state to check if detector exists
            _LOGGER.debug(f"Requesting state for detector {entry.entry_id}")
            
            # Send a get_state message to check detector status
            if await client._send_message({"type": "get_state"}):
                # We've sent the request, wait for the response
                # The server will respond with detector_not_found if it doesn't exist
                await asyncio.sleep(1)  # Give server time to process
                
                # If we receive detector_not_found, server will send this as a message
                # Our current connection may be valid but detector might not be running
                # Let's check the last update time to see if we're getting updates
                current_time = dt.datetime.now().timestamp()
                last_update = client.last_update_time
                
                if (not last_update or 
                    (current_time - last_update > 60) or  # No updates in the last minute
                    client.connection_status == "disconnected"):
                    
                    _LOGGER.warning(f"Detector {entry.entry_id} appears to be missing or not running, recreating...")
                    # Recreate the detector by reinitializing
                    await client.async_initialize(config)
                    _LOGGER.info(f"Detector {entry.entry_id} recreation attempt completed")
            else:
                _LOGGER.warning(f"Failed to send state request to check detector {entry.entry_id}")
    
    # Schedule periodic check
    entry.async_on_unload(
        async_track_time_interval(hass, check_detector_status, DETECTOR_CHECK_INTERVAL)
    )
    
    # Run initial check after a short delay
    async def initial_check() -> None:
        """Run initial detector status check after a delay."""
        await asyncio.sleep(30)  # Wait 30 seconds before first check
        await check_detector_status()
    
    hass.async_create_task(initial_check())
    
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