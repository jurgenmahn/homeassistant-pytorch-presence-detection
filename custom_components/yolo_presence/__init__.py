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
    CONF_DETECTION_INTERVAL,
    DEFAULT_PROCESSING_SERVER_PORT,
    DEFAULT_USE_TCP_CONNECTION,
    DEFAULT_DETECTION_INTERVAL_CPU,
)
from .api_client import YoloProcessingApiClient

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.BINARY_SENSOR, Platform.SENSOR]
# We'll calculate the actual check interval dynamically based on the config


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up YOLO Presence from a config entry."""
    _LOGGER.debug(f"Setting up entry {entry.title}")
    
    try:
        # Make sure the domain data structure exists
        hass.data.setdefault(DOMAIN, {})
        
        # Check if this entry is already set up (prevent duplicate setup)
        if entry.entry_id in hass.data[DOMAIN]:
            _LOGGER.warning(f"Entry {entry.title} already set up, cleaning up first")
            await async_unload_entry(hass, entry)
        
        # Get configuration by merging data and options
        config = {**entry.data, **entry.options}
        
        # Get required parameters
        server_url = config.get(CONF_PROCESSING_SERVER)
        server_port = config.get(
            CONF_PROCESSING_SERVER_PORT, DEFAULT_PROCESSING_SERVER_PORT
        )
        use_tcp = config.get(CONF_USE_TCP_CONNECTION, DEFAULT_USE_TCP_CONNECTION)
        
        # Get detection interval from config
        detection_interval = config.get(
            CONF_DETECTION_INTERVAL, 
            DEFAULT_DETECTION_INTERVAL_CPU
        )
        
        # Create API client with all necessary parameters
        _LOGGER.debug(f"Creating API client for {entry.title} with detection interval: {detection_interval}s")
        client = YoloProcessingApiClient(
            hass=hass,
            server_url=server_url,
            detector_id=entry.entry_id,
            update_interval=detection_interval,  # Use the configured detection interval
            server_port=server_port,
            use_tcp=use_tcp,
        )
        
        # Initialize the client first before storing it
        _LOGGER.debug(f"Initializing client for {entry.title}")
        if not await client.async_initialize(config):
            _LOGGER.warning(
                f"Failed to initialize YOLO processing client for {entry.title}. Check the processing server status."
            )
            return False
        
        # Only store the client if initialization succeeded
        hass.data[DOMAIN][entry.entry_id] = client
        
        # Load integration platforms (sensors)
        _LOGGER.debug(f"Setting up platforms for {entry.title}")
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
        
        # Register reload handler
        entry.async_on_unload(entry.add_update_listener(async_reload_entry))
        
        # Set up periodic detector check
        async def check_detector_status(now: dt.datetime = None) -> None:
            """Check if detector is running on server and recreate if needed."""
            try:
                _LOGGER.debug(f"Polling detector {entry.title} with configured interval of {detection_interval}s")
                
                # Make sure client still exists
                if entry.entry_id not in hass.data.get(DOMAIN, {}):
                    _LOGGER.warning(f"Client for {entry.title} no longer exists, skipping poll")
                    return
                    
                client = hass.data[DOMAIN][entry.entry_id]
                
                # With HTTP-based polling, we just need to call async_update
                # This will send the poll request to the server
                await client.async_update()
                
                # Check if we're connected after the update
                if not client.is_connected:
                    _LOGGER.debug(f"Client not connected after update for {entry.title}")
                    
                    # Check last update time to see if we're getting updates
                    current_time = dt.datetime.now().timestamp()
                    last_update = client.last_update_time
                    
                    if (
                        not last_update
                        or (current_time - last_update > 60)  # No updates in the last minute
                        or client.connection_status == "disconnected"
                    ):
                        _LOGGER.warning(
                            f"Detector {entry.entry_id} appears to be missing or not running, recreating..."
                        )
                        # Recreate the detector by reinitializing
                        await client.async_initialize(config)
                        _LOGGER.info(f"Detector {entry.entry_id} recreation attempt completed")
            except Exception as ex:
                _LOGGER.error(f"Error in check_detector_status for {entry.title}: {ex}")
        
        # Schedule periodic check based on the configured detection interval
        # Convert detection_interval from seconds to a timedelta
        check_interval = timedelta(seconds=detection_interval)
        _LOGGER.info(f"Setting up detector check with interval: {detection_interval} seconds")
        
        entry.async_on_unload(
            async_track_time_interval(hass, check_detector_status, check_interval)
        )
        
        # Run initial check after a short delay
        async def initial_check() -> None:
            """Run initial detector status check after a delay."""
            await asyncio.sleep(30)  # Wait 30 seconds before first check
            await check_detector_status()
            
        hass.async_create_task(initial_check())
        
        # Setup was successful
        _LOGGER.info(f"Successfully set up {entry.title}")
        return True
        
    except Exception as ex:
        _LOGGER.error(f"Error setting up entry {entry.title}: {ex}")
        # Clean up any partial setup
        if entry.entry_id in hass.data.get(DOMAIN, {}):
            hass.data[DOMAIN].pop(entry.entry_id)
        return False

# This section has been moved inside the try block above


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    _LOGGER.debug(f"Unloading entry {entry.title}")
    
    try:
        # Unload sensors and binary sensors
        unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
        
        # Even if platform unloading fails, try to clean up resources
        if entry.entry_id in hass.data.get(DOMAIN, {}):
            try:
                # Get the client
                client = hass.data[DOMAIN][entry.entry_id]
                
                # Properly shut down the client
                _LOGGER.debug(f"Shutting down client for {entry.title}")
                await client.async_shutdown()
                
                # Remove the entry from the domain data
                hass.data[DOMAIN].pop(entry.entry_id)
            except Exception as ex:
                _LOGGER.error(f"Error shutting down client for {entry.title}: {ex}")
                # Don't raise the error, we want to continue cleanup
        
        return unload_ok
    except Exception as ex:
        _LOGGER.error(f"Error unloading entry {entry.title}: {ex}")
        return False


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    _LOGGER.info(f"Reloading entry {entry.title} to apply configuration changes")
    # Make sure we completely unload the integration before setting up again
    await async_unload_entry(hass, entry)
    
    # Remove any existing data to prevent conflicts
    if entry.entry_id in hass.data[DOMAIN]:
        hass.data[DOMAIN].pop(entry.entry_id)
    
    # Set up the entry again
    await async_setup_entry(hass, entry)
