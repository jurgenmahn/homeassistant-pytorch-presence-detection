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
    hass.data.setdefault(DOMAIN, {})

    # Get configuration
    config = {**entry.data, **entry.options}

    # Create API client instance
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
    
    client = YoloProcessingApiClient(
        hass=hass,
        server_url=server_url,
        detector_id=entry.entry_id,
        update_interval=detection_interval,  # Use the configured detection interval
        server_port=server_port,
        use_tcp=use_tcp,
    )

    # Store client instance
    hass.data[DOMAIN][entry.entry_id] = client

    # Initialize the client and connect to the processing server
    if not await client.async_initialize(config):
        _LOGGER.warning(
            f"Failed to initialize YOLO processing client for {entry.title}. Check the processing server status."
        )
        return False

    # Load integration platforms (sensors)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register reload handler
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    # Set up periodic detector check
    async def check_detector_status(now: dt.datetime = None) -> None:
        """Check if detector is running on server and recreate if needed."""
        _LOGGER.debug(f"Polling detector {entry.title} with configured interval of {detection_interval}s")

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
    _LOGGER.info(f"Reloading entry {entry.title} to apply configuration changes")
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
